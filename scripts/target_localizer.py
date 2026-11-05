#!/usr/bin/env python3
"""RS1 thermal victim localizer, ROS 2 Humble.

Reads the mono16 Gazebo thermal stream independently of the original detector,
finds separated warm components, projects each estimated foot pixel onto a
configured horizontal ground plane using timestamped TF, emits one alert per
new location, and saves a per-mission CSV in ~/rescue_logs.

This is an *estimated* localisation proof of concept. It isn't real GPS and
cannot account for uneven terrain, occlusion or targets not touching ground.
"""

import csv
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException

from target_geometry import pixel_to_ground
from target_tracks import StableTargetTracker
from thermal_shape import classify_hot_silhouette, classify_simple_thermal_candidate
from thermal_processing import decode_mono16, connected_hot_components, raw_to_celsius


class TargetLocalizer(Node):
    def __init__(self):
        super().__init__('target_localizer')
        self.declare_parameter('thermal_topic', '/parrot1/camera/thermal/image')
        self.declare_parameter('camera_frame', 'parrot1_thermal_camera_link')
        self.declare_parameter('fixed_frame', 'parrot1_odom')
        self.declare_parameter('horizontal_fov_rad', 0.698132)
        self.declare_parameter('ground_z_m', 0.0)
        self.declare_parameter('threshold_kelvin', 305.0)
        self.declare_parameter('min_hot_pixels', 100)
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('dedup_radius_m', 10)
        self.declare_parameter('max_target_distance_m', 35.0)
        self.declare_parameter('output_directory', '~/rescue_logs')
        self.declare_parameter('confirm_span_s', 0.50)
        self.declare_parameter('max_scatter_m', 0.45)
        self.declare_parameter('candidate_radius_m', 0.8)
        self.declare_parameter('min_bbox_height_px', 24)
        self.declare_parameter('require_unclipped', True)
        self.declare_parameter('require_human_shape', False)  # opt-in to preserve legacy tests
        self.declare_parameter('expected_human_temp_c', 37.0)
        self.declare_parameter('human_temp_tolerance_c', 1.0)
        self.declare_parameter('human_min_aspect_ratio', 1.20)
        self.declare_parameter('human_max_aspect_ratio', 4.0)

        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.hfov = float(self.get_parameter('horizontal_fov_rad').value)
        self.ground_z = float(self.get_parameter('ground_z_m').value)
        self.threshold_raw = int(round(float(self.get_parameter('threshold_kelvin').value) * 100))
        self.min_pixels = int(self.get_parameter('min_hot_pixels').value)
        self.max_distance = float(self.get_parameter('max_target_distance_m').value)
        self.tracker = StableTargetTracker(
            confirm_frames=self.get_parameter('confirm_frames').value,
            confirm_span_s=self.get_parameter('confirm_span_s').value,
            min_hot_pixels=self.get_parameter('min_hot_pixels').value,
            min_bbox_height_px=self.get_parameter('min_bbox_height_px').value,
            candidate_radius_m=self.get_parameter('candidate_radius_m').value,
            max_scatter_m=self.get_parameter('max_scatter_m').value,
            dedup_radius_m=self.get_parameter('dedup_radius_m').value,
            require_unclipped=self.get_parameter('require_unclipped').value)
        self.min_pixels = self.tracker.min_hot_pixels
        self.require_human_shape = bool(self.get_parameter('require_human_shape').value)
        self.expected_human_temp_c = float(self.get_parameter('expected_human_temp_c').value)
        self.human_temp_tolerance_c = float(self.get_parameter('human_temp_tolerance_c').value)
        self.human_min_aspect_ratio = float(self.get_parameter('human_min_aspect_ratio').value)
        self.human_max_aspect_ratio = float(
            self.get_parameter('human_max_aspect_ratio').value
        )

        if (
            not all(math.isfinite(v) for v in (
                self.expected_human_temp_c,
                self.human_temp_tolerance_c,
                self.human_min_aspect_ratio,
                self.human_max_aspect_ratio
            ))
            or self.human_temp_tolerance_c < 0
            or self.human_min_aspect_ratio <= 0
            or self.human_max_aspect_ratio < self.human_min_aspect_ratio
        ):
            raise ValueError('Invalid thermal temperature or H/W ratio limits')
        self.last_shape_event = {}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.frames = deque(maxlen=10)  # hold images briefly until pose TF catches up
        self.targets = []
        self.next_id = 1
        self.last_warning = 0.0
        self.last_unsupported_format = None

        self.create_subscription(
            Image, self.get_parameter('thermal_topic').value,
            self.on_image, qos_profile_sensor_data)
        self.timer = self.create_timer(0.12, self.process_frames)

        self.pose_pub = self.create_publisher(PoseStamped, '/parrot1/rescue/target_pose', 10)
        self.alert_pub = self.create_publisher(String, '/parrot1/rescue/target_alert', 10)
        self.shape_pub = self.create_publisher(String, '/parrot1/rescue/shape_candidate', 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/parrot1/rescue/markers', latched)

        out_dir = os.path.expanduser(self.get_parameter('output_directory').value)
        os.makedirs(out_dir, exist_ok=True)
        mission_stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        self.csv_path = os.path.join(out_dir, f'targets_{mission_stamp}.csv')
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'id', 'sim_time_sec', 'frame_id', 'x_m', 'y_m', 'z_m',
            'max_temp_c', 'foot_pixel_u', 'foot_pixel_v', 'hot_pixels', 'edge_clipped',
            'classification', 'shape_aspect_ratio', 'shape_score',
            'temp_match', 'shape_match', 'classification_reason'])
        self.csv_writer.writeheader()
        self.csv_file.flush()
        self.get_logger().info(f'Localizer ready: fixed_frame={self.fixed_frame}; camera={self.camera_frame}')
        self.get_logger().info(f'Saving detections to {self.csv_path}')
        self.get_logger().info('Human-shape filter: %s' % ('ENABLED' if self.require_human_shape else 'DISABLED'))
        self.get_logger().info(
            'Quality gate: %d distinct frames over >= %.2fs; min %d hot pixels; '
            'unclipped=%s; dedup %.2fm' % (
                self.tracker.confirm_frames, self.tracker.confirm_span_s,
                self.tracker.min_hot_pixels, self.tracker.require_unclipped,
                self.tracker.dedup_radius_m))

    def warn_periodically(self, message):
        now = time.monotonic()
        if now - self.last_warning > 3:
            self.last_warning = now
            self.get_logger().warning(message)

    def on_image(self, image):
        if image.encoding.lower() != 'mono16':
            if image.encoding != self.last_unsupported_format:
                self.get_logger().warning(f'Expected mono16, received {image.encoding!r}; image ignored')
                self.last_unsupported_format = image.encoding
            return
        self.frames.append(image)

    def process_frames(self):
        if not self.frames:
            return
        image = self.frames[0]
        stamp = Time.from_msg(image.header.stamp)
        if stamp.nanoseconds == 0:
            # Static only: images without a timestamp can't be localised accurately during movement.
            self.warn_periodically('Thermal image has a zero timestamp; using latest pose (stationary testing only)')
            stamp = Time()
        try:
            tf = self.tf_buffer.lookup_transform(self.fixed_frame, self.camera_frame, stamp)
        except TransformException as exc:
            self.warn_periodically(f'Waiting for TF {self.fixed_frame} <- {self.camera_frame}: {exc}')
            return
        self.frames.popleft()
        try:
            arr = decode_mono16(image)
            components, labels, stats = connected_hot_components(arr, self.threshold_raw)
        except (ValueError, TypeError, cv2.error) as exc:
            self.warn_periodically(f'Cannot decode thermal image: {exc}')
            return

        tr = tf.transform.translation
        q = tf.transform.rotation
        origin = (tr.x, tr.y, tr.z)
        quaternion = (q.x, q.y, q.z, q.w)
        observations = []

        for component in range(1, components):  # 0 is background
            left, top, width, height, count = [int(s) for s in stats[component]]
            if count < self.min_pixels:
                continue
            shape = classify_hot_silhouette(labels[top:top+height, left:left+width] == component)
            if self.require_human_shape and shape['classification'] != 'human_like':
                stamp_sec = image.header.stamp.sec + image.header.stamp.nanosec * 1e-9
                self.publish_shape_event(shape, (left, top, width, height), stamp_sec)
                continue
                        # Reject invalid thermal candidates before tracking.
            hottest = int(arr[labels == component].max())
            temp_c = raw_to_celsius(hottest)

            gate = self.strict_candidate(temp_c, shape, count)

            if gate['classification'] != 'possible_human':
                stamp_sec = (
                    image.header.stamp.sec
                    + image.header.stamp.nanosec * 1e-9
                )

                metrics = dict(
                    shape,
                    max_temp_c=round(temp_c, 2),
                    temp_match=gate['temp_match'],
                    shape_match=gate['shape_match'],
                    gate_reason=gate['reason'],
                )

                self.publish_shape_event(
                    metrics,
                    (left, top, width, height),
                    stamp_sec
                )
                continue
            bottom_row = top + height - 1
            # Approximate ground-contact pixel: mean of the lowest 4 hot-pixel rows.
            # More accurate than the centroid for a standing target, but not exact.
            y0 = max(top, bottom_row - 3)
            strip = labels[y0:bottom_row + 1, left:left + width]
            ys, xs = np.where(strip == component)
            foot_u = float(left + xs.mean()) if len(xs) else left + width / 2.0
            foot_v = float(bottom_row)
            estimate = pixel_to_ground(
                foot_u, foot_v, image.width, image.height, self.hfov,
                origin, quaternion, self.ground_z, self.max_distance)
            if estimate is None:
                continue
            edge_clipped = (left == 0 or top == 0 or left + width >= image.width or top + height >= image.height)
            observations.append(dict(
                xyz=estimate, temp_c=temp_c, foot_u=foot_u, foot_v=foot_v,
                pixels=count, bbox_height=height, edge_clipped=edge_clipped, shape=shape))

        stamp_sec = image.header.stamp.sec + image.header.stamp.nanosec * 1e-9
        for confirmed in self.tracker.observe(
                observations, stamp_sec, [t['xyz'] for t in self.targets]):
            self.add_target(
                confirmed['xyz'], confirmed['temp_c'],
                confirmed['foot_u'], confirmed['foot_v'],
                confirmed['pixels'], confirmed['edge_clipped'], image.header.stamp, confirmed.get('shape'))

    def strict_candidate(self, temp_c, shape, pixels):
        """
        A candidate is accepted only when temperature AND H/W ratio
        are both inside their configured inclusive ranges.
        """

        try:
            temp_c = float(temp_c)
            ratio = float(shape.get('aspect_ratio', float('nan')))

            temp_ok = (
                math.isfinite(temp_c)
                and abs(temp_c - self.expected_human_temp_c)
                <= self.human_temp_tolerance_c + 1e-9
            )

            ratio_ok = (
                math.isfinite(ratio)
                and self.human_min_aspect_ratio
                <= ratio
                <= self.human_max_aspect_ratio
                and pixels >= self.min_pixels
            )

        except (TypeError, ValueError, OverflowError):
            temp_ok = False
            ratio_ok = False

        if temp_ok and ratio_ok:
            classification = 'possible_human'
            reason = 'temperature_and_ratio_match'

        elif temp_ok:
            classification = 'uncertain'
            reason = 'ratio_out_of_range'

        elif ratio_ok:
            classification = 'uncertain'
            reason = 'temperature_out_of_range'

        else:
            classification = 'other_thermal_target'
            reason = 'temperature_and_ratio_rejected'

        return {
            'classification': classification,
            'temp_match': bool(temp_ok),
            'shape_match': bool(ratio_ok),
            'reason': reason,
        }

    def publish_shape_event(self, metrics, bbox, stamp_sec):
        # Throttle diagnostics per image region to avoid flooding PyQt / ROS.
        key = (metrics['classification'], bbox[0] // 40, bbox[1] // 40)
        if stamp_sec - self.last_shape_event.get(key, -1e9) < 3.0:
            return
        self.last_shape_event[key] = stamp_sec
        payload = dict(metrics, bbox_xywh=list(bbox), sim_time_sec=round(stamp_sec, 3))
        self.shape_pub.publish(String(data=json.dumps(payload)))

    def add_target(self, xyz, temp_c, u, v, pixels, clipped, stamp, shape=None):
        # Compare max component temperature + height/width of the same blob.
        # Final safety check before allocating a target ID.
        metrics = shape or {}

        classification = self.strict_candidate(
            temp_c,
            metrics,
            pixels
        )

        # Both conditions MUST pass.
        if classification['classification'] != 'possible_human':
            self.get_logger().debug(
                f'Rejected {classification["reason"]}: '
                f'{temp_c:.2f} C; '
                f'H/W={metrics.get("aspect_ratio", 0.0)}, '
                f'{pixels} hot pixels '
                f'(no CSV/alert/marker)'
            )
            return
        # RS1_REJECT_OTHER_THERMAL_TARGET_V1
        # Do not allocate an ID, save CSV, publish rescue alert/pose, or add a
        # marker for a target that fails BOTH simple classifier checks.
        if classification['classification'] == 'other_thermal_target':
            self.get_logger().debug(
                f'Rejected other_thermal_target: {temp_c:.2f} C; '
                f'{pixels} hot pixels (no CSV/alert/marker)')
            return
        target_id = self.next_id
        self.next_id += 1
        self.targets.append({'id': target_id, 'xyz': xyz})
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        record = {
            'id': target_id, 'sim_time_sec': f'{stamp_sec:.3f}', 'frame_id': self.fixed_frame,
            'x_m': f'{xyz[0]:.3f}', 'y_m': f'{xyz[1]:.3f}', 'z_m': f'{xyz[2]:.3f}',
            'max_temp_c': f'{temp_c:.2f}', 'foot_pixel_u': f'{u:.1f}',
            'foot_pixel_v': f'{v:.1f}', 'hot_pixels': pixels, 'edge_clipped': clipped,
            'classification': classification['classification'],
            'shape_aspect_ratio': (f"{shape['aspect_ratio']:.3f}" if shape else ''),
            'shape_score': (str(shape['shape_score']) if shape else ''),
            'temp_match': classification['temp_match'],
            'shape_match': classification['shape_match'],
            'classification_reason': classification['reason']}
        self.csv_writer.writerow(record)
        self.csv_file.flush()

        pose = PoseStamped()
        pose.header.frame_id = self.fixed_frame
        pose.header.stamp = stamp
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        alert = String()
        alert.data = json.dumps(record)
        self.alert_pub.publish(alert)
        self.publish_markers()
        self.get_logger().info(
            f'TARGET #{target_id} PING: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) '
            f'[{self.fixed_frame}], max={temp_c:.2f} C, edge_clipped={clipped}, '
            f'class={classification["classification"]}, SAVED')

    def publish_markers(self):
        msg = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for target in self.targets:
            x, y, z = target['xyz']
            marker = Marker()
            marker.header.frame_id = self.fixed_frame
            marker.header.stamp = stamp
            marker.ns = 'rescued_targets'
            marker.id = target['id']
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = x, y, z + 0.5
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = 0.8, 0.8, 0.8
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.1, 0.1, 1.0
            msg.markers.append(marker)

            label = Marker()
            label.header.frame_id = self.fixed_frame
            label.header.stamp = stamp
            label.ns = 'rescued_target_labels'
            label.id = target['id']
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y, z + 1.3
            label.pose.orientation.w = 1.0
            label.scale.z = 0.5
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f'TARGET #{target["id"]}'
            msg.markers.append(label)
        self.marker_pub.publish(msg)

    def destroy_node(self):
        if not self.csv_file.closed:
            self.csv_file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TargetLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
