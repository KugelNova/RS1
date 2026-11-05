#!/usr/bin/env python3
"""Confirm human rescue targets using paired thermal and RGB-D observations."""

import csv
from collections import deque
from datetime import datetime, timezone
import json
import os
import time

import cv2
import message_filters
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from camera_fusion import decode_depth_image, match_rgbd_thermal, transform_point
from lidar_target_fusion import decode_xyz_points, refine_human_position
from rescue_alert import TARGET_ALERT_TOPIC, qos_profile_rescue_alert
from rescue_population.ground_waypoint import GroundWaypoints
from rgb_detection import RgbDetector
from target_tracks import StableTargetTracker
from thermal_processing import connected_hot_components, decode_mono16, raw_to_celsius
from thermal_shape import classify_hot_silhouette


def rgb_as_bgr(image):
    encoding = image.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(encoding)
    if not channels or image.width < 1 or image.height < 1 or image.step < image.width * channels:
        raise ValueError(f'Unsupported RGB image encoding or dimensions: {image.encoding!r}')
    if len(image.data) < image.step * image.height:
        raise ValueError('RGB image data is shorter than its declared row strides')
    array = np.ndarray((image.height, image.width, channels), dtype=np.uint8,
                       buffer=image.data, strides=(image.step, channels, 1))
    if encoding == 'bgr8':
        return array
    conversions = {'rgb8': cv2.COLOR_RGB2BGR, 'rgba8': cv2.COLOR_RGBA2BGR,
                   'bgra8': cv2.COLOR_BGRA2BGR}
    return cv2.cvtColor(array, conversions[encoding])


def pose_tuple(stamped):
    translation = stamped.transform.translation
    rotation = stamped.transform.rotation
    return ((translation.x, translation.y, translation.z),
            (rotation.x, rotation.y, rotation.z, rotation.w))


class TargetLocalizer(Node):
    def __init__(self):
        super().__init__('target_localizer')
        self.declare_parameter('thermal_topic', '/parrot1/camera/thermal/image')
        self.declare_parameter('rgb_topic', '/parrot1/camera/image')
        self.declare_parameter('depth_topic', '/parrot1/camera/depth/image')
        self.declare_parameter('lidar_topic', '/parrot1/rescue/lidar/points')
        self.declare_parameter('lidar_frame', 'parrot1_rescue_lidar_link')
        self.declare_parameter('ground_mesh', '')
        self.declare_parameter('ground_frame', 'parrot1_odom')
        self.declare_parameter('camera_frame', 'parrot1_thermal_camera_link')
        self.declare_parameter('rgb_frame', 'parrot1_camera_link')
        self.declare_parameter('fixed_frame', 'parrot1_odom')
        self.declare_parameter('horizontal_fov_rad', 0.610865)
        self.declare_parameter('rgb_horizontal_fov_rad', 2.0944)
        self.declare_parameter('threshold_kelvin', 305.0)
        self.declare_parameter('min_hot_pixels', 20)
        self.declare_parameter('min_bbox_height_px', 8)
        self.declare_parameter('confirm_frames', 5)
        self.declare_parameter('confirm_span_s', 0.35)
        self.declare_parameter('dedup_radius_m', 1.5)
        self.declare_parameter('output_directory', '~/rescue_logs')
        self.declare_parameter('max_scatter_m', 0.45)
        self.declare_parameter('candidate_radius_m', 0.8)
        self.declare_parameter('require_unclipped', True)
        self.declare_parameter('require_human_shape', False)

        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.rgb_frame = self.get_parameter('rgb_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.ground_frame = self.get_parameter('ground_frame').value
        self.ground = None
        ground_mesh = self.get_parameter('ground_mesh').value
        if ground_mesh:
            try:
                self.ground = GroundWaypoints.from_mesh(ground_mesh)
            except (OSError, ValueError) as exc:
                self.get_logger().warning(f'Ground rescue waypoints unavailable: {exc}')
        self.thermal_hfov = self.get_parameter('horizontal_fov_rad').value
        self.rgb_hfov = self.get_parameter('rgb_horizontal_fov_rad').value
        self.threshold_raw = round(self.get_parameter('threshold_kelvin').value * 100)
        self.require_human_shape = self.get_parameter('require_human_shape').value
        self.tracker = StableTargetTracker(
            confirm_frames=max(2, self.get_parameter('confirm_frames').value),
            confirm_span_s=max(0.1, self.get_parameter('confirm_span_s').value),
            min_hot_pixels=self.get_parameter('min_hot_pixels').value,
            min_bbox_height_px=self.get_parameter('min_bbox_height_px').value,
            candidate_radius_m=self.get_parameter('candidate_radius_m').value,
            max_scatter_m=self.get_parameter('max_scatter_m').value,
            dedup_radius_m=self.get_parameter('dedup_radius_m').value,
            expire_s=0.65,
            require_unclipped=self.get_parameter('require_unclipped').value,
            required_classification='human',
        )
        package_dir = get_package_share_directory('41068_ignition_bringup')
        self.detector = RgbDetector(package_dir=package_dir)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.frames = deque(maxlen=10)
        self.lidar_scans = deque(maxlen=8)
        self.create_subscription(PointCloud2, self.get_parameter('lidar_topic').value,
                                 self.on_lidar, qos_profile_sensor_data)
        self.last_frame_stamps = (0, 0, 0)
        self.last_warning = 0.0
        self.targets = []
        self.next_id = 1

        self.subscribers = [
            message_filters.Subscriber(
                self, Image, self.get_parameter(parameter).value,
                qos_profile=qos_profile_sensor_data)
            for parameter in ('thermal_topic', 'rgb_topic', 'depth_topic')
        ]
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=12, slop=0.055)
        self.synchronizer.registerCallback(self.on_frames)
        self.timer = self.create_timer(0.05, self.process_frames)

        self.pose_pub = self.create_publisher(PoseStamped, '/parrot1/rescue/target_pose', 10)
        self.alert_pub = self.create_publisher(String, TARGET_ALERT_TOPIC, qos_profile_rescue_alert)
        self.waypoint_pub = self.create_publisher(
            PoseStamped, '/parrot1/rescue/waypoint', qos_profile_rescue_alert)
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
            'classification', 'confidence', 'detector_label', 'shape_aspect_ratio', 'shape_score',
            'location_source', 'lidar_frames', 'waypoint_frame_id',
            'waypoint_x_m', 'waypoint_y_m', 'waypoint_z_m'])
        self.csv_writer.writeheader()
        self.csv_file.flush()
        self.get_logger().info(
            f'Human confirmation ready in {self.fixed_frame}. Saving confirmed alerts to {self.csv_path}')

    def warn_periodically(self, message):
        now = time.monotonic()
        if now - self.last_warning >= 3.0:
            self.get_logger().warning(message)
            self.last_warning = now

    def on_lidar(self, cloud):
        stamp = Time.from_msg(cloud.header.stamp)
        if stamp.nanoseconds > 0:
            self.lidar_scans.append((stamp, cloud))

    def lidar_in_rgb_frame(self, rgb_stamp):
        nearby = [(abs(stamp.nanoseconds - rgb_stamp.nanoseconds), stamp, cloud)
                  for stamp, cloud in self.lidar_scans]
        for delay, stamp, cloud in sorted(nearby, key=lambda item: item[0]):
            if delay > 130_000_000:
                break
            try:
                tf = self.tf_buffer.lookup_transform_full(
                    self.rgb_frame, rgb_stamp, self.lidar_frame, stamp, self.fixed_frame)
            except TransformException:
                continue
            try:
                return stamp, decode_xyz_points(cloud), pose_tuple(tf)
            except ValueError as exc:
                self.warn_periodically(f'Ignoring invalid rescue LiDAR cloud: {exc}')
        return None

    def on_frames(self, thermal, rgb, depth):
        thermal_stamp, rgb_stamp, depth_stamp = (
            Time.from_msg(image.header.stamp) for image in (thermal, rgb, depth))
        # RGB and depth are produced by the same sensor. Avoid using a depth image
        # from a different exposure for projection even if thermal arrived nearby.
        if abs(rgb_stamp.nanoseconds - depth_stamp.nanoseconds) > 5_000_000:
            return
        stamps = (thermal_stamp.nanoseconds, rgb_stamp.nanoseconds, depth_stamp.nanoseconds)
        if any(stamp <= 0 for stamp in stamps):
            return
        self.frames.append((thermal, rgb, depth, thermal_stamp, rgb_stamp, stamps, time.monotonic()))

    def process_frames(self):
        if not self.frames:
            return
        thermal, rgb, depth, thermal_stamp, rgb_stamp, stamps, queued_at = self.frames[0]
        if any(stamp <= previous for stamp, previous in zip(stamps, self.last_frame_stamps)):
            self.frames.popleft()
            return
        try:
            rgb_to_fixed = self.tf_buffer.lookup_transform(self.fixed_frame, self.rgb_frame, rgb_stamp)
            rgb_to_thermal = self.tf_buffer.lookup_transform_full(
                self.camera_frame, thermal_stamp, self.rgb_frame, rgb_stamp, self.fixed_frame)
        except TransformException as exc:
            if time.monotonic() - queued_at > 1.0:
                self.frames.popleft()
                self.warn_periodically(f'Dropping synchronized images without camera TF: {exc}')
            return
        self.frames.popleft()
        self.last_frame_stamps = stamps
        try:
            thermal_image = decode_mono16(thermal)
            components, labels, stats = connected_hot_components(thermal_image, self.threshold_raw)
            if components <= 1:
                return
            depth_image = decode_depth_image(depth)
            if (rgb.height, rgb.width) != depth_image.shape:
                raise ValueError('RGB and registered depth images have different dimensions')
            detections = self.detector.detect(rgb_as_bgr(rgb))
            evidence = match_rgbd_thermal(
                depth_image, detections, labels,
                rgb_to_thermal=pose_tuple(rgb_to_thermal), rgb_to_fixed=pose_tuple(rgb_to_fixed),
                rgb_hfov=self.rgb_hfov, thermal_hfov=self.thermal_hfov)
        except (ValueError, TypeError, cv2.error) as exc:
            self.warn_periodically(f'Cannot process synchronized camera images: {exc}')
            return

        observations = []
        lidar = None
        if any(match['classification'] == 'human' for match in evidence.values()):
            lidar = self.lidar_in_rgb_frame(rgb_stamp)
        for component, match in evidence.items():
            left, top, width, height, count = map(int, stats[component])
            mask = labels[top:top + height, left:left + width] == component
            shape = classify_hot_silhouette(mask) if match['classification'] == 'human' else None
            if self.require_human_shape and shape and shape['classification'] != 'human_like':
                continue
            foot_rows, foot_columns = np.nonzero(mask[-4:])
            foot_u = left + float(foot_columns.mean()) if foot_rows.size else left + width / 2.0
            clipped = (left == 0 or top == 0 or left + width >= thermal.width
                       or top + height >= thermal.height)
            if lidar is not None and match['classification'] == 'human' and not clipped:
                scan_stamp, points, lidar_to_rgb = lidar
                refined = refine_human_position(
                    match['xyz'], points, depth_image, labels, component,
                    lidar_to_rgb=lidar_to_rgb, rgb_to_thermal=pose_tuple(rgb_to_thermal),
                    rgb_to_fixed=pose_tuple(rgb_to_fixed),
                    lidar_stamp_s=scan_stamp.nanoseconds * 1e-9,
                    rgb_stamp_s=rgb_stamp.nanoseconds * 1e-9,
                    rgb_hfov=self.rgb_hfov, thermal_hfov=self.thermal_hfov)
                if refined is not None:
                    match['xyz'] = refined['xyz']
                    match['lidar_stamp_ns'] = scan_stamp.nanoseconds

            hottest = thermal_image[top:top + height, left:left + width][mask].max()
            observations.append(dict(
                match, temp_c=raw_to_celsius(hottest), foot_u=foot_u, foot_v=top + height - 1,
                pixels=count, bbox_height=height, edge_clipped=clipped, shape=shape))

        for confirmed in self.tracker.observe(
                observations, thermal_stamp.nanoseconds * 1e-9, [t['xyz'] for t in self.targets]):
            self.add_target(confirmed, thermal.header.stamp)

    def ground_waypoint(self, xyz, stamp):
        if self.ground is None:
            return None
        if self.ground_frame != self.fixed_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.ground_frame, self.fixed_frame, Time.from_msg(stamp))
                xyz = transform_point(xyz, pose_tuple(tf))
            except TransformException as exc:
                self.warn_periodically(f'Cannot transform rescue target into the ground frame: {exc}')
                return None
        return self.ground.at(xyz)

    def add_target(self, detection, stamp):
        target_id = self.next_id
        self.next_id += 1
        xyz = detection['xyz']
        waypoint = self.ground_waypoint(xyz, stamp)
        self.targets.append({'id': target_id, 'xyz': xyz, 'waypoint': waypoint})
        lidar_frames = detection['lidar_frames']
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        shape = detection['shape']
        record = {
            'id': target_id, 'sim_time_sec': f'{stamp_sec:.3f}', 'frame_id': self.fixed_frame,
            'x_m': f'{xyz[0]:.3f}', 'y_m': f'{xyz[1]:.3f}', 'z_m': f'{xyz[2]:.3f}',
            'max_temp_c': f"{detection['temp_c']:.2f}", 'foot_pixel_u': f"{detection['foot_u']:.1f}",
            'foot_pixel_v': f"{detection['foot_v']:.1f}", 'hot_pixels': detection['pixels'],
            'edge_clipped': detection['edge_clipped'], 'classification': detection['classification'],
            'confidence': round(detection['confidence'], 3), 'detector_label': detection['label'],
            'shape_aspect_ratio': f"{shape['aspect_ratio']:.3f}" if shape else '',
            'shape_score': str(shape['shape_score']) if shape else '',
            'location_source': 'rgbd_lidar' if lidar_frames else 'rgbd',
            'lidar_frames': lidar_frames,
        }
        if waypoint is not None:
            record.update(
                waypoint_frame_id=self.ground_frame,
                waypoint_x_m=waypoint[0], waypoint_y_m=waypoint[1], waypoint_z_m=waypoint[2])
            ground_pose = PoseStamped()
            ground_pose.header.frame_id = self.ground_frame
            ground_pose.header.stamp = stamp
            ground_pose.pose.position.x, ground_pose.pose.position.y, ground_pose.pose.position.z = waypoint
            ground_pose.pose.orientation.w = 1.0
            self.waypoint_pub.publish(ground_pose)
        self.csv_writer.writerow(record)
        self.csv_file.flush()

        pose = PoseStamped()
        pose.header.frame_id = self.fixed_frame
        pose.header.stamp = stamp
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)
        self.alert_pub.publish(String(data=json.dumps(record)))
        self.publish_markers()
        self.get_logger().info(
            f'CONFIRMED HUMAN #{target_id}: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) '
            f'[{self.fixed_frame}], {lidar_frames} matched LiDAR scans')
        if waypoint is not None:
            self.get_logger().info(
                f'Ground waypoint #{target_id}: ({waypoint[0]:.2f}, {waypoint[1]:.2f}, '
                f'{waypoint[2]:.2f}) [{self.ground_frame}]')

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
            label.text = f'HUMAN #{target["id"]}'
            msg.markers.append(label)

            if target['waypoint'] is not None:
                waypoint = Marker()
                waypoint.header.frame_id = self.ground_frame
                waypoint.header.stamp = stamp
                waypoint.ns = 'rescue_ground_waypoints'
                waypoint.id = target['id']
                waypoint.type = Marker.CYLINDER
                waypoint.pose.position.x, waypoint.pose.position.y, waypoint.pose.position.z = target['waypoint']
                waypoint.pose.position.z += 0.04
                waypoint.pose.orientation.w = 1.0
                waypoint.scale.x, waypoint.scale.y, waypoint.scale.z = 0.8, 0.8, 0.08
                waypoint.color.r, waypoint.color.g, waypoint.color.b, waypoint.color.a = 0.1, 0.9, 0.7, 1.0
                msg.markers.append(waypoint)
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
