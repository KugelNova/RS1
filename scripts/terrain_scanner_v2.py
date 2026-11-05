#!/usr/bin/env python3
"""RS1 terrain elevation sampling from Gazebo PointCloud2 + timestamped ROS TF.

Output is the LOWEST OBSERVED SURFACE in each cell, not verified bare-earth
terrain. Trees, people, occlusion, depth noise, and unseen areas require later
filtering and validation.

Run with the TF namespace remappings described in README_mapping_v2.md.
"""

from datetime import datetime, timezone
import csv
import math
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener, TransformException


def quaternion_to_matrix(x, y, z, w):
    """Return rotation matrix R so world_xyz = R @ camera_xyz + t."""
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid TF rotation quaternion')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


class TerrainScannerV2(Node):
    def __init__(self):
        super().__init__('terrain_scanner_v2')
        self.declare_parameter('cloud_topic', '/parrot1/camera/depth/points')
        self.declare_parameter('map_frame', 'parrot1_odom')
        self.declare_parameter('output_directory', '~/rescue_logs/maps')
        self.declare_parameter('grid_size_m', 0.5)
        self.declare_parameter('max_range_m', 25.0)
        self.declare_parameter('min_range_m', 0.3)
        self.declare_parameter('point_stride', 20)
        self.declare_parameter('save_every_s', 5.0)
        self.declare_parameter('map_min_x', -22.0)
        self.declare_parameter('map_max_x', 22.0)
        self.declare_parameter('map_min_y', -22.0)
        self.declare_parameter('map_max_y', 22.0)

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.grid = float(self.get_parameter('grid_size_m').value)
        self.min_range = float(self.get_parameter('min_range_m').value)
        self.max_range = float(self.get_parameter('max_range_m').value)
        self.stride = int(self.get_parameter('point_stride').value)
        self.x_min = float(self.get_parameter('map_min_x').value)
        self.x_max = float(self.get_parameter('map_max_x').value)
        self.y_min = float(self.get_parameter('map_min_y').value)
        self.y_max = float(self.get_parameter('map_max_y').value)
        self.save_every = float(self.get_parameter('save_every_s').value)
        if not all(math.isfinite(v) for v in (self.grid, self.min_range, self.max_range,
                                               self.x_min, self.x_max, self.y_min, self.y_max,
                                               self.save_every)):
            raise ValueError('All mapping parameters must be finite')
        if (self.grid <= 0 or self.min_range < 0 or self.max_range <= self.min_range
                or self.stride < 1 or self.save_every <= 0
                or self.x_min >= self.x_max or self.y_min >= self.y_max):
            raise ValueError('Invalid mapping parameters')

        output_directory = os.path.expanduser(str(self.get_parameter('output_directory').value))
        os.makedirs(output_directory, exist_ok=True)
        self.filename = os.path.join(output_directory,
            'terrain_profile_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.csv')
        # (cell_x, cell_y) -> [lowest_observed_z, total_sample_count, distinct_frame_count]
        self.cells = {}
        self.frames_processed = 0
        self.latest_cloud = None
        self.last_processed_stamp_ns = None
        self.last_log = 0.0
        self.dirty = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cloud_sub = self.create_subscription(
            PointCloud2, str(self.get_parameter('cloud_topic').value),
            self.cloud_callback, qos_profile_sensor_data)
        self.process_timer = self.create_timer(0.2, self.process_cloud)
        self.save_timer = self.create_timer(self.save_every, self.save_csv)
        self.get_logger().info(
            f'Mapper ready. map_frame={self.map_frame}, grid={self.grid}m, '
            f'bounds X[{self.x_min}, {self.x_max}], Y[{self.y_min}, {self.y_max}]. '
            f'Output: {self.filename}')

    def cloud_callback(self, msg):
        # Keep only the newest unprocessed cloud to control CPU/memory usage.
        self.latest_cloud = msg

    def process_cloud(self):
        msg = self.latest_cloud
        if msg is None:
            return
        stamp_ns = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        if stamp_ns == 0:
            self._log_throttled('Point cloud timestamp is zero; refusing unlocated measurements')
            self.latest_cloud = None
            return
        if self.last_processed_stamp_ns is not None:
            if stamp_ns == self.last_processed_stamp_ns:
                self.latest_cloud = None
                return
            if stamp_ns < self.last_processed_stamp_ns:
                self.get_logger().warning('Simulation clock went backwards: saving old map and starting a new one')
                self.save_csv(force=True)
                self.cells.clear()
                self.dirty = False
                self.filename = os.path.join(
                    os.path.dirname(self.filename),
                    'terrain_profile_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.csv')
                self.last_processed_stamp_ns = None

        camera_frame = msg.header.frame_id
        if not camera_frame:
            self._log_throttled('Point cloud has empty frame_id; cannot map')
            self.latest_cloud = None
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, camera_frame, Time.from_msg(msg.header.stamp))
        except TransformException as exc:
            self._log_throttled(f'Waiting for TF {self.map_frame} <- {camera_frame}: {exc}')
            return  # retry the cloud if its timestamp is still available

        self.latest_cloud = None
        q = tf.transform.rotation
        t = tf.transform.translation
        try:
            rotation = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        except ValueError as exc:
            self._log_throttled(str(exc))
            return
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)

        raw = []
        try:
            for i, p in enumerate(point_cloud2.read_points(
                    msg, field_names=('x', 'y', 'z'), skip_nans=True)):
                if i % self.stride:
                    continue
                raw.append((float(p[0]), float(p[1]), float(p[2])))
        except (ValueError, TypeError) as exc:
            self._log_throttled(f'Could not decode PointCloud2: {exc}')
            return
        if not raw:
            self.last_processed_stamp_ns = stamp_ns
            return

        camera_xyz = np.asarray(raw, dtype=np.float64)
        distance_sq = np.einsum('ij,ij->i', camera_xyz, camera_xyz)
        valid = (np.isfinite(camera_xyz).all(axis=1) &
                 (distance_sq >= self.min_range**2) &
                 (distance_sq <= self.max_range**2))
        camera_xyz = camera_xyz[valid]
        if not camera_xyz.size:
            self.last_processed_stamp_ns = stamp_ns
            return
        world_xyz = camera_xyz @ rotation.T + translation
        valid = (np.isfinite(world_xyz).all(axis=1) &
                 (world_xyz[:, 0] >= self.x_min) & (world_xyz[:, 0] < self.x_max) &
                 (world_xyz[:, 1] >= self.y_min) & (world_xyz[:, 1] < self.y_max) &
                 (world_xyz[:, 2] < translation[2] + 0.5))
        world_xyz = world_xyz[valid]

        seen = set()
        for x, y, z in world_xyz:
            key = (math.floor(x / self.grid), math.floor(y / self.grid))
            if key not in self.cells:
                self.cells[key] = [float(z), 1, 0]
            else:
                c = self.cells[key]
                c[0] = min(c[0], float(z))
                c[1] += 1
            seen.add(key)
        for key in seen:
            self.cells[key][2] += 1
        self.frames_processed += 1
        self.dirty = self.dirty or bool(seen)
        self.last_processed_stamp_ns = stamp_ns
        now = time.monotonic()
        if now - self.last_log >= 5:
            self.get_logger().info(
                f'Cloud #{self.frames_processed}: {len(world_xyz)} accepted points, '
                f'{len(self.cells)} mapped cells')
            self.last_log = now

    def _log_throttled(self, text):
        now = time.monotonic()
        if now - self.last_log >= 5:
            self.get_logger().warning(text)
            self.last_log = now

    def save_csv(self, force=False):
        if not self.cells or (not force and not self.dirty):
            return
        temp_path = self.filename + '.tmp'
        try:
            with open(temp_path, 'w', newline='', encoding='utf-8') as fp:
                writer = csv.writer(fp)
                writer.writerow(['x', 'y', 'terrain_z', 'sample_count',
                                 'frame_count', 'map_frame'])
                for (cx, cy), (z, sample_count, frames) in sorted(self.cells.items()):
                    writer.writerow([(cx + 0.5) * self.grid, (cy + 0.5) * self.grid,
                                     f'{z:.5f}', sample_count, frames, self.map_frame])
            os.replace(temp_path, self.filename)
            self.dirty = False
            self.get_logger().info(f'Saved {len(self.cells)} cells to {self.filename}')
        except OSError as exc:
            self.get_logger().error(f'Could not save elevation map: {exc}')

    def destroy_node(self):
        self.save_csv(force=True)
        return super().destroy_node()


def main():
    rclpy.init()
    node = TerrainScannerV2()
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
