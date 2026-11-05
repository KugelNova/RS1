#!/usr/bin/env python3
"""RS1 Mapping V3: timestamped TF + evidence-aware observed surface grid.

This is an experimental 2.5D surface mapper, not validated ground/obstacle
classification. Vegetation can hide bare earth; unseen ground stays unknown.
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

from terrain_surface_core_v3 import SurfaceGrid


def quaternion_to_matrix(x, y, z, w):
    """Matches the V2 camera-link TF conversion validated in flat/rough world."""
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid TF rotation quaternion')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


class TerrainScannerV3(Node):
    def __init__(self):
        super().__init__('terrain_scanner_v3')
        defaults = {
            'cloud_topic': '/parrot1/camera/depth/points',
            'map_frame': 'parrot1_odom',
            'output_directory': '~/rescue_logs/maps',
            'grid_size_m': 0.5,
            'min_range_m': 0.3,
            'max_range_m': 25.0,
            'point_stride': 20,
            'save_every_s': 5.0,
            'map_min_x': -22.0,
            'map_max_x': 22.0,
            'map_min_y': -22.0,
            'map_max_y': 22.0,
            'min_frames_per_cell': 3,
            'max_height_iqr_m': 1.0,
            'neighbor_jump_m': 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        self.map_frame = str(p('map_frame'))
        self.grid = float(p('grid_size_m'))
        self.min_range = float(p('min_range_m'))
        self.max_range = float(p('max_range_m'))
        self.stride = int(p('point_stride'))
        self.save_every = float(p('save_every_s'))
        self.x_min, self.x_max = float(p('map_min_x')), float(p('map_max_x'))
        self.y_min, self.y_max = float(p('map_min_y')), float(p('map_max_y'))
        if (self.stride < 1 or self.min_range < 0 or self.max_range <= self.min_range
                or self.save_every <= 0 or self.x_min >= self.x_max or self.y_min >= self.y_max
                or not all(map(math.isfinite, (self.min_range, self.max_range, self.save_every,
                                            self.x_min, self.x_max, self.y_min, self.y_max)))):
            raise ValueError('Invalid mapping parameters')
        self.gridmap = SurfaceGrid(
            grid=self.grid, min_frames=int(p('min_frames_per_cell')),
            max_iqr_m=float(p('max_height_iqr_m')),
            neighbor_jump_m=float(p('neighbor_jump_m')))
        out_dir = os.path.expanduser(str(p('output_directory')))
        os.makedirs(out_dir, exist_ok=True)
        self.output_dir = out_dir
        self.filename = self._new_filename()
        self.frames_processed = 0
        self.latest_cloud = None
        self.last_processed_stamp_ns = None
        self.last_log = 0.0
        self.dirty = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cloud_sub = self.create_subscription(
            PointCloud2, str(p('cloud_topic')), self.on_cloud, qos_profile_sensor_data)
        self.process_timer = self.create_timer(0.2, self.process_cloud)
        self.save_timer = self.create_timer(self.save_every, self.save_csv)
        self.get_logger().info(
            f'V3 mapper ready: frame={self.map_frame}, grid={self.grid}m, '
            f'X[{self.x_min}, {self.x_max}] Y[{self.y_min}, {self.y_max}]; '
            f'output={self.filename}')

    def _new_filename(self):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        return os.path.join(self.output_dir, f'terrain_profile_v3_{stamp}.csv')

    def on_cloud(self, msg):
        self.latest_cloud = msg

    def process_cloud(self):
        msg = self.latest_cloud
        if msg is None:
            return
        stamp_ns = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        if stamp_ns == 0 or not msg.header.frame_id:
            self.latest_cloud = None
            self.warn_throttled('Missing cloud timestamp or camera frame')
            return
        if self.last_processed_stamp_ns is not None:
            if stamp_ns == self.last_processed_stamp_ns:
                self.latest_cloud = None
                return
            if stamp_ns < self.last_processed_stamp_ns:
                self.get_logger().warning('Simulation time moved backwards: starting a new V3 map')
                self.save_csv(force=True)
                self.gridmap.cells.clear()
                self.filename = self._new_filename()
                self.last_processed_stamp_ns = None
                self.dirty = False
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp))
        except TransformException as exc:
            self.warn_throttled(f'Waiting for TF: {exc}')
            return
        self.latest_cloud = None
        q, t = tf.transform.rotation, tf.transform.translation
        try:
            rotation = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        except ValueError as exc:
            self.warn_throttled(str(exc))
            return
        translation = np.array((t.x, t.y, t.z), dtype=np.float64)
        raw = []
        try:
            for i, p in enumerate(point_cloud2.read_points(
                    msg, field_names=('x', 'y', 'z'), skip_nans=True)):
                if i % self.stride == 0:
                    raw.append((float(p[0]), float(p[1]), float(p[2])))
        except (ValueError, TypeError) as exc:
            self.warn_throttled(f'Cannot read cloud: {exc}')
            return
        self.last_processed_stamp_ns = stamp_ns
        if not raw:
            return
        xyz = np.asarray(raw, dtype=np.float64)
        dist_sq = np.einsum('ij,ij->i', xyz, xyz)
        valid = (np.isfinite(xyz).all(axis=1) &
                 (dist_sq >= self.min_range**2) & (dist_sq <= self.max_range**2))
        xyz = xyz[valid]
        if not xyz.size:
            return
        world_xyz = xyz @ rotation.T + translation
        valid = (np.isfinite(world_xyz).all(axis=1) &
                 (world_xyz[:, 0] >= self.x_min) & (world_xyz[:, 0] < self.x_max) &
                 (world_xyz[:, 1] >= self.y_min) & (world_xyz[:, 1] < self.y_max) &
                 (world_xyz[:, 2] < translation[2] + 0.5))
        world_xyz = world_xyz[valid]
        mapped = self.gridmap.add_cloud(world_xyz)
        self.frames_processed += 1
        self.dirty = self.dirty or bool(mapped)
        now = time.monotonic()
        if now - self.last_log >= 5:
            self.get_logger().info(
                f'Cloud #{self.frames_processed}: {len(world_xyz)} points, '
                f'{len(self.gridmap.cells)} observed cells')
            self.last_log = now

    def warn_throttled(self, message):
        now = time.monotonic()
        if now - self.last_log >= 5:
            self.get_logger().warning(message)
            self.last_log = now

    def save_csv(self, force=False):
        if not self.gridmap.cells or (not force and not self.dirty):
            return
        rows = self.gridmap.summaries()
        path_tmp = self.filename + '.tmp'
        try:
            with open(path_tmp, 'w', newline='', encoding='utf-8') as fp:
                columns = ['x', 'y', 'observed_z', 'ground_candidate_z',
                           'sample_count', 'frame_count', 'height_iqr_m',
                           'status', 'map_frame']
                writer = csv.DictWriter(fp, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    rec = dict(row, map_frame=self.map_frame)
                    rec['ground_candidate_z'] = (
                        '' if rec['ground_candidate_z'] is None
                        else f'{rec["ground_candidate_z"]:.5f}')
                    rec['observed_z'] = f'{rec["observed_z"]:.5f}'
                    rec['height_iqr_m'] = f'{rec["height_iqr_m"]:.5f}'
                    writer.writerow(rec)
            os.replace(path_tmp, self.filename)
            self.dirty = False
            counts = {}
            for row in rows:
                counts[row['status']] = counts.get(row['status'], 0) + 1
            self.get_logger().info(f'Saved {len(rows)} observed cells {counts}: {self.filename}')
        except OSError as exc:
            self.get_logger().error(f'Could not save V3 map: {exc}')

    def destroy_node(self):
        self.save_csv(force=True)
        return super().destroy_node()


def main():
    rclpy.init()
    node = TerrainScannerV3()
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
