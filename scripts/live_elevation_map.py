"""Live top-down elevation display for RS1's existing PyQt dashboard.

Uses the validated camera-link -> parrot1_odom TF conversion. Displays the
highest observed surface in each 0.5 m cell, including tree canopies and rocks.
No CSV, classification, or external mapping process required.
"""

import json
import math
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException
from tf2_msgs.msg import TFMessage


class LiveElevationMap(QWidget):
    """Qt widget; attaches two ROS subscriptions to the UI's existing node."""

    MIN_X = MIN_Y = -22.0
    MAX_X = MAX_Y = 22.0
    RESOLUTION = 0.5
    VMIN = -10.0
    VMAX = 15.0

    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.node = node
        self.height_grid = np.full((88, 88), -np.inf, dtype=np.float32)
        self.latest_cloud = None
        self.last_stamp_ns = None
        self.last_image = None
        self.total_clouds = 0
        self.tf_buffer = Buffer()
        self.tf_sub = node.create_subscription(
            TFMessage, '/parrot1/tf', self._on_tf, 100)
        self.tf_static_sub = node.create_subscription(
            TFMessage, '/parrot1/tf_static', self._on_tf_static,
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.cloud_sub = node.create_subscription(
            PointCloud2, '/parrot1/camera/depth/points',
            self._on_cloud, qos_profile_sensor_data)

        # RS1_LIVE_MAP_MISSION_OVERLAYS_V1: independent, read-only overlays.
        self.drone_xy = None
        self.drone_yaw = 0.0
        self.flight_trail = []
        self.target_markers = {}
        self.last_odom_stamp_ns = None
        self.odom_sub = node.create_subscription(
            Odometry, '/parrot1/odometry', self._on_odom, qos_profile_sensor_data)
        self.target_sub = node.create_subscription(
            String, '/parrot1/rescue/target_alert', self._on_target_alert, 10)

        layout = QVBoxLayout(self)
        self.image_label = QLabel('Waiting for depth camera and TF...')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(240, 220)
        self.image_label.setStyleSheet('background: #070b15; color: #9babc0;')
        layout.addWidget(self.image_label, 1)
        self.legend = QLabel('LOW −10 m  (blue)     0 m  (green)     HIGH +15 m  (red)')
        self.legend.setAlignment(Qt.AlignCenter)
        self.legend.setText(self.legend.text() + ' | △ Drone   — Trail   ● Targets')
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        self.status = QLabel('Highest observed elevation / cell  |  X: −22..22 m, Y: −22..22 m')
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)
        self.status.hide()  # RS1_UI_QUIET_MAP_STATUS_V1

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_map)
        self.timer.start(750)
        # Redraw overlays even when a new depth frame has not yet arrived.
        self.overlay_timer = QTimer(self)
        self.overlay_timer.timeout.connect(self._scale_image)
        self.overlay_timer.start(250)
        self._render()

    def _on_tf(self, msg):
        for transform in msg.transforms:
            self.tf_buffer.set_transform(transform, 'live_elevation')

    def _on_tf_static(self, msg):
        for transform in msg.transforms:
            self.tf_buffer.set_transform_static(transform, 'live_elevation')

    def _on_cloud(self, msg):
        # Wait for the matching TF instead of replacing the pending cloud.
        if self.latest_cloud is None:
            self.latest_cloud = msg
            return

        old = self.latest_cloud.header.stamp
        new = msg.header.stamp

        if (new.sec, new.nanosec) < (old.sec, old.nanosec):
            # Simulation restarted.
            self.latest_cloud = msg
            self.last_stamp_ns = None
            self.height_grid.fill(-np.inf)
            self.total_clouds = 0

    @staticmethod
    def _rotation(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n < 1e-9 or not math.isfinite(n):
            raise ValueError('Invalid camera quaternion')
        x, y, z, w = x/n, y/n, z/n, w/n
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ], dtype=np.float64)

    @staticmethod
    def _sample_xyz(msg):
        """Read every 4th pixel directly, avoiding slow Python point loops."""
        fields = {field.name: field for field in msg.fields}
        if (not all(k in fields for k in ('x', 'y', 'z')) or
                not all(fields[k].datatype == PointField.FLOAT32 for k in ('x', 'y', 'z'))):
            raise ValueError('PointCloud2 requires FLOAT32 x/y/z fields')
        if msg.width < 1 or msg.height < 1 or msg.point_step < 12:
            raise ValueError('Invalid point-cloud dimensions')
        offsets = [fields[k].offset for k in ('x', 'y', 'z')]
        if any(off < 0 or off + 4 > msg.point_step for off in offsets):
            raise ValueError('Invalid xyz field offsets')
        endian = '>' if msg.is_bigendian else '<'
        dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [endian + 'f4'] * 3,
            'offsets': offsets,
            'itemsize': msg.point_step,
        })
        view = np.ndarray(
            (msg.height, msg.width), dtype=dtype, buffer=msg.data,
            strides=(msg.row_step, msg.point_step))
        sampled = view[::4, ::4]
        return np.column_stack([sampled[k].reshape(-1) for k in ('x', 'y', 'z')]).astype(np.float64)

    def _update_map(self):
        msg = self.latest_cloud
        if msg is None:
            return
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns <= 0 or not msg.header.frame_id:
            self.status.setText('Waiting for timestamped depth cloud...')
            self.latest_cloud = None
            return
        if self.last_stamp_ns == stamp_ns:
            self.latest_cloud = None
            return
        if self.last_stamp_ns is not None and stamp_ns < self.last_stamp_ns:
            # Gazebo reset; don't mix runs in one map.
            self.height_grid.fill(-np.inf)
            self.total_clouds = 0
        try:
            tf = self.tf_buffer.lookup_transform(
                'parrot1_odom', msg.header.frame_id, Time.from_msg(msg.header.stamp))
        except TransformException as exc:
            self.status.setText(f'TF waiting: {str(exc)[:130]}')
            return
        try:
            xyz = self._sample_xyz(msg)
            R = self._rotation(tf.transform.rotation)
        except (ValueError, TypeError, BufferError) as exc:
            self.status.setText(f'Invalid depth cloud: {exc}')
            self.latest_cloud = None
            return
        self.latest_cloud = None
        self.last_stamp_ns = stamp_ns
        if not xyz.size:
            return
        valid = np.isfinite(xyz).all(axis=1)
        dist2 = np.einsum('ij,ij->i', xyz, xyz)
        valid &= (dist2 >= 0.3**2) & (dist2 <= 25.0**2)
        xyz = xyz[valid]
        if not len(xyz):
            return
        translation = tf.transform.translation
        world = xyz @ R.T + np.array([translation.x, translation.y, translation.z])
        valid = (np.isfinite(world).all(axis=1) &
                 (world[:, 0] >= self.MIN_X) & (world[:, 0] < self.MAX_X) &
                 (world[:, 1] >= self.MIN_Y) & (world[:, 1] < self.MAX_Y))
        xyz = world[valid]
        if not len(xyz):
            return
        ix = np.floor((xyz[:, 0] - self.MIN_X) / self.RESOLUTION).astype(np.int32)
        iy = np.floor((xyz[:, 1] - self.MIN_Y) / self.RESOLUTION).astype(np.int32)
        # Height is the highest observed return, not a ground estimate.
        np.maximum.at(self.height_grid, (iy, ix), xyz[:, 2])
        self.total_clouds += 1
        self._render()

    def _on_odom(self, msg):
        # These coordinates already use parrot1_odom, same as the height grid.
        if msg.header.frame_id != 'parrot1_odom':
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        x, y = float(p.x), float(p.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if stamp_ns > 0:
            if (self.last_odom_stamp_ns is not None and
                    stamp_ns < self.last_odom_stamp_ns - 1_000_000_000):
                # Gazebo restarted: don't draw a trail joining different runs.
                self.flight_trail.clear()
                self.target_markers.clear()
                self.drone_xy = None
                self.height_grid.fill(-np.inf)
                self.total_clouds = 0
                self.last_stamp_ns = None
                self.latest_cloud = None
                self._render()
            self.last_odom_stamp_ns = stamp_ns

        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm > 1e-9 and math.isfinite(norm):
            xq, yq, zq, wq = q.x/norm, q.y/norm, q.z/norm, q.w/norm
            self.drone_yaw = math.atan2(
                2 * (wq*zq + xq*yq), 1 - 2 * (yq*yq + zq*zq))
        self.drone_xy = (x, y)
        # Keep a bounded, spatially decimated actual flight trace.
        if (not self.flight_trail or
                math.hypot(x-self.flight_trail[-1][0],
                           y-self.flight_trail[-1][1]) >= 0.25):
            self.flight_trail.append((x, y))
            if len(self.flight_trail) > 2000:
                self.flight_trail = self.flight_trail[-2000:]

    def _on_target_alert(self, msg):
        try:
            alert = json.loads(msg.data)
            if alert.get('frame_id') != 'parrot1_odom':
                return
            tid = int(alert['id'])
            x, y = float(alert['x_m']), float(alert['y_m'])
            if tid < 1 or not (math.isfinite(x) and math.isfinite(y)):
                return
        except (ValueError, TypeError, KeyError, AttributeError):
            return
        # Confirmed rescue alerts only; repeated alerts for the same ID update it.
        self.target_markers[tid] = (x, y)

    def _map_pixel(self, x, y, width, height):
        px = (x - self.MIN_X) / (self.MAX_X - self.MIN_X) * width
        py = (self.MAX_Y - y) / (self.MAX_Y - self.MIN_Y) * height
        return QPointF(px, py)

    def _render(self):
        z = self.height_grid[::-1, :]
        known = np.isfinite(z)
        scale = np.clip((np.where(known, z, self.VMIN) - self.VMIN) /
                        (self.VMAX - self.VMIN), 0.0, 1.0)
        stops = [0, 0.25, 0.5, 0.75, 1]
        colours = np.array([
            [35, 45, 120], [23, 138, 211], [42, 195, 130],
            [246, 216, 70], [204, 45, 45]
        ], dtype=np.float64)
        img = np.empty((*z.shape, 3), dtype=np.uint8)
        for c in range(3):
            img[..., c] = np.interp(scale, stops, colours[:, c]).astype(np.uint8)
        img[~known] = (8, 12, 23)
        qimg = QImage(img.data, img.shape[1], img.shape[0],
                      img.shape[1]*3, QImage.Format_RGB888).copy()
        self.last_image = QPixmap.fromImage(qimg)
        self._scale_image()
        self.status.setText(
            f'{int(known.sum())} / {known.size} mapped cells  |  '
            f'{self.total_clouds} depth frames  |  X/Y ±22 m')

    def _scale_image(self):
        if self.last_image is None or not self.isVisible():
            return
        base = self.last_image.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        # Paint on the scaled map, so icons/labels stay legible at any UI size.
        result = base.copy()
        width, height = result.width(), result.height()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Cyan trail: actual odometry samples, not the planned route.
        if len(self.flight_trail) > 1:
            painter.setPen(QPen(QColor(25, 238, 246, 215), 2))
            for a, b in zip(self.flight_trail[:-1], self.flight_trail[1:]):
                painter.drawLine(
                    self._map_pixel(a[0], a[1], width, height),
                    self._map_pixel(b[0], b[1], width, height))

        # Orange numbered circles: confirmed rescue / target_alert messages.
        for tid, (x, y) in self.target_markers.items():
            if not (self.MIN_X <= x <= self.MAX_X and
                    self.MIN_Y <= y <= self.MAX_Y):
                continue
            pt = self._map_pixel(x, y, width, height)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QColor(235, 91, 35))
            painter.drawEllipse(pt, 12, 12)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                int(pt.x()-12), int(pt.y()-12), 24, 24,
                Qt.AlignCenter, str(tid))

        # White directional triangle: current drone position and heading.
        if self.drone_xy is not None:
            x, y = self.drone_xy
            if self.MIN_X <= x <= self.MAX_X and self.MIN_Y <= y <= self.MAX_Y:
                centre = self._map_pixel(x, y, width, height)
                fx, fy = math.cos(self.drone_yaw), -math.sin(self.drone_yaw)
                sx, sy = -fy, fx
                vertices = QPolygonF([
                    QPointF(centre.x()+14*fx, centre.y()+14*fy),
                    QPointF(centre.x()-9*fx+8*sx, centre.y()-9*fy+8*sy),
                    QPointF(centre.x()-9*fx-8*sx, centre.y()-9*fy-8*sy)
                ])
                painter.setPen(QPen(QColor(5, 15, 35), 2))
                painter.setBrush(QColor(255, 255, 255))
                painter.drawPolygon(vertices)

        painter.end()
        self.image_label.setPixmap(result)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_image()
