#!/usr/bin/env python3

import sys
import csv
import json
import math as _math
from collections import deque
import math
import time
import subprocess
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String

from PyQt5.QtCore import Qt, QTimer, QUrl, QEvent
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QFont,
    QPainter,
    QPen,
    QColor,
    QDesktopServices,
)
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QMessageBox,
    QTabWidget,
    QSplitter,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QHeaderView,
    QLineEdit,
    QCheckBox,
)


from pyqt_rescue_records import rescue_log_dir, list_missions, read_mission, mission_path
from rs1_coverage_route import plan_lawnmower, route_stats
from live_elevation_map import LiveElevationMap

# RS1_PYQT_OPERATOR_DASHBOARD_V2

LINEAR_SPEED = 0.5
ANGULAR_SPEED = 0.8
VERTICAL_SPEED = 0.35  # m/s, manual climb/descent

# Thermal display range.
# 285 K = ~11.9 C
# 315 K = ~41.9 C
THERMAL_MIN_RAW = 28500
THERMAL_MAX_RAW = 31500


class DroneUiNode(Node):

    def __init__(self):
        super().__init__(
            'drone_ui',
            cli_args=['--ros-args',
                      '-r', '/tf:=/parrot1/tf',
                      '-r', '/tf_static:=/parrot1/tf_static'])

        self.latest_rgb_image = None
        self.latest_thermal_image = None
        self.latest_scan = None
        self.live_rescue_alerts = deque(maxlen=100)

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw_deg = 0.0

        self.last_rgb_time = 0.0
        self.last_thermal_time = 0.0
        self.last_odom_time = 0.0
        self.last_lidar_time = 0.0

        self.cmd_pub = self.create_publisher(
            Twist,
            '/rs1/manual/cmd_vel',
            10
        )

        self.create_subscription(
            Image,
            '/parrot1/camera/image',
            self.rgb_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Image,
            '/parrot1/camera/thermal/image',
            self.thermal_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            String,
            '/parrot1/rescue/target_alert',
            self.rescue_alert_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/parrot1/odometry',
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            LaserScan,
            '/parrot1/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.mission_request_pub = self.create_publisher(
            String, '/rs1/control/command', 10)
        self.supervisor_status = {}
        self.last_supervisor_status = 0.0
        self.create_subscription(String, '/rs1/control/status',
                                 self.on_supervisor_status, 10)

        self.get_logger().info('Drone UI ROS node started')

    def on_supervisor_status(self, msg):
        try:
            status = json.loads(msg.data)
            if isinstance(status, dict):
                self.supervisor_status = status
                self.last_supervisor_status = time.monotonic()
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    def supervisor_online(self):
        return time.monotonic() - self.last_supervisor_status < 1.5

    def request(self, action, **kwargs):
        self.mission_request_pub.publish(
            String(data=json.dumps({'action': action, **kwargs})))


    # --------------------------------------------------
    # RESCUE ALERTS (ROS callback; PyQt update occurs on UI timer)
    # --------------------------------------------------

    def rescue_alert_callback(self, msg):
        try:
            obj = json.loads(msg.data)
            alert = {
                'id': int(obj['id']),
                'frame_id': str(obj['frame_id']),
                'x_m': float(obj['x_m']),
                'y_m': float(obj['y_m']),
                'z_m': float(obj['z_m']),
                'max_temp_c': float(obj['max_temp_c']),
                'edge_clipped': str(obj.get('edge_clipped', 'False')).lower() == 'true',
                # RS1_SIMPLE_TEMP_SHAPE_CLASSIFIER_V1
                'classification': str(obj.get('classification', 'thermal_unverified')),
                'shape_aspect_ratio': str(obj.get('shape_aspect_ratio', '')),
            }
            if not alert['frame_id'] or not all(
                _math.isfinite(alert[k]) for k in ('x_m', 'y_m', 'z_m', 'max_temp_c')
            ):
                raise ValueError('Invalid rescue alert data')
            self.live_rescue_alerts.append(alert)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f'Ignored invalid rescue alert: {exc}')


    # --------------------------------------------------
    # RGB CAMERA
    # --------------------------------------------------

    def rgb_callback(self, msg):

        try:
            encoding = msg.encoding.lower()

            if encoding in ('rgb8', 'bgr8'):

                raw = np.frombuffer(
                    msg.data,
                    dtype=np.uint8
                )

                image = raw.reshape(
                    msg.height,
                    msg.step
                )

                image = image[:, :msg.width * 3]

                image = image.reshape(
                    msg.height,
                    msg.width,
                    3
                )

                if encoding == 'bgr8':
                    image = image[:, :, ::-1]

                image = np.ascontiguousarray(image)

                qimage = QImage(
                    image.data,
                    msg.width,
                    msg.height,
                    image.strides[0],
                    QImage.Format_RGB888
                ).copy()

            elif encoding == 'rgba8':

                raw = np.frombuffer(
                    msg.data,
                    dtype=np.uint8
                )

                image = raw.reshape(
                    msg.height,
                    msg.step
                )

                image = image[:, :msg.width * 4]

                image = image.reshape(
                    msg.height,
                    msg.width,
                    4
                )

                image = np.ascontiguousarray(image)

                qimage = QImage(
                    image.data,
                    msg.width,
                    msg.height,
                    image.strides[0],
                    QImage.Format_RGBA8888
                ).copy()

            elif encoding == 'bgra8':

                raw = np.frombuffer(
                    msg.data,
                    dtype=np.uint8
                )

                image = raw.reshape(
                    msg.height,
                    msg.step
                )

                image = image[:, :msg.width * 4]

                image = image.reshape(
                    msg.height,
                    msg.width,
                    4
                )

                image = image[:, :, [2, 1, 0, 3]]

                image = np.ascontiguousarray(image)

                qimage = QImage(
                    image.data,
                    msg.width,
                    msg.height,
                    image.strides[0],
                    QImage.Format_RGBA8888
                ).copy()

            elif encoding == 'mono8':

                raw = np.frombuffer(
                    msg.data,
                    dtype=np.uint8
                )

                image = raw.reshape(
                    msg.height,
                    msg.step
                )

                image = image[:, :msg.width]

                image = np.ascontiguousarray(image)

                qimage = QImage(
                    image.data,
                    msg.width,
                    msg.height,
                    image.strides[0],
                    QImage.Format_Grayscale8
                ).copy()

            else:
                self.get_logger().warning(
                    f'Unsupported RGB encoding: '
                    f'{msg.encoding}'
                )
                return

            self.latest_rgb_image = qimage
            self.last_rgb_time = time.monotonic()

        except Exception as exc:
            self.get_logger().error(
                f'RGB conversion failed: {exc}'
            )


    # --------------------------------------------------
    # THERMAL CAMERA
    # --------------------------------------------------

    def thermal_callback(self, msg):

        try:

            if msg.encoding.lower() not in (
                'mono16',
                '16uc1'
            ):
                self.get_logger().warning(
                    f'Unexpected thermal encoding: '
                    f'{msg.encoding}'
                )
                return

            if msg.is_bigendian:
                dtype = np.dtype('>u2')
            else:
                dtype = np.dtype('<u2')

            raw = np.frombuffer(
                msg.data,
                dtype=dtype
            )

            row_width = msg.step // 2

            thermal = raw.reshape(
                msg.height,
                row_width
            )

            thermal = thermal[:, :msg.width]

            # Convert thermal values to visible grayscale.
            clipped = np.clip(
                thermal,
                THERMAL_MIN_RAW,
                THERMAL_MAX_RAW
            )

            display = (
                (
                    clipped - THERMAL_MIN_RAW
                )
                * 255.0
                /
                (
                    THERMAL_MAX_RAW -
                    THERMAL_MIN_RAW
                )
            )

            display = display.astype(
                np.uint8
            )

            # Invalid pixels stay black.
            display[thermal == 0] = 0

            display = np.ascontiguousarray(
                display
            )

            qimage = QImage(
                display.data,
                msg.width,
                msg.height,
                display.strides[0],
                QImage.Format_Grayscale8
            ).copy()

            self.latest_thermal_image = qimage
            self.last_thermal_time = time.monotonic()

        except Exception as exc:
            self.get_logger().error(
                f'Thermal conversion failed: {exc}'
            )


    # --------------------------------------------------
    # ODOMETRY
    # --------------------------------------------------

    def odom_callback(self, msg):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        self.x = position.x
        self.y = position.y
        self.z = position.z

        siny_cosp = 2.0 * (
            orientation.w * orientation.z +
            orientation.x * orientation.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            orientation.y * orientation.y +
            orientation.z * orientation.z
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        self.yaw_deg = math.degrees(yaw)

        self.last_odom_time = time.monotonic()


    # --------------------------------------------------
    # LIDAR
    # --------------------------------------------------

    def scan_callback(self, msg):

        self.latest_scan = msg
        self.last_lidar_time = time.monotonic()


    # --------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------

    def send_velocity(self, linear_x=0.0, angular_z=0.0, linear_z=0.0):
        """Publish a complete 3-axis manual velocity command."""
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.z = float(linear_z)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.send_velocity(0.0, 0.0, 0.0)


class DroneUI(QWidget):
    """Compact ROS2 operator console. UI-only changes; ROS topics unchanged."""

    SENSOR_TIMEOUT_S = 5.0
    KEY_TO_ACTION = {
        Qt.Key_W: 'forward',
        Qt.Key_S: 'backward',
        Qt.Key_A: 'left',
        Qt.Key_D: 'right',
        Qt.Key_R: 'ascend',
        Qt.Key_F: 'descend',
    }

    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node
        self.manual_enabled = False
        self._held_controls = set()
        self._last_cmd_active = False
        self.live_alert_count = 0
        self.selected_mission_name = None
        self.current_rescue_records = []
        self._history_signature = None
        self._sensor_states = {}
        self._supervisor_proc = None
        self._close_after_hold = False

        self.setWindowTitle('RS1 | Search & Rescue Operator Console')
        self.setMinimumSize(720, 480)
        screen = QApplication.primaryScreen()
        if screen is not None:
            bounds = screen.availableGeometry()
            width = max(720, min(1120, int(bounds.width() * 0.88)))
            height = max(480, min(700, int(bounds.height() * 0.88)))
            self.resize(width, height)
        else:
            self.resize(1050, 660)
        self.setFocusPolicy(Qt.StrongFocus)

        self.build_ui()
        # Intercepts movement keys even when a button or camera selector owns focus.
        QApplication.instance().installEventFilter(self)
        QApplication.instance().applicationStateChanged.connect(self._app_state_changed)

        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.process_ros)
        self.ros_timer.start(20)

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)

        self.command_timer = QTimer(self)
        self.command_timer.timeout.connect(self.publish_current_command)
        self.command_timer.start(100)

        self.mission_refresh_timer = QTimer(self)
        self.mission_refresh_timer.timeout.connect(self.refresh_rescue_history)
        self.mission_refresh_timer.start(1500)
        self.refresh_rescue_history()

    def build_ui(self):
        self.setStyleSheet("""
            QWidget { background: #141b27; color: #e6edf7;
                      font-family: 'DejaVu Sans', Arial; font-size: 12px; }
            QGroupBox { border: 1px solid #303d50; border-radius: 8px;
                        margin-top: 14px; padding: 9px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px;
                               padding: 0 4px; color: #b9c9de; }
            QPushButton { background: #233347; border: 1px solid #40536e;
                          border-radius: 5px; padding: 5px 8px; min-height: 25px; }
            QPushButton:hover { background: #30465e; }
            QPushButton:pressed { background: #405e7d; }
            QPushButton:disabled { background: #1c2735; color: #64748b;
                                   border-color: #29394d; }
            QPushButton#manualArm:checked { background: #294d43; border-color: #40977e; }
            QPushButton#manualStop { background: #70343d; border-color: #b15f68; }
            QPushButton#emergencyAbort { background: #98363b; border: 1px solid #eb7a83;
                                         font-weight: bold; min-height: 30px; }
            QLabel#sensorCanvas { background: #060b12; color: #8191a6;
                                  border: 1px solid #28364b; border-radius: 5px; }
            QLabel#alertIdle { background: #1b2b3b; border-radius: 5px;
                               padding: 8px; color: #afc6dc; }
            QLabel#alertActive { background: #68353b; border-radius: 5px;
                                 padding: 8px; color: #ffffff; font-weight: 600; }
            QLabel#alertAck { background: #24483f; border-radius: 5px;
                              padding: 8px; color: #b7f1da; }
            QComboBox { background: #1e2b3d; border: 1px solid #41536d;
                        border-radius: 5px; padding: 5px; }
            QComboBox QAbstractItemView { background: #1e2b3d; color: #e6edf7; }
            QTabWidget::pane { border: 1px solid #303d50; border-radius: 5px; }
            QTabBar::tab { background: #1a2739; padding: 8px 18px; margin-right: 3px; }
            QTabBar::tab:selected { background: #30465e; }
            QTableWidget { background: #182638; alternate-background-color: #213248;
                           gridline-color: #334155; }
            QHeaderView::section { background: #25374d; color: #dce9f8; padding: 5px;
                                   border: 1px solid #354960; }
            QScrollArea { border: none; background: transparent; }
            QFrame#videoFrame { background: #121d2a; border: 1px solid #2d4054;
                                border-radius: 5px; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(7)

        top = QHBoxLayout()
        title = QLabel('RS1  /  SEARCH & RESCUE')
        title.setFont(QFont('DejaVu Sans', 15, QFont.Bold))
        top.addWidget(title)
        top.addStretch(1)
        self.position_label = QLabel('X --   Y --   ALT --')
        self.position_label.setStyleSheet('color: #d1e0ee; font-weight: 600;')
        top.addWidget(self.position_label)
        self.emergency_button = QPushButton('ABORT / LAND')
        self.emergency_button.setObjectName('emergencyAbort')
        self.emergency_button.setToolTip('Aborts managed flight, then lands at current XY if ground below is confirmed clear')
        self.emergency_button.clicked.connect(self._emergency_abort)
        top.addWidget(self.emergency_button)
        root.addLayout(top)

        self.pages = QTabWidget()
        self.pages.setDocumentMode(True)
        self.pages.addTab(self._dashboard_page(), 'Live Dashboard')
        self.pages.addTab(self._history_page(), 'Rescue History')
        self.pages.currentChanged.connect(self._on_page_changed)
        root.addWidget(self.pages, 1)

        alert_row = QHBoxLayout()
        self.rescue_alert_label = QLabel('No live rescue alerts')
        self.rescue_alert_label.setObjectName('alertIdle')
        self.rescue_alert_label.setWordWrap(True)
        alert_row.addWidget(self.rescue_alert_label, 1)
        self.acknowledge_button = QPushButton('Acknowledge')
        self.acknowledge_button.setEnabled(False)
        self.acknowledge_button.clicked.connect(self.acknowledge_rescue_alert)
        alert_row.addWidget(self.acknowledge_button)
        root.addLayout(alert_row)

    def _dashboard_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(5)
        # Operational actions stay visible above live feeds, even on small screens.
        outer.addWidget(self._mission_dashboard_panel(), 0)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(5)

        selector = QHBoxLayout()
        selector.addWidget(QLabel('Sensor view'))
        self.view_selector = QComboBox()
        self.view_selector.addItems(['RGB + Thermal', 'RGB Camera', 'IR / Thermal', 'LiDAR', 'Elevation Map', 'RGB + Thermal + Elevation'])
        self.view_selector.currentIndexChanged.connect(self._on_view_changed)
        selector.addWidget(self.view_selector)
        selector.addStretch(1)
        left_layout.addLayout(selector)

        self.rgb_label = self._make_camera_label('Waiting for RGB camera...')
        self.thermal_label = self._make_camera_label('Waiting for thermal camera...')
        self.lidar_label = self._make_camera_label('Waiting for LiDAR...')
        self.single_rgb_label = self._make_camera_label('Waiting for RGB camera...')
        self.single_thermal_label = self._make_camera_label('Waiting for thermal camera...')

        self.camera_pages = QTabWidget()
        self.camera_pages.tabBar().hide()
        self.camera_pages.setDocumentMode(True)

        dual = QWidget()
        dual_layout = QHBoxLayout(dual)
        dual_layout.setContentsMargins(3, 3, 3, 3)
        dual_layout.setSpacing(4)
        dual_layout.addWidget(self._video_panel('RGB', self.rgb_label), 1)
        dual_layout.addWidget(self._video_panel('THERMAL', self.thermal_label), 1)
        self.camera_pages.addTab(dual, 'Dual')
        self.camera_pages.addTab(self._video_panel('RGB CAMERA', self.single_rgb_label), 'RGB')
        self.camera_pages.addTab(self._video_panel('IR / THERMAL', self.single_thermal_label), 'Thermal')
        self.camera_pages.addTab(self._video_panel('LIDAR', self.lidar_label), 'LiDAR')
        # RS1_UI_TRIPLE_VIEW_V1: share ONE running live map between both views.
        self.elevation_map = LiveElevationMap(self.node)
        self.map_only_page = QWidget()
        self.map_only_layout = QVBoxLayout(self.map_only_page)
        self.map_only_layout.setContentsMargins(0, 0, 0, 0)
        self.map_only_layout.addWidget(self.elevation_map)
        self.camera_pages.addTab(self.map_only_page, 'Elevation')

        # RGB and thermal share the left column; live elevation fills the right.
        self.combo_rgb_label = self._make_camera_label('Waiting for RGB camera...')
        self.combo_thermal_label = self._make_camera_label('Waiting for thermal camera...')
        combined = QWidget()
        combined_layout = QHBoxLayout(combined)
        combined_layout.setContentsMargins(2, 2, 2, 2)
        combined_layout.setSpacing(4)
        feeds = QWidget()
        feeds_layout = QVBoxLayout(feeds)
        feeds_layout.setContentsMargins(0, 0, 0, 0)
        feeds_layout.setSpacing(4)
        feeds_layout.addWidget(self._video_panel('RGB CAMERA', self.combo_rgb_label), 1)
        feeds_layout.addWidget(self._video_panel('IR / THERMAL', self.combo_thermal_label), 1)
        combined_layout.addWidget(feeds, 2)
        self.combo_map_container = QWidget()
        self.combo_map_layout = QVBoxLayout(self.combo_map_container)
        self.combo_map_layout.setContentsMargins(0, 0, 0, 0)
        combined_layout.addWidget(self.combo_map_container, 3)
        self.camera_pages.addTab(combined, 'RGB + Thermal + Elevation')
        left_layout.addWidget(self.camera_pages, 1)

        note = QLabel('Sensor view is read-only. Keyboard movement requires Manual Control enabled.')
        note.setStyleSheet('color: #8da0ba; font-size: 11px;')
        note.setWordWrap(True)
        left_layout.addWidget(note)
        splitter.addWidget(left)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(252)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(5, 2, 5, 2)
        right_layout.setSpacing(7)

        telemetry = QGroupBox('Drone Telemetry')
        t = QVBoxLayout(telemetry)
        t.setSpacing(5)
        self.heading_label = QLabel('Heading: --')
        self.altitude_label = QLabel('Altitude: --')
        t.addWidget(self.heading_label)
        t.addWidget(self.altitude_label)
        right_layout.addWidget(telemetry)

        health = QGroupBox('Sensors')
        health_layout = QGridLayout(health)
        self.rgb_status = QLabel('RGB --')
        self.thermal_status = QLabel('Thermal --')
        self.lidar_status = QLabel('LiDAR --')
        self.odom_status = QLabel('Odometry --')
        health_layout.addWidget(self.rgb_status, 0, 0)
        health_layout.addWidget(self.thermal_status, 0, 1)
        health_layout.addWidget(self.lidar_status, 1, 0)
        health_layout.addWidget(self.odom_status, 1, 1)
        right_layout.addWidget(health)
        right_layout.addWidget(self._control_panel())
        right_layout.addStretch(1)

        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes([730, 325])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        outer.addLayout(layout, 1)
        return page

    def _video_panel(self, title, image_label):
        frame = QFrame()
        frame.setObjectName('videoFrame')
        v = QVBoxLayout(frame)
        v.setContentsMargins(5, 5, 5, 5)
        v.setSpacing(4)
        label = QLabel(title)
        label.setStyleSheet('color: #adbfda; font-size: 11px; font-weight: 600;')
        v.addWidget(label)
        v.addWidget(image_label, 1)
        return frame

    def _make_camera_label(self, text):
        label = QLabel(text)
        label.setObjectName('sensorCanvas')
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        label.setMinimumSize(80, 80)
        return label

    def _control_panel(self):
        group = QGroupBox('Manual Control')
        outer = QVBoxLayout(group)
        outer.setSpacing(6)
        self.arm_button = QPushButton('Enable manual control')
        self.arm_button.setObjectName('manualArm')
        self.arm_button.setCheckable(True)
        self.arm_button.toggled.connect(self._toggle_manual)
        outer.addWidget(self.arm_button)

        caution = QLabel('Manual commands are routed through Mission Supervisor; enable only in IDLE/HOLD.')
        caution.setWordWrap(True)
        caution.setStyleSheet('color: #e6b66d; font-size: 11px;')
        outer.addWidget(caution)

        grid = QGridLayout()
        grid.setSpacing(5)
        self._movement_buttons = []

        def make_button(text, action, row, col, colspan=1):
            button = QPushButton(text)
            button.setEnabled(False)
            button.setFocusPolicy(Qt.NoFocus)
            button.pressed.connect(lambda a=action: self._set_held(a, True))
            button.released.connect(lambda a=action: self._set_held(a, False))
            grid.addWidget(button, row, col, 1, colspan)
            self._movement_buttons.append(button)
            return button

        make_button('Forward  W', 'forward', 0, 1)
        make_button('Yaw Left  A', 'left', 1, 0)
        self.stop_button = QPushButton('STOP  Space')
        self.stop_button.setObjectName('manualStop')
        self.stop_button.setEnabled(False)
        self.stop_button.setFocusPolicy(Qt.NoFocus)
        self.stop_button.clicked.connect(self.stop_command)
        grid.addWidget(self.stop_button, 1, 1)
        make_button('Yaw Right  D', 'right', 1, 2)
        make_button('Backward  S', 'backward', 2, 1)
        make_button('ASCEND  R', 'ascend', 3, 0)
        make_button('DESCEND  F', 'descend', 3, 2)
        outer.addLayout(grid)

        self.manual_status = QLabel('Manual control: DISABLED')
        self.manual_status.setStyleSheet('color: #93a7bf; font-size: 11px;')
        outer.addWidget(self.manual_status)
        return group

    def _history_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(7)

        bar = QHBoxLayout()
        bar.addWidget(QLabel('Mission'))
        self.mission_selector = QComboBox()
        self.mission_selector.addItem('Latest (auto)', None)
        self.mission_selector.currentIndexChanged.connect(self.render_rescue_history)
        bar.addWidget(self.mission_selector, 1)
        refresh = QPushButton('Refresh')
        refresh.clicked.connect(self.refresh_rescue_history)
        bar.addWidget(refresh)
        open_csv = QPushButton('Open CSV')
        open_csv.clicked.connect(self.open_rescue_csv)
        bar.addWidget(open_csv)
        copy_btn = QPushButton('Copy coordinates')
        copy_btn.clicked.connect(self.copy_selected_target)
        bar.addWidget(copy_btn)
        goto_btn = QPushButton('Go to selected target')
        goto_btn.clicked.connect(self._goto_selected_target)
        bar.addWidget(goto_btn)
        layout.addLayout(bar)

        self.rescue_table = QTableWidget(0, 11)
        self.rescue_table.setHorizontalHeaderLabels([
            'ID', 'Sim time (s)', 'X (m)', 'Y (m)', 'Z (m)',
            'Max °C', 'Frame', 'Edge clipped', 'Hot pixels', 'Classification', 'H/W ratio'
        ])
        self.rescue_table.setAlternatingRowColors(True)
        self.rescue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.rescue_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.rescue_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rescue_table.verticalHeader().setVisible(False)
        self.rescue_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.rescue_table.horizontalHeader().setStretchLastSection(True)
        self.rescue_table.setColumnWidth(0, 50)
        self.rescue_table.setColumnWidth(1, 105)
        for col in (2, 3, 4, 5):
            self.rescue_table.setColumnWidth(col, 80)
        layout.addWidget(self.rescue_table, 1)
        self.rescue_path_label = QLabel('Logs: ' + str(rescue_log_dir()))
        self.rescue_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.rescue_path_label.setWordWrap(True)
        self.rescue_path_label.setStyleSheet('color: #9fb3cd; font-size: 11px;')
        layout.addWidget(self.rescue_path_label)
        return page

    # ------------------------------------------------------------------
    # Mission controls: always visible above the camera feeds.
    # ------------------------------------------------------------------
    def _mission_dashboard_panel(self):
        group = QGroupBox('Mission Control')
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 7, 8, 6)
        outer.setSpacing(4)

        # Row 1: supervisor and sensor process lifecycle.
        row = QHBoxLayout()
        row.setSpacing(5)
        self.supervisor_info = QLabel('SUPERVISOR: OFFLINE')
        self.supervisor_info.setWordWrap(False)
        self.supervisor_info.setMinimumWidth(85)
        self.supervisor_info.setStyleSheet('color: #9fc2db; font-weight: 600;')
        row.addWidget(self.supervisor_info, 1)
        self.start_supervisor_button = QPushButton('Start Supervisor')
        self.start_supervisor_button.setToolTip('Starts the RS1 command supervisor; Gazebo must be running')
        self.start_supervisor_button.clicked.connect(self._start_supervisor)
        row.addWidget(self.start_supervisor_button)
        self.sensors_start_button = QPushButton('Start Sensors')
        self.sensors_start_button.setToolTip('Start thermal detector and localizer. Recorder starts only with search.')
        self.sensors_start_button.clicked.connect(self._start_sensors)
        row.addWidget(self.sensors_start_button)
        self.sensors_stop_button = QPushButton('Stop Sensors')
        self.sensors_stop_button.setToolTip('Stop supervisor-managed detector and localizer when no mission is active')
        self.sensors_stop_button.clicked.connect(lambda: self.node.request('stop_sensors'))
        row.addWidget(self.sensors_stop_button)
        outer.addLayout(row)

        # Row 2: search, abort/hold, and optional controlled descent.
        # The mission recorder creates timestamped terrain CSVs automatically.
        row = QHBoxLayout()
        row.setSpacing(5)
        self.mission_start_button = QPushButton('Start Search')
        self.mission_start_button.setToolTip('Start selected search route through Mission Supervisor; starts recorder automatically')
        self.mission_start_button.clicked.connect(self._start_search)
        row.addWidget(self.mission_start_button)
        self.hold_button = QPushButton('Abort / Hold')
        self.hold_button.setToolTip('Cancel the UI-managed mission and hold position')
        self.hold_button.clicked.connect(self._hold_mission)
        row.addWidget(self.hold_button)
        row.addStretch(1)
        self.clear_land_box = QCheckBox('Clear below (sim)')
        self.clear_land_box.setToolTip(
            'Confirm the area directly beneath the drone is clear. No downward obstacle sensor is installed.')
        row.addWidget(self.clear_land_box)
        self.land_button = QPushButton('Land')
        self.land_button.setToolTip('Controlled landing after abort/hold, if the area below is confirmed clear')
        self.land_button.clicked.connect(self._controlled_land)
        row.addWidget(self.land_button)
        outer.addLayout(row)

        # Terrain search is the only UI-managed search route.
        # The XY defaults are in parrot1_odom, not world/map coordinates.
        row = QHBoxLayout()
        row.setSpacing(5)
        row.addWidget(QLabel('Terrain sweep'))
        self.route_fields = {}
        for key, label, default in (
                ('min_x', 'Xmin', '-22'), ('max_x', 'Xmax', '22'),
                ('min_y', 'Ymin', '-22'), ('max_y', 'Ymax', '22'),
                ('lane_spacing', 'Lane', '3'), ('altitude', 'Alt', '12')):
            row.addWidget(QLabel(label))
            field = QLineEdit(default)
            field.setMaximumWidth(52)
            field.setToolTip('XY bounds in parrot1_odom metres; altitude measured in odom Z')
            self.route_fields[key] = field
            row.addWidget(field)
        self.route_preview_button = QPushButton('Preview')
        self.route_preview_button.clicked.connect(self._preview_search_route)
        row.addWidget(self.route_preview_button)
        self.route_clearance_box = QCheckBox('Climb & route clear (sim)')
        self.route_clearance_box.setToolTip(
            'You have checked takeoff climb corridor, altitude, canopy, '
            'obstacles and route in the actual running world. No downward sensor.')
        row.addWidget(self.route_clearance_box)
        for field in self.route_fields.values():
            field.textChanged.connect(lambda _text: self.route_clearance_box.setChecked(False))
        outer.addLayout(row)

        # Row 4: navigation to specified XY or selected saved rescue target.
        row = QHBoxLayout()
        row.setSpacing(5)
        row.addWidget(QLabel('Go to'))
        row.addWidget(QLabel('X'))
        self.target_x = QLineEdit()
        self.target_x.setPlaceholderText('X m')
        self.target_x.setMaximumWidth(90)
        self.target_x.setToolTip('parrot1_odom X coordinate, metres')
        row.addWidget(self.target_x)
        row.addWidget(QLabel('Y'))
        self.target_y = QLineEdit()
        self.target_y.setPlaceholderText('Y m')
        self.target_y.setMaximumWidth(90)
        self.target_y.setToolTip('parrot1_odom Y coordinate, metres')
        row.addWidget(self.target_y)
        self.goto_button = QPushButton('Go XY')
        self.goto_button.setToolTip('Go to XY at 12 m survey altitude; no path planning, front LiDAR stop only')
        self.goto_button.clicked.connect(self._goto_xy)
        row.addWidget(self.goto_button)
        self.goto_target_button = QPushButton('Go to Saved Target')
        self.goto_target_button.setToolTip('Uses selected row from Rescue History in parrot1_odom')
        self.goto_target_button.clicked.connect(self._goto_selected_target)
        row.addWidget(self.goto_target_button)
        row.addStretch(1)
        note = QLabel('Frame: parrot1_odom')
        note.setStyleSheet('color: #8498b1; font-size: 10px;')
        row.addWidget(note)
        outer.addLayout(row)
        return group

    def _start_supervisor(self):
        if self.node.supervisor_online():
            return
        if self._supervisor_proc and self._supervisor_proc.poll() is None:
            return
        logpath = Path.home() / 'rescue_logs' / 'rs1_supervisor_ui.log'
        logpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with logpath.open('a', encoding='utf-8') as log:
                self._supervisor_proc = subprocess.Popen(
                    ['ros2', 'run', '41068_ignition_bringup', 'rs1_mission_supervisor'],
                    stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True)
        except OSError as e:
            QMessageBox.warning(self, 'Mission supervisor', f'Cannot start: {e}')

    def _start_sensors(self):
        self.node.request('start_sensors', trial='terrain')

    def _terrain_request(self):
        values = {}
        for key, field in self.route_fields.items():
            values[key] = float(field.text().strip())
        if not math.isfinite(values['altitude']) or not 5 <= values['altitude'] <= 25:
            raise ValueError('Altitude must be 5..25 m')
        start = (self.node.x, self.node.y)
        path = plan_lawnmower(*(values[key] for key in
            ('min_x', 'max_x', 'min_y', 'max_y', 'lane_spacing')), start=start)
        return values, route_stats(path, start=start)

    def _preview_search_route(self):
        try:
            values, stats = self._terrain_request()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Terrain route', str(exc))
            return
        QMessageBox.information(self, 'Terrain route preview',
            f"Bounds X [{values['min_x']}, {values['max_x']}], "
            f"Y [{values['min_y']}, {values['max_y']}] m\n"
            f"Lane spacing: {values['lane_spacing']} m\n"
            f"Waypoint count: {stats['waypoints']}\n"
            f"Route including transit: ~{stats['distance_m']:.0f} m\n\n"
            'This does NOT validate canopy clearance, actual thermal '
            'coverage, or world-to-odom frame alignment.')

    def _start_search(self):
        try:
            values, _ = self._terrain_request()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Terrain search', str(exc))
            return
        if not self.route_clearance_box.isChecked():
            QMessageBox.warning(self, 'Terrain search',
                'Verify the climb corridor, terrain clearance and planned '
                'search area in the actual sim, then check the clearance box.')
            return
        self.node.request('start_search', trial='terrain', search_mode='terrain',
                          clearance_verified=True, **values)

    def _hold_mission(self):
        self.stop_command()
        self.arm_button.setChecked(False)
        self.node.request('hold')

    def _emergency_abort(self):
        if not self.node.supervisor_online():
            QMessageBox.warning(self, 'No supervisor',
                'The mission supervisor is offline. This button CANNOT stop '
                'a flight controller started in another terminal.')
            return
        self.stop_command()
        self.arm_button.setChecked(False)
        self.node.request('abort_land', clear_landing_area=self.clear_land_box.isChecked())

    def _controlled_land(self):
        self.node.request('land', clear_landing_area=self.clear_land_box.isChecked())

    def _goto_xy(self):
        try:
            x, y = float(self.target_x.text()), float(self.target_y.text())
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError('Coordinates must be finite')
        except ValueError:
            QMessageBox.warning(self, 'Invalid coordinates', 'Enter numeric finite X and Y values.')
            return
        self.node.request('goto', x=x, y=y, frame_id='parrot1_odom')

    def _goto_selected_target(self):
        row = self.rescue_table.currentRow()
        if row < 0 or row >= len(self.current_rescue_records):
            QMessageBox.information(self, 'Select target',
                                    'Select a row in Rescue History first.')
            return
        record = self.current_rescue_records[row]
        if record.get('frame_id') != 'parrot1_odom':
            QMessageBox.warning(self, 'Coordinate frame mismatch',
                'Target frame is not parrot1_odom. Transform coordinates before navigation.')
            return
        self.node.request('goto', x=record['x_m'], y=record['y_m'],
                          frame_id=record['frame_id'])
        self.pages.setCurrentIndex(0)  # Return to live dashboard after target navigation.

    def _on_page_changed(self, idx):
        self.stop_command()
        if idx != 0 and self.arm_button.isChecked():
            self.arm_button.setChecked(False)

    def _on_view_changed(self, idx):
        # Move the one elevation widget instead of creating a second subscriber.
        if idx == 5:
            self.map_only_layout.removeWidget(self.elevation_map)
            self.combo_map_layout.addWidget(self.elevation_map)
        elif idx == 4:
            self.combo_map_layout.removeWidget(self.elevation_map)
            self.map_only_layout.addWidget(self.elevation_map)
        self.camera_pages.setCurrentIndex(idx)
        if idx in (4, 5):
            self.elevation_map.show()

    # ------------------------------------------------------------------
    # Rescue alerts and read-only history. Existing CSV format preserved.
    # ------------------------------------------------------------------
    def consume_rescue_alerts(self):
        while self.node.live_rescue_alerts:
            alert = self.node.live_rescue_alerts.popleft()
            self.live_alert_count += 1
            self.rescue_alert_label.setText(
                f'{alert["classification"].upper()}  |  TARGET #{alert["id"]}  |  '
                f'X {alert["x_m"]:.2f}  Y {alert["y_m"]:.2f}  Z {alert["z_m"]:.2f} m  |  '
                f'{alert["max_temp_c"]:.2f}°C  [{alert["frame_id"]}]'
                + ('  EDGE CLIPPED' if alert['edge_clipped'] else '')
            )
            self.rescue_alert_label.setObjectName('alertActive')
            self._refresh_alert_style()
            self.acknowledge_button.setEnabled(True)
            QApplication.beep()
            self._history_signature = None
            self.refresh_rescue_history()

    def _refresh_alert_style(self):
        self.rescue_alert_label.style().unpolish(self.rescue_alert_label)
        self.rescue_alert_label.style().polish(self.rescue_alert_label)

    def acknowledge_rescue_alert(self):
        self.acknowledge_button.setEnabled(False)
        self.rescue_alert_label.setText(
            f'Acknowledged  |  {self.live_alert_count} alert(s) in this UI session'
        )
        self.rescue_alert_label.setObjectName('alertAck')
        self._refresh_alert_style()

    def refresh_rescue_history(self):
        try:
            choices = list_missions()
            old_selection = self.mission_selector.currentData()
            current_choices = [self.mission_selector.itemData(i)
                               for i in range(1, self.mission_selector.count())]
            if choices != current_choices:
                self.mission_selector.blockSignals(True)
                self.mission_selector.clear()
                self.mission_selector.addItem('Latest (auto)', None)
                for name in choices:
                    self.mission_selector.addItem(name, name)
                if old_selection in choices:
                    self.mission_selector.setCurrentIndex(choices.index(old_selection) + 1)
                self.mission_selector.blockSignals(False)
            self.render_rescue_history()
        except (OSError, ValueError, csv.Error) as exc:
            self.rescue_path_label.setText('Cannot read rescue logs: ' + str(exc))

    def render_rescue_history(self, *_):
        choices = list_missions()
        chosen = self.mission_selector.currentData()
        name = chosen if chosen in choices else (choices[0] if choices else None)
        if not name:
            self.selected_mission_name = None
            self.current_rescue_records = []
            self.rescue_table.setRowCount(0)
            self.rescue_path_label.setText('No rescue CSV files yet: ' + str(rescue_log_dir()))
            return
        try:
            path = mission_path(name)
            info = path.stat()
            signature = (name, info.st_mtime_ns, info.st_size)
            if signature == self._history_signature:
                return
            records = read_mission(name)
            self._history_signature = signature
            self.selected_mission_name = name
            self.current_rescue_records = records
            self.rescue_table.setRowCount(len(records))
            for i, record in enumerate(records):
                columns = (
                    str(record['id']), f'{record["sim_time_sec"]:.3f}',
                    f'{record["x_m"]:.3f}', f'{record["y_m"]:.3f}',
                    f'{record["z_m"]:.3f}', f'{record["max_temp_c"]:.2f}',
                    record['frame_id'], 'YES' if record['edge_clipped'] else 'NO',
                    str(record['hot_pixels']),
                    record.get('classification', 'thermal_unverified'),
                    str(record.get('shape_aspect_ratio') or '—'),
                )
                for j, value in enumerate(columns):
                    self.rescue_table.setItem(i, j, QTableWidgetItem(value))
            if records:
                self.rescue_table.selectRow(len(records) - 1)
            self.rescue_path_label.setText(f'{len(records)} saved target(s)  |  CSV: {path}')
        except (OSError, ValueError, csv.Error) as exc:
            self._history_signature = None
            self.rescue_path_label.setText('Cannot read rescue CSV: ' + str(exc))

    def copy_selected_target(self):
        row = self.rescue_table.currentRow()
        if row < 0 or row >= len(self.current_rescue_records):
            QMessageBox.information(self, 'Select a target', 'Select a target row first.')
            return
        record = self.current_rescue_records[row]
        copied = (f'Target #{record["id"]}: '
                  f'({record["x_m"]:.3f}, {record["y_m"]:.3f}, {record["z_m"]:.3f}) m '
                  f'[{record["frame_id"]}], sim time={record["sim_time_sec"]:.3f}s')
        QApplication.clipboard().setText(copied)

    def open_rescue_csv(self):
        if not self.selected_mission_name:
            QMessageBox.information(self, 'No mission', 'No saved rescue CSV yet.')
            return
        try:
            path = mission_path(self.selected_mission_name)
        except (ValueError, FileNotFoundError):
            QMessageBox.warning(self, 'Missing file', 'This mission CSV is unavailable.')
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.information(self, 'CSV path', f'Open the CSV manually:\n{path}')

    # ------------------------------------------------------------------
    # ROS processing and display
    # ------------------------------------------------------------------
    def process_ros(self):
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            print(f'ROS callback error: {exc}')

    def update_ui(self):
        self.consume_rescue_alerts()
        self.position_label.setText(
            f'X {self.node.x:.1f} m   Y {self.node.y:.1f} m   ALT {self.node.z:.1f} m'
        )
        self.heading_label.setText(f'Heading: {self.node.yaw_deg:.1f}°')
        self.altitude_label.setText(f'Altitude: {self.node.z:.2f} m')

        now = time.monotonic()
        self.set_status(self.rgb_status, 'RGB', now - self.node.last_rgb_time < self.SENSOR_TIMEOUT_S)
        self.set_status(self.thermal_status, 'Thermal', now - self.node.last_thermal_time < self.SENSOR_TIMEOUT_S)
        self.set_status(self.lidar_status, 'LiDAR', now - self.node.last_lidar_time < self.SENSOR_TIMEOUT_S)
        self.set_status(self.odom_status, 'Odometry', now - self.node.last_odom_time < self.SENSOR_TIMEOUT_S)

        online = self.node.supervisor_online()
        status = self.node.supervisor_status if online else {}
        state = status.get('state', 'OFFLINE')
        reason = status.get('reason', 'Start Mission Supervisor to enable controls')
        proc = status.get('processes', {})
        # Keep the dashboard compact; full rejection reason and process states
        # remain available via tooltip instead of expanding the camera layout.
        self.supervisor_info.setText(f'SUPERVISOR: {state}')  # RS1_UI_QUIET_SUPERVISOR_V1
        self.supervisor_info.setToolTip(
            f'State: {state}\nReason: {reason}\nManaged: ' +
            ', '.join(f'{k}={v}' for k,v in proc.items()))
        self.start_supervisor_button.setEnabled(not online)
        self.sensors_start_button.setEnabled(online)
        self.sensors_stop_button.setEnabled(online and state not in ('SEARCH','GOTO','LANDING'))
        self.mission_start_button.setEnabled(online and state in ('IDLE','HOLD','ARRIVED'))
        self.goto_button.setEnabled(online and state in ('IDLE','HOLD','ARRIVED'))
        self.goto_target_button.setEnabled(online and state in ('IDLE','HOLD','ARRIVED'))
        self.hold_button.setEnabled(online)
        self.land_button.setEnabled(online and state in ('IDLE','HOLD','ARRIVED','LANDED_ESTIMATED'))
        # Emergency never claims to cancel external/direct publishers.
        if self.manual_enabled and online and state in ('SEARCH','GOTO','LANDING','LANDED_ESTIMATED'):
            self.arm_button.setChecked(False)
        if self._close_after_hold and online and state not in ('SEARCH', 'GOTO', 'LANDING', 'MANUAL'):
            self._close_after_hold = False
            QTimer.singleShot(0, self.close)
            return
        if self.pages.currentIndex() != 0:
            return
        view = self.view_selector.currentIndex()
        if view == 0:
            self.show_image(self.rgb_label, self.node.latest_rgb_image, 'Waiting for RGB...')
            self.show_image(self.thermal_label, self.node.latest_thermal_image, 'Waiting for thermal...')
        elif view == 1:
            self.show_image(self.single_rgb_label, self.node.latest_rgb_image, 'Waiting for RGB...')
        elif view == 2:
            self.show_image(self.single_thermal_label, self.node.latest_thermal_image, 'Waiting for thermal...')
        elif view == 3:
            self.show_lidar(self.lidar_label)
        elif view == 5:
            self.show_image(self.combo_rgb_label, self.node.latest_rgb_image, 'Waiting for RGB...')
            self.show_image(self.combo_thermal_label, self.node.latest_thermal_image, 'Waiting for thermal...')
        # Elevation view renders itself on its own timer.

    def show_image(self, label, image, waiting_text):
        if image is None:
            label.clear()
            label.setText(waiting_text)
            return
        pixmap = QPixmap.fromImage(image).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def show_lidar(self, label):
        scan = self.node.latest_scan
        if scan is None:
            label.clear()
            label.setText('Waiting for LiDAR...')
            return
        width = max(80, label.width())
        height = max(80, label.height())
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(6, 11, 18))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        cx, cy = width / 2.0, height / 2.0
        distance_max = min(scan.range_max, 20.0)
        if not math.isfinite(distance_max) or distance_max <= 0:
            painter.end()
            label.setPixmap(pixmap)
            return
        px_m = min(width, height) * .42 / distance_max
        painter.setPen(QPen(QColor(60, 83, 105), 1))
        for metres in (5, 10, 15, 20):
            if metres <= distance_max:
                radius = metres * px_m
                painter.drawEllipse(int(cx-radius), int(cy-radius), int(2*radius), int(2*radius))
        painter.drawLine(int(cx), 0, int(cx), height)
        painter.drawLine(0, int(cy), width, int(cy))
        painter.setPen(QPen(QColor(79, 235, 152), 3))
        angle = scan.angle_min
        for distance in scan.ranges:
            if math.isfinite(distance) and scan.range_min < distance < distance_max:
                painter.drawPoint(
                    int(cx + distance * math.sin(angle) * px_m),
                    int(cy - distance * math.cos(angle) * px_m)
                )
            angle += scan.angle_increment
        painter.setPen(QPen(QColor(255, 216, 99), 6))
        painter.drawPoint(int(cx), int(cy))
        painter.setPen(QPen(QColor(255, 216, 99), 2))
        painter.drawLine(int(cx), int(cy), int(cx), int(cy - 30))
        painter.end()
        label.setPixmap(pixmap)

    def set_status(self, label, name, online):
        state = (name, online)
        if self._sensor_states.get(label) == state:
            return
        self._sensor_states[label] = state
        label.setText(f'{"●" if online else "○"} {name}')
        label.setStyleSheet('color: #83e2b8;' if online else 'color: #e69292;')

    # ------------------------------------------------------------------
    # Manual flight: hold-to-move. No idle cmd_vel output while disabled.
    # ------------------------------------------------------------------
    def _toggle_manual(self, enabled):
        if enabled:
            status = self.node.supervisor_status if self.node.supervisor_online() else {}
            if status.get('state') not in ('IDLE', 'HOLD', 'ARRIVED', 'MANUAL'):
                self.arm_button.blockSignals(True)
                self.arm_button.setChecked(False)
                self.arm_button.blockSignals(False)
                QMessageBox.warning(self, 'Manual control unavailable',
                    'Start Mission Supervisor and abort autonomous flight first.')
                return
            self.node.request('manual_on')
        else:
            self.stop_command()
            if self.node.supervisor_online():
                self.node.request('manual_off')
        self.manual_enabled = enabled
        for button in self._movement_buttons:
            button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.arm_button.setText('Disable manual control' if enabled else 'Enable manual control')
        self.manual_status.setText('Manual control: ENABLED' if enabled else 'Manual control: DISABLED')
        self.manual_status.setStyleSheet(
            'color: #8de2b8; font-size: 11px;' if enabled else
            'color: #93a7bf; font-size: 11px;'
        )
        if enabled:
            self.setFocus(Qt.OtherFocusReason)

    def _set_held(self, action, pressed):
        if not self.manual_enabled:
            return
        if pressed:
            self._held_controls.add(action)
        else:
            self._held_controls.discard(action)
        self.publish_current_command(force=True)

    def _velocity(self):
        held = self._held_controls
        vx = LINEAR_SPEED * (int('forward' in held) - int('backward' in held))
        vz = VERTICAL_SPEED * (int('ascend' in held) - int('descend' in held))
        yaw = ANGULAR_SPEED * (int('left' in held) - int('right' in held))
        return vx, yaw, vz

    def publish_current_command(self, force=False):
        if not self.manual_enabled:
            return
        active = bool(self._held_controls)
        if active:
            self.node.send_velocity(*self._velocity())
        elif self._last_cmd_active:
            # Send one final zero, then stop publishing so autonomous commands
            # are not repeatedly overridden by an idle operator UI.
            self.node.stop()
        self._last_cmd_active = active

    def stop_command(self):
        was_active = self._last_cmd_active or bool(self._held_controls)
        self._held_controls.clear()
        if was_active:
            self.node.stop()
        self._last_cmd_active = False

    def eventFilter(self, obj, event):
        typ = event.type()
        if typ not in (QEvent.KeyPress, QEvent.KeyRelease):
            return super().eventFilter(obj, event)
        if not self.manual_enabled or not self.isActiveWindow() or self.pages.currentIndex() != 0:
            return super().eventFilter(obj, event)
        if QApplication.activePopupWidget() is not None:
            self.stop_command()
            return super().eventFilter(obj, event)
        # Never interpret typing XY coordinates as manual flight commands.
        # The global event filter otherwise captures R/F/W/A/S/D in text fields.
        if isinstance(QApplication.focusWidget(), QLineEdit):
            self.stop_command()
            return super().eventFilter(obj, event)
        if event.isAutoRepeat():
            return False
        key = event.key()
        if key == Qt.Key_Space and typ == QEvent.KeyPress:
            self.stop_command()
            return True
        action = self.KEY_TO_ACTION.get(key)
        if action is not None:
            self._set_held(action, typ == QEvent.KeyPress)
            return True
        return super().eventFilter(obj, event)

    def _app_state_changed(self, state):
        if state != Qt.ApplicationActive:
            self.stop_command()

    def closeEvent(self, event):
        if self.node.supervisor_online() and self.node.supervisor_status.get('state') in ('SEARCH','GOTO','LANDING'):
            reply = QMessageBox.question(self, 'Mission still running',
                'Mission Supervisor will remain running after this UI closes.\n'
                'Abort and hold before closing?',
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.node.request('hold')
            self._close_after_hold = True
            event.ignore()
            return
        self.stop_command()
        if self.manual_enabled and self.node.supervisor_online():
            self.node.request('manual_off')
        self.ros_timer.stop()
        self.command_timer.stop()
        self.ui_timer.stop()
        self.mission_refresh_timer.stop()
        QApplication.instance().removeEventFilter(self)
        event.accept()


def main():
    rclpy.init()
    node = DroneUiNode()
    app = QApplication(sys.argv)
    window = DroneUI(node)
    window.show()
    result = app.exec_()
    # Avoid publishing cmd_vel merely because a read-only UI closes.
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(result)


if __name__ == '__main__':
    main()
