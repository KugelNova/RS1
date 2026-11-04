#!/usr/bin/env python3

import sys
import math
import time
import numpy as np

import rclpy

# --- COMMENTED OUT FOR MERGE ---
# from rclpy.node import Node
# from rclpy.qos import qos_profile_sensor_data
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import Image, LaserScan
# -------------------------------

# --- ADDED FOR MERGE ---
from rescue_drone_node import RescueDroneNode
# -----------------------

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QFont,
    QPainter,
    QPen,
    QColor,
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
)


LINEAR_SPEED = 0.5
ANGULAR_SPEED = 0.8

# --- COMMENTED OUT FOR MERGE ---
# Thermal display range is now handled by RescueDroneNode
# THERMAL_MIN_RAW = 28500
# THERMAL_MAX_RAW = 31500

"""
# --- COMMENTED OUT FOR MERGE ---
class DroneUiNode(Node):

    def __init__(self):
        super().__init__('drone_ui')
        ... (All subscriptions, odom callbacks, and raw image conversions removed)
        
    def rgb_callback(self, msg): ...
    def thermal_callback(self, msg): ...
    def odom_callback(self, msg): ...
    def scan_callback(self, msg): ...
    def send_velocity(self, linear_x, angular_z): ...
    def stop(self): ...
# -------------------------------
"""

class DroneUI(QWidget):

    def __init__(self, ros_node):

        super().__init__()

        self.node = ros_node

        self.linear_cmd = 0.0
        self.angular_cmd = 0.0

        self.setWindowTitle(
            'Search & Rescue Drone Control'
        )

        self.resize(
            1100,
            750
        )

        self.setFocusPolicy(
            Qt.StrongFocus
        )

        self.build_ui()

        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.process_ros)
        self.ros_timer.start(20)

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)

        self.command_timer = QTimer(self)
        self.command_timer.timeout.connect(self.publish_current_command)
        self.command_timer.start(100)

    # --------------------------------------------------
    # UI
    # --------------------------------------------------

    def build_ui(self):

        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #eeeeee; font-size: 15px; }
            QGroupBox { border: 1px solid #555555; border-radius: 8px; margin-top: 10px; padding: 10px; font-weight: bold; }
            QPushButton { background-color: #353535; border: 1px solid #666666; border-radius: 6px; padding: 12px; min-height: 38px; }
            QPushButton:pressed { background-color: #606060; }
            QPushButton#stopButton { background-color: #8a3030; font-weight: bold; }
            QComboBox { background-color: #303030; border: 1px solid #666666; border-radius: 5px; padding: 8px; min-width: 180px; }
            QLabel#camera { background-color: #000000; border: 1px solid #555555; }
            QLabel#online { color: #72d572; }
            QLabel#offline { color: #dd6666; }
        """)

        main_layout = QVBoxLayout()

        title = QLabel('SEARCH & RESCUE DRONE CONTROL')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Arial', 22, QFont.Bold))
        main_layout.addWidget(title)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel('Sensor View:'))
        self.view_selector = QComboBox()
        self.view_selector.addItems(['RGB Camera', 'IR / Thermal', 'LiDAR'])
        selector_layout.addWidget(self.view_selector)
        selector_layout.addStretch()
        main_layout.addLayout(selector_layout)

        content_layout = QHBoxLayout()

        camera_group = QGroupBox('Sensor Display')
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel('Waiting for sensor data...')
        self.camera_label.setObjectName('camera')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(700, 450)
        camera_layout.addWidget(self.camera_label)
        camera_group.setLayout(camera_layout)
        content_layout.addWidget(camera_group, 3)

        status_group = QGroupBox('Drone Status')
        status_layout = QVBoxLayout()
        self.position_label = QLabel()
        self.heading_label = QLabel()

        self.rgb_status = QLabel()
        self.thermal_status = QLabel()
        self.odom_status = QLabel()
        self.lidar_status = QLabel()

        status_layout.addWidget(self.position_label)
        status_layout.addWidget(self.heading_label)
        status_layout.addSpacing(20)
        status_layout.addWidget(self.rgb_status)
        status_layout.addWidget(self.thermal_status)
        status_layout.addWidget(self.lidar_status)
        status_layout.addWidget(self.odom_status)
        status_layout.addStretch()
        status_group.setLayout(status_layout)
        content_layout.addWidget(status_group, 1)

        main_layout.addLayout(content_layout)

        control_group = QGroupBox('Manual Control')
        controls = QGridLayout()

        self.forward_button = QPushButton('▲\nForward\n(W)')
        self.left_button = QPushButton('↶\nTurn Left\n(A)')
        self.stop_button = QPushButton('STOP\n(Space)')
        self.stop_button.setObjectName('stopButton')
        self.right_button = QPushButton('↷\nTurn Right\n(D)')
        self.back_button = QPushButton('▼\nBackward\n(S)')

        controls.addWidget(self.forward_button, 0, 1)
        controls.addWidget(self.left_button, 1, 0)
        controls.addWidget(self.stop_button, 1, 1)
        controls.addWidget(self.right_button, 1, 2)
        controls.addWidget(self.back_button, 2, 1)
        control_group.setLayout(controls)
        main_layout.addWidget(control_group)

        self.setLayout(main_layout)

        self.forward_button.pressed.connect(lambda: self.set_command(LINEAR_SPEED, 0.0))
        self.forward_button.released.connect(self.stop_command)
        self.back_button.pressed.connect(lambda: self.set_command(-LINEAR_SPEED, 0.0))
        self.back_button.released.connect(self.stop_command)
        self.left_button.pressed.connect(lambda: self.set_command(0.0, ANGULAR_SPEED))
        self.left_button.released.connect(self.stop_command)
        self.right_button.pressed.connect(lambda: self.set_command(0.0, -ANGULAR_SPEED))
        self.right_button.released.connect(self.stop_command)
        self.stop_button.clicked.connect(self.stop_command)

    # --------------------------------------------------
    # ROS PROCESSING
    # --------------------------------------------------

    def process_ros(self):
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            print(f'ROS callback error: {exc}')

    # --------------------------------------------------
    # DISPLAY UPDATE
    # --------------------------------------------------

    def update_ui(self):

        # --- MODIFIED FOR MERGE ---
        self.position_label.setText(
            f'Position\n'
            f'X: {self.node.x:7.2f} m\n'
            f'Y: {self.node.y:7.2f} m\n'
            f'Z: {self.node.z:7.2f} m'
        )

        self.heading_label.setText(
            f'Heading: '
            f'{math.degrees(self.node.yaw):6.1f}°'
        )

        # Map directly to RescueDroneNode's image buffers to verify active feeds
        self.set_status(self.rgb_status, 'RGB Camera', self.node.rgb_frame is not None)
        self.set_status(self.thermal_status, 'IR Camera', self.node.thermal_frame is not None)
        self.set_status(self.lidar_status, 'LiDAR', self.node.lidar_frame is not None)
        self.set_status(self.odom_status, 'Odometry', True)

        selected = self.view_selector.currentText()

        if selected == 'RGB Camera':
            self.show_image(self.node.rgb_frame, 'Waiting for RGB camera...')
        elif selected == 'IR / Thermal':
            self.show_image(self.node.thermal_frame, 'Waiting for thermal camera...')
        elif selected == 'LiDAR':
            self.show_lidar()

    def show_image(self, image_data, waiting_text):

        # --- MODIFIED FOR MERGE ---
        if image_data is None:
            self.camera_label.clear()
            self.camera_label.setText(waiting_text)
            return

        # Load OpenCV-generated jpg byte array natively into PyQt
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)

        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(pixmap)

    # --------------------------------------------------
    # LIDAR DISPLAY
    # --------------------------------------------------

    def show_lidar(self):
        # --- MODIFIED FOR MERGE ---
        # Forward pre-rendered LiDAR OpenCV byte array to standard image viewer
        self.show_image(self.node.lidar_frame, 'Waiting for LiDAR...')
        
        """
        # --- COMMENTED OUT FOR MERGE ---
        scan = self.node.latest_scan
        if scan is None: ...
        width = max(self.camera_label.width(), 400)
        height = max(self.camera_label.height(), 300)
        pixmap = QPixmap(width, height)
        ... [Original QPainter Logic] ...
        painter.end()
        self.camera_label.setPixmap(pixmap)
        # -------------------------------
        """

    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------

    def set_status(self, label, name, online):
        if online:
            label.setText(f'● {name}: ONLINE')
            label.setObjectName('online')
        else:
            label.setText(f'● {name}: OFFLINE')
            label.setObjectName('offline')

        label.style().unpolish(label)
        label.style().polish(label)

    # --------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------

    def set_command(self, linear, angular):
        self.linear_cmd = linear
        self.angular_cmd = angular
        self.publish_current_command()

    def publish_current_command(self):
        # --- MODIFIED FOR MERGE ---
        self.node.set_command(
            lx=self.linear_cmd,
            ly=0.0,
            lz=0.0,
            ax=0.0,
            ay=0.0,
            az=self.angular_cmd
        )

    def stop_command(self):
        self.linear_cmd = 0.0
        self.angular_cmd = 0.0
        
        # --- MODIFIED FOR MERGE ---
        self.node.set_command(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # --------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        key = event.key()

        if key == Qt.Key_W: self.set_command(LINEAR_SPEED, 0.0)
        elif key == Qt.Key_S: self.set_command(-LINEAR_SPEED, 0.0)
        elif key == Qt.Key_A: self.set_command(0.0, ANGULAR_SPEED)
        elif key == Qt.Key_D: self.set_command(0.0, -ANGULAR_SPEED)
        elif key == Qt.Key_Space: self.stop_command()
        elif key == Qt.Key_1: self.view_selector.setCurrentIndex(0)
        elif key == Qt.Key_2: self.view_selector.setCurrentIndex(1)
        elif key == Qt.Key_3: self.view_selector.setCurrentIndex(2)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): return
        if event.key() in (Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D):
            self.stop_command()

    # --------------------------------------------------
    # CLOSE
    # --------------------------------------------------

    def closeEvent(self, event):
        self.stop_command()
        self.ros_timer.stop()
        self.command_timer.stop()
        self.ui_timer.stop()
        event.accept()


def main():

    rclpy.init()

    # --- MODIFIED FOR MERGE ---
    # Instantiate RescueDroneNode instead of DroneUiNode
    node = RescueDroneNode()
    # --------------------------

    app = QApplication(
        sys.argv
    )

    window = DroneUI(
        node
    )

    window.show()

    result = app.exec_()

    # --- MODIFIED FOR MERGE ---
    node.set_command(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # --------------------------

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()

    sys.exit(result)


if __name__ == '__main__':
    main()