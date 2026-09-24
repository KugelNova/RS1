#!/usr/bin/env python3

import sys
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan

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

# Thermal display range.
# 285 K = ~11.9 C
# 315 K = ~41.9 C
THERMAL_MIN_RAW = 28500
THERMAL_MAX_RAW = 31500


class DroneUiNode(Node):

    def __init__(self):
        super().__init__('drone_ui')

        self.latest_rgb_image = None
        self.latest_thermal_image = None
        self.latest_scan = None

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
            '/parrot1/cmd_vel',
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
            '/parrot1/thermal/image_raw',
            self.thermal_callback,
            qos_profile_sensor_data
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

        self.get_logger().info(
            'Drone UI ROS node started'
        )


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

    def send_velocity(
        self,
        linear_x,
        angular_z
    ):

        msg = Twist()

        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)

        self.cmd_pub.publish(msg)


    def stop(self):
        self.send_velocity(
            0.0,
            0.0
        )


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

        self.ros_timer.timeout.connect(
            self.process_ros
        )

        self.ros_timer.start(20)

        self.ui_timer = QTimer(self)

        self.ui_timer.timeout.connect(
            self.update_ui
        )

        self.ui_timer.start(100)

        self.command_timer = QTimer(self)

        self.command_timer.timeout.connect(
            self.publish_current_command
        )

        self.command_timer.start(100)


    # --------------------------------------------------
    # UI
    # --------------------------------------------------

    def build_ui(self):

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #eeeeee;
                font-size: 15px;
            }

            QGroupBox {
                border: 1px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }

            QPushButton {
                background-color: #353535;
                border: 1px solid #666666;
                border-radius: 6px;
                padding: 12px;
                min-height: 38px;
            }

            QPushButton:pressed {
                background-color: #606060;
            }

            QPushButton#stopButton {
                background-color: #8a3030;
                font-weight: bold;
            }

            QComboBox {
                background-color: #303030;
                border: 1px solid #666666;
                border-radius: 5px;
                padding: 8px;
                min-width: 180px;
            }

            QLabel#camera {
                background-color: #000000;
                border: 1px solid #555555;
            }

            QLabel#online {
                color: #72d572;
            }

            QLabel#offline {
                color: #dd6666;
            }
        """)

        main_layout = QVBoxLayout()

        title = QLabel(
            'SEARCH & RESCUE DRONE CONTROL'
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setFont(
            QFont(
                'Arial',
                22,
                QFont.Bold
            )
        )

        main_layout.addWidget(title)

        # -------------------------
        # VIEW SELECTOR
        # -------------------------

        selector_layout = QHBoxLayout()

        selector_layout.addWidget(
            QLabel('Sensor View:')
        )

        self.view_selector = QComboBox()

        self.view_selector.addItems([
            'RGB Camera',
            'IR / Thermal',
            'LiDAR'
        ])

        selector_layout.addWidget(
            self.view_selector
        )

        selector_layout.addStretch()

        main_layout.addLayout(
            selector_layout
        )

        # -------------------------
        # DISPLAY + STATUS
        # -------------------------

        content_layout = QHBoxLayout()

        camera_group = QGroupBox(
            'Sensor Display'
        )

        camera_layout = QVBoxLayout()

        self.camera_label = QLabel(
            'Waiting for sensor data...'
        )

        self.camera_label.setObjectName(
            'camera'
        )

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setMinimumSize(
            700,
            450
        )

        camera_layout.addWidget(
            self.camera_label
        )

        camera_group.setLayout(
            camera_layout
        )

        content_layout.addWidget(
            camera_group,
            3
        )

        # -------------------------
        # STATUS
        # -------------------------

        status_group = QGroupBox(
            'Drone Status'
        )

        status_layout = QVBoxLayout()

        self.position_label = QLabel()
        self.heading_label = QLabel()

        self.rgb_status = QLabel()
        self.thermal_status = QLabel()
        self.odom_status = QLabel()
        self.lidar_status = QLabel()

        status_layout.addWidget(
            self.position_label
        )

        status_layout.addWidget(
            self.heading_label
        )

        status_layout.addSpacing(20)

        status_layout.addWidget(
            self.rgb_status
        )

        status_layout.addWidget(
            self.thermal_status
        )

        status_layout.addWidget(
            self.lidar_status
        )

        status_layout.addWidget(
            self.odom_status
        )

        status_layout.addStretch()

        status_group.setLayout(
            status_layout
        )

        content_layout.addWidget(
            status_group,
            1
        )

        main_layout.addLayout(
            content_layout
        )

        # -------------------------
        # CONTROLS
        # -------------------------

        control_group = QGroupBox(
            'Manual Control'
        )

        controls = QGridLayout()

        self.forward_button = QPushButton(
            '▲\nForward\n(W)'
        )

        self.left_button = QPushButton(
            '↶\nTurn Left\n(A)'
        )

        self.stop_button = QPushButton(
            'STOP\n(Space)'
        )

        self.stop_button.setObjectName(
            'stopButton'
        )

        self.right_button = QPushButton(
            '↷\nTurn Right\n(D)'
        )

        self.back_button = QPushButton(
            '▼\nBackward\n(S)'
        )

        controls.addWidget(
            self.forward_button,
            0,
            1
        )

        controls.addWidget(
            self.left_button,
            1,
            0
        )

        controls.addWidget(
            self.stop_button,
            1,
            1
        )

        controls.addWidget(
            self.right_button,
            1,
            2
        )

        controls.addWidget(
            self.back_button,
            2,
            1
        )

        control_group.setLayout(
            controls
        )

        main_layout.addWidget(
            control_group
        )

        self.setLayout(
            main_layout
        )

        # -------------------------
        # BUTTON SIGNALS
        # -------------------------

        self.forward_button.pressed.connect(
            lambda: self.set_command(
                LINEAR_SPEED,
                0.0
            )
        )

        self.forward_button.released.connect(
            self.stop_command
        )

        self.back_button.pressed.connect(
            lambda: self.set_command(
                -LINEAR_SPEED,
                0.0
            )
        )

        self.back_button.released.connect(
            self.stop_command
        )

        self.left_button.pressed.connect(
            lambda: self.set_command(
                0.0,
                ANGULAR_SPEED
            )
        )

        self.left_button.released.connect(
            self.stop_command
        )

        self.right_button.pressed.connect(
            lambda: self.set_command(
                0.0,
                -ANGULAR_SPEED
            )
        )

        self.right_button.released.connect(
            self.stop_command
        )

        self.stop_button.clicked.connect(
            self.stop_command
        )


    # --------------------------------------------------
    # ROS PROCESSING
    # --------------------------------------------------

    def process_ros(self):

        try:

            rclpy.spin_once(
                self.node,
                timeout_sec=0.0
            )

        except Exception as exc:

            print(
                f'ROS callback error: {exc}'
            )


    # --------------------------------------------------
    # DISPLAY UPDATE
    # --------------------------------------------------

    def update_ui(self):

        self.position_label.setText(
            f'Position\n'
            f'X: {self.node.x:7.2f} m\n'
            f'Y: {self.node.y:7.2f} m\n'
            f'Z: {self.node.z:7.2f} m'
        )

        self.heading_label.setText(
            f'Heading: '
            f'{self.node.yaw_deg:6.1f}°'
        )

        now = time.monotonic()

        self.set_status(
            self.rgb_status,
            'RGB Camera',
            now - self.node.last_rgb_time < 1.5
        )

        self.set_status(
            self.thermal_status,
            'IR Camera',
            now - self.node.last_thermal_time < 1.5
        )

        self.set_status(
            self.lidar_status,
            'LiDAR',
            now - self.node.last_lidar_time < 1.5
        )

        self.set_status(
            self.odom_status,
            'Odometry',
            now - self.node.last_odom_time < 1.5
        )

        selected = self.view_selector.currentText()

        if selected == 'RGB Camera':

            self.show_image(
                self.node.latest_rgb_image,
                'Waiting for RGB camera...'
            )

        elif selected == 'IR / Thermal':

            self.show_image(
                self.node.latest_thermal_image,
                'Waiting for thermal camera...'
            )

        elif selected == 'LiDAR':

            self.show_lidar()


    def show_image(
        self,
        image,
        waiting_text
    ):

        if image is None:

            self.camera_label.clear()

            self.camera_label.setText(
                waiting_text
            )

            return

        pixmap = QPixmap.fromImage(
            image
        )

        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(
            pixmap
        )


    # --------------------------------------------------
    # LIDAR DISPLAY
    # --------------------------------------------------

    def show_lidar(self):

        scan = self.node.latest_scan

        if scan is None:

            self.camera_label.clear()

            self.camera_label.setText(
                'Waiting for LiDAR...'
            )

            return

        width = max(
            self.camera_label.width(),
            400
        )

        height = max(
            self.camera_label.height(),
            300
        )

        pixmap = QPixmap(
            width,
            height
        )

        pixmap.fill(
            QColor(
                5,
                5,
                5
            )
        )

        painter = QPainter(
            pixmap
        )

        painter.setRenderHint(
            QPainter.Antialiasing
        )

        centre_x = width / 2.0
        centre_y = height / 2.0

        maximum_display_range = min(
            scan.range_max,
            20.0
        )

        pixels_per_metre = (
            min(width, height)
            * 0.42
            / maximum_display_range
        )

        # Range circles.
        painter.setPen(
            QPen(
                QColor(
                    60,
                    60,
                    60
                ),
                1
            )
        )

        for metres in (
            5,
            10,
            15,
            20
        ):

            if metres > maximum_display_range:
                continue

            radius = (
                metres *
                pixels_per_metre
            )

            painter.drawEllipse(
                int(centre_x - radius),
                int(centre_y - radius),
                int(radius * 2),
                int(radius * 2)
            )

        # Crosshair.
        painter.drawLine(
            int(centre_x),
            0,
            int(centre_x),
            height
        )

        painter.drawLine(
            0,
            int(centre_y),
            width,
            int(centre_y)
        )

        # LiDAR scan points.
        painter.setPen(
            QPen(
                QColor(
                    50,
                    230,
                    120
                ),
                3
            )
        )

        angle = scan.angle_min

        for distance in scan.ranges:

            if (
                math.isfinite(distance)
                and
                scan.range_min < distance < maximum_display_range
            ):

                # Forward = upward on screen.
                screen_x = (
                    centre_x
                    +
                    distance
                    *
                    math.sin(angle)
                    *
                    pixels_per_metre
                )

                screen_y = (
                    centre_y
                    -
                    distance
                    *
                    math.cos(angle)
                    *
                    pixels_per_metre
                )

                painter.drawPoint(
                    int(screen_x),
                    int(screen_y)
                )

            angle += scan.angle_increment

        # Drone marker.
        painter.setPen(
            QPen(
                QColor(
                    255,
                    220,
                    80
                ),
                6
            )
        )

        painter.drawPoint(
            int(centre_x),
            int(centre_y)
        )

        # Forward direction marker.
        painter.setPen(
            QPen(
                QColor(
                    255,
                    220,
                    80
                ),
                2
            )
        )

        painter.drawLine(
            int(centre_x),
            int(centre_y),
            int(centre_x),
            int(centre_y - 30)
        )

        painter.end()

        self.camera_label.setPixmap(
            pixmap
        )


    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------

    def set_status(
        self,
        label,
        name,
        online
    ):

        if online:

            label.setText(
                f'● {name}: ONLINE'
            )

            label.setObjectName(
                'online'
            )

        else:

            label.setText(
                f'● {name}: OFFLINE'
            )

            label.setObjectName(
                'offline'
            )

        label.style().unpolish(
            label
        )

        label.style().polish(
            label
        )


    # --------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------

    def set_command(
        self,
        linear,
        angular
    ):

        self.linear_cmd = linear
        self.angular_cmd = angular

        self.publish_current_command()


    def publish_current_command(self):

        self.node.send_velocity(
            self.linear_cmd,
            self.angular_cmd
        )


    def stop_command(self):

        self.linear_cmd = 0.0
        self.angular_cmd = 0.0

        self.node.stop()


    # --------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------

    def keyPressEvent(
        self,
        event
    ):

        if event.isAutoRepeat():
            return

        key = event.key()

        if key == Qt.Key_W:

            self.set_command(
                LINEAR_SPEED,
                0.0
            )

        elif key == Qt.Key_S:

            self.set_command(
                -LINEAR_SPEED,
                0.0
            )

        elif key == Qt.Key_A:

            self.set_command(
                0.0,
                ANGULAR_SPEED
            )

        elif key == Qt.Key_D:

            self.set_command(
                0.0,
                -ANGULAR_SPEED
            )

        elif key == Qt.Key_Space:

            self.stop_command()

        # Sensor shortcuts.
        elif key == Qt.Key_1:

            self.view_selector.setCurrentIndex(
                0
            )

        elif key == Qt.Key_2:

            self.view_selector.setCurrentIndex(
                1
            )

        elif key == Qt.Key_3:

            self.view_selector.setCurrentIndex(
                2
            )


    def keyReleaseEvent(
        self,
        event
    ):

        if event.isAutoRepeat():
            return

        if event.key() in (
            Qt.Key_W,
            Qt.Key_S,
            Qt.Key_A,
            Qt.Key_D
        ):

            self.stop_command()


    # --------------------------------------------------
    # CLOSE
    # --------------------------------------------------

    def closeEvent(
        self,
        event
    ):

        self.stop_command()

        self.ros_timer.stop()
        self.command_timer.stop()
        self.ui_timer.stop()

        event.accept()


def main():

    rclpy.init()

    node = DroneUiNode()

    app = QApplication(
        sys.argv
    )

    window = DroneUI(
        node
    )

    window.show()

    result = app.exec_()

    node.stop()

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()

    sys.exit(result)


if __name__ == '__main__':
    main()