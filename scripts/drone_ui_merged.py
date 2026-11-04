#!/usr/bin/env python3

import sys
import math
import time
import numpy as np

import rclpy
from rescue_drone_node import RescueDroneNode
from mission_control import MissionController

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
    QTextEdit,
    QLineEdit,
)


LINEAR_SPEED = 0.5
ANGULAR_SPEED = 0.8

class DroneUI(QWidget):

    def __init__(self, ros_node):

        super().__init__()

        self.node = ros_node

        self.cmd_lx = 0.0
        self.cmd_ly = 0.0
        self.cmd_lz = 0.0
        self.cmd_ax = 0.0
        self.cmd_ay = 0.0
        self.cmd_az = 0.0
        
        self.prev_human_count = 0
        self.prev_fire_rgb = False
        self.prev_fire_thermal = False
        self.prev_flood = False
        self.pip_active = False
        self.is_auto_mode = False
        
        self.mission_ctrl = MissionController(self.node, self.log_event, self.run_survey)

        self.setWindowTitle('Search & Rescue Drone Control')
        self.resize(1150, 750)
        self.setFocusPolicy(Qt.StrongFocus)

        self.build_ui()

        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.process_ros)
        self.ros_timer.start(20)

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)

    # --------------------------------------------------
    # UI
    # --------------------------------------------------

    def build_ui(self):

        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #eeeeee; font-size: 15px; }
            QGroupBox { border: 1px solid #555555; border-radius: 8px; margin-top: 5px; padding: 5px; font-weight: bold; }
            QPushButton { background-color: #353535; border: 1px solid #666666; border-radius: 4px; padding: 6px; min-height: 20px; font-size: 12px; }
            QPushButton:pressed { background-color: #606060; }
            QPushButton:checked { background-color: #4a6b4a; border: 2px solid #72d572; }
            QPushButton#stopButton { background-color: #8a3030; font-weight: bold; font-size: 14px; min-height: 40px; min-width: 60px; }
            QPushButton#actionButton { background-color: #005588; font-weight: bold; border-color: #0088cc; }
            QComboBox { background-color: #303030; border: 1px solid #666666; border-radius: 5px; padding: 8px; min-width: 180px; }
            QLineEdit { background-color: #111111; color: #fff; border: 1px solid #666; border-radius: 4px; padding: 4px; text-align: center; }
            QLabel#camera { background-color: #000000; border: 1px solid #555555; }
            QLabel#online { color: #72d572; font-size: 13px; }
            QLabel#offline { color: #dd6666; font-size: 13px; }
            QTextEdit { background-color: #111111; color: #aaffaa; border: 1px solid #444444; font-family: monospace; font-size: 12px; }
        """)

        main_layout = QVBoxLayout()

        title = QLabel('SEARCH & RESCUE DRONE CONTROL')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Arial', 18, QFont.Bold))
        main_layout.addWidget(title)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel('Sensor View:'))
        self.view_selector = QComboBox()
        self.view_selector.addItems(['RGB Camera', 'IR / Thermal', 'LiDAR'])
        selector_layout.addWidget(self.view_selector)
        
        self.pip_button = QPushButton('Toggle PIP')
        self.pip_button.setCheckable(True)
        self.pip_button.clicked.connect(self.toggle_pip)
        selector_layout.addWidget(self.pip_button)

        self.swap_pip_button = QPushButton('Swap PIP Views ⮀')
        self.swap_pip_button.clicked.connect(self.swap_pip)
        selector_layout.addWidget(self.swap_pip_button)
        
        selector_layout.addSpacing(20)

        mission_group = QGroupBox('Operation Center: Mission Control')
        mission_layout = QHBoxLayout()
        mission_layout.setContentsMargins(5, 5, 5, 5)
        
        mission_layout.addWidget(QLabel('Target X:'))
        self.target_x_input = QLineEdit('2.0')
        self.target_x_input.setMaximumWidth(50)
        mission_layout.addWidget(self.target_x_input)
        
        mission_layout.addWidget(QLabel('Y:'))
        self.target_y_input = QLineEdit('0.0')
        self.target_y_input.setMaximumWidth(50)
        mission_layout.addWidget(self.target_y_input)
        
        self.launch_btn = QPushButton('🚀 Launch')
        self.launch_btn.setObjectName('actionButton')
        self.launch_btn.clicked.connect(self.launch_mission)
        mission_layout.addWidget(self.launch_btn)
        
        self.survey_btn = QPushButton('🔄 Survey')
        self.survey_btn.setObjectName('actionButton')
        self.survey_btn.clicked.connect(self.run_survey)
        mission_layout.addWidget(self.survey_btn)

        # UPDATED UI LABEL
        self.rtb_btn = QPushButton('🏠 RTB (-10.0, 7.0)')
        self.rtb_btn.clicked.connect(self.return_to_base)
        mission_layout.addWidget(self.rtb_btn)
        
        mission_group.setLayout(mission_layout)
        selector_layout.addWidget(mission_group)
        
        selector_layout.addStretch()
        main_layout.addLayout(selector_layout)

        content_layout = QHBoxLayout()

        camera_group = QGroupBox('Sensor Display')
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel('Waiting for sensor data...')
        self.camera_label.setObjectName('camera')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(600, 380)
        camera_layout.addWidget(self.camera_label)
        camera_group.setLayout(camera_layout)
        content_layout.addWidget(camera_group, 3)

        right_panel_layout = QVBoxLayout()
        
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
        status_layout.addSpacing(5)
        status_layout.addWidget(self.rgb_status)
        status_layout.addWidget(self.thermal_status)
        status_layout.addWidget(self.lidar_status)
        status_layout.addWidget(self.odom_status)
        status_group.setLayout(status_layout)
        right_panel_layout.addWidget(status_group)
        
        log_group = QGroupBox('Detection Logs')
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_panel_layout.addWidget(log_group)

        content_layout.addLayout(right_panel_layout, 1)
        main_layout.addLayout(content_layout)

        bottom_controls_layout = QHBoxLayout()

        alt_rot_group = QGroupBox('Altitude & Rotation')
        alt_rot_layout = QGridLayout()
        self.up_button = QPushButton('△ Fly Up (R)')
        self.down_button = QPushButton('▽ Fly Down (F)')
        self.ccw_button = QPushButton('↶ Turn CCW (Q)')
        self.cw_button = QPushButton('↷ Turn CW (E)')
        
        alt_rot_layout.addWidget(self.up_button, 0, 0)
        alt_rot_layout.addWidget(self.ccw_button, 0, 1)
        alt_rot_layout.addWidget(self.down_button, 1, 0)
        alt_rot_layout.addWidget(self.cw_button, 1, 1)
        alt_rot_group.setLayout(alt_rot_layout)
        bottom_controls_layout.addWidget(alt_rot_group)

        trans_group = QGroupBox('Translation')
        trans_main_layout = QVBoxLayout()
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel('Mode:'))
        self.manual_btn = QPushButton('Manual')
        self.manual_btn.setCheckable(True)
        self.manual_btn.setChecked(True)
        self.manual_btn.clicked.connect(self.set_manual_mode)
        
        self.auto_btn = QPushButton('Auto')
        self.auto_btn.setCheckable(True)
        self.auto_btn.clicked.connect(self.set_auto_mode)
        
        mode_layout.addWidget(self.manual_btn)
        mode_layout.addWidget(self.auto_btn)
        mode_layout.addStretch() 
        trans_main_layout.addLayout(mode_layout)

        nav_center_layout = QHBoxLayout()
        nav_center_layout.addStretch() 

        trans_layout = QGridLayout()
        self.forward_button = QPushButton('▲ Fwd (W)')
        self.left_button = QPushButton('◀ Left (A)')
        self.right_button = QPushButton('▶ Right (D)')
        self.back_button = QPushButton('▼ Back (S)')
        self.stop_button = QPushButton('STOP (Space)')
        self.stop_button.setObjectName('stopButton')

        trans_layout.addWidget(self.forward_button, 0, 1)
        trans_layout.addWidget(self.left_button, 1, 0)
        trans_layout.addWidget(self.stop_button, 1, 1)
        trans_layout.addWidget(self.right_button, 1, 2)
        trans_layout.addWidget(self.back_button, 2, 1)
        
        nav_center_layout.addLayout(trans_layout)
        nav_center_layout.addStretch() 
        
        trans_main_layout.addLayout(nav_center_layout)
        trans_group.setLayout(trans_main_layout)
        bottom_controls_layout.addWidget(trans_group)

        att_group = QGroupBox('Attitude Control (Pitch/Roll/Yaw)')
        att_layout = QGridLayout()
        self.pitch_down = QPushButton('Pitch Down (-ay)')
        self.pitch_up = QPushButton('Pitch Up (+ay)')
        self.roll_left = QPushButton('Roll Left (-ax)')
        self.roll_right = QPushButton('Roll Right (+ax)')
        self.yaw_left = QPushButton('Yaw Left (+az)')
        self.yaw_right = QPushButton('Yaw Right (-az)')

        att_layout.addWidget(self.pitch_down, 0, 0)
        att_layout.addWidget(self.roll_left, 0, 1)
        att_layout.addWidget(self.yaw_left, 0, 2)
        att_layout.addWidget(self.pitch_up, 1, 0)
        att_layout.addWidget(self.roll_right, 1, 1)
        att_layout.addWidget(self.yaw_right, 1, 2)
        att_group.setLayout(att_layout)
        bottom_controls_layout.addWidget(att_group)

        main_layout.addLayout(bottom_controls_layout)
        self.setLayout(main_layout)

        self.forward_button.pressed.connect(lambda: self.set_command(lx=LINEAR_SPEED))
        self.forward_button.released.connect(self.handle_release)
        self.back_button.pressed.connect(lambda: self.set_command(lx=-LINEAR_SPEED))
        self.back_button.released.connect(self.handle_release)
        self.left_button.pressed.connect(lambda: self.set_command(ly=LINEAR_SPEED))
        self.left_button.released.connect(self.handle_release)
        self.right_button.pressed.connect(lambda: self.set_command(ly=-LINEAR_SPEED))
        self.right_button.released.connect(self.handle_release)
        
        self.up_button.pressed.connect(lambda: self.set_command(lz=LINEAR_SPEED))
        self.up_button.released.connect(self.handle_release)
        self.down_button.pressed.connect(lambda: self.set_command(lz=-LINEAR_SPEED))
        self.down_button.released.connect(self.handle_release)
        self.ccw_button.pressed.connect(lambda: self.set_command(az=ANGULAR_SPEED))
        self.ccw_button.released.connect(self.handle_release)
        self.cw_button.pressed.connect(lambda: self.set_command(az=-ANGULAR_SPEED))
        self.cw_button.released.connect(self.handle_release)
        self.stop_button.clicked.connect(self.stop_command)

        self.pitch_down.pressed.connect(lambda: self.set_command(ay=-ANGULAR_SPEED))
        self.pitch_down.released.connect(self.handle_release)
        self.pitch_up.pressed.connect(lambda: self.set_command(ay=ANGULAR_SPEED))
        self.pitch_up.released.connect(self.handle_release)
        
        self.roll_left.pressed.connect(lambda: self.set_command(ax=-ANGULAR_SPEED))
        self.roll_left.released.connect(self.handle_release)
        self.roll_right.pressed.connect(lambda: self.set_command(ax=ANGULAR_SPEED))
        self.roll_right.released.connect(self.handle_release)
        
        self.yaw_left.pressed.connect(lambda: self.set_command(az=ANGULAR_SPEED))
        self.yaw_left.released.connect(self.handle_release)
        self.yaw_right.pressed.connect(lambda: self.set_command(az=-ANGULAR_SPEED))
        self.yaw_right.released.connect(self.handle_release)

    # --------------------------------------------------
    # DELEGATED MISSION CONTROL LOGIC
    # --------------------------------------------------
    def launch_mission(self):
        try:
            tx = float(self.target_x_input.text())
            ty = float(self.target_y_input.text())
            self.mission_ctrl.start_mission(tx, ty)
        except ValueError:
            self.log_event("❌ ERROR: Invalid target coordinates!")

    def return_to_base(self):
        # UPDATED RTB COORDINATES
        self.mission_ctrl.start_rtb(-10.0, 7.0)

    def run_survey(self):
        self.mission_ctrl.cancel_mission()
        self.log_event("🔄 SURVEY INITIATED: Commencing 360° sensor sweep...")
        self.set_auto_mode()
        self.set_command(az=0.5)

    # --------------------------------------------------
    # MODES & TOGGLES
    # --------------------------------------------------
    def set_manual_mode(self):
        self.is_auto_mode = False
        self.manual_btn.setChecked(True)
        self.auto_btn.setChecked(False)
        self.stop_command()

    def set_auto_mode(self):
        self.is_auto_mode = True
        self.auto_btn.setChecked(True)
        self.manual_btn.setChecked(False)

    def handle_release(self):
        if not self.is_auto_mode:
            self.stop_command()

    def toggle_pip(self, checked):
        self.pip_active = checked

    def swap_pip(self):
        current = self.view_selector.currentText()
        if current == 'RGB Camera':
            self.view_selector.setCurrentText('IR / Thermal')
        elif current == 'IR / Thermal':
            self.view_selector.setCurrentText('RGB Camera')
        
    def log_event(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()

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
        state_str = getattr(self.node, 'flight_state', 'UNKNOWN')

        self.position_label.setText(
            f'Position (X, Y, Z)\n'
            f'{self.node.x:6.2f}, {self.node.y:6.2f}, {self.node.z:6.2f}\n'
            f'State: {state_str}'
        )

        self.heading_label.setText(
            f'Heading: '
            f'{math.degrees(self.node.yaw):6.1f}°'
        )

        self.mission_ctrl.update(state_str)
        
        current_human_count = getattr(self.node, 'human_count', 0)
        if current_human_count > self.prev_human_count:
            for i in range(self.prev_human_count + 1, current_human_count + 1):
                self.log_event(f"HUMAN #{i} DETECTED at X:{self.node.x:.1f}, Y:{self.node.y:.1f}")
        self.prev_human_count = current_human_count
        
        if self.node.fire_rgb and not self.prev_fire_rgb:
            self.log_event(f"FIRE (RGB) DETECTED at X:{self.node.x:.1f}, Y:{self.node.y:.1f}")
        self.prev_fire_rgb = self.node.fire_rgb
        
        if getattr(self.node, 'fire_thermal', False) and not self.prev_fire_thermal:
            self.log_event(f"FIRE (THERMAL) CONFIRMED at X:{self.node.x:.1f}, Y:{self.node.y:.1f}")
        self.prev_fire_thermal = getattr(self.node, 'fire_thermal', False)
        
        if self.node.flood_detected and not self.prev_flood:
            self.log_event(f"FLOOD HAZARD DETECTED at X:{self.node.x:.1f}, Y:{self.node.y:.1f}")
        self.prev_flood = self.node.flood_detected

        self.set_status(self.rgb_status, 'RGB Camera', self.node.rgb_frame is not None)
        self.set_status(self.thermal_status, 'IR Camera', self.node.thermal_frame is not None)
        self.set_status(self.lidar_status, 'LiDAR', self.node.lidar_frame is not None)
        self.set_status(self.odom_status, 'Odometry', True)

        selected = self.view_selector.currentText()

        if selected == 'RGB Camera':
            self.show_image(self.node.rgb_frame, self.node.thermal_frame, 'Waiting for RGB camera...')
        elif selected == 'IR / Thermal':
            self.show_image(self.node.thermal_frame, self.node.rgb_frame, 'Waiting for thermal camera...')
        elif selected == 'LiDAR':
            self.show_image(self.node.lidar_frame, None, 'Waiting for LiDAR...')

    def show_image(self, main_data, pip_data, waiting_text):
        if main_data is None:
            self.camera_label.clear()
            self.camera_label.setText(waiting_text)
            return

        main_pixmap = QPixmap()
        main_pixmap.loadFromData(main_data)
        main_pixmap = main_pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        if self.pip_active and pip_data is not None:
            pip_pixmap = QPixmap()
            pip_pixmap.loadFromData(pip_data)
            
            pip_target_w = main_pixmap.width() // 4
            pip_target_h = main_pixmap.height() // 4
            pip_pixmap = pip_pixmap.scaled(pip_target_w, pip_target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            actual_pip_w = pip_pixmap.width()
            actual_pip_h = pip_pixmap.height()
            
            painter = QPainter(main_pixmap)
            
            margin = 15
            x = main_pixmap.width() - actual_pip_w - margin
            y = main_pixmap.height() - actual_pip_h - margin
            
            painter.setPen(QPen(Qt.white, 2))
            painter.drawRect(x - 1, y - 1, actual_pip_w + 2, actual_pip_h + 2)
            painter.drawPixmap(x, y, pip_pixmap)
            painter.end()

        self.camera_label.setPixmap(main_pixmap)

    # --------------------------------------------------
    # STATUS & COMMAND
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

    def set_command(self, lx=0.0, ly=0.0, lz=0.0, ax=0.0, ay=0.0, az=0.0):
        self.mission_ctrl.cancel_mission() 
        self.cmd_lx = lx
        self.cmd_ly = ly
        self.cmd_lz = lz
        self.cmd_ax = ax
        self.cmd_ay = ay
        self.cmd_az = az
        self.publish_current_command()

    def publish_current_command(self):
        self.node.set_command(
            lx=self.cmd_lx,
            ly=self.cmd_ly,
            lz=self.cmd_lz,
            ax=self.cmd_ax,
            ay=self.cmd_ay,
            az=self.cmd_az
        )

    def stop_command(self):
        self.mission_ctrl.cancel_mission()
        self.cmd_lx = 0.0
        self.cmd_ly = 0.0
        self.cmd_lz = 0.0
        self.cmd_ax = 0.0
        self.cmd_ay = 0.0
        self.cmd_az = 0.0
        self.node.set_command(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # --------------------------------------------------
    # KEYBOARD & CLOSE
    # --------------------------------------------------

    def keyPressEvent(self, event):
        if event.isAutoRepeat(): return
        key = event.key()

        if key == Qt.Key_W: self.set_command(lx=LINEAR_SPEED)
        elif key == Qt.Key_S: self.set_command(lx=-LINEAR_SPEED)
        elif key == Qt.Key_A: self.set_command(ly=LINEAR_SPEED)
        elif key == Qt.Key_D: self.set_command(ly=-LINEAR_SPEED)
        elif key == Qt.Key_R: self.set_command(lz=LINEAR_SPEED)
        elif key == Qt.Key_F: self.set_command(lz=-LINEAR_SPEED)
        elif key == Qt.Key_Q: self.set_command(az=ANGULAR_SPEED)
        elif key == Qt.Key_E: self.set_command(az=-ANGULAR_SPEED)
        
        elif key == Qt.Key_Space: self.stop_command()
        elif key == Qt.Key_1: self.view_selector.setCurrentIndex(0)
        elif key == Qt.Key_2: self.view_selector.setCurrentIndex(1)
        elif key == Qt.Key_3: self.view_selector.setCurrentIndex(2)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): return
        if event.key() in (Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E, Qt.Key_R, Qt.Key_F):
            self.handle_release()

    def closeEvent(self, event):
        self.stop_command()
        self.ros_timer.stop()
        self.ui_timer.stop()
        event.accept()


def main():
    rclpy.init()
    node = RescueDroneNode()
    app = QApplication(sys.argv)
    window = DroneUI(node)
    window.show()

    result = app.exec_()
    node.set_command(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(result)

if __name__ == '__main__':
    main()