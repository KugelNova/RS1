import os
import threading
import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import cv2
import numpy as np

class WebTeleopNode(Node):
    def __init__(self):
        super().__init__('web_teleop_node')
        
        self.cmd_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', 10)
        self.current_twist = Twist()
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.rgb_frame = None
        self.thermal_frame = None
        self.lidar_frame = None
        
        # Telemetry & State
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0
        self.flight_state = "STOPPED / LEVELED"
        
        # Detection State Variables
        self.human_detected = False
        self.detect_x = 0.0
        self.detect_y = 0.0
        self.last_snap_time = 0.0
        os.makedirs("/tmp/rescue_snaps", exist_ok=True)
        
        # Autopilot Variables
        self.is_navigating = False
        self.target_x, self.target_y, self.target_z = 0.0, 0.0, 0.0
        self.target_speed = 0.5
        
        self.create_subscription(Image, '/parrot1/camera/image', self.rgb_cb, 10)
        self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_cb, 10)
        self.create_subscription(Odometry, '/parrot1/odometry', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/parrot1/scan', self.lidar_cb, 10)

    def set_command(self, lx=0.0, ly=0.0, lz=0.0, ax=0.0, ay=0.0, az=0.0):
        self.is_navigating = False 
        
        if lx == 0.0 and ly == 0.0 and lz == 0.0 and ax == 0.0 and ay == 0.0 and az == 0.0:
            self.flight_state = "STOPPED / LEVELED"
        else:
            self.flight_state = "MANUAL OVERRIDE"

        self.current_twist.linear.x = float(lx)
        self.current_twist.linear.y = float(ly)
        self.current_twist.linear.z = float(lz)
        self.current_twist.angular.x = float(ax)
        self.current_twist.angular.y = float(ay)
        self.current_twist.angular.z = float(az)
        
    def goto_target(self, x, y, z, speed):
        self.target_x, self.target_y, self.target_z = float(x), float(y), float(z)
        self.target_speed = float(speed)
        self.is_navigating = True
        self.flight_state = "AUTOPILOT RUNNING"

    def timer_callback(self):
        if self.is_navigating:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dz = self.target_z - self.z
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 0.3 and abs(dz) < 0.2:
                self.is_navigating = False
                self.flight_state = "TARGET REACHED"
                self.current_twist = Twist() 
            else:
                lz = max(-self.target_speed, min(self.target_speed, dz * 0.5))
                target_yaw = math.atan2(dy, dx)
                yaw_error = math.atan2(math.sin(target_yaw - self.yaw), math.cos(target_yaw - self.yaw))
                az = max(-0.5, min(0.5, yaw_error * 1.2))
                
                if abs(yaw_error) < 0.5:
                    lx = max(-self.target_speed, min(self.target_speed, distance * 0.5))
                else:
                    lx = 0.0
                    
                self.current_twist.linear.x = float(lx)
                self.current_twist.linear.y = 0.0
                self.current_twist.linear.z = float(lz)
                self.current_twist.angular.x = 0.0
                self.current_twist.angular.y = 0.0
                self.current_twist.angular.z = float(az)

        self.cmd_pub.publish(self.current_twist)
        
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        
        q = msg.pose.pose.orientation
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = math.sqrt(1 + 2 * (q.w * q.y - q.x * q.z))
        cosp = math.sqrt(1 - 2 * (q.w * q.y - q.x * q.z))
        self.pitch = 2 * math.atan2(sinp, cosp) - math.pi / 2
        
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def rgb_cb(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img)
            self.rgb_frame = buffer.tobytes()
        except Exception: pass
            
    def thermal_cb(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            
            # 1. Visualization: Create the colored image for the web UI
            clipped = np.clip(img, 90, 115)
            normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
            color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            
            # 2. Detection: Use RAW absolute heat to prevent false positives
            # Apply Gaussian Blur to remove sensor noise and small hot artifacts
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Threshold strictly on raw pixel intensity (100) to find genuine heat sources
            _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_this_frame = False
            for contour in contours:
                # Increased minimum area to 150 to ignore small rocks and leaves
                if cv2.contourArea(contour) > 30: 
                    detected_this_frame = True
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.drawMarker(color_img, (x + w//2, y + h//2), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(color_img, f"DETECTED X:{self.x:.1f} Y:{self.y:.1f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.human_detected = detected_this_frame
            if detected_this_frame:
                self.detect_x = self.x
                self.detect_y = self.y

            # 3. Snapshotting
            if self.human_detected:
                current_time = time.time()
                if current_time - self.last_snap_time > 5.0:
                    snap_path = f"/tmp/rescue_snaps/human_detect_{int(current_time)}.png"
                    cv2.imwrite(snap_path, color_img)
                    self.last_snap_time = current_time

            _, buffer = cv2.imencode('.jpg', color_img)
            self.thermal_frame = buffer.tobytes()
        except Exception: pass

    def lidar_cb(self, msg):
        try:
            img_size = 400
            radar_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            center = (img_size // 2, img_size // 2)
            cv2.circle(radar_img, center, 4, (0, 0, 255), -1)
            scale = 15.0 
            cv2.circle(radar_img, center, int(5 * scale), (50, 50, 50), 1)
            cv2.circle(radar_img, center, int(10 * scale), (50, 50, 50), 1)
            angle = msg.angle_min
            for r in msg.ranges:
                if msg.range_min < r < msg.range_max and not np.isinf(r):
                    x_ros = r * np.cos(angle)
                    y_ros = r * np.sin(angle)
                    img_x = int(center[0] - y_ros * scale)
                    img_y = int(center[1] - x_ros * scale)
                    if 0 <= img_x < img_size and 0 <= img_y < img_size:
                        cv2.circle(radar_img, (img_x, img_y), 2, (0, 255, 0), -1)
                angle += msg.angle_increment
            _, buffer = cv2.imencode('.jpg', radar_img)
            self.lidar_frame = buffer.tobytes()
        except Exception: pass