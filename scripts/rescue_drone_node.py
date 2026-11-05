import os
import threading
import time
import math
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import cv2
import numpy as np

class RescueDroneNode(Node):
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
        self.fire_detected = False
        self.flood_detected = False
        
        self.fire_rgb = False
        self.fire_thermal = False

        os.makedirs("/tmp/rescue_snaps", exist_ok=True)
        
        # Autopilot Variables
        self.is_navigating = False
        self.target_x, self.target_y, self.target_z = 0.0, 0.0, 0.0
        self.target_speed = 0.5
        
        self.create_subscription(Image, '/parrot1/camera/image', self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/parrot1/odometry', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/parrot1/scan', self.lidar_cb, qos_profile_sensor_data)

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

        # --- HARDWARE SAFETY INTERLOCK ---
        # Refuse to descend below 5.0m if fire or flood is currently detected
        if self.is_navigating and (getattr(self, 'fire_thermal', False) or getattr(self, 'flood_detected', False)):
            if self.target_z < 5.0:
                self.target_z = 5.0  # Force drone to stay at least 5m high
               
       
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
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # --- DYNAMIC HORIZON MASKING ---
            dynamic_horizon = int(msg.height * 0.45 + (self.pitch * msg.height))
            horizon = max(0, min(dynamic_horizon, msg.height))
            
            hsv_masked = hsv.copy()
            hsv_masked[0:horizon, :] = 0
            
            # --- 1. Flood & Water Body Detection ---
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            water_mask = cv2.inRange(hsv_masked, lower_blue, upper_blue)
            
            ground_area = msg.width * (msg.height - horizon)
            if ground_area <= 0: ground_area = 1 
            
            water_ratio = cv2.countNonZero(water_mask) / ground_area
            self.flood_detected = (water_ratio > 0.50)
            
            if self.flood_detected:
                cv2.putText(img_bgr, f"FLOOD HAZARD: {water_ratio*100:.1f}%", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            elif water_ratio > 0.03:
                cv2.putText(img_bgr, f"WATER BODY: {water_ratio*100:.1f}%", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            # --- 2. Unified Fire Detection (Orange/Red) ---
            lower_fire = np.array([5, 200, 220])
            upper_fire = np.array([25, 255, 255])
            fire_mask = cv2.inRange(hsv_masked, lower_fire, upper_fire)
            
            fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.fire_rgb = False
            self.rgb_fire_center = None
            
            valid_fire_contours = [c for c in fire_contours if cv2.contourArea(c) > 50]
            
            if valid_fire_contours:
                self.fire_rgb = True
                all_points = np.vstack(valid_fire_contours)
                x, y, w, h = cv2.boundingRect(all_points)
                self.rgb_fire_center = (x + w // 2, y + h // 2)
                
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(img_bgr, "FIRE (VISUAL)", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', img_bgr)
            self.rgb_frame = buffer.tobytes()
        except Exception: pass

    def thermal_cb(self, msg):
        try:
            if msg.encoding == '32FC1':
                img = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding in ['16UC1', 'mono16']:
                img = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            
            # --- Dynamic Horizon for Thermal ---
            dynamic_horizon = int(msg.height * 0.45 + (self.pitch * msg.height))
            horizon = max(0, min(dynamic_horizon, msg.height))
            
            # --- 1. Identify Fire Pixels ---
            # Removed the flawed 16-bit overflow logic that flagged cold objects as fire
            fire_mask_base = cv2.inRange(img, 35000, 65535)
            fire_mask_base[0:horizon, :] = 0  # Ignore dynamic sky
            
            # --- 2. Render UI Image (Force fire to max brightness) ---
            ui_img = img.copy()
            ui_img[fire_mask_base > 0] = 32500
            
            clipped_vis = np.clip(ui_img, 28000, 32500)
            img_normalized = cv2.normalize(clipped_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_img = cv2.applyColorMap(img_normalized, cv2.COLORMAP_INFERNO)
            
            kernel = np.ones((15, 15), np.uint8)
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            
            # --- 3. Human Detection ---
            human_mask = cv2.inRange(blurred, 29000, 32000)
            fire_halo = cv2.dilate(fire_mask_base, kernel, iterations=6)
            human_mask = cv2.bitwise_and(human_mask, cv2.bitwise_not(fire_halo))
            
            human_thresh = cv2.morphologyEx(human_mask, cv2.MORPH_CLOSE, kernel)
            human_thresh = cv2.dilate(human_thresh, kernel, iterations=2)
            human_contours, _ = cv2.findContours(human_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_this_frame = False
            target_count = 0
            for contour in human_contours:
                if cv2.contourArea(contour) > 400:
                    detected_this_frame = True
                    target_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.drawMarker(color_img, (x + w // 2, y + h // 2), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
                    cv2.putText(color_img, f"HUMAN #{target_count}", (x, max(28, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            
            self.human_count = target_count
            self.human_detected = detected_this_frame

            if detected_this_frame:
                self.detect_x = self.x
                self.detect_y = self.y
                cv2.rectangle(color_img, (0, 0), (color_img.shape[1], 20), (0, 0, 0), -1)
                cv2.putText(color_img, f"TARGETS: {target_count} | GPS: X:{self.x:.1f} Y:{self.y:.1f}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

            # --- 4. Fire Detection (Thermal Confirmation via Sensor Fusion) ---
            fire_thresh_hot = cv2.morphologyEx(fire_mask_base, cv2.MORPH_CLOSE, kernel)
            fire_thresh_hot = cv2.dilate(fire_thresh_hot, kernel, iterations=2)
            fire_contours, _ = cv2.findContours(fire_thresh_hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            self.fire_thermal = False
            for contour in fire_contours:
                if cv2.contourArea(contour) > 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    t_cx, t_cy = x + w // 2, y + h // 2
                    
                    if getattr(self, 'fire_rgb', False) and getattr(self, 'rgb_fire_center', None) is not None:
                        rx, ry = self.rgb_fire_center
                        if abs(t_cx - rx) < 80 and abs(t_cy - ry) < 80:
                            self.fire_thermal = True
                            cv2.rectangle(color_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.drawMarker(color_img, (t_cx, t_cy), (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
                            cv2.putText(color_img, "FIRE CONFIRMED", (x, max(28, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', color_img)
            self.thermal_frame = buffer.tobytes()
        except Exception as e:
            import traceback
            print("\n--- THERMAL CALLBACK CRASH ---")
            traceback.print_exc()
        
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