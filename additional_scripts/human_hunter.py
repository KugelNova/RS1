import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import subprocess
from nav_msgs.msg import Odometry

class HumanHunterNode(Node):
    def __init__(self):
        super().__init__('celsius_human_hunter')
        
        self.image_pub = self.create_publisher(Image, '/parrot1/camera/thermal/image_color', 10)
        self.vel_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_callback, 10)
        
        # RGB camera subscriber
        self.latest_rgb = None
        self.rgb_sub = self.create_subscription(
            Image, 
            '/parrot1/camera/image', 
            self.rgb_callback, 
            10
        )
        self.known_humans = []
        self.human_counter = 0

        self.current_altitude = 0.0
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/parrot1/odometry', 
            self.odom_callback,
            10
        )
        
        self.known_humans = []  # List of (x, y) coordinates
        self.target_coords = None
        self.state = "SEARCHING"
        self.scan_start_time = None
        
        self.frame_count = 0
        
        self.get_logger().info('Celsius Human Hunter Active: Searching for ~32 C human signatures...')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_altitude = msg.pose.pose.position.z
        
    
    def rgb_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        try:
            self.latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        except Exception:
            pass

        
    
    def save_human_snapshots(self, gray_img, inferno_img, rgb_img, human_id, temp):
        import os
        os.makedirs("/tmp/rescue_snaps", exist_ok=True)
        
        # Save views
        cv2.imwrite(f"/tmp/rescue_snaps/human_{human_id}_gray.png", gray_img)
        cv2.imwrite(f"/tmp/rescue_snaps/human_{human_id}_inferno.png", inferno_img)
        if rgb_img is not None:
            # Convert RGB to BGR for OpenCV saving
            cv2.imwrite(f"/tmp/rescue_snaps/human_{human_id}_rgb.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f"📸 RGB Snapshot successfully saved for Human #{human_id}")
        else:
            self.get_logger().warn(f"⚠️ RGB Snapshot was None for Human #{human_id} (Check topic)")
       
       
       # Flash coordinates & status
        print("\n" + "!"*40)
        print(f"🚨 HUMAN #{human_id} FOUND! 🚨")
        print(f"🌡️ Temperature: {temp}")
        print(f"📍 Location Coordinates -> X: {self.current_x:.2f}m, Y: {self.current_y:.2f}m, Alt: {self.current_altitude:.2f}m")
        print(f"📸 Snapshots saved to /tmp/rescue_snaps/")
        print("!"*40 + "\n")
    
    def thermal_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        
        
        # height/width definitions
        h, w = msg.height, msg.width 

        # --- Background/Ambient Temperature Logging ---
        min_pixel = np.min(img)
        mean_pixel = np.mean(img)
        max_frame_pixel = np.max(img)

        min_temp_c = (min_pixel * 3.0) - 273.15
        mean_temp_c = (mean_pixel * 3.0) - 273.15
        
        
        # 1. Colorization for RViz HUD
        clipped_img = np.clip(img, 90, 115)
        normalized_img = cv2.normalize(clipped_img, None, 0, 255, cv2.NORM_MINMAX)
        color_img = cv2.applyColorMap(normalized_img, cv2.COLORMAP_INFERNO)

        # 2. Object Detection (Isolate specific temperature bands)
        # Human band: ~26.8C to ~41.8C (Raw 100 to 105)
        human_mask = cv2.inRange(img, 100, 105)

        # Fire band: > 470C (Raw 250 to 255)
        fire_mask = cv2.inRange(img, 250, 255)
        
        # Combine both masks to completely ignore the hot 196 background
        mask = cv2.bitwise_or(human_mask, fire_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        human_found = False
        target_x = 0
        target_h = 0 
        largest_area = 0   
        status = "SEARCHING..."
        terminal_output = [] # FIX 1: Restored this line

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            
            if area > 15: 
                roi_img = img[y:y+bh, x:x+bw]
                roi_mask = mask[y:y+bh, x:x+bw]
                valid_pixels = roi_img[roi_mask == 255]
                
                if len(valid_pixels) == 0:
                    continue
                    
                max_val = np.max(valid_pixels)
                max_c = (max_val * 3.0) - 273.15
                
                if max_val >= 250:
                    temp_str = "> 491.8 C"
                    color = (0, 0, 255)
                    label = "BURNING OBJECT"
                elif 25 < max_c < 45:
                    temp_str = f"{max_c:.1f} C"
                    color = (0, 255, 0)
                    label = "HUMAN DETECTED"
                    
                    if area > largest_area:
                        largest_area = area
                        target_x = x + (bw / 2)
                        target_h = bh
                        human_found = True
                                                
                else:
                    temp_str = f"{max_c:.1f} C"
                    color = (255, 255, 0)
                    label = "Warm Object"
                
                
                # --- Widen the box to match human physical proportions (~45% of height) ---
                draw_w = max(bw, int(bh * 0.45))
                draw_x = max(0, int(x + (bw / 2) - (draw_w / 2)))
                
                cv2.rectangle(color_img, (draw_x, y), (draw_x + draw_w, y + bh), color, 2)
                # -------------------------------------------------------------------------------
                #cv2.putText(color_img, f"{label}: {temp_str}", (x, y - 5), font, 0.5, (255, 255, 255), 1)
                    
                terminal_output.append(f"{label} at {temp_str}")

        
        # 3. Autonomous Flight Logic with Safe Hover
        twist = Twist()

        # --- Proportional Altitude Control ---
        target_altitude = 1.8  # Target height in meters
        altitude_error = target_altitude - self.current_altitude

        # Calculate raw speed, then clamp it between -0.5 and 0.5
        raw_z_speed = float(altitude_error * 0.8)
        twist.linear.z = max(-0.5, min(0.5, raw_z_speed)) # Adjust vertical speed
        # ------------------------------------------

        if human_found:
            # 1. Turn to center the human in the camera view
            error_x = (w / 2) - target_x
            twist.angular.z = float(error_x * 0.005) # Proportional turning speed
            
            target_height = 90 
            
            if target_h < (target_height - 10):
                twist.linear.x = 0.5  # Too far: Move forward
                status = "HUMAN FOUND - MOVING CLOSER"
                self.get_logger().info('Tracking: Moving closer...',throttle_duration_sec=2.0)
            elif target_h > (target_height + 10):
                twist.linear.x = -0.5 # Too close: Back up
                status = "HUMAN FOUND - BACKING UP"
                self.get_logger().info('Tracking: Backing up...', throttle_duration_sec=2.0)
            else:
                twist.linear.x = 0.0  # Perfect distance: Hover
                #twist.angular.z = 0.0
                status = "HUMAN FOUND - HOVERING"
                self.get_logger().info('HOVERING: Target safely reached.', throttle_duration_sec=3.0)
                
                # --- TRIGGER SNAPSHOT ONLY WHEN HOVERING ---
                is_new_human = True
                for hx, hy in self.known_humans:
                    dist = np.sqrt((self.current_x - hx)**2 + (self.current_y - hy)**2)
                    if dist < 2.0:  # Within 2 meters is the same person
                        is_new_human = False
                        break
                
                if is_new_human:
                    self.known_humans.append((self.current_x, self.current_y))
                    self.human_counter += 1
                    # Pass the current frames to the snapshot saver
                    self.save_human_snapshots(img, color_img, self.latest_rgb, self.human_counter, "Verified Target")
                # -------------------------------------------

        else:
            # If no human is seen, spin in place to search
            status = "SEARCHING..."
            twist.angular.z = 0.3
            twist.linear.x = 0.0
            
        # Publish the movement commands to the drone
        self.vel_pub.publish(twist)

        cv2.putText(color_img, f"Status: {status}", (10, 30), font, 0.6, (255, 255, 255), 2)
        
        self.frame_count += 1
        

        # Publish to RViz
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height, out_msg.width = h, w
        out_msg.encoding = 'bgr8'
        out_msg.step = w * 3
        out_msg.data = color_img.tobytes()
        self.image_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    #node = CelsiusHumanHunter()
    node = HumanHunterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        #node.vel_pub.publish(Twist()) # Stop the drone gracefully
        node.get_logger().info('Keyboard interrupt detected. Halting drone...')   
    finally:
        # 1. Publish the stop command BEFORE shutting down
        node.vel_pub.publish(Twist())

        # 2. Destroy the node and shutdown the context
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()