import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import subprocess

class CelsiusHumanHunter(Node):
    def __init__(self):
        super().__init__('celsius_human_hunter')
        
        self.image_pub = self.create_publisher(Image, '/parrot1/camera/thermal/image_color', 10)
        self.vel_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_callback, 10)
        
        self.frame_count = 0

        
        self.get_logger().info('Celsius Human Hunter Active: Searching for ~32 C human signatures...')

    def thermal_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        
        # Fix: Re-added height/width definitions so the flight logic doesn't crash
        h, w = msg.height, msg.width 

        # --- NEW: Background/Ambient Temperature Logging ---
        min_pixel = np.min(img)
        mean_pixel = np.mean(img)
        max_frame_pixel = np.max(img)

        min_temp_c = (min_pixel * 3.0) - 273.15
        mean_temp_c = (mean_pixel * 3.0) - 273.15
        
        # Print the background and environmental temperatures to the terminal
        self.get_logger().info(f"Environment Ambient (Min): ~{min_temp_c:.1f} C")
        self.get_logger().info(f"Frame Average: ~{mean_temp_c:.1f} C | Frame Max: {max_frame_pixel} (Raw)")
        # ---------------------------------------------------

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
                
                #cv2.rectangle(color_img, (x, y), (x+bw, y+bh), color, 2)
                # --- NEW: Widen the box to match human physical proportions (~45% of height) ---
                draw_w = max(bw, int(bh * 0.45))
                draw_x = max(0, int(x + (bw / 2) - (draw_w / 2)))
                
                cv2.rectangle(color_img, (draw_x, y), (draw_x + draw_w, y + bh), color, 2)
                # -------------------------------------------------------------------------------
                #cv2.putText(color_img, f"{label}: {temp_str}", (x, y - 5), font, 0.5, (255, 255, 255), 1)
                    
                terminal_output.append(f"{label} at {temp_str}")

        # FIX 2: Flight logic un-indented so it runs after the loop finishes
        # 3. Autonomous Flight Logic with Safe Hover
        twist = Twist()

        if human_found:
            # 1. Turn to center the human in the camera view
            error_x = (w / 2) - target_x
            twist.angular.z = float(error_x * 0.005) # Proportional turning speed
            
            target_height = 70 
            
            if target_h < (target_height - 10):
                twist.linear.x = 0.5  # Too far: Move forward
                status = "HUMAN FOUND - MOVING CLOSER"
                self.get_logger().info('Tracking: Moving closer...')
            elif target_h > (target_height + 10):
                twist.linear.x = -0.5 # Too close: Back up
                status = "HUMAN FOUND - BACKING UP"
                self.get_logger().info('Tracking: Backing up...')
            else:
                twist.linear.x = 0.0  # Perfect distance: Hover
                #twist.angular.z = 0.0
                status = "HUMAN FOUND - HOVERING"
                self.get_logger().info('HOVERING: Target safely reached.')
                
        else:
            # If no human is seen, spin in place to search
            status = "SEARCHING..."
            twist.angular.z = 0.3
            twist.linear.x = 0.0
            
        # Publish the movement commands to the drone
        self.vel_pub.publish(twist)

        cv2.putText(color_img, f"Status: {status}", (10, 30), font, 0.6, (255, 255, 255), 2)
        
        # 4. Terminal Output (Throttled to ~2 seconds)
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            print('========================================')
            print(f'Drone Status: {status}')
            for out in terminal_output:
                print(f" - {out}")
            print('========================================')

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
    node = CelsiusHumanHunter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.vel_pub.publish(Twist()) # Stop the drone gracefully
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()