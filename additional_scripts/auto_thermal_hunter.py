import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class AutoThermalHunter(Node):
    def __init__(self):
        super().__init__('auto_thermal_hunter')
        
        # 1. Publisher for the RViz video feed
        self.image_pub = self.create_publisher(Image, '/parrot1/camera/thermal/image_color', 10)
        
        # 2. Publisher for Drone Flight Controls
        self.vel_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', 10)
        
        # 3. Subscriber for Raw Thermal Data
        self.subscription = self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_callback, 10)
        
        self.get_logger().info('Auto-Hunter Active: Scanning environment for heat signatures...')

    def thermal_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        h, w = msg.height, msg.width

        # Find the coordinates and value of the hottest pixel
        max_val = np.max(img)
        max_k = max_val * 3.0
        max_y, max_x = np.unravel_index(np.argmax(img), img.shape)

        # ==========================================
        # AUTONOMOUS FLIGHT LOGIC (Visual Servoing)
        # ==========================================
        twist = Twist()
        
        # If a significant heat source is detected (above ambient grass)
        if max_val > 98:
            # Calculate how far off-center the heat source is
            error_x = (w / 2) - max_x
            
            # Proportional control: Steer toward the target
            twist.angular.z = error_x * 0.005
            
            # If the target is roughly centered in the camera, fly forward
            if abs(error_x) < 40:
                twist.linear.x = 0.5
        else:
            # No heat detected: Rotate slowly in place to search the area
            twist.angular.z = 0.3

        # Execute the flight command
        self.vel_pub.publish(twist)

        # ==========================================
        # COLORIZATION & HUD RENDERING
        # ==========================================
        clipped_img = np.clip(img, 90, 115)
        normalized_img = cv2.normalize(clipped_img, None, 0, 255, cv2.NORM_MINMAX)
        color_img = cv2.applyColorMap(normalized_img, cv2.COLORMAP_INFERNO)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Convert Kelvin to Celsius
        max_c_display = max_k - 273.15
        
        # Draw tracking crosshairs and temperature text
        cv2.circle(color_img, (max_x, max_y), 15, (255, 0, 0), 2)
        cv2.line(color_img, (w//2, 0), (w//2, h), (0, 255, 0), 1) # Center alignment line
        
        # status_text = "Searching..." if max_val <= 98 else f"Tracking: {max_k} K" #previous statement before changed to below line
        status_text = "Searching..." if max_val <= 98 else f"Tracking: {max_c_display:.1f} C"
        cv2.putText(color_img, status_text, (10, 30), font, 0.6, (255, 255, 255), 2)

        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height, out_msg.width = h, w
        out_msg.encoding = 'bgr8'
        out_msg.step = w * 3
        out_msg.data = color_img.tobytes()
        self.image_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutoThermalHunter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Stop the drone before shutting down
        node.vel_pub.publish(Twist()) 
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()