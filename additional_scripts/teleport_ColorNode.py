import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import subprocess

class UnifiedThermalManager(Node):
    def __init__(self):
        super().__init__('unified_thermal_manager')
        self.teleport_robot()
        self.publisher_ = self.create_publisher(Image, '/parrot1/camera/thermal/image_color', 10)
        self.subscription = self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_callback, 10)
        
        # Counter to throttle terminal spam
        self.frame_count = 0
        self.get_logger().info('Unified Thermal Node Active. Checking feed...')

    def teleport_robot(self):
        cmd = [
            'ign', 'service', '-s', '/world/simple_trees/set_pose',
            '--reqtype', 'ignition.msgs.Pose',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '2000',
            '--req', "name: 'parrot1', position: {x: 0.0, y: 1.0, z: 2.0}, orientation: {x: 0.174, y: 0.0, z: -0.700, w: 0.693}"
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=3)

    def thermal_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        h, w = msg.height, msg.width
        
        mid_band = img[int(h*0.3):int(h*0.7), :]
        left_third = mid_band[:, :w//3]
        center_third = mid_band[:, w//3:2*w//3]
        right_third = mid_band[:, 2*w//3:]
        
        # 1. Throttle terminal prints to once every 20 frames (~2 seconds)
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            print('========================================')
            print(f'Left: {np.max(left_third)} | Center: {np.max(center_third)} | Right (House): {np.max(right_third)}')
            print('========================================')

        # 2. Clip the contrast range before normalizing
        # We clip at 115 so the 103 center box gets bright contrast. 
        # The 255 house will be clipped to 115, which normalizes to pure white.
        clipped_img = np.clip(img, 90, 115)
        normalized_img = cv2.normalize(clipped_img, None, 0, 255, cv2.NORM_MINMAX)
        color_img = cv2.applyColorMap(normalized_img, cv2.COLORMAP_INFERNO)
        
        # 3. Draw exact temperatures onto the RViz frame
        # Gazebo's default L8 thermal resolution is 3.0 Kelvin per pixel intensity.
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255) # White text
        
        cv2.putText(color_img, f"{np.max(left_third) * 3} K", (w//6 - 30, h//2), font, 0.6, color, 2)
        cv2.putText(color_img, f"{np.max(center_third) * 3} K", (w//2 - 30, h//2), font, 0.6, color, 2)
        
        # House maxes out the 8-bit sensor at 255 (765 K), so we indicate it is clipping
        cv2.putText(color_img, f"> 765 K", (5*w//6 - 40, h//2), font, 0.6, color, 2)
        
        # Publish to RViz
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height, out_msg.width = h, w
        out_msg.encoding = 'bgr8'
        out_msg.step = w * 3
        out_msg.data = color_img.tobytes()
        self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedThermalManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()