import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class ThermalScanner(Node):
    def __init__(self):
        super().__init__('thermal_scanner')
        # Subscribe to the drone's thermal camera topic
        self.subscription = self.create_subscription(
            Image,
            '/model/parrot1/thermal_camera',
            self.listener_callback,
            10)
        self.get_logger().info('Thermal scanner active. Looking for heat signatures...')

    def listener_callback(self, msg):
        # Convert the raw byte buffer into a 16-bit integer array
        img_16bit = np.frombuffer(msg.data, dtype=np.uint16)
        
        # Find the absolute hottest pixel in the current camera frame
        max_pixel_value = np.max(img_16bit)
        
        # Gazebo encodes the 16-bit thermal data as (Temperature in Kelvin * 100)
        max_temp_kelvin = max_pixel_value / 100.0
        
        # Check if the hottest object exceeds a threshold (e.g., 500 K)
        if max_temp_kelvin > 500.0:
            self.get_logger().warn(f'🔥 FIRE DETECTED! Target acquired at {max_temp_kelvin} K')
        else:
            self.get_logger().info(f'Clear. Hottest object in view: {max_temp_kelvin} K')

def main(args=None):
    rclpy.init(args=args)
    node = ThermalScanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()