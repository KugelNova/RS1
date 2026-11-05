#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Point

from thermal_processing import decode_mono16, summarize_hot_pixels


class ThermalDetector(Node):

    def __init__(self):
        super().__init__('thermal_detector')

        # Gazebo thermal camera uses 0.01 K per raw count.
        # 305 K = approximately 31.85 C.
        self.threshold_raw = 30500

        # Ignore very small hot regions / noise.
        self.min_pixels = 50

        self.subscription = self.create_subscription(
            Image,
            '/parrot1/camera/thermal/image',
            self.image_callback,
            10
        )

        self.detected_pub = self.create_publisher(
            Bool,
            '/parrot1/thermal/target_detected',
            10
        )

        self.pixel_pub = self.create_publisher(
            Point,
            '/parrot1/thermal/target_pixel',
            10
        )

        self.last_state = None

        self.get_logger().info(
            'Thermal detector started'
        )

        self.get_logger().info(
            'Listening to /parrot1/camera/thermal/image'
        )

        self.get_logger().info(
            'Detection threshold: 305.00 K / 31.85 C'
        )


    def image_callback(self, msg):

        if msg.encoding != 'mono16':
            self.get_logger().warning(
                f'Unexpected image encoding: {msg.encoding}'
            )
            return

        try:
            image = decode_mono16(msg)
            hot = summarize_hot_pixels(image, self.threshold_raw, self.min_pixels)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f'Invalid thermal image: {exc}')
            return

        detected_msg = Bool()
        detected_msg.data = hot is not None
        self.detected_pub.publish(detected_msg)

        if hot is None:
            if self.last_state is not False:
                self.get_logger().info('No thermal target detected')
            self.last_state = False
            return

        # x/y identify the average pixel across all hot regions, preserving
        # the existing topic contract (not a per-person detection).
        pixel_msg = Point()
        pixel_msg.x = hot['centre_x']
        pixel_msg.y = hot['centre_y']
        pixel_msg.z = hot['max_temp_c']
        self.pixel_pub.publish(pixel_msg)

        self.get_logger().info(
            f'TARGET DETECTED | '
            f'pixels={hot["pixels"]} | '
            f'centre=({hot["centre_x"]:.1f},{hot["centre_y"]:.1f}) | '
            f'box=({hot["x_min"]},{hot["y_min"]})-'
            f'({hot["x_max"]},{hot["y_max"]}) | '
            f'max={hot["max_temp_c"]:.2f} C'
        )

        self.last_state = True


def main(args=None):

    rclpy.init(args=args)

    node = ThermalDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()