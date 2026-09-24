#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Point


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
            '/parrot1/thermal/image_raw',
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
            'Listening to /parrot1/thermal/image_raw'
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

        # Handle image byte order correctly.
        if msg.is_bigendian:
            dtype = np.dtype('>u2')
        else:
            dtype = np.dtype('<u2')

        image = np.frombuffer(
            msg.data,
            dtype=dtype
        ).reshape(
            msg.height,
            msg.width
        )

        # Hot pixels above threshold.
        # Zero pixels are automatically excluded.
        hot_mask = image >= self.threshold_raw

        ys, xs = np.where(hot_mask)

        detected = len(xs) >= self.min_pixels

        detected_msg = Bool()
        detected_msg.data = detected

        self.detected_pub.publish(detected_msg)

        if not detected:

            if self.last_state is not False:
                self.get_logger().info(
                    'No thermal target detected'
                )

            self.last_state = False
            return

        # Bounding box.
        x_min = int(xs.min())
        x_max = int(xs.max())

        y_min = int(ys.min())
        y_max = int(ys.max())

        # Target centre in image coordinates.
        centre_x = float(xs.mean())
        centre_y = float(ys.mean())

        # Maximum temperature in detected region.
        hottest_raw = int(image[hot_mask].max())

        hottest_kelvin = hottest_raw * 0.01
        hottest_celsius = hottest_kelvin - 273.15

        # Publish pixel position.
        #
        # x = horizontal image pixel
        # y = vertical image pixel
        # z = maximum temperature in Celsius
        pixel_msg = Point()

        pixel_msg.x = centre_x
        pixel_msg.y = centre_y
        pixel_msg.z = hottest_celsius

        self.pixel_pub.publish(pixel_msg)

        # Only print continuously while target is present,
        # useful during development.
        self.get_logger().info(
            f'TARGET DETECTED | '
            f'pixels={len(xs)} | '
            f'centre=({centre_x:.1f},{centre_y:.1f}) | '
            f'box=({x_min},{y_min})-({x_max},{y_max}) | '
            f'max={hottest_celsius:.2f} C'
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