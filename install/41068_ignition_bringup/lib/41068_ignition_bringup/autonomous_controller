#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


class AutonomousController(Node):

    def __init__(self):
        super().__init__('autonomous_controller')

        self.cmd_pub = self.create_publisher(
            Twist,
            '/parrot1/cmd_vel',
            10
        )

        self.create_subscription(
            TFMessage,
            '/world/large_demo/dynamic_pose/info',
            self.pose_callback,
            10
        )

        self.create_subscription(
            LaserScan,
            '/parrot1/scan',
            self.scan_callback,
            10
        )

        self.target_height = 2.0
        self.travel_distance = 5.0
        self.obstacle_distance = 2.0

        self.current_x = None
        self.current_y = None
        self.current_z = None

        self.start_x = None
        self.start_y = None

        self.front_distance = math.inf
        self.finished = False

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Autonomous controller started')

    def pose_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == 'parrot1':
                position = transform.transform.translation

                self.current_x = position.x
                self.current_y = position.y
                self.current_z = position.z

                if self.start_x is None:
                    self.start_x = self.current_x
                    self.start_y = self.current_y

                break

    def scan_callback(self, msg):
        front_ranges = []
        half_angle = math.radians(15.0)

        for index, distance in enumerate(msg.ranges):
            angle = msg.angle_min + index * msg.angle_increment

            # The scan runs from 0 to 2*pi. Angles close to
            # zero represent the region directly ahead.
            is_front = (
                angle <= half_angle
                or angle >= (2.0 * math.pi - half_angle)
            )

            is_valid = (
                math.isfinite(distance)
                and msg.range_min <= distance <= msg.range_max
            )

            if is_front and is_valid:
                front_ranges.append(distance)

        if front_ranges:
            self.front_distance = min(front_ranges)
        else:
            self.front_distance = math.inf

    def control_loop(self):
        if self.current_x is None or self.current_z is None:
            return

        command = Twist()

        if self.finished:
            self.cmd_pub.publish(command)
            return

        # First reach and stabilise at the target altitude.
        height_error = self.target_height - self.current_z

        if abs(height_error) > 0.15:
            command.linear.z = max(
                -0.4,
                min(0.4, 0.5 * height_error)
            )

            self.cmd_pub.publish(command)

            self.get_logger().info(
                f'Adjusting altitude: z={self.current_z:.2f} m'
            )
            return

        # Stop when an obstacle is detected directly ahead.
        if self.front_distance < self.obstacle_distance:
            self.cmd_pub.publish(command)

            self.get_logger().warning(
                f'Obstacle detected at '
                f'{self.front_distance:.2f} m'
            )
            return

        distance_travelled = math.hypot(
            self.current_x - self.start_x,
            self.current_y - self.start_y
        )

        if distance_travelled < self.travel_distance:
            command.linear.x = 0.4

            # Small altitude correction while moving.
            command.linear.z = max(
                -0.2,
                min(0.2, 0.5 * height_error)
            )

            self.cmd_pub.publish(command)

            self.get_logger().info(
                f'Distance: {distance_travelled:.2f} m, '
                f'altitude: {self.current_z:.2f} m, '
                f'front: {self.front_distance:.2f} m'
            )
            return

        self.finished = True
        self.cmd_pub.publish(command)
        self.get_logger().info('Target distance reached')


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()