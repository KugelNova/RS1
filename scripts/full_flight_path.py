#!/usr/bin/env python3

#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


class FullFlightPath(Node):

    def __init__(self):
        super().__init__('full_flight_path')

        # Movement publisher
        self.cmd_pub = self.create_publisher(
            Twist,
            '/parrot1/cmd_vel',
            10
        )

        # True Gazebo 3D position
        self.create_subscription(
            TFMessage,
            '/world/large_demo/dynamic_pose/info',
            self.pose_callback,
            10
        )

        # Horizontal LiDAR
        self.create_subscription(
            LaserScan,
            '/parrot1/scan',
            self.scan_callback,
            10
        )

        # Search-area settings
        self.minimum_x = 2.0
        self.maximum_x = 18.0
        self.minimum_y = -10.0
        self.maximum_y = 10.0
        self.lane_spacing = 3.0

        # Flight settings
        self.survey_altitude = 3.0
        self.maximum_altitude = 6.0
        self.obstacle_distance = 2.5
        self.waypoint_tolerance = 0.5

        self.maximum_forward_speed = 0.6
        self.maximum_turn_speed = 0.5
        self.maximum_vertical_speed = 0.4

        # Current drone state
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.current_yaw = None

        self.front_distance = math.inf

        self.waypoints = self.generate_lawnmower_path()
        self.current_waypoint_index = 0

        self.state = 'TAKEOFF'
        self.obstacle_warning_printed = False

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            f'Generated {len(self.waypoints)} waypoints'
        )

        self.get_logger().info(
            'Full flight path controller started'
        )

    def generate_lawnmower_path(self):
        waypoints = []

        current_y = self.minimum_y
        move_positive_x = True

        while current_y <= self.maximum_y:
            if move_positive_x:
                waypoints.append((
                    self.minimum_x,
                    current_y,
                    self.survey_altitude
                ))

                waypoints.append((
                    self.maximum_x,
                    current_y,
                    self.survey_altitude
                ))
            else:
                waypoints.append((
                    self.maximum_x,
                    current_y,
                    self.survey_altitude
                ))

                waypoints.append((
                    self.minimum_x,
                    current_y,
                    self.survey_altitude
                ))

            move_positive_x = not move_positive_x
            current_y += self.lane_spacing

        return waypoints

    def pose_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == 'parrot1':
                translation = transform.transform.translation
                rotation = transform.transform.rotation

                self.current_x = translation.x
                self.current_y = translation.y
                self.current_z = translation.z

                self.current_yaw = self.quaternion_to_yaw(
                    rotation.x,
                    rotation.y,
                    rotation.z,
                    rotation.w
                )

                break

    def quaternion_to_yaw(self, x, y, z, w):
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)

        return math.atan2(sin_yaw, cos_yaw)

    def scan_callback(self, msg):
        front_ranges = []
        half_angle = math.radians(20.0)

        for index, distance in enumerate(msg.ranges):
            angle = (
                msg.angle_min
                + index * msg.angle_increment
            )

            # Convert angles into the range -pi to +pi.
            normalised_angle = math.atan2(
                math.sin(angle),
                math.cos(angle)
            )

            is_front = abs(normalised_angle) <= half_angle

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

    def normalise_angle(self, angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    def clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def altitude_command(self):
        height_error = (
            self.survey_altitude
            - self.current_z
        )

        return self.clamp(
            0.5 * height_error,
            -self.maximum_vertical_speed,
            self.maximum_vertical_speed
        )

    def stop_drone(self):
        self.cmd_pub.publish(Twist())

    def control_loop(self):
        if (
            self.current_x is None
            or self.current_y is None
            or self.current_z is None
            or self.current_yaw is None
        ):
            return

        command = Twist()

        # Stage 1: take off to the survey altitude.
        if self.state == 'TAKEOFF':
            height_error = (
                self.survey_altitude
                - self.current_z
            )

            if abs(height_error) > 0.15:
                command.linear.z = self.altitude_command()
                self.cmd_pub.publish(command)

                self.get_logger().info(
                    f'Taking off: '
                    f'z={self.current_z:.2f} m'
                )
                return

            self.state = 'FOLLOW_PATH'
            self.get_logger().info(
                'Survey altitude reached'
            )

        if self.state == 'FINISHED':
            self.stop_drone()
            return

        if self.state == 'BLOCKED':
            self.stop_drone()
            return

        # Stop safely when an obstacle is ahead.
        if self.front_distance < self.obstacle_distance:
            self.state = 'BLOCKED'
            self.stop_drone()

            if not self.obstacle_warning_printed:
                self.get_logger().warning(
                    f'Obstacle detected at '
                    f'{self.front_distance:.2f} m. '
                    f'Drone stopped.'
                )

                self.obstacle_warning_printed = True

            return

        if self.current_waypoint_index >= len(self.waypoints):
            self.state = 'FINISHED'
            self.stop_drone()

            self.get_logger().info(
                'Lawnmower search completed'
            )
            return

        target = self.waypoints[
            self.current_waypoint_index
        ]

        target_x, target_y, target_z = target

        difference_x = target_x - self.current_x
        difference_y = target_y - self.current_y

        horizontal_distance = math.hypot(
            difference_x,
            difference_y
        )

        # Move to the next waypoint when close enough.
        if horizontal_distance < self.waypoint_tolerance:
            self.get_logger().info(
                f'Reached waypoint '
                f'{self.current_waypoint_index + 1}/'
                f'{len(self.waypoints)}: '
                f'({target_x:.1f}, {target_y:.1f}, '
                f'{target_z:.1f})'
            )

            self.current_waypoint_index += 1
            self.stop_drone()
            return

        target_yaw = math.atan2(
            difference_y,
            difference_x
        )

        yaw_error = self.normalise_angle(
            target_yaw - self.current_yaw
        )

        command.linear.z = self.altitude_command()

        # Turn toward the waypoint before flying forward.
        if abs(yaw_error) > math.radians(10.0):
            command.angular.z = self.clamp(
                1.2 * yaw_error,
                -self.maximum_turn_speed,
                self.maximum_turn_speed
            )
        else:
            command.linear.x = min(
                self.maximum_forward_speed,
                0.5 * horizontal_distance
            )

            command.angular.z = self.clamp(
                0.8 * yaw_error,
                -0.25,
                0.25
            )

        self.cmd_pub.publish(command)

        self.get_logger().info(
            f'Waypoint '
            f'{self.current_waypoint_index + 1}/'
            f'{len(self.waypoints)} | '
            f'distance={horizontal_distance:.2f} m | '
            f'z={self.current_z:.2f} m | '
            f'front={self.front_distance:.2f} m'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FullFlightPath()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_drone()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

