#!/usr/bin/env python3


import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


class FullFlightPath(Node):


    # Here we create our subscriptions for what we need so our pose and lidar scan and also our publisher as we use our lidar scans 
    # and pose to decide what commands we publish 
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
        self.obstacle_distance = 2.0
        self.waypoint_tolerance = 0.5

        self.maximum_forward_speed = 4.0
        self.maximum_turn_speed = 3.0
        self.maximum_vertical_speed = 2.5

        # Current drone state
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.current_yaw = None

        self.front_distance = math.inf
        self.left_distance = math.inf
        self.right_distance = math.inf

        self.avoidance_distance = 3.0
        self.avoidance_start_x = None
        self.avoidance_start_y = None
        self.avoidance_target_yaw = None
        self.avoidance_attempts = 0
        self.maximum_avoidance_attempts = 5

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
        # here we define out way points that ultimately set up our back and forth search pattern
        # set so that we create a point at the start and end of the row row length is defined above
        # then we set lane spcaing so the drone can fly up the lane and then reverse its direction creating  a lawn mower effect.
        # but here we just define the way points at the start and end of each lane and then control loop flies the drone 
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
        # here we subsricbe to pose of parrot1/ drojne 
        # we read in its current pose from the new messages it recieves from Gazebo 
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
        # here we convert our quat which is our pose recieved and convert to yaw we can then better understadn 
        # this method is called in pose call back 
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)

        return math.atan2(sin_yaw, cos_yaw)

    def scan_callback(self, msg):
        # we read in lidar scans Front, Left, Right
        front_ranges = []
        left_ranges = []
        right_ranges = []

        sector_width = math.radians(20.0)

        # we then check each lidar measurement 
        # we use it if it falls within the senesor allwoed range to ensure we use only reliable data. 
        for index, distance in enumerate(msg.ranges):
            is_valid = (
                math.isfinite(distance)
                and msg.range_min <= distance <= msg.range_max
            )

            if not is_valid:
                continue

            angle = msg.angle_min + index * msg.angle_increment

            angle = self.normalise_angle(angle)

            # Forward sector centred on 0 degrees.
            if abs(angle) <= sector_width:
                front_ranges.append(distance)

            # Left sector centred on +90 degrees.
            left_error = self.normalise_angle(
                angle - math.pi / 2.0
            )

            if abs(left_error) <= sector_width:
                left_ranges.append(distance)

            # Right sector centred on -90 degrees.
            right_error = self.normalise_angle(
                angle + math.pi / 2.0
            )

            if abs(right_error) <= sector_width:
                right_ranges.append(distance)

        self.front_distance = (
            min(front_ranges)
            if front_ranges
            else math.inf
        )

        self.left_distance = (
            min(left_ranges)
            if left_ranges
            else math.inf
        )

        self.right_distance = (
            min(right_ranges)
            if right_ranges
            else math.inf
        )



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
        # set our command by finding what heigtht we define above and setting it as an error until we reach it 
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


    def begin_avoidance(self):
        self.stop_drone()

        if self.avoidance_attempts >= self.maximum_avoidance_attempts:
            self.state = 'BLOCKED'

            self.get_logger().error(
                'Maximum avoidance attempts reached. Drone stopped.'
            )
            return

        left_blocked = (
            self.left_distance < self.obstacle_distance
        )

        right_blocked = (
            self.right_distance < self.obstacle_distance
        )

        if left_blocked and right_blocked:
            self.state = 'BLOCKED'

            self.get_logger().error(
                'Front, left and right directions are blocked.'
            )
            return

        if self.left_distance >= self.right_distance:
            turn_direction = 1.0
            direction_name = 'left'
        else:
            turn_direction = -1.0
            direction_name = 'right'

        self.avoidance_target_yaw = self.normalise_angle(
            self.current_yaw
            + turn_direction * math.pi / 2.0
        )

        self.avoidance_attempts += 1
        self.state = 'AVOID_TURN'

        self.get_logger().warning(
            f'Obstacle at {self.front_distance:.2f} m. '
            f'Turning {direction_name}.'
        )


    def control_avoidance(self):
        command = Twist()
        command.linear.z = self.altitude_command()

        if self.state == 'AVOID_TURN':
            yaw_error = self.normalise_angle(
                self.avoidance_target_yaw
                - self.current_yaw
            )

            if abs(yaw_error) > math.radians(10.0):
                command.angular.z = self.clamp(
                    1.2 * yaw_error,
                    -self.maximum_turn_speed,
                    self.maximum_turn_speed
                )

                self.cmd_pub.publish(command)
                return

            self.avoidance_start_x = self.current_x
            self.avoidance_start_y = self.current_y
            self.state = 'AVOID_SIDE'

            self.stop_drone()

            self.get_logger().info(
                'Avoidance turn complete. Starting detour.'
            )
            return

        if self.state == 'AVOID_SIDE':
            # Something is now blocking the detour direction.
            if self.front_distance < 1.5:
                self.state = 'BLOCKED'
                self.stop_drone()

                self.get_logger().error(
                    'Detour direction is also blocked. Drone stopped.'
                )
                return

            distance_moved = math.hypot(
                self.current_x - self.avoidance_start_x,
                self.current_y - self.avoidance_start_y
            )

            if distance_moved < self.avoidance_distance:
                command.linear.x = 0.4
                self.cmd_pub.publish(command)
                return

            self.state = 'FOLLOW_PATH'
            self.stop_drone()

            self.get_logger().info(
                'Detour complete. Returning to original waypoint.'
            )   

    def control_loop(self):
            
            if (
                self.current_x is None
                or self.current_y is None
                or self.current_z is None
                or self.current_yaw is None
            ):
                return

            command = Twist()

            # we take off to the survey altitude.
            # using altitude command and publish commands until we reach it 
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

            if self.state in ('AVOID_TURN', 'AVOID_SIDE'):
                self.control_avoidance()
                return

            # Stop safely when an obstacle is ahead.
            # then we calculate angle to next way point 

            if self.front_distance < self.obstacle_distance:
                self.begin_avoidance()
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
                self.avoidance_attempts = 0
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

