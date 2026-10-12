#!/usr/bin/env python3
"""ROS 2 above-canopy search-and-rescue *simulation* controller.

Uses existing thermal_detector node; does not infer a ground target position or
attempt an unsensed descent through trees. Check canopy and initial climb corridor
before operating. Adjust camera orientation/FOV and survey altitude separately.
"""
import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage


class FullFlightPath(Node):
    def __init__(self):
        super().__init__('full_flight_path')

        # WORLD AND FLIGHT PARAMETERS: adjust after checking the Gazebo model.
        self.pose_topic = '/world/simple_trees/dynamic_pose/info'
        self.search_altitude = 12.0   # EXAMPLE ONLY; verify canopy + climb clearance
        self.maximum_altitude = 15.0
        self.min_x, self.max_x = -3.0, 8.0
        self.min_y, self.max_y = -3.0, 6.0
        self.lane_spacing = 2.0
        self.waypoint_tolerance = 0.5
        self.altitude_tolerance = 0.2
        self.forward_speed = 0.6
        self.vertical_speed = 0.6
        self.turn_speed = 0.8
        self.obstacle_distance = 2.0
        self.sensor_timeout = 1.0
        self.inspect_seconds = 6.0
        self.confirm_frames = 3
        self.thermal_timeout = 1.0
        self.thermal_rearm_seconds = 8.0

        # SAFETY GATE: must be set True after verifying that launch -> survey
        # altitude and the survey corridor are clear in *this* Gazebo world.
        self.canopy_route_verified = True

        self.cmd_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', 10)
        self.report_pub = self.create_publisher(
            String, '/parrot1/thermal/investigation_report', 10)
        self.create_subscription(TFMessage, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(LaserScan, '/parrot1/scan', self.scan_cb, 10)
        self.create_subscription(
            Bool, '/parrot1/thermal/target_detected', self.thermal_cb, 10)
        self.create_subscription(
            Point, '/parrot1/thermal/target_pixel', self.pixel_cb, 10)

        self.x = self.y = self.z = self.yaw = None
        self.last_pose = self.last_scan = self.last_thermal = None
        self.front_distance = 0.0
        self.have_scan = False
        self.thermal_detected = False
        self.positive_frames = 0
        self.latest_pixel = None
        self.latest_pixel_time = None
        self.event = None
        self.inspect_started = None
        self.rearm_until = 0.0
        self.last_hold_log = -1e9
        self.waypoints = self.make_waypoints()
        self.waypoint_index = 0
        self.state = 'WAIT_FOR_CLEARANCE' if not self.canopy_route_verified else 'TAKEOFF'
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            f'Created {len(self.waypoints)} search waypoints; '
            f'flight state={self.state}; survey altitude={self.search_altitude:.1f} m')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def make_waypoints(self):
        waypoints = []
        y = self.min_y
        positive = True
        while y <= self.max_y + 1e-6:
            pair = (self.min_x, self.max_x) if positive else (self.max_x, self.min_x)
            waypoints.extend((x, y) for x in pair)
            positive = not positive
            y += self.lane_spacing
        return waypoints

    def pose_cb(self, msg):
        for tr in msg.transforms:
            if tr.child_frame_id.strip('/') != 'parrot1':
                continue
            t = tr.transform.translation
            q = tr.transform.rotation
            self.x, self.y, self.z = t.x, t.y, t.z
            self.yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y*q.y + q.z*q.z))
            self.last_pose = self.now()
            return

    def scan_cb(self, msg):
        half_angle = math.radians(20.0)
        front = []
        for i, distance in enumerate(msg.ranges):
            angle = self.wrap(msg.angle_min + i * msg.angle_increment)
            if abs(angle) > half_angle or math.isnan(distance):
                continue
            # Gazebo may use +infinity to indicate no LiDAR return.
            if math.isinf(distance):
                if distance > 0 and msg.range_max > 0:
                    front.append(float(msg.range_max))
            elif msg.range_min <= distance <= msg.range_max:
                front.append(float(distance))
        self.front_distance = min(front) if front else 0.0  # unknown = stop
        self.have_scan = True
        self.last_scan = self.now()

    def thermal_cb(self, msg):
        self.last_thermal = self.now()
        self.thermal_detected = bool(msg.data)
        self.positive_frames = self.positive_frames + 1 if msg.data else 0

    def pixel_cb(self, msg):
        self.latest_pixel = (float(msg.x), float(msg.y), float(msg.z))
        self.latest_pixel_time = self.now()

    def stop(self):
        self.cmd_pub.publish(Twist())

    def sensors_ready(self):
        t = self.now()
        return (
            self.x is not None and self.have_scan
            and self.last_pose is not None and self.last_scan is not None
            and 0.0 <= t - self.last_pose <= self.sensor_timeout
            and 0.0 <= t - self.last_scan <= self.sensor_timeout)

    def altitude_control(self):
        return self.clamp(
            0.5 * (self.search_altitude - self.z),
            -self.vertical_speed, self.vertical_speed)

    def start_inspection(self):
        self.stop()
        self.state = 'INSPECT'
        self.inspect_started = self.now()
        self.event = {
            'status': 'unverified_thermal_candidate',
            'detected_time_ros_seconds': round(self.now(), 3),
            'drone_position_at_detection_m': [self.x, self.y, self.z],
            'drone_yaw_rad': self.yaw,
            'waypoint_index_1based': self.waypoint_index + 1,
            'target_pixel_xy': list(self.latest_pixel[:2]) if self.latest_pixel else None,
            'max_hot_region_temp_C': self.latest_pixel[2] if self.latest_pixel else None,
        }
        self.get_logger().warning(
            'Thermal candidate detected; hovering for second observation')

    def finish_inspection(self):
        # This checks if the thermal flag stayed active during the hover;
        # it does NOT establish human identity or a world-frame target location.
        thermal_fresh = (self.last_thermal is not None and
                         self.now() - self.last_thermal <= self.thermal_timeout)
        self.event['still_detected_after_hover'] = bool(
            thermal_fresh and self.thermal_detected)
        self.event['inspection_duration_seconds'] = self.inspect_seconds
        msg = String()
        msg.data = json.dumps(self.event)
        self.report_pub.publish(msg)
        self.get_logger().warning(f'THERMAL REPORT: {msg.data}')
        self.event = None
        self.state = 'SEARCH'
        self.positive_frames = 0
        self.rearm_until = self.now() + self.thermal_rearm_seconds

    def control_loop(self):
        if self.state == 'WAIT_FOR_CLEARANCE':
            self.stop()
            if self.now() - self.last_hold_log >= 5.0:
                self.get_logger().error(
                    'Flight disabled: verify canopy height and ascent/search '
                    'corridors, then set canopy_route_verified=True')
                self.last_hold_log = self.now()
            return

        if not self.sensors_ready():
            self.stop()
            return
        if self.state in ('BLOCKED', 'FINISHED'):
            self.stop()
            return

        if self.state == 'TAKEOFF':
            # Initial climb is allowed only by the independent clearance gate.
            if self.search_altitude > self.maximum_altitude:
                self.state = 'BLOCKED'
                self.get_logger().error('Survey altitude exceeds configured maximum')
                self.stop()
                return
            if abs(self.search_altitude - self.z) > self.altitude_tolerance:
                cmd = Twist()
                cmd.linear.z = self.altitude_control()
                self.cmd_pub.publish(cmd)
                return
            self.stop()
            self.state = 'SEARCH'
            self.get_logger().info('Survey altitude reached; starting thermal scan')
            return

        if self.state == 'INSPECT':
            self.stop()  # inspect at existing altitude; NO automatic descent
            if self.now() - self.inspect_started >= self.inspect_seconds:
                self.finish_inspection()
            return

        if self.waypoint_index >= len(self.waypoints):
            self.state = 'FINISHED'
            self.stop()
            self.get_logger().info('Lawnmower scan completed')
            return

        if self.state != 'SEARCH':
            self.stop()
            return

        thermal_fresh = (
            self.last_thermal is not None
            and self.now() - self.last_thermal <= self.thermal_timeout)
        pixel_fresh = (
            self.latest_pixel_time is not None
            and self.now() - self.latest_pixel_time <= self.thermal_timeout)
        if (self.now() >= self.rearm_until and thermal_fresh and pixel_fresh
                and self.positive_frames >= self.confirm_frames):
            self.start_inspection()
            return

        tx, ty = self.waypoints[self.waypoint_index]
        dx, dy = tx - self.x, ty - self.y
        distance = math.hypot(dx, dy)
        if distance <= self.waypoint_tolerance:
            self.get_logger().info(
                f'Reached waypoint {self.waypoint_index + 1}/{len(self.waypoints)}')
            self.waypoint_index += 1
            self.stop()
            return

        target_yaw = math.atan2(dy, dx)
        error = self.wrap(target_yaw - self.yaw)
        cmd = Twist()
        cmd.linear.z = self.altitude_control()
        if abs(error) > math.radians(10):
            cmd.angular.z = self.clamp(1.2 * error, -self.turn_speed, self.turn_speed)
            self.cmd_pub.publish(cmd)
            return

        if self.front_distance < self.obstacle_distance:
            self.state = 'BLOCKED'
            self.stop()
            self.get_logger().error(
                f'Obstacle/invalid front scan at {self.front_distance:.2f} m '
                'above canopy; no safe local route implemented')
            return

        clearance = max(0.0, self.front_distance - self.obstacle_distance)
        cmd.linear.x = min(self.forward_speed, 0.5 * distance, 0.4 * clearance)
        cmd.angular.z = self.clamp(0.8 * error, -0.2, 0.2)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = FullFlightPath()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
