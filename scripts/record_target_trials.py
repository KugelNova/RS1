#!/usr/bin/env python3
"""Diagnostics only: correlate rescue target alerts with drone height at alert time.

Does not publish flight commands or modify the target localizer. Stores a separate
CSV in ~/rescue_logs so original mission CSV format/UI remains unchanged.
"""
import argparse
import csv
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import String

FIELDS = [
    'target_id', 'alert_sim_time_sec', 'frame_id', 'target_x_m', 'target_y_m',
    'target_z_m', 'drone_odom_time_sec', 'drone_x_m', 'drone_y_m', 'drone_z_m',
    'odom_alert_delta_s', 'altitude_observation', 'max_temp_c',
    'hot_pixels', 'edge_clipped',
]


class TrialRecorder(Node):
    def __init__(self, trial, out_dir):
        super().__init__('rs1_trial_recorder')
        self.trial = trial
        self.samples = deque(maxlen=500)
        self.count = 0
        Path(out_dir).expanduser().mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self.path = Path(out_dir).expanduser() / f'rs1_trial_{trial}_{stamp}.csv'
        self.file = self.path.open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.file.flush()
        self.create_subscription(Odometry, '/parrot1/odom', self.on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/parrot1/rescue/target_alert',
                                 self.on_alert, 10)
        self.get_logger().info(f'Trial={trial}; recorder CSV={self.path}')

    def on_odom(self, msg):
        sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        if sec > 0:
            self.samples.append((sec, p.x, p.y, p.z))

    def on_alert(self, msg):
        try:
            alert = json.loads(msg.data)
            stamp = float(alert['sim_time_sec'])
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().warning(f'Ignored malformed alert: {exc}')
            return
        best = min(self.samples, key=lambda o: abs(o[0] - stamp), default=None)
        if best and abs(best[0] - stamp) > 1.0:
            best = None
        obs = 'unknown_no_aligned_odom'
        if best:
            # This is a height hint, not a controller state; survey altitude is 12 m.
            obs = 'below_11.8m' if best[3] < 11.8 else 'at_or_above_11.8m'
        record = {
            'target_id': alert.get('id', ''),
            'alert_sim_time_sec': alert.get('sim_time_sec', ''),
            'frame_id': alert.get('frame_id', ''),
            'target_x_m': alert.get('x_m', ''),
            'target_y_m': alert.get('y_m', ''),
            'target_z_m': alert.get('z_m', ''),
            'drone_odom_time_sec': f'{best[0]:.3f}' if best else '',
            'drone_x_m': f'{best[1]:.3f}' if best else '',
            'drone_y_m': f'{best[2]:.3f}' if best else '',
            'drone_z_m': f'{best[3]:.3f}' if best else '',
            'odom_alert_delta_s': f'{abs(best[0]-stamp):.3f}' if best else '',
            'altitude_observation': obs,
            'max_temp_c': alert.get('max_temp_c', ''),
            'hot_pixels': alert.get('hot_pixels', ''),
            'edge_clipped': alert.get('edge_clipped', ''),
        }
        self.writer.writerow(record)
        self.file.flush()
        self.count += 1
        self.get_logger().warning(
            f"ALERT #{record['target_id']} t={stamp:.2f}s "
            f"drone_z={record['drone_z_m'] or '?'}m [{obs}] "
            f"target=({record['target_x_m']},{record['target_y_m']})")

    def destroy_node(self):
        if not self.file.closed:
            self.file.close()
        self.get_logger().info(f'{self.count} alerts written to {self.path}')
        return super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', choices=['empty', 'ascent', 'spaced', 'far_single', 'far_pair'], required=True)
    parser.add_argument('--output-dir', default='~/rescue_logs')
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = TrialRecorder(args.trial, args.output_dir)
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
