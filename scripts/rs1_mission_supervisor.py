#!/usr/bin/env python3
"""RS1 simulation-only mission controller.

Single ownership of /parrot1/cmd_vel for UI-managed operations. Existing
far-test mission is run with a ROS2 remap to /rs1/auto/cmd_vel, so no team
flight script is changed. This is NOT a certified safety/landing controller.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from rs1_mission_logic import (finite_xy, goto_step, descent_speed,
                               ODOM_TIMEOUT_S, SCAN_TIMEOUT_S, wrap_angle)
from rs1_coverage_route import plan_lawnmower, route_stats

PACKAGE = '41068_ignition_bringup'


class MissionSupervisor(Node):
    def __init__(self):
        super().__init__('rs1_mission_supervisor')
        self.cmd_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/rs1/control/status', 10)
        self.create_subscription(String, '/rs1/control/command', self.on_request, 10)
        self.create_subscription(Twist, '/rs1/manual/cmd_vel', self.manual_cb, 10)
        self.create_subscription(Twist, '/rs1/auto/cmd_vel', self.auto_cb, 10)
        self.create_subscription(Odometry, '/parrot1/odom', self.pose_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/parrot1/scan', self.scan_cb, qos_profile_sensor_data)
        self.state = 'IDLE'
        self.reason = 'Supervisor ready; start sensors or search from the UI'
        self.tasks = {}
        self._log_handles = {}
        self.pose = None
        self.pose_at = 0.0
        self.scan_at = 0.0
        self.front = 0.0
        self.latest_manual = Twist()
        self.latest_auto = Twist()
        self.manual_at = 0.0
        self.auto_at = 0.0
        self.goto_xy = None
        self.landing_confirmed = False
        self.recorder_trial = 'far_pair'
        self._last_cmd = 0.0
        self.create_timer(0.1, self.tick)
        self.create_timer(0.25, self.publish_status)
        self.get_logger().info('RS1 supervisor started; cmd_vel owner in managed mode')

    def pose_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y),
                         1 - 2*(q.y*q.y + q.z*q.z))
        self.pose = (float(p.x), float(p.y), float(p.z), yaw)
        self.pose_at = time.monotonic()

    def scan_cb(self, msg):
        front = []
        for i, d in enumerate(msg.ranges):
            angle = wrap_angle(msg.angle_min + i * msg.angle_increment)
            if abs(angle) > math.radians(20):
                continue
            if math.isinf(d) and d > 0:
                front.append(float(msg.range_max))
            elif math.isfinite(d) and msg.range_min <= d <= msg.range_max:
                front.append(float(d))
        self.front = min(front) if front else 0.0
        self.scan_at = time.monotonic()

    def pose_ok(self):
        return self.pose is not None and time.monotonic() - self.pose_at <= ODOM_TIMEOUT_S

    def scan_ok(self):
        return self.scan_at and time.monotonic() - self.scan_at <= SCAN_TIMEOUT_S

    def foreign_publishers(self):
        # The old CLI controller and old UI publish directly here; never compete.
        try:
            return [p.node_name for p in self.get_publishers_info_by_topic('/parrot1/cmd_vel')
                    if p.node_name != self.get_name()]
        except Exception:
            return ['unknown (graph inspection failed)']

    def velocity(self, x=0.0, z=0.0, yaw=0.0):
        t = Twist()
        t.linear.x, t.linear.z, t.angular.z = float(x), float(z), float(yaw)
        self.cmd_pub.publish(t)
        self._last_cmd = time.monotonic()

    def hold(self, reason, state='HOLD'):
        was_search = self.state == 'SEARCH'
        # Critical: revoke velocity ownership and publish zero BEFORE waiting
        # for a subprocess to exit, or an abort could continue moving.
        self.state = state
        self.goto_xy = None
        self.reason = reason
        self.velocity()
        if was_search:
            self.managed_stop('mission')
        # Recorder belongs to SEARCH only, regardless of why flight stopped.
        if 'recorder' in self.tasks:
            self.managed_stop('recorder')
        self.get_logger().warning(f'{state}: {reason}')

    def managed_start(self, key, args):
        existing = self.tasks.get(key)
        if existing and existing.poll() is None:
            return True
        log_dir = Path.home() / 'rescue_logs' / 'supervisor'
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / f'{key}.log').open('a', encoding='utf-8')
        try:
            p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
        except (OSError, ValueError) as e:
            log.close()
            self.reason = f'Cannot start {key}: {e}'
            self.get_logger().error(self.reason)
            return False
        self.tasks[key] = p
        self._log_handles[key] = log
        self.get_logger().info(f'Started managed process {key} PID {p.pid}')
        return True

    def managed_stop(self, key):
        p = self.tasks.pop(key, None)
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGINT)
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=1.0)
            except ProcessLookupError:
                pass
        log = self._log_handles.pop(key, None)
        if log:
            log.close()

    def existing_nodes(self):
        return {name for name, _namespace in self.get_node_names_and_namespaces()}

    def start_sensors(self, trial):
        # Sensor startup is independent of mission recording.
        self.recorder_trial = trial
        if 'recorder' in self.tasks and self.state != 'SEARCH':
            self.managed_stop('recorder')  # Clear a recorder from a prior UI version.
        # Do not stop externally started sensors; report them as external.
        names = self.existing_nodes()
        if 'thermal_detector' not in names:
            self.managed_start('thermal', ['ros2', 'run', PACKAGE, 'thermal_detector'])
        if 'target_localizer' not in names:
            self.managed_start('localizer', ['ros2', 'launch', PACKAGE,
                '41068_target_localization.launch.py', 'require_human_shape:=false'])
        self.reason = 'Thermal detector and localizer requested; recorder starts with SEARCH'

    def on_request(self, msg):
        try:
            req = json.loads(msg.data)
            if not isinstance(req, dict):
                raise ValueError('Expected JSON object')
            action = req.get('action')
            if action == 'abort_land':
                self.abort(bool(req.get('clear_landing_area', False)))
                return
            if action == 'hold':
                self.hold('Mission cancelled; holding position')
                return
            if action == 'start_sensors':
                trial = str(req.get('trial', 'far_pair'))
                if trial not in ('far_pair', 'far_single', 'terrain', 'empty', 'ascent', 'spaced'):
                    raise ValueError('Invalid recorder trial')
                self.start_sensors(trial)
                return
            if action == 'stop_sensors':
                if self.state in ('SEARCH', 'GOTO', 'LANDING'):
                    raise ValueError('Abort active flight before stopping sensors')
                for key in ('recorder', 'localizer', 'thermal'):
                    self.managed_stop(key)
                self.reason = 'Stopped supervisor-managed sensor processes'
                return
            if action == 'manual_on':
                self.require_idle()
                self.state = 'MANUAL'
                self.reason = 'Manual control armed (hold-to-move)'
                return
            if action == 'manual_off':
                if self.state == 'MANUAL':
                    self.hold('Manual control disarmed', 'IDLE')
                return
            if action == 'start_search':
                self.require_idle(require_scan=True)
                if not {'thermal_detector', 'target_localizer'}.issubset(self.existing_nodes()):
                    raise ValueError('Start thermal detector and target localizer before search')
                if 'full_flight_path_far_test' in self.existing_nodes():
                    raise ValueError('An external far-flight controller is already running')
                trial = str(req.get('trial', self.recorder_trial))
                if trial not in ('far_pair', 'far_single', 'terrain', 'empty', 'ascent', 'spaced'):
                    raise ValueError('Invalid recorder trial')
                self.recorder_trial = trial
                mode = str(req.get('search_mode', 'far_pair'))
                if mode not in ('far_pair', 'terrain'):
                    raise ValueError('Unknown search_mode')
                mission_args = ['ros2', 'run', PACKAGE,
                    'full_flight_path_far_test', '--ros-args',
                    '-r', '/parrot1/cmd_vel:=/rs1/auto/cmd_vel']
                route_summary = 'Legacy far-pair route'
                if mode == 'terrain':
                    if req.get('clearance_verified') is not True:
                        raise ValueError('Terrain mission requires verified climb/route clearance')
                    try:
                        bounds = [float(req[key]) for key in
                                  ('min_x', 'max_x', 'min_y', 'max_y', 'lane_spacing')]
                        altitude = float(req['altitude'])
                    except (KeyError, ValueError, TypeError) as exc:
                        raise ValueError('Invalid terrain search bounds/altitude') from exc
                    if not math.isfinite(altitude) or not 5 <= altitude <= 25:
                        raise ValueError('Survey altitude must be 5..25 m')
                    initial_xy = self.pose[:2]
                    route = plan_lawnmower(*bounds, start=initial_xy)
                    stats = route_stats(route, start=initial_xy)
                    params = dict(zip((
                        'coverage_min_x', 'coverage_max_x',
                        'coverage_min_y', 'coverage_max_y',
                        'coverage_lane_spacing'), bounds))
                    params.update(search_mode='terrain', coverage_altitude=altitude,
                                  coverage_start_x=initial_xy[0],
                                  coverage_start_y=initial_xy[1],
                                  coverage_clearance_verified=True)
                    for key, value in params.items():
                        mission_args.extend(['-p', f'{key}:={str(value).lower() if isinstance(value, bool) else value}'])
                    route_summary = ('Terrain sweep: %d waypoints, %.0f m; '
                                     '>=%.1f simulated minutes at 1.2 m/s, '
                                     'excluding turns/inspections') % (
                                         stats['waypoints'], stats['distance_m'],
                                         stats['optimistic_sim_minutes'])
                ok = self.managed_start('mission', mission_args)
                if ok:
                    self.auto_at = 0.
                    self.state = 'SEARCH'
                    if not self.managed_start('recorder', ['ros2', 'run', PACKAGE,
                        'record_target_trials', '--trial', trial]):
                        self.hold('Search cancelled: recorder failed to start')
                    else:
                        self.reason = f'{route_summary}; {trial} recorder started'
                return
            if action == 'goto':
                self.require_idle(require_scan=True)
                if str(req.get('frame_id', 'parrot1_odom')) != 'parrot1_odom':
                    raise ValueError('Only parrot1_odom coordinates are supported')
                self.goto_xy = finite_xy(req.get('x'), req.get('y'))
                self.state = 'GOTO'
                self.reason = f'Experimental go-to {self.goto_xy} at survey altitude'
                return
            if action == 'land':
                if self.state in ('SEARCH', 'GOTO', 'MANUAL'):
                    raise ValueError('Use ABORT / HOLD first')
                self.begin_landing(bool(req.get('clear_landing_area', False)))
                return
            raise ValueError(f'Unknown action: {action}')
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.reason = f'REJECTED: {exc}'
            self.get_logger().warning(self.reason)

    def require_idle(self, require_scan=False):
        if self.state not in ('IDLE', 'HOLD', 'ARRIVED'):
            raise ValueError(f'Cannot start: current mode is {self.state}')
        mission = self.tasks.get('mission')
        if mission is not None and mission.poll() is None:
            raise ValueError('Previous managed mission process has not exited')
        foreign = self.foreign_publishers()
        if foreign:
            raise ValueError('Other cmd_vel publisher(s): ' + ', '.join(foreign))
        if not self.pose_ok():
            raise ValueError('Fresh /parrot1/odom required')
        if require_scan and not self.scan_ok():
            raise ValueError('Fresh /parrot1/scan required')

    def abort(self, landing_clear):
        self.hold('ABORT: managed mission cancelled; checking landing conditions')
        if landing_clear:
            self.begin_landing(True)
        else:
            self.reason = 'ABORT/HOLD: confirm the area directly below is clear to land'

    def begin_landing(self, clearing):
        # Landing is permitted at any XY; no downward obstacle sensor exists.
        # Operator must confirm the ground below is clear in this SIMULATION.
        if self.foreign_publishers():
            self.hold('LANDING BLOCKED: another publisher controls cmd_vel')
        elif not self.pose_ok():
            self.hold('LANDING BLOCKED: odometry unavailable/stale')
        elif not clearing:
            self.hold('LANDING BLOCKED: operator has not confirmed clear ground below')
        else:
            self.state = 'LANDING'
            self.reason = ('Controlled descent at <=0.20 m/s at current XY; '
                           'terrain clearance and touchdown unverified')
            self.landing_confirmed = True

    def manual_cb(self, msg):
        if self.state == 'MANUAL':
            self.latest_manual = msg
            self.manual_at = time.monotonic()

    def auto_cb(self, msg):
        if self.state == 'SEARCH':
            self.latest_auto = msg
            self.auto_at = time.monotonic()

    def tick(self):
        if self.state == 'IDLE':
            if 'recorder' in self.tasks:
                self.managed_stop('recorder')
            return
        if self.foreign_publishers():
            self.hold('CONFLICT: unmanaged cmd_vel publisher detected')
            # Continue publishing hold while conflicting; cannot prevent external cmd_vel.
            return
        if self.state == 'SEARCH':
            p = self.tasks.get('mission')
            rec = self.tasks.get('recorder')
            if p is None:
                self.hold('Search process missing; check supervisor/mission.log')
            elif p.poll() is not None:
                if p.returncode == 0:
                    # Successful mission exit stops the recorder and frees the UI.
                    self.hold('Lawnmower scan completed; recorder stopped', 'IDLE')
                else:
                    self.hold(
                        f'Search process exited with code {p.returncode}; '
                        'check supervisor/mission.log')
            elif not rec or rec.poll() is not None:
                self.hold('Search stopped: recorder unavailable; see supervisor/recorder.log')
            elif not self.pose_ok():
                self.hold('Search hold: odometry stale')
            elif self.auto_at and time.monotonic() - self.auto_at < 0.7:
                t = self.latest_auto
                self.velocity(x=t.linear.x, z=t.linear.z, yaw=t.angular.z)
            else:
                self.velocity()  # Auto startup or watchdog; no stale motion
        elif self.state == 'MANUAL':
            if not self.pose_ok():
                self.hold('Manual stopped: odometry stale')
            elif time.monotonic() - self.manual_at < 0.5:
                t = self.latest_manual
                self.velocity(x=max(-0.5, min(0.5, t.linear.x)),
                              z=max(-0.35, min(0.35, t.linear.z)),
                              yaw=max(-0.8, min(0.8, t.angular.z)))
            else:
                self.velocity()
        elif self.state == 'GOTO':
            if not self.pose_ok() or not self.scan_ok():
                self.hold('Go-to stopped: stale odometry or LiDAR')
                return
            x,y,z,yaw = self.pose
            vx,wz,vz,done,err = goto_step(x,y,z,yaw,*self.goto_xy,self.front)
            if err:
                self.hold('Go-to stopped: ' + err)
            elif done:
                self.hold('Destination reached (XY tolerance 0.65 m)', 'ARRIVED')
            else:
                self.velocity(vx,vz,wz)
        elif self.state == 'LANDING':
            if not self.pose_ok():
                self.hold('Landing stopped: odometry stale')
                return
            _x, _y, z, _yaw = self.pose
            dz, done = descent_speed(z)
            if done:
                self.hold('Touchdown altitude estimated; motors NOT disarmed', 'LANDED_ESTIMATED')
            else:
                self.velocity(z=dz)
        elif self.state in ('HOLD', 'ARRIVED', 'LANDED_ESTIMATED'):
            self.velocity()

    def publish_status(self):
        proc = {k: ('running' if p.poll() is None else f'exited:{p.returncode}')
                for k,p in self.tasks.items()}
        self.status_pub.publish(String(data=json.dumps({
            'state': self.state, 'reason': self.reason,
            'processes': proc,
            'position': list(self.pose[:3]) if self.pose else None,
            'pose_fresh': bool(self.pose_ok()),
            'scan_fresh': bool(self.scan_ok()),
            'foreign_publishers': self.foreign_publishers(),
            'destination': list(self.goto_xy) if self.goto_xy else None,
            'stamp_unix': time.time(),
        })))

    def destroy_node(self):
        self.managed_stop('mission')
        for key in ('recorder', 'localizer', 'thermal'):
            self.managed_stop(key)
        self.velocity()
        return super().destroy_node()


def main():
    rclpy.init()
    node = MissionSupervisor()
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
