"""Pure helpers for RS1 simulation-only mission supervisor.

No ROS dependencies: these checks can run in ordinary unit tests.
"""
import math

HOME_XY = (2.0, 0.0)  # Far-test world launch clearing; verify per world
HOME_RADIUS_M = 3.0
MAX_XY_M = 45.0
SURVEY_Z_M = 12.0
MAX_Z_M = 15.0
LAND_Z_M = 0.85  # Approximate base link touchdown; not motor disarm
ODOM_TIMEOUT_S = 3.0
SCAN_TIMEOUT_S = 3.0


def finite_xy(x, y):
    try:
        a, b = float(x), float(y)
    except (TypeError, ValueError):
        raise ValueError('Coordinates must be numeric') from None
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('Coordinates must be finite')
    if abs(a) > MAX_XY_M or abs(b) > MAX_XY_M:
        raise ValueError('Destination outside supported ±45 m far-test map')
    return a, b


def near_launch(x, y):
    return math.hypot(float(x) - HOME_XY[0], float(y) - HOME_XY[1]) <= HOME_RADIUS_M


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(value, low, high):
    return min(high, max(low, value))


def goto_step(x, y, z, yaw, target_x, target_y, front_distance,
              target_z=SURVEY_Z_M, maximum_z=MAX_Z_M):
    """Return (vx, wz, vz, arrived, error). Body-frame Twist; no route planner.

    A forward LiDAR cannot verify the ground below the vehicle or see through
    obstacles. An invalid or blocked scan prevents horizontal motion.
    """
    distance = math.hypot(target_x - x, target_y - y)
    vz = clamp(0.45 * (target_z - z), -0.35, 0.45)
    if z > maximum_z:
        return 0., 0., 0., False, 'Altitude exceeds configured limit'
    if distance <= 0.65 and abs(z - target_z) <= 0.5:
        return 0., 0., 0., True, ''
    if abs(z - target_z) > 0.5:
        # Vertical-first avoids horizontal transit at unverified altitude.
        return 0., 0., vz, False, ''
    error = wrap_angle(math.atan2(target_y-y, target_x-x) - yaw)
    if abs(error) > math.radians(10):
        return 0., clamp(1.1 * error, -0.5, 0.5), vz, False, ''
    if not math.isfinite(front_distance) or front_distance <= 2.5:
        return 0., 0., 0., False, 'Obstacle or invalid forward LiDAR (<2.5 m)'
    vx = max(0., min(0.65, 0.4 * distance, 0.3 * (front_distance - 2.5)))
    return vx, clamp(0.7 * error, -0.15, 0.15), vz, False, ''


def descent_speed(z):
    if not math.isfinite(z):
        raise ValueError('Altitude invalid')
    if z <= LAND_Z_M + 0.08:
        return 0., True
    return -min(0.20, max(0.06, (z - LAND_Z_M) * 0.35)), False


# RS1 simulated proximity and overflight assistance
MANUAL_STANDOFF_M = 0.5
MANUAL_BUFFER_M = 0.15  # scan period + velocity lag allowance (not a guarantee)
MANUAL_MAX_SPEED_MPS = 0.5


def manual_axis_speed(requested, distance, scan_fresh):
    """Limit forward/reverse body-X motion based on the matching LiDAR sector.

    A true 0.5m physical gap cannot be guaranteed without knowing body radius,
    laser mounting position, latency, and actual stopping dynamics.
    """
    requested = float(requested)
    if not math.isfinite(requested):
        return 0.0
    requested = max(-MANUAL_MAX_SPEED_MPS,
                    min(MANUAL_MAX_SPEED_MPS, requested))
    if abs(requested) < 1e-9:
        return 0.0
    if not scan_fresh or not math.isfinite(distance) or distance <= 0.0:
        return 0.0
    # Reduce speed progressively before entering the 0.5m stand-off region.
    available = max(0.0, distance - MANUAL_STANDOFF_M - MANUAL_BUFFER_M)
    permitted = min(MANUAL_MAX_SPEED_MPS, 0.75 * available)
    return math.copysign(min(abs(requested), permitted), requested) if permitted > 1e-6 else 0.0


class OverflightAssist:
    """Sim-only stop, climb, verify front clearance, gain extra height, resume.

    Never permits XY translation while actively climbing; bounds ascent and
    fails closed if scan becomes invalid. Does NOT check overhead clearance.
    """
    def __init__(self, blocked_m=2.5, cleared_m=3.5,
                 maximum_z=25.0, maximum_gain_m=8.0,
                 margin_over_top_m=1.0, climb_speed=0.35):
        if cleared_m <= blocked_m or maximum_gain_m <= margin_over_top_m:
            raise ValueError('Invalid overflight settings')
        self.blocked_m = float(blocked_m)
        self.cleared_m = float(cleared_m)
        self.maximum_z = float(maximum_z)
        self.maximum_gain_m = float(maximum_gain_m)
        self.margin_over_top_m = float(margin_over_top_m)
        self.climb_speed = float(climb_speed)
        self.reset()

    def reset(self):
        self.phase = 'IDLE'
        self.initial_z = None
        self.lift_to_z = None
        self.good_scans = 0

    @property
    def active(self):
        return self.phase != 'IDLE'

    def begin(self, z):
        if not math.isfinite(z) or z + self.margin_over_top_m + 0.5 > self.maximum_z:
            return False
        self.phase = 'CLIMB'
        self.initial_z = float(z)
        self.lift_to_z = None
        self.good_scans = 0
        return True

    def step(self, z, front_distance):
        """Return (vertical_speed, completed, error). XY must be zero."""
        if not self.active:
            return 0.0, True, ''
        if not math.isfinite(z) or not math.isfinite(front_distance) or front_distance <= 0:
            return 0.0, False, 'Overflight stopped: missing/invalid LiDAR or altitude'
        ascent_limit = min(self.maximum_z, self.initial_z + self.maximum_gain_m)
        if self.phase == 'CLIMB':
            if front_distance >= self.cleared_m and z >= self.initial_z + 0.5:
                self.good_scans += 1
            else:
                self.good_scans = 0
            if self.good_scans >= 3:
                self.lift_to_z = z + self.margin_over_top_m
                if self.lift_to_z > ascent_limit:
                    return 0.0, False, 'Overflight stopped: insufficient altitude for clearance margin'
                self.phase = 'EXTRA_LIFT'
            elif z >= ascent_limit - 0.10:
                return 0.0, False, 'Overflight stopped: no free forward corridor below altitude limit'
            else:
                return min(self.climb_speed, 0.7 * (ascent_limit - z)), False, ''
        if self.phase == 'EXTRA_LIFT':
            if front_distance < self.blocked_m:
                self.phase = 'CLIMB'
                self.good_scans = 0
                if z >= ascent_limit - 0.10:
                    return 0.0, False, 'Overflight stopped: reblocked at altitude limit'
                return max(0.0, min(self.climb_speed, 0.7 * (ascent_limit - z))), False, ''
            if z < self.lift_to_z - 0.10:
                return min(self.climb_speed, 0.7 * (self.lift_to_z - z)), False, ''
            self.reset()
            return 0.0, True, ''
        return 0.0, False, 'Overflight stopped: invalid state'
