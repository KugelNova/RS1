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


def goto_step(x, y, z, yaw, target_x, target_y, front_distance):
    """Return (vx, wz, vz, arrived, error). Body-frame Twist; no route planner.

    A forward LiDAR cannot verify the ground below the vehicle or see through
    obstacles. An invalid or blocked scan prevents horizontal motion.
    """
    distance = math.hypot(target_x - x, target_y - y)
    vz = clamp(0.45 * (SURVEY_Z_M - z), -0.35, 0.45)
    if z > MAX_Z_M:
        return 0., 0., 0., False, 'Altitude exceeds 15 m limit'
    if distance <= 0.65 and abs(z - SURVEY_Z_M) <= 0.5:
        return 0., 0., 0., True, ''
    if abs(z - SURVEY_Z_M) > 0.5:
        # Vertical-first avoids horizontal transit at unverified altitude.
        return 0., 0., vz, False, ''
    error = wrap_angle(math.atan2(target_y-y, target_x-x) - yaw)
    if abs(error) > math.radians(10):
        return 0., clamp(1.1 * error, -0.5, 0.5), vz, False, ''
    if not math.isfinite(front_distance) or front_distance < 2.5:
        return 0., 0., 0., False, 'Obstacle or invalid forward LiDAR (<2.5 m)'
    vx = max(0., min(0.65, 0.4 * distance, 0.3 * (front_distance - 2.5)))
    return vx, clamp(0.7 * error, -0.15, 0.15), vz, False, ''


def descent_speed(z):
    if not math.isfinite(z):
        raise ValueError('Altitude invalid')
    if z <= LAND_Z_M + 0.08:
        return 0., True
    return -min(0.20, max(0.06, (z - LAND_Z_M) * 0.35)), False
