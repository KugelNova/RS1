"""Wire format for confirmed human rescue alerts."""
import json
import math

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

TARGET_ALERT_TOPIC = '/parrot1/rescue/target_alert'
WAYPOINT_COORDINATES = ('waypoint_x_m', 'waypoint_y_m', 'waypoint_z_m')
qos_profile_rescue_alert = QoSProfile(
    depth=100, reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)


def parse_confirmed_alert(data):
    """Normalize a confirmed alert. Legacy thermal alerts and other classes are ignored."""
    alert = json.loads(data)
    if not isinstance(alert, dict):
        raise ValueError('Expected a rescue alert object')
    if alert.get('classification') != 'human':
        return None
    alert['id'] = int(alert['id'])
    frame = alert.get('frame_id')
    if alert['id'] < 1 or not isinstance(frame, str) or not frame.strip():
        raise ValueError('Rescue alert needs a positive ID and a coordinate frame')
    for field in ('sim_time_sec', 'x_m', 'y_m', 'z_m', 'max_temp_c'):
        alert[field] = float(alert[field])
        if not math.isfinite(alert[field]):
            raise ValueError(f'Rescue alert {field} must be finite')
    alert['edge_clipped'] = str(alert.get('edge_clipped', False)).lower() == 'true'
    # A missing or invalid optional waypoint never suppresses the confirmed body estimate.
    waypoint_frame = alert.pop('waypoint_frame_id', None)
    waypoint_values = [alert.pop(field, None) for field in WAYPOINT_COORDINATES]
    if isinstance(waypoint_frame, str) and waypoint_frame.strip():
        try:
            xyz = tuple(map(float, waypoint_values))
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if all(map(math.isfinite, xyz)):
                alert.update(zip(WAYPOINT_COORDINATES, xyz))
                alert['waypoint_frame_id'] = waypoint_frame.strip()
    return alert


def format_ground_waypoint(alert, decimals=2):
    frame = alert.get('waypoint_frame_id')
    if not frame:
        return ''
    xyz = (alert[field] for field in WAYPOINT_COORDINATES)
    coordinates = '  '.join(f'{axis} {value:.{decimals}f}' for axis, value in zip('XYZ', xyz))
    return f'Ground waypoint  {coordinates} m  [{frame}]'
