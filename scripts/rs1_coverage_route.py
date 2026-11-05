"""Pure-Python rectangular lawnmower planner for RS1 simulated search areas.

Coordinates must already be in the controller's /parrot1/odom XY frame.
The planner does not check terrain, canopy, obstacles, heat-camera footprint,
or ensure a safe transit to the first waypoint. Those require preflight review.
"""

import argparse
import math


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label} must be a finite number') from None
    if not math.isfinite(result):
        raise ValueError(f'{label} must be a finite number')
    return result


def _positions(low, high, spacing):
    lines = [low]
    while lines[-1] + spacing < high - 1e-9:
        lines.append(lines[-1] + spacing)
    if high - lines[-1] > 1e-9:
        lines.append(high)
    return lines


def route_length(waypoints, start=None):
    points = ([] if start is None else [start]) + list(waypoints)
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))


def plan_lawnmower(min_x, max_x, min_y, max_y, spacing, start=None):
    """Return a deterministic rectangle sweep with lane gaps <= spacing.

    Sweeps along the longer axis to reduce turns. Search starts from the
    nearest of four corner/orientation possibilities; corners are included.
    """
    min_x = _number(min_x, 'min_x')
    max_x = _number(max_x, 'max_x')
    min_y = _number(min_y, 'min_y')
    max_y = _number(max_y, 'max_y')
    spacing = _number(spacing, 'lane spacing')
    if not min_x < max_x or not min_y < max_y:
        raise ValueError('Search bounds must have min < max')
    if not 0.5 <= spacing <= 20.0:
        raise ValueError('Lane spacing must be between 0.5 and 20 metres')
    if max(max_x - min_x, max_y - min_y) > 300.0:
        raise ValueError('Search area exceeds 300 m per axis')
    if start is not None:
        if len(start) != 2:
            raise ValueError('Start must contain X,Y')
        start = (_number(start[0], 'start x'), _number(start[1], 'start y'))
    along_x = max_x - min_x >= max_y - min_y
    lanes = _positions(min_y, max_y, spacing) if along_x else _positions(min_x, max_x, spacing)
    if len(lanes) * 2 > 600:
        raise ValueError('Too many waypoints; increase lane spacing or reduce search area')
    if along_x:
        def build(sequence, forward):
            path = []
            for i, y in enumerate(sequence):
                a, b = (min_x, max_x) if (forward == (i % 2 == 0)) else (max_x, min_x)
                path.extend(((a, y), (b, y)))
            return path
    else:
        def build(sequence, forward):
            path = []
            for i, x in enumerate(sequence):
                a, b = (min_y, max_y) if (forward == (i % 2 == 0)) else (max_y, min_y)
                path.extend(((x, a), (x, b)))
            return path
    candidates = [build(seq, fwd)
                  for seq in (lanes, list(reversed(lanes)))
                  for fwd in (True, False)]
    if start is None:
        return candidates[0]
    return min(candidates, key=lambda path: (route_length(path, start), tuple(path[0])))


def route_stats(path, start=None, speed=1.2):
    speed = _number(speed, 'Speed')
    if speed <= 0:
        raise ValueError('Speed must be positive')
    length = route_length(path, start)
    return {
        'waypoints': len(path),
        'distance_m': round(length, 1),
        'optimistic_sim_minutes': round(length / speed / 60, 1),
        'turns': max(0, len(path) // 2 - 1),
    }


def main():
    p = argparse.ArgumentParser(description='Preview XY sweep; does not start a flight')
    for name in ('min_x', 'max_x', 'min_y', 'max_y', 'spacing'):
        p.add_argument('--' + name.replace('_', '-'), type=float, required=True)
    p.add_argument('--start-x', type=float, default=-10.0)
    p.add_argument('--start-y', type=float, default=7.0)
    p.add_argument('--speed', type=float, default=1.2)
    args = p.parse_args()
    start = (args.start_x, args.start_y)
    route = plan_lawnmower(args.min_x, args.max_x, args.min_y, args.max_y,
                           args.spacing, start=start)
    stats = route_stats(route, start=start, speed=args.speed)
    for k, v in stats.items():
        print(f'{k}: {v}')
    print('first waypoints:', route[:4])
    print('last waypoints:', route[-4:])
    print('WARNING: travel time is optimistic and does not include turns, '
          'inspection hovers or low Gazebo real-time factor.')
    print('WARNING: FOV/detection range and terrain/canopy clearance are NOT verified.')


if __name__ == '__main__':
    main()
