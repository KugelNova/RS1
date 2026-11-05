"""Read-only helper for the ROS2 rescue target localizer's mission CSV files.

Used by scripts/drone_ui.py. Safe to import without ROS or PyQt5.
"""
import csv
import math
import os
import re
from pathlib import Path

MISSION_RE = re.compile(r'^targets_\d{8}_\d{6}_\d{6}\.csv$')
NUMERIC = ('sim_time_sec', 'x_m', 'y_m', 'z_m', 'max_temp_c',
           'foot_pixel_u', 'foot_pixel_v')


def rescue_log_dir():
    return Path(os.path.expanduser(os.environ.get('RESCUE_LOG_DIR', '~/rescue_logs')))


def list_missions(directory=None):
    root = Path(directory) if directory is not None else rescue_log_dir()
    if not root.is_dir():
        return []
    candidates = []
    for p in root.iterdir():
        if not p.is_file() or p.is_symlink() or not MISSION_RE.fullmatch(p.name):
            continue
        candidates.append((p.stat().st_mtime_ns, p.name))
    return [name for _, name in sorted(candidates, reverse=True)]


def mission_path(name, directory=None):
    root = Path(directory) if directory is not None else rescue_log_dir()
    if not MISSION_RE.fullmatch(str(name)):
        raise ValueError('Invalid mission filename')
    if name not in list_missions(root):
        raise FileNotFoundError('Mission log not found')
    return root / name


def read_mission(name, directory=None):
    """Return complete valid CSV rows; ignore an unfinished final row.

    A localizer flushes each completed row. Its active CSV can be inspected
    while being written. The caller should show the associated frame_id.
    """
    path = mission_path(name, directory)
    records = []
    with path.open('r', encoding='utf-8', newline='') as stream:
        for line in csv.DictReader(stream):
            try:
                record = {
                    'id': int(line['id']),
                    'frame_id': line['frame_id'],
                    'hot_pixels': int(line['hot_pixels']),
                    'edge_clipped': line['edge_clipped'].strip().lower() == 'true',
                }
                for field in NUMERIC:
                    record[field] = float(line[field])
                if not record['frame_id']:
                    continue
                # Old logs and targets outside terrain_new may have no waypoint.
                waypoint_frame = (line.get('waypoint_frame_id') or '').strip()
                if waypoint_frame:
                    fields = ('waypoint_x_m', 'waypoint_y_m', 'waypoint_z_m')
                    try:
                        waypoint = {field: float(line[field]) for field in fields}
                    except (KeyError, TypeError, ValueError, OverflowError):
                        waypoint = {}
                    if waypoint and all(map(math.isfinite, waypoint.values())):
                        record.update(waypoint, waypoint_frame_id=waypoint_frame)
                records.append(record)
            except (KeyError, ValueError, TypeError, AttributeError):
                continue
    return records
