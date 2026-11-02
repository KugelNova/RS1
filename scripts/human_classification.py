"""World-coordinate association of RGB object labels with thermal targets.

No Gazebo ground truth is used. Returns human, animal, or unknown; a hot
object or an absence of RGB evidence never counts as a confirmed person.
"""
import math


def match_evidence(target_xyz, evidence, stamp_sec, *, max_age_s=2.0,
                   radius_m=1.5, min_confidence=0.35):
    unknown = {'classification': 'unknown', 'label': '',
               'confidence': 0.0, 'reason': 'no_recent_rgb_evidence'}
    if not isinstance(evidence, dict):
        return unknown
    if not isinstance(evidence.get('objects'), list):
        return unknown
    evidence_sec = evidence.get('sim_time_sec')
    if not isinstance(evidence_sec, (int, float)) or not math.isfinite(evidence_sec):
        return unknown
    delta = float(stamp_sec) - float(evidence_sec)
    if delta < -0.25 or delta > max_age_s:
        return unknown
    matches = []
    for obj in evidence['objects']:
        if not isinstance(obj, dict):
            continue
        cls = obj.get('classification')
        if cls not in ('human', 'animal'):
            continue
        try:
            conf = float(obj['confidence'])
            x, y = [float(v) for v in obj['xyz_m'][:2]]
            dist = math.hypot(float(target_xyz[0])-x, float(target_xyz[1])-y)
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if all(math.isfinite(v) for v in (conf, x, y, dist)) and conf >= min_confidence and dist <= radius_m:
            matches.append((dist, -conf, cls, obj))
    if not matches:
        return {'classification': 'unknown', 'label': '', 'confidence': 0.0,
                'reason': 'no_spatially_matched_rgb_object'}
    # A spatially ambiguous overlap is not a safe human confirmation.
    if len({m[2] for m in matches}) > 1:
        return {'classification': 'unknown', 'label': '', 'confidence': 0.0,
                'reason': 'ambiguous_human_animal_overlap'}
    dist, negconf, cls, obj = min(matches, key=lambda m: (m[0], m[1]))
    return {'classification': cls, 'label': str(obj.get('label', cls)),
            'confidence': -negconf, 'distance_m': dist, 'reason': 'rgb_thermal_world_match'}
