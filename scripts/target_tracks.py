"""Confirm spatially stable detections across distinct frames.

An optional required class limits which observations can confirm a track.
An observed animal cancels a nearby pending human confirmation.
"""
from collections import deque
from statistics import median
from math import hypot


class StableTargetTracker:
    def __init__(self, *, confirm_frames=6, confirm_span_s=0.65,
                 min_hot_pixels=100, min_bbox_height_px=24,
                 candidate_radius_m=0.80, max_scatter_m=0.45,
                 dedup_radius_m=1.90, expire_s=2.0, require_unclipped=True,
                 required_classification=None):
        self.confirm_frames = int(confirm_frames)
        self.confirm_span_s = float(confirm_span_s)
        self.min_hot_pixels = int(min_hot_pixels)
        self.min_bbox_height_px = int(min_bbox_height_px)
        self.candidate_radius_m = float(candidate_radius_m)
        self.max_scatter_m = float(max_scatter_m)
        self.dedup_radius_m = float(dedup_radius_m)
        self.expire_s = float(expire_s)
        self.require_unclipped = bool(require_unclipped)
        self.required_classification = required_classification
        self.pending = []
        self.rejected = {'small': 0, 'clipped': 0, 'unstable': 0, 'duplicate': 0}

    @staticmethod
    def distance(a, b):
        return hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def median_position(samples):
        return (float(median(s['xyz'][0] for s in samples)),
                float(median(s['xyz'][1] for s in samples)),
                float(median(s['xyz'][2] for s in samples)))

    def observe(self, observations, stamp_sec, confirmed_xyz):
        """Return new, confirmed observations from one DISTINCT thermal image.

        observations: dicts with xyz, pixels, bbox_height, edge_clipped,
          temp_c, foot_u, foot_v, stamp (ROS stamp).  Sim time is numeric.
        confirmed_xyz: existing confirmed locations supplied by node.
        """
        if stamp_sec <= 0:
            return []  # No frame timestamp: cannot verify distinct frames/duration.
        self.pending = [p for p in self.pending
                        if 0 <= stamp_sec - p['last_stamp'] <= self.expire_s]
        animals = []
        if self.required_classification == 'human':
            animals = [obs['xyz'] for obs in observations if obs.get('classification') == 'animal']
        if animals:
            self.pending = [
                p for p in self.pending
                if all(self.distance(self.median_position(p['samples']), animal)
                       > self.candidate_radius_m for animal in animals)
            ]
        consumed = set()  # A component cannot count twice for the same candidate in one image.
        fresh = []
        for obs in observations:
            if self.required_classification and obs.get('classification') != self.required_classification:
                continue
            if any(self.distance(obs['xyz'], animal) <= self.candidate_radius_m for animal in animals):
                continue
            if obs['pixels'] < self.min_hot_pixels or obs['bbox_height'] < self.min_bbox_height_px:
                self.rejected['small'] += 1
                continue
            if self.require_unclipped and obs['edge_clipped']:
                self.rejected['clipped'] += 1
                continue
            pos = obs['xyz']
            if any(self.distance(pos, c) < self.dedup_radius_m for c in (*confirmed_xyz, *[f['xyz'] for f in fresh])):
                self.rejected['duplicate'] += 1
                continue
            candidates = [(self.distance(pos, self.median_position(p['samples'])), i)
                          for i, p in enumerate(self.pending)
                          if i not in consumed and p['last_stamp'] < stamp_sec]
            distance, index = min(candidates, default=(float('inf'), None))
            if distance > self.candidate_radius_m:
                index = None
            if index is None:
                p = {'samples': deque(maxlen=24), 'last_stamp': stamp_sec}
                self.pending.append(p)
                index = len(self.pending) - 1
            else:
                p = self.pending[index]
            consumed.add(index)
            p['samples'].append(dict(obs, stamp_sec=stamp_sec))
            p['last_stamp'] = stamp_sec
            samples = p['samples']
            if len(samples) < self.confirm_frames:
                continue
            if stamp_sec - samples[0]['stamp_sec'] < self.confirm_span_s:
                continue
            m = self.median_position(samples)
            scatter = max(self.distance(m, s['xyz']) for s in samples)
            if scatter > self.max_scatter_m:
                self.rejected['unstable'] += 1
                # Discard unstable samples and permit reconfirmation when view settles.
                p['samples'] = deque(list(samples)[-2:], maxlen=24)
                continue
            lidar_stamps = {s.get('lidar_stamp_ns') for s in samples} - {None}
            result = dict(obs, xyz=m, lidar_frames=len(lidar_stamps))
            fresh.append(result)
        self.pending = [p for p in self.pending
                        if not any(self.distance(self.median_position(p['samples']), f['xyz'])
                                   < self.candidate_radius_m for f in fresh)]
        return fresh
