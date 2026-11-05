#!/usr/bin/env python3
"""Pure-Python/Numpy evidence-aware surface grid for RS1 mapping V3.

This is NOT a ground segmentation oracle. Observations may include trees and
rocks. Only candidate surface cells are exported as ground candidates and all
other cases are left unknown, never assigned invented ground elevation.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class Cell:
    observations: deque = field(default_factory=lambda: deque(maxlen=128))
    sample_count: int = 0
    frame_count: int = 0


class SurfaceGrid:
    def __init__(self, grid=0.5, min_frames=3, max_iqr_m=1.0,
                 neighbor_jump_m=2.0, max_frame_history=128):
        if not (math.isfinite(grid) and grid > 0):
            raise ValueError('grid must be positive')
        if min_frames < 1 or max_frame_history < 3:
            raise ValueError('Invalid observation limits')
        if not all(math.isfinite(v) and v >= 0 for v in (max_iqr_m, neighbor_jump_m)):
            raise ValueError('Invalid thresholds')
        self.grid = float(grid)
        self.min_frames = int(min_frames)
        self.max_iqr_m = float(max_iqr_m)
        self.neighbor_jump_m = float(neighbor_jump_m)
        self.max_frame_history = int(max_frame_history)
        self.cells = {}

    def add_cloud(self, world_xyz):
        """Take one statistically independent per-cell lower-surface reading per cloud."""
        a = np.asarray(world_xyz, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError('world_xyz must have shape (N, 3)')
        a = a[np.isfinite(a).all(axis=1)]
        if len(a) == 0:
            return 0
        grouped = {}
        for x, y, z in a:
            key = (math.floor(x / self.grid), math.floor(y / self.grid))
            grouped.setdefault(key, []).append(z)
        for key, zs in grouped.items():
            if key not in self.cells:
                self.cells[key] = Cell(observations=deque(maxlen=self.max_frame_history))
            cell = self.cells[key]
            # Low within-cloud percentile resists a one-off low pixel in a cell.
            low_surface = float(np.percentile(zs, 20))
            cell.observations.append(low_surface)
            cell.sample_count += len(zs)
            cell.frame_count += 1
        return len(grouped)

    def summaries(self):
        """Build surface candidates. Never claim bare earth has been verified."""
        estimates = {}
        for key, cell in self.cells.items():
            samples = np.asarray(cell.observations, dtype=np.float64)
            height = float(np.percentile(samples, 20))
            iqr = float(np.percentile(samples, 75) - np.percentile(samples, 25))
            estimates[key] = (height, iqr)
        result = []
        for key, cell in sorted(self.cells.items()):
            z, iqr = estimates[key]
            if cell.frame_count < self.min_frames:
                status = 'insufficient_views'
            elif iqr > self.max_iqr_m:
                status = 'unstable_surface'
            else:
                # Mark sharp positive spikes where multiple nearby cells give
                # substantially lower returns. Does NOT solve wide canopies.
                lower_neighbors = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        adj = (key[0] + dx, key[1] + dy)
                        if adj not in estimates:
                            continue
                        other = self.cells[adj]
                        if other.frame_count < self.min_frames:
                            continue
                        zz, ii = estimates[adj]
                        if ii > self.max_iqr_m:
                            continue
                        if z - zz > self.neighbor_jump_m:
                            lower_neighbors += 1
                status = ('possible_obstacle' if lower_neighbors >= 2
                          else 'surface_candidate')
            result.append({
                'x': (key[0] + 0.5) * self.grid,
                'y': (key[1] + 0.5) * self.grid,
                'observed_z': z,
                'ground_candidate_z': z if status == 'surface_candidate' else None,
                'sample_count': cell.sample_count,
                'frame_count': cell.frame_count,
                'height_iqr_m': iqr,
                'status': status,
            })
        return result
