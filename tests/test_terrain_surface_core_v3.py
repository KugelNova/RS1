"""Pure-Numpy regression tests for RS1 terrain V3 core; no ROS required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import numpy as np
from terrain_surface_core_v3 import SurfaceGrid


def test_flat_ground_candidacy():
    g = SurfaceGrid()
    for i in range(6):
        g.add_cloud([[0.1, 0.1, 0.001 * i], [0.2, 0.2, 0.001 * i]])
    r = g.summaries()[0]
    assert r['status'] == 'surface_candidate'
    assert abs(r['ground_candidate_z']) < 0.01
    assert r['frame_count'] == 6
    assert r['sample_count'] == 12


def test_single_low_outlier_not_kept_forever():
    g = SurfaceGrid()
    g.add_cloud([[0.1, 0.1, -10]])
    for _ in range(12):
        g.add_cloud([[0.1, 0.1, 0]])
    r = g.summaries()[0]
    assert r['status'] == 'surface_candidate'
    assert abs(r['observed_z']) < 0.1


def test_mixed_surface_flagged_unstable():
    g = SurfaceGrid()
    for z in [0, 0, 0, 0, 8, 8, 8, 8]:
        g.add_cloud([[0.1, 0.1, z]])
    r = g.summaries()[0]
    assert r['status'] == 'unstable_surface'
    assert r['ground_candidate_z'] is None


def test_elevated_cell_flagged_against_neighbors():
    g = SurfaceGrid()
    for _ in range(5):
        g.add_cloud([[0.1, 0.1, 7], [-0.1, 0.1, 0], [0.1, -0.1, 0]])
    center = next(r for r in g.summaries() if r['x'] == 0.25 and r['y'] == 0.25)
    assert center['status'] == 'possible_obstacle'
    assert center['ground_candidate_z'] is None


def test_insufficient_views_unknown():
    g = SurfaceGrid()
    g.add_cloud([[0.1, 0.1, 0]])
    r = g.summaries()[0]
    assert r['status'] == 'insufficient_views'
    assert r['ground_candidate_z'] is None


def test_nonfinite_points_ignored():
    g = SurfaceGrid()
    g.add_cloud([[0, 0, 1], [np.nan, 1, 1], [1, 1, np.inf]])
    assert len(g.cells) == 1
    assert next(iter(g.cells.values())).sample_count == 1
