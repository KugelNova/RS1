"""Focused LiDAR range refinement tests with camera-space synthetic surfaces."""

from pathlib import Path
from types import SimpleNamespace
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lidar_target_fusion import decode_xyz_points, refine_human_position


IDENTITY = ((0., 0., 0.), (0., 0., 0., 1.))


def run(points, *, depth=None, labels=None, lidar_stamp=1.02, lidar_to_rgb=IDENTITY,
        camera_xyz=(5.2, 0., 0.)):
    if depth is None:
        depth = np.full((100, 100), 5.2)
    if labels is None:
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[44:56, 44:56] = 1
    return refine_human_position(
        camera_xyz, np.asarray(points, dtype=float), depth, labels, 1,
        lidar_to_rgb=lidar_to_rgb, rgb_to_thermal=IDENTITY, rgb_to_fixed=IDENTITY,
        lidar_stamp_s=lidar_stamp, rgb_stamp_s=1.0,
        rgb_hfov=np.pi / 2, thermal_hfov=np.pi / 2)


@pytest.fixture
def hits():
    return np.array([(5., -.3, 0.), (5., -.1, .12), (5., .1, -.12), (5., .3, 0.)])


def test_agreeing_points_reduce_camera_range_bias(hits):
    match = run(hits)
    assert match['xyz'] == pytest.approx((5.1, 0., 0.))
    assert match['lidar_points'] == 4
    assert match['depth_correction_m'] == pytest.approx(-0.1)


def test_transform_applied_before_matching(hits):
    lidar_origin_in_rgb = (0.1, 0.15, -0.1)
    translated = hits - lidar_origin_in_rgb
    match = run(translated, lidar_to_rgb=(lidar_origin_in_rgb, IDENTITY[1]))
    assert match['xyz'] == pytest.approx((5.1, 0., 0.))


def test_stale_and_missing_points_do_not_refine(hits):
    assert run(hits, lidar_stamp=0.8) is None
    assert run(np.empty((0, 3))) is None
    assert run(np.repeat(hits[:1], 6, axis=0)) is None


def test_foreground_background_and_other_components_do_not_refine(hits):
    assert run(hits - (0.6, 0., 0.)) is None
    assert run(hits + (0.6, 0., 0.)) is None
    other_target = np.full((100, 100), 2, dtype=np.int32)
    assert run(hits, labels=other_target) is None
    assert run(hits, depth=np.full((100, 100), np.inf)) is None


def test_inconsistent_depth_layers_do_not_refine(hits):
    hits[:2, 0] = 4.92
    hits[2:, 0] = 5.45
    assert run(hits) is None


def test_decoder_honors_field_offsets_endianness_and_row_padding():
    fields = [SimpleNamespace(name=name, datatype=7, count=1, offset=offset)
              for name, offset in [('z', 4), ('y', 8), ('x', 12)]]
    first = struct.pack('>Ifff', 42, 3., 2., 1.)
    second = struct.pack('>Ifff', 42, float('nan'), 0., 0.)
    cloud = SimpleNamespace(fields=fields, is_bigendian=True, width=1, height=2,
                            point_step=16, row_step=20, data=first + b'pad!' + second + b'pad!')
    assert decode_xyz_points(cloud).tolist() == [[1., 2., 3.]]
    cloud.data = cloud.data[:-1]
    with pytest.raises(ValueError, match='strides'):
        decode_xyz_points(cloud)
