"""Regression coverage for projected RGB-D and thermal association."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from camera_fusion import decode_depth_image, match_rgbd_thermal


IDENTITY = ((0, 0, 0), (0, 0, 0, 1))


def detection(box, classification='human'):
    return {'bbox_xyxy': box, 'classification': classification,
            'label': 'person' if classification == 'human' else 'deer',
            'confidence': 0.9}


def match(depth, detections, labels, rgb_to_thermal=IDENTITY, rgb_to_fixed=IDENTITY):
    return match_rgbd_thermal(depth, detections, labels, rgb_to_thermal=rgb_to_thermal,
                              rgb_to_fixed=rgb_to_fixed, rgb_hfov=math.pi / 2,
                              thermal_hfov=math.pi / 2)


def test_depth_decoder_respects_stride_and_endianness():
    image = SimpleNamespace(encoding='32FC1', width=2, height=2, step=12,
                            is_bigendian=True,
                            data=np.asarray([[1.5, 2.5, 0], [4.5, 5.5, 0]], '>f4').tobytes())
    assert decode_depth_image(image) == pytest.approx(np.array([[1.5, 2.5], [4.5, 5.5]]))
    image.encoding = '16UC1'
    image.step = 6
    image.is_bigendian = False
    image.data = np.asarray([[1500, 2500, 0], [4500, 5500, 0]], '<u2').tobytes()
    assert decode_depth_image(image) == pytest.approx(np.array([[1.5, 2.5], [4.5, 5.5]]))


def test_separate_labels_and_localize_visible_surface_without_ground_assumption():
    depth = np.full((64, 64), 8.0)
    labels = np.zeros_like(depth, dtype=np.int32)
    labels[16:40, 22:33] = 1
    labels[25:39, 45:56] = 2
    yaw90 = (0, 0, math.sqrt(0.5), math.sqrt(0.5))
    matches = match(depth, [detection((22, 16, 33, 40)), detection((45, 25, 56, 39), 'animal')],
                    labels, rgb_to_fixed=((10, 20, 3), yaw90))
    assert matches[1]['classification'] == 'human'
    assert matches[1]['coverage'] == 1
    assert matches[1]['xyz'] == pytest.approx((8.875, 28, 4))
    assert matches[2]['classification'] == 'animal'
    assert matches[2]['label'] == 'deer'


def test_spatial_mismatch_insufficient_support_and_invalid_depth_return_no_match():
    depth = np.full((64, 64), 8.0)
    labels = np.zeros_like(depth, dtype=np.int32)
    labels[16:40, 22:33] = 1
    assert not match(depth, [detection((40, 15, 62, 42))], labels)
    assert not match(depth, [detection((21, 15, 25, 42))], labels)
    depth[16:40, 22:33] = np.inf
    depth[16:40, 25:30] = np.nan
    assert not match(depth, [detection((22, 16, 33, 40))], labels)


def test_competing_labels_or_one_box_covering_multiple_targets_are_ambiguous():
    depth = np.full((64, 64), 8.0)
    labels = np.zeros_like(depth, dtype=np.int32)
    labels[16:40, 22:33] = 1
    labels[25:39, 45:56] = 2
    assert not match(depth, [detection((22, 16, 33, 40)), detection((22, 16, 33, 40), 'animal')],
                     labels)
    assert not match(depth, [detection((20, 15, 60, 43))], labels)


def test_foreground_and_background_depth_layers_are_not_one_target():
    depth = np.full((64, 64), 8.0)
    labels = np.zeros_like(depth, dtype=np.int32)
    labels[16:40, 22:33] = 1
    depth[28:40, 22:33] = 11.0
    assert not match(depth, [detection((22, 16, 33, 40))], labels)


def test_narrow_thermal_lens_and_offset_pitch_have_dense_calibrated_support():
    # Intersect each thermal ray with a planar RGB-depth target to derive its
    # analytical silhouette. Use the Parrot lens sizes and relative pitch.
    depth = np.full((480, 720), 8.0, dtype=np.float32)
    v, u = np.indices((240, 320), dtype=float)
    thermal_f = 160 / math.tan(0.610865 / 2)
    rgb_f = 360 / math.tan(2.0944 / 2)
    thermal_rays = np.stack((np.ones_like(u), (159.5 - u) / thermal_f,
                             (119.5 - v) / thermal_f), axis=-1)
    pitch = math.radians(-10)
    rotation = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                         [-math.sin(pitch), 0, math.cos(pitch)]])
    translation = np.array((-0.012, 0, 0.070))
    origins = -translation @ rotation
    directions = thermal_rays @ rotation
    surface = origins + ((8.0 - origins[0]) / directions[..., 0])[..., None] * directions
    rgb_u = 359.5 - rgb_f * surface[..., 1] / 8.0
    rgb_v = 239.5 - rgb_f * surface[..., 2] / 8.0
    box = (347, 260, 370, 291)
    left, top, right, bottom = box
    inside = ((left <= rgb_u) & (rgb_u < right)
              & (top <= rgb_v) & (rgb_v < bottom))
    labels = inside.astype(np.int32)
    assert inside.sum() > 2000
    matches = match_rgbd_thermal(
        depth, [detection(box)], labels, rgb_to_fixed=IDENTITY,
        rgb_to_thermal=(translation, (0, math.sin(pitch / 2), 0, math.cos(pitch / 2))))
    assert matches[1]['coverage'] > 0.94
    assert matches[1]['xyz'] == pytest.approx(np.median(surface[inside], axis=0), abs=0.08)
