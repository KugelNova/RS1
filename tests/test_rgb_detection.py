"""Regression using RGB frames captured from the seven models in Gazebo."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR / 'scripts'))

from rgb_detection import RgbDetector


@pytest.fixture(scope='module')
def detector():
    return RgbDetector(package_dir=PACKAGE_DIR)


@pytest.mark.parametrize('row,column,classification', [
    (0, 0, 'human'),
    (0, 1, 'human'),
    (0, 2, 'human'),   # Off-centre view from the Parrot camera
    (1, 0, 'animal'),  # Bear
    (1, 1, 'animal'),  # Deer
    (1, 2, 'animal'),  # Bird
    (2, 0, 'animal'),  # Mouse
    (2, 1, 'animal'),  # Kangaroo
    (2, 2, 'animal'),  # Horse
])
def test_recognizes_rendered_model(detector, row, column, classification):
    atlas_path = Path(__file__).parent / 'fixtures' / 'rgb_targets.png'
    atlas = cv2.imread(str(atlas_path), cv2.IMREAD_COLOR)
    tile_size = 112
    y, x = row * tile_size, column * tile_size
    image = atlas[y:y + tile_size, x:x + tile_size]
    detections = detector.detect(image)
    assert [d['classification'] for d in detections] == [classification]
    assert all(0.7 <= d['confidence'] <= 1.0 for d in detections)


def test_red_terrain_or_ambient_light_does_not_confirm_human(detector):
    image = np.full((120, 160, 3), 40, dtype=np.uint8)
    image[30:80, 55:83] = (12, 15, 175)
    assert all(d['classification'] != 'human' for d in detector.detect(image))


def test_clipped_silhouette_is_not_evidence(detector):
    atlas_path = Path(__file__).parent / 'fixtures' / 'rgb_targets.png'
    atlas = cv2.imread(str(atlas_path), cv2.IMREAD_COLOR)
    # Cut off the lower half of a person at the bottom edge of the frame.
    assert detector.detect(atlas[:56, :112]) == []