"""Image-based RGB recognition for the bundled Gazebo human and wildlife meshes.

The actors in terrain_new share a red material, which provides image proposals.
Classification compares their observed silhouettes with projected views of all
seven supported meshes. No Gazebo poses, entity names, or labels are read.
"""
from pathlib import Path

import cv2
import numpy as np

from rescue_population.entities import SUPPORTED_ANIMALS, fitted_mesh


_MASK_SIZE = 48
_MASK_PADDING = 3
_MIN_PIXELS = 25
_MIN_ANIMAL_SIMILARITY = 0.70
_MIN_HUMAN_SIMILARITY = 0.80
_MIN_MARGIN = 0.055
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def _read_mesh(mesh):
    vertices = []
    faces = []
    with Path(mesh.path).open(encoding='utf-8') as source:
        for line in source:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'v':
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif fields[0] == 'f':
                face = [int(value.split('/')[0]) for value in fields[1:]]
                faces.append(np.array([i - 1 if i > 0 else len(vertices) + i for i in face]))
    xyz = np.asarray(vertices)[:, [0, 2, 1]]
    xyz[:, 1] *= -1
    xyz = xyz * mesh.scale + mesh.translation
    xyz[:, 2] -= mesh.dimensions[2] / 2.0
    return xyz, faces


def _fit_image(mask):
    """Center a cropped silhouette in a square while preserving aspect ratio."""
    height, width = mask.shape
    content_size = _MASK_SIZE - 2 * _MASK_PADDING
    scale = content_size / max(width, height)
    fit_width = max(1, round(width * scale))
    fit_height = max(1, round(height * scale))
    resized = cv2.resize(mask, (fit_width, fit_height), interpolation=cv2.INTER_LINEAR)
    result = np.zeros((_MASK_SIZE, _MASK_SIZE), dtype=np.uint8)
    y = (_MASK_SIZE - fit_height) // 2
    x = (_MASK_SIZE - fit_width) // 2
    result[y:y + fit_height, x:x + fit_width] = resized
    return result


def _draw_projection(points, faces):
    pixels = _MASK_SIZE - 2 * _MASK_PADDING - 1
    extent = np.ptp(points, axis=0)
    points = (points - (points.max(axis=0) + points.min(axis=0)) / 2) * (pixels / extent.max())
    # Subpixel vertices keep narrow features consistent between adjacent poses.
    points = np.rint((points + (_MASK_SIZE - 1) / 2) * 16).astype(np.int32)
    mask = np.zeros((_MASK_SIZE, _MASK_SIZE), dtype=np.uint8)
    for face in faces:
        cv2.fillPoly(mask, [points[face]], 1, shift=4)
    return mask


def _mesh_views(mesh):
    xyz, faces = _read_mesh(mesh)
    for azimuth in np.deg2rad(np.arange(0, 360, 12)):
        cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)
        right = np.array([-sin_az, cos_az, 0.0])
        for elevation_deg in (-15, 0, 15, 30, 45, 52.5, 60, 75, 90):
            elevation = np.deg2rad(elevation_deg)
            # Sample the finer off-centre views around the Parrot's usual
            # viewing angles, where narrow limb gaps change with camera roll.
            rolls = (-16, -8, 0, 8, 16) if 45 <= elevation_deg <= 60 else (-16, 0, 16)
            cos_el, sin_el = np.cos(elevation), np.sin(elevation)
            toward_camera = np.array([cos_az * cos_el, sin_az * cos_el, sin_el])
            up = np.cross(toward_camera, right)
            xy = np.column_stack((xyz @ right, -(xyz @ up)))
            along = xyz @ toward_camera
            # A close perspective view and a distant, approximately orthographic view.
            for distance in (mesh.dimensions[2] * 3.0, mesh.dimensions[2] * 14.0):
                perspective = xy / (distance - along[:, None])
                for roll in np.deg2rad(rolls):
                    c, s = np.cos(roll), np.sin(roll)
                    yield _draw_projection(perspective @ ((c, -s), (s, c)), faces)


class RgbDetector:
    """Return human/animal detections supported by the RGB silhouette alone."""

    def __init__(self, *, package_dir):
        templates = []
        human_templates = []
        for label in ('person', *SUPPORTED_ANIMALS):
            is_human = label == 'person'
            mesh = (fitted_mesh('human', package_dir) if is_human
                    else fitted_mesh('animal', package_dir, label))
            for mask in _mesh_views(mesh):
                templates.append(np.packbits(mask))
                human_templates.append(is_human)
        self._templates = np.array(templates)
        self._template_areas = _POPCOUNT[self._templates].sum(axis=1)
        self._human_templates = np.array(human_templates)

    def _classify(self, mask):
        observed = np.packbits(_fit_image(mask))
        area = _POPCOUNT[observed].sum()
        # A large area mismatch cannot meet the minimum IoU even if perfectly
        # aligned. Skip these templates before taking pixel intersections.
        eligible = ((self._template_areas >= area * _MIN_ANIMAL_SIMILARITY) &
                    (self._template_areas * _MIN_ANIMAL_SIMILARITY <= area))
        template_areas = self._template_areas[eligible]
        intersections = _POPCOUNT[self._templates[eligible] & observed].sum(axis=1)
        scores = intersections / (template_areas + area - intersections)
        is_human = self._human_templates[eligible]
        best_human = float(scores[is_human].max(initial=0.0))
        best_animal = float(scores[~is_human].max(initial=0.0))
        # Human confirmation uses the stricter match threshold. Scores are
        # silhouette overlap, not calibrated classification probabilities.
        if best_human >= _MIN_HUMAN_SIMILARITY and best_human - best_animal >= _MIN_MARGIN:
            return 'human', 'person', best_human
        if best_animal >= _MIN_ANIMAL_SIMILARITY and best_animal - best_human >= _MIN_MARGIN:
            return 'animal', 'animal', best_animal
        return None

    def detect(self, bgr):
        """Return xyxy boxes in original RGB pixel coordinates (upper bounds exclusive)."""
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError('RGB detector expects an 8-bit, three-channel BGR image')
        blue, green, red = (bgr[:, :, channel].astype(np.int16) for channel in range(3))
        other = np.maximum(blue, green)
        red_mask = ((red - other >= 14) & (red * 3 >= other * 4)).astype(np.uint8)
        components, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        image_height, image_width = red_mask.shape
        detections = []
        for component in range(1, components):
            left, top, width, height, count = (int(value) for value in stats[component])
            right, bottom = left + width, top + height
            if (count < _MIN_PIXELS or min(width, height) < 4 or left == 0 or top == 0 or
                    right == image_width or bottom == image_height):
                continue
            result = self._classify((labels[top:bottom, left:right] == component).astype(np.uint8))
            if result is None:
                continue
            classification, label, confidence = result
            detections.append({
                'bbox_xyxy': (left, top, right, bottom),
                'classification': classification,
                'label': label,
                'confidence': confidence,
            })
        return detections