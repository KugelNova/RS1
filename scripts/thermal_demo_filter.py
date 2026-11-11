"""Thermal demo display classifier: source-independent, no ROS/Qt imports.

Show every sufficiently large hot component or only temperature + H/W matches.
This is a VISUAL DEMONSTRATION, not the rescue target confirmation pipeline.
"""

from __future__ import annotations

import math
import numpy as np


def find_hot_objects(image, *, threshold_raw=30500, min_pixels=100,
                     temperature_range=(36.0, 38.0), ratio_range=(1.2, 4.0)):
    import cv2

    values = np.asarray(image)
    if values.ndim != 2:
        return []
    mask = (values >= int(threshold_raw)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((3, 3), dtype=np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    objects = []
    for component in range(1, n):
        x, y, w, h, pixels = (int(v) for v in stats[component])
        if pixels < min_pixels or w <= 0:
            continue
        crop = values[y:y+h, x:x+w]
        selected = crop[labels[y:y+h, x:x+w] == component]
        if not selected.size:
            continue
        temp = float(selected.max()) / 100.0 - 273.15
        ratio = h / float(w)
        candidate = (math.isfinite(temp) and temperature_range[0] <= temp <= temperature_range[1]
                     and ratio_range[0] <= ratio <= ratio_range[1])
        objects.append({'bbox': (x, y, w, h), 'hot_pixels': pixels,
                        'temp_c': temp, 'aspect_ratio': ratio,
                        'human_candidate': bool(candidate)})
    return objects
