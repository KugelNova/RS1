"""Explainable binary thermal silhouette classifier for controlled RS1 demo.

No model downloads / ML dependencies. Input is a cropped binary connected component
representing ONE hot object. Distinguishes upright human-like silhouettes from
horizontal non-human-like silhouettes. Unknown is NOT evidence of an animal.

This is a view-dependent demo heuristic, not a general wildlife/person recognizer.
"""
from statistics import median


def _runs(values):
    """Count foreground runs in a boolean sequence."""
    count = 0
    previous = False
    for value in values:
        current = bool(value)
        if current and not previous:
            count += 1
        previous = current
    return count


def classify_hot_silhouette(mask):
    """Return status + explainable image-shape measurements.

    mask: any 2-D numpy array/list of booleans for a *single* cropped component.
    Returns 'human_like', 'non_human_like', or 'unknown'. The returned score is a
    heuristic, not a calibrated probability. Avoid assigning animal species.
    """
    import numpy as np  # already used by the thermal localizer

    a = np.asarray(mask, dtype=bool)
    if a.ndim != 2 or a.size == 0 or not a.any():
        return {'classification': 'unknown', 'reason': 'empty', 'aspect_ratio': 0.0,
                'head_torso_ratio': None, 'leg_split': False, 'fill_ratio': 0.0,
                'shape_score': 0}
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = a.shape
    area = int(a.sum())
    aspect = h / float(w)
    fill = area / float(h*w)
    result = {'classification': 'unknown', 'reason': 'ambiguous_orientation_or_shape',
              'aspect_ratio': round(aspect, 3), 'head_torso_ratio': None,
              'leg_split': False, 'fill_ratio': round(fill, 3), 'shape_score': 0}
    if h < 24 or w < 6 or area < 90:
        result['reason'] = 'too_small'
        return result

    def widths(lo, hi):
        lo_i = max(0, min(h-1, int(h * lo)))
        hi_i = min(h, max(lo_i + 1, int(h * hi)))
        values = [int(a[r].sum()) for r in range(lo_i, hi_i) if a[r].any()]
        return float(median(values)) if values else 0.0

    # Head segment tends to be narrower than the shoulder/torso band in
    # a front/side view of an upright person. This is a *heuristic*.
    head_w = widths(0.07, 0.20)
    torso_w = widths(0.30, 0.52)
    head_torso = head_w / torso_w if torso_w else float('inf')
    result['head_torso_ratio'] = round(head_torso, 3) if head_torso != float('inf') else None

    # Two separated legs, when visible, add corroboration but are not required.
    lo = int(h * 0.70)
    hi = max(lo+1, int(h * 0.95))
    split_rows = sum(1 for row in a[lo:hi] if _runs(row) >= 2)
    leg_split = split_rows >= max(2, (hi-lo)//4)
    result['leg_split'] = leg_split

    if aspect <= 0.95:
        result.update(classification='non_human_like', reason='horizontal_silhouette', shape_score=0)
        return result

    score = int(aspect >= 1.55) + int(aspect >= 1.95)
    score += int(head_torso <= 0.85)
    score += int(leg_split)
    result['shape_score'] = score

    # Support the project's stylised standing human meshes: limbs/head can
    # be fragmented by thresholding, leaving a tall sparse silhouette.
    tall_sparse = aspect >= 2.20 and 0.20 <= fill <= 0.75
    structured = aspect >= 1.55 and head_torso <= 0.85 and score >= 2
    if tall_sparse or structured:
        result.update(classification='human_like', reason=('tall_sparse_upright_profile'
            if tall_sparse else 'upright_head_torso_profile'))
    elif aspect < 1.2 and head_torso > 0.95:
        result.update(classification='non_human_like', reason='wide_non_upright_profile')
    else:
        result['reason'] = 'ambiguous_orientation_or_shape'
    return result


# RS1_SIMPLE_TEMP_SHAPE_CLASSIFIER_V1
# Transparent, simulation-specific classifier: BOTH temperature and a tall
# bounding rectangle are required for a possible-human label.
def classify_simple_thermal_candidate(max_temp_c, aspect_ratio, hot_pixels, *,
                                     expected_temp_c=37.82,
                                     temp_tolerance_c=0.15,
                                     min_aspect_ratio=1.60,
                                     min_hot_pixels=100):
    """Return classification and both constituent tests; no ML or ROS needed.

    aspect_ratio is height / width of ONE segmented hot component, not the
    image aspect ratio. max_temp_c is the maximum within that same component.
    A false-negative label here is not proof of an animal or an absent person.
    """
    import math
    try:
        temp = float(max_temp_c)
        ratio = float(aspect_ratio)
        pixels = int(hot_pixels)
        expected = float(expected_temp_c)
        tolerance = float(temp_tolerance_c)
        aspect_min = float(min_aspect_ratio)
        min_pixels = int(min_hot_pixels)
    except (TypeError, ValueError, OverflowError):
        return {'classification': 'uncertain', 'temp_match': False,
                'shape_match': False, 'reason': 'invalid_measurement'}

    if (not all(math.isfinite(v) for v in (temp, ratio, expected, tolerance, aspect_min))
            or tolerance < 0 or aspect_min <= 0 or pixels < 0 or min_pixels <= 0):
        return {'classification': 'uncertain', 'temp_match': False,
                'shape_match': False, 'reason': 'invalid_measurement'}

    # Inclusive tolerance allows the exact boundary after thermal rounding.
    temp_match = abs(temp - expected) <= tolerance + 1e-9
    shape_match = pixels >= min_pixels and ratio >= aspect_min
    if temp_match and shape_match:
        label = 'possible_human'
        reason = 'temperature_and_upright_rectangle_match'
    elif temp_match or shape_match:
        label = 'uncertain'
        reason = ('temperature_only' if temp_match else 'shape_only')
    else:
        label = 'other_thermal_target'
        reason = 'neither_temperature_nor_shape_match'
    return {'classification': label,
            'temp_match': bool(temp_match), 'shape_match': bool(shape_match),
            'reason': reason}
