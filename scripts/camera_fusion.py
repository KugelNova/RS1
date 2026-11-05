"""Associate Gazebo RGB detections with thermal components using RGB-D geometry."""

import math

import numpy as np


def decode_depth_image(image):
    """Read an axial depth image in metres, respecting ROS strides and byte order."""
    encoding = image.encoding.lower()
    endian = '>' if image.is_bigendian else '<'
    if encoding == '32fc1':
        dtype = np.dtype(endian + 'f4')
    elif encoding in ('16uc1', 'mono16'):
        dtype = np.dtype(endian + 'u2')
    else:
        raise ValueError(f'Unsupported depth encoding {image.encoding!r}')
    if image.width <= 0 or image.height <= 0 or image.step < image.width * dtype.itemsize:
        raise ValueError('Invalid depth image dimensions or row stride')
    pixels = np.ndarray((image.height, image.width), dtype=dtype, buffer=image.data,
                        strides=(image.step, dtype.itemsize))
    if encoding == '32fc1':
        return pixels
    return pixels.astype(np.float32) * np.float32(0.001)


def _pose_matrix(transform):
    translation, quaternion = transform
    translation = np.asarray(translation, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if (translation.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(translation).all() or not np.isfinite(norm) or norm <= 0):
        raise ValueError('Expected a finite translation and nonzero XYZW rotation')
    x, y, z, w = quaternion / norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return rotation, translation


def transform_point(point, pose):
    """Transform one XYZ point between TF frames."""
    rotation, translation = _pose_matrix(pose)
    return tuple(map(float, rotation @ point + translation))


def _pinhole(shape, hfov):
    height, width = shape
    if not math.isfinite(hfov) or not 0 < hfov < math.pi:
        raise ValueError('Horizontal camera FOV must be between 0 and pi radians')
    return width / (2.0 * math.tan(hfov / 2.0)), (width - 1.0) / 2.0, (height - 1.0) / 2.0


def _project_box(depth, box, rgb_pinhole, thermal_pinhole, thermal_shape, transform):
    """Return one nearest RGB-D point per projected thermal pixel."""
    left, top, right, bottom = box
    crop = depth[top:bottom, left:right]
    rows, cols = np.nonzero(np.isfinite(crop) & (crop > 0))
    x = crop[rows, cols]
    rgb_f, rgb_cx, rgb_cy = rgb_pinhole
    # The Gazebo depth value is forward-axis distance. Camera-link Y is left
    # and Z is up, so increasing image coordinates negate those components.
    points = np.column_stack((x, (rgb_cx - (cols + left)) * x / rgb_f,
                              (rgb_cy - (rows + top)) * x / rgb_f))
    rotation, translation = transform
    thermal_points = points @ rotation.T + translation
    # Project the footprint of each RGB pixel, not just its centre. The
    # narrow thermal camera can resolve several pixels per RGB-D sample.
    half_pixel_corners = np.array([[0, sy, sz] for sy in (-0.5, 0.5)
                                   for sz in (-0.5, 0.5)]) / rgb_f
    corner_offsets = half_pixel_corners @ rotation.T
    corners = thermal_points[:, None, :] + x[:, None, None] * corner_offsets
    forward = (corners[:, :, 0] > 0).all(axis=1)
    points = points[forward]
    tx, ty, tz = corners[forward].transpose(2, 0, 1)
    thermal_f, thermal_cx, thermal_cy = thermal_pinhole
    u = thermal_cx - thermal_f * ty / tx
    v = thermal_cy - thermal_f * tz / tx
    height, width = thermal_shape
    min_u = np.ceil(np.clip(u.min(axis=1), 0, width)).astype(np.intp)
    max_u = np.floor(np.clip(u.max(axis=1), -1, width - 1)).astype(np.intp)
    min_v = np.ceil(np.clip(v.min(axis=1), 0, height)).astype(np.intp)
    max_v = np.floor(np.clip(v.max(axis=1), -1, height - 1)).astype(np.intp)
    span_u = max_u - min_u + 1
    span_v = max_v - min_v + 1
    # Discard footprints that cross the near plane or become unreasonably
    # large from foreshortening instead of projecting across the entire image.
    usable = (span_u > 0) & (span_v > 0) & (span_u <= 64) & (span_v <= 64)
    sizes = np.where(usable, span_u * span_v, 0)
    source = np.repeat(np.arange(len(sizes)), sizes)
    offset = np.arange(len(source)) - np.repeat(sizes.cumsum() - sizes, sizes)
    dest_u = min_u[source] + offset % span_u[source]
    dest_v = min_v[source] + offset // span_u[source]
    pixel = dest_v * width + dest_u

    # Use the nearest sample when multiple RGB footprints overlap the same
    # thermal pixel, and count each thermal pixel only once.
    nearest_first = np.argsort(thermal_points[forward, 0][source])
    _, unique = np.unique(pixel[nearest_first], return_index=True)
    chosen = nearest_first[unique]
    return pixel[chosen], points[source[chosen]]


def match_rgbd_thermal(depth_m, detections, thermal_labels, *, rgb_to_thermal,
                       rgb_to_fixed, rgb_hfov=2.0944, thermal_hfov=0.610865,
                       min_confidence=0.35, min_coverage=0.55, min_thermal_pixels=8):
    """Match classified RGB boxes to thermal components and locate their visible surface.

    Boxes have exclusive right/bottom bounds. Depth is registered to the RGB
    image. Transform inputs are (translation_xyz, quaternion_xyzw), each taking
    points from the RGB camera link into its destination frame. For moving
    cameras, callers should evaluate each link at its image acquisition time.

    A match needs valid depth projected onto most of the component. A box that
    materially overlaps multiple components or a component supported by
    competing classes is left unclassified. Results are keyed by positive
    component ID. Their xyz is the median visible surface in the fixed frame.
    """
    depth = np.asarray(depth_m)
    labels = np.asarray(thermal_labels)
    if depth.ndim != 2 or labels.ndim != 2 or not depth.size or not labels.size:
        raise ValueError('Expected nonempty 2D depth and thermal label images')
    rgb_pinhole = _pinhole(depth.shape, rgb_hfov)
    thermal_pinhole = _pinhole(labels.shape, thermal_hfov)
    thermal_pose = _pose_matrix(rgb_to_thermal)
    fixed_rotation, fixed_translation = _pose_matrix(rgb_to_fixed)
    if not detections:
        return {}

    labels_flat = labels.ravel()
    area = np.bincount(labels_flat)
    height, width = depth.shape
    options = {}
    possible_classes = {}
    for detection in detections:
        try:
            classification = detection['classification']
            confidence = float(detection['confidence'])
            left, top, right, bottom = map(float, detection['bbox_xyxy'])
        except (KeyError, ValueError, TypeError):
            continue
        if (classification not in ('human', 'animal')
                or not min_confidence <= confidence <= 1.0
                or not all(map(math.isfinite, (left, top, right, bottom)))
                or left >= right or top >= bottom):
            continue
        box = (max(0, math.floor(left)), max(0, math.floor(top)),
               min(width, math.ceil(right)), min(height, math.ceil(bottom)))
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        thermal_pixels, points = _project_box(
            depth, box, rgb_pinhole, thermal_pinhole, labels.shape, thermal_pose)
        projected_labels = labels_flat[thermal_pixels]
        supported, counts = np.unique(projected_labels[projected_labels > 0], return_counts=True)
        coverage = counts / area[supported]
        plausible = (counts >= max(3, min_thermal_pixels // 2)) & (coverage >= 0.25)
        for component in supported[plausible]:
            possible_classes.setdefault(component, set()).add(classification)
        if np.count_nonzero(plausible) != 1:
            continue
        index = np.flatnonzero(plausible)[0]
        if counts[index] < min_thermal_pixels or coverage[index] < min_coverage:
            continue
        component = supported[index]
        inliers = points[projected_labels == component]
        # Reject projections with a wide depth discontinuity, such as a
        # foreground branch mixed with a warm object several metres behind it.
        near, far = np.quantile(inliers[:, 0], [0.1, 0.9])
        if far - near > 1.8:
            continue
        # Choose the strongest observation if the detector produced duplicate
        # boxes for the same class. Disagreeing classes are removed below.
        strength = (coverage[index], confidence)
        if component in options and strength <= options[component][0]:
            continue
        fixed_points = inliers @ fixed_rotation.T + fixed_translation
        options[component] = (strength, {
            'xyz': tuple(map(float, np.median(fixed_points, axis=0))),
            'classification': classification,
            'label': str(detection.get('label', classification)),
            'confidence': confidence,
            'rgb_pixels': int(counts[index]),
            'coverage': float(coverage[index]),
        })

    return {int(component): result for component, (_, result) in options.items()
            if len(possible_classes[component]) == 1}
