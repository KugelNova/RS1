"""Conservatively refine an RGB-D human position with corroborating LiDAR ranges."""

import math

import numpy as np

from camera_fusion import _pinhole, _pose_matrix


def decode_xyz_points(cloud):
    """Read finite XYZ points from a PointCloud2, preserving layout and byte order."""
    fields = {field.name: field for field in cloud.fields}
    float_types = {7: 'f4', 8: 'f8'}  # sensor_msgs/PointField FLOAT32 and FLOAT64
    endian = '>' if cloud.is_bigendian else '<'
    formats, offsets = [], []
    for name in ('x', 'y', 'z'):
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in float_types:
            raise ValueError('LiDAR point cloud needs scalar floating-point XYZ fields')
        dtype = np.dtype(endian + float_types[field.datatype])
        if not 0 <= field.offset <= cloud.point_step - dtype.itemsize:
            raise ValueError('LiDAR XYZ field exceeds the point stride')
        formats.append(dtype)
        offsets.append(field.offset)
    if (cloud.width < 0 or cloud.height < 0 or cloud.row_step < cloud.width * cloud.point_step
            or len(cloud.data) < cloud.row_step * cloud.height):
        raise ValueError('LiDAR point cloud has invalid dimensions or row strides')
    if not cloud.width or not cloud.height:
        return np.empty((0, 3))

    dtype = np.dtype({'names': ('x', 'y', 'z'), 'formats': formats,
                      'offsets': offsets, 'itemsize': cloud.point_step})
    points = np.ndarray((cloud.height, cloud.width), dtype=dtype, buffer=cloud.data,
                        strides=(cloud.row_step, cloud.point_step))
    xyz = np.column_stack([points[name].ravel() for name in ('x', 'y', 'z')])
    return xyz[np.isfinite(xyz).all(axis=1)]


def _image_pixels(points, shape, hfov):
    """Project forward-facing camera-link points and flag in-frame pixels."""
    focal, cx, cy = _pinhole(shape, hfov)
    px = np.rint(cx - focal * points[:, 1] / points[:, 0])
    py = np.rint(cy - focal * points[:, 2] / points[:, 0])
    height, width = shape
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return valid, py[valid].astype(np.intp), px[valid].astype(np.intp)


def refine_human_position(camera_xyz, points, depth_m, thermal_labels, component_id, *,
                          lidar_to_rgb, rgb_to_thermal, rgb_to_fixed,
                          lidar_stamp_s, rgb_stamp_s, rgb_hfov=2.0944,
                          thermal_hfov=0.610865, max_age_s=0.13):
    """Return a range correction for an existing matched human, or None.

    Transforms are (translation XYZ, quaternion XYZW), evaluated at the
    corresponding sensor timestamps. Each accepted LiDAR hit must land in
    the same thermal component and agree with the RGB-D surface at its pixel.
    The median range correction keeps the direction of the RGB-D estimate.
    """
    if (not math.isfinite(lidar_stamp_s) or not math.isfinite(rgb_stamp_s)
            or not math.isfinite(max_age_s) or min(lidar_stamp_s, rgb_stamp_s) <= 0
            or abs(lidar_stamp_s - rgb_stamp_s) > max_age_s):
        return None

    min_hits = 3
    points = np.asarray(points)
    position = np.asarray(camera_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected an N × 3 LiDAR point array')
    if len(points) < min_hits or position.shape != (3,) or not np.isfinite(position).all():
        return None
    if component_id <= 0:
        return None
    if depth_m.ndim != 2 or thermal_labels.ndim != 2:
        raise ValueError('Expected two-dimensional depth and thermal label images')

    lidar_rotation, lidar_translation = _pose_matrix(lidar_to_rgb)
    thermal_rotation, thermal_translation = _pose_matrix(rgb_to_thermal)
    fixed_rotation, fixed_translation = _pose_matrix(rgb_to_fixed)
    centre = (position - fixed_translation) @ fixed_rotation
    if centre[0] <= 0.2:
        return None

    # Nearby returns only, so a beam from the background at the same bearing
    # cannot influence the correction.
    lidar = points[np.isfinite(points).all(axis=1)]
    rgb_points = lidar @ lidar_rotation.T + lidar_translation
    near = (rgb_points[:, 0] > 0.2) & (np.linalg.norm(rgb_points - centre, axis=1) <= 0.8)
    rgb_points = rgb_points[near]
    thermal_points = rgb_points @ thermal_rotation.T + thermal_translation
    in_front = thermal_points[:, 0] > 0.2
    rgb_points = rgb_points[in_front]
    thermal_points = thermal_points[in_front]
    if len(rgb_points) < min_hits:
        return None

    on_image, rows, cols = _image_pixels(thermal_points, thermal_labels.shape, thermal_hfov)
    same_target = thermal_labels[rows, cols] == component_id
    rgb_points = rgb_points[on_image][same_target]

    on_image, rows, cols = _image_pixels(rgb_points, depth_m.shape, rgb_hfov)
    observed_depth = depth_m[rows, cols]
    residual = rgb_points[on_image, 0] - observed_depth
    agree = np.isfinite(observed_depth) & (observed_depth > 0.2) & (np.abs(residual) <= 0.30)
    residual = residual[agree]
    # Multiple returns at the same RGB pixel are one surface observation.
    pixel = rows[agree] * depth_m.shape[1] + cols[agree]
    nearest_first = np.argsort(np.abs(residual))
    _, unique = np.unique(pixel[nearest_first], return_index=True)
    residual = residual[nearest_first[unique]]
    if residual.size < min_hits:
        return None
    low, high = np.quantile(residual, (0.1, 0.9))
    if high - low > 0.18:
        return None

    # Keep the RGB-D direction and use LiDAR as an independent range check,
    # blending half the observed axial bias rather than replacing the estimate.
    correction = float(np.median(residual)) * 0.5
    corrected = position + (position - fixed_translation) * (correction / centre[0])
    return {'xyz': tuple(map(float, corrected)), 'lidar_points': int(residual.size),
            'depth_correction_m': correction}
