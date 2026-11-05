"""Shared pure thermal-image operations for RS1.

This module has no ROS node, topics, publishers, or state. Keeping the
thermal detector and localizer as independent nodes avoids changing the
existing flight/UI integration. ROS Image-like messages are accepted, but
rclpy itself is not required to import or unit-test these helpers.
"""

import numpy as np


RAW_KELVIN_SCALE = 0.01  # Gazebo simulated thermal image: 0.01 K/count
CELSIUS_OFFSET = 273.15


def decode_mono16(msg):
    """Return a zero-copy 2-D thermal array, handling endian and row stride.

    The caller decides whether to accept or reject a message encoding. This
    accepts `mono16` only; other 16-bit ROS formats may use different units.
    """
    if str(msg.encoding).lower() != 'mono16':
        raise ValueError(f'Expected mono16 thermal data, got {msg.encoding!r}')
    width, height, step = int(msg.width), int(msg.height), int(msg.step)
    if width <= 0 or height <= 0 or step < 2 * width or step % 2:
        raise ValueError('Invalid thermal image dimensions or row stride')
    if len(msg.data) < step * height:
        raise ValueError('Thermal buffer shorter than image row strides')
    dtype = np.dtype('>u2') if msg.is_bigendian else np.dtype('<u2')
    return np.ndarray(
        shape=(height, width), dtype=dtype,
        buffer=msg.data, strides=(step, 2))


def hot_pixel_mask(image, threshold_raw):
    """Boolean threshold mask; zeros are excluded for positive thresholds."""
    return image >= threshold_raw


def raw_to_celsius(raw):
    """Convert Gazebo thermal raw counts to Celsius."""
    return float(raw) * RAW_KELVIN_SCALE - CELSIUS_OFFSET


def summarize_hot_pixels(image, threshold_raw, min_pixels):
    """Summarize ALL hot pixels, retaining legacy detector behaviour.

    Multiple separate hot objects are deliberately *not* split here: the
    existing `/parrot1/thermal/target_pixel` topic reports their mean pixel.
    The target localizer uses per-component segmentation instead.
    Returns None when there are fewer than min_pixels hot pixels.
    """
    mask = hot_pixel_mask(image, threshold_raw)
    ys, xs = np.where(mask)
    count = len(xs)
    if count < min_pixels:
        return None
    return {
        'pixels': count,
        'centre_x': float(xs.mean()),
        'centre_y': float(ys.mean()),
        'x_min': int(xs.min()),
        'x_max': int(xs.max()),
        'y_min': int(ys.min()),
        'y_max': int(ys.max()),
        'max_temp_c': raw_to_celsius(int(image[mask].max())),
    }


def connected_hot_components(image, threshold_raw):
    """Connected components after the localizer's original 3x3 closing.

    Returns (component_count, integer_labels, stats). Uses cv2 only here so
    basic detector imports need not load OpenCV. Component 0 is background.
    """
    import cv2
    binary = hot_pixel_mask(image, threshold_raw).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    n, labels, stats, _centres = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    return n, labels, stats
