"""Pure thermal helper tests; no ROS 2 installation required."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from thermal_processing import (
    connected_hot_components, decode_mono16, hot_pixel_mask,
    raw_to_celsius, summarize_hot_pixels,
)


def fake_image(values, endian=False, pad_bytes=0):
    arr = np.asarray(values, dtype=np.uint16)
    h, w = arr.shape
    dtype = '>u2' if endian else '<u2'
    payload = b''.join(arr.astype(dtype).tobytes()[row*w*2:(row+1)*w*2]
                       + b'\x00'*pad_bytes for row in range(h))
    return SimpleNamespace(encoding='mono16', is_bigendian=endian,
                           width=w, height=h, step=w*2+pad_bytes, data=payload)


class ThermalProcessingTests(unittest.TestCase):
    def test_decode_plain(self):
        vals = [[0, 30500], [31500, 28000]]
        self.assertEqual(decode_mono16(fake_image(vals)).tolist(), vals)

    def test_decode_big_endian(self):
        vals = [[30500, 31000], [0, 32000]]
        self.assertEqual(decode_mono16(fake_image(vals, endian=True)).tolist(), vals)

    def test_decode_padded_rows(self):
        vals = [[30500, 0, 1], [2, 31000, 3]]
        self.assertEqual(decode_mono16(fake_image(vals, pad_bytes=4)).tolist(), vals)

    def test_invalid_image_rejected(self):
        msg = fake_image([[30500, 0]])
        msg.data = msg.data[:2]
        with self.assertRaises(ValueError):
            decode_mono16(msg)

    def test_threshold_and_temperature(self):
        arr = decode_mono16(fake_image([[0, 30499, 30500]]))
        self.assertEqual(hot_pixel_mask(arr, 30500).tolist(), [[False, False, True]])
        self.assertAlmostEqual(raw_to_celsius(30500), 31.85)

    def test_detector_summary_both_regions_combined(self):
        arr = decode_mono16(fake_image([[30500, 0, 30600], [0, 0, 0]]))
        self.assertIsNone(summarize_hot_pixels(arr, 30500, 3))
        summary = summarize_hot_pixels(arr, 30500, 2)
        self.assertEqual(summary['pixels'], 2)
        self.assertEqual((summary['centre_x'], summary['centre_y']), (1.0, 0.0))
        self.assertAlmostEqual(summary['max_temp_c'], 32.85)

    def test_connected_components_keep_separate_objects(self):
        arr = np.zeros((20, 22), dtype=np.uint16)
        arr[2:7, 3:8] = 30500
        arr[12:17, 14:19] = 31500
        count, labels, stats = connected_hot_components(arr, 30500)
        self.assertEqual(count, 3)  # including background
        self.assertEqual(sorted(stats[1:, 4].tolist()), [25, 25])
        self.assertNotEqual(labels[4, 5], labels[14, 16])


if __name__ == '__main__':
    unittest.main()
