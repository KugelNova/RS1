import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from target_geometry import pixel_to_ground


class GroundProjectionTests(unittest.TestCase):
    def test_observed_human2_like_bbox(self):
        # Camera base (2,0,0.8), camera offset (.2,0,.2), target foot at ~(304,170)
        pt = pixel_to_ground(304.0, 170.0, 320, 240, 2.0944,
                             (2.2, 0.0, 1.0), (0, 0, 0, 1), ground_z=0)
        self.assertIsNotNone(pt)
        self.assertAlmostEqual(pt[0], 4.02, delta=0.15)
        self.assertAlmostEqual(pt[1], -2.87, delta=0.2)
        self.assertEqual(pt[2], 0.0)

    def test_upward_ray_is_skipped(self):
        self.assertIsNone(pixel_to_ground(160, 20, 320, 240, 2.0944,
                                          (0, 0, 1), (0, 0, 0, 1)))

    def test_yaw_rotation(self):
        q = (0, 0, math.sin(math.pi/4), math.cos(math.pi/4))
        pt = pixel_to_ground(160, 170, 320, 240, 2.0944, (0, 0, 1), q)
        self.assertIsNotNone(pt)
        self.assertGreater(pt[1], 0)
        self.assertAlmostEqual(pt[0], 0.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
