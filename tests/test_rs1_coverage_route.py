"""No ROS/Gazebo dependency: python3 -m unittest tests/test_rs1_coverage_route.py"""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rs1_coverage_route import plan_lawnmower, route_length, route_stats


class TestCoverageRoute(unittest.TestCase):
    def test_rectangle_corners_and_spacing(self):
        route = plan_lawnmower(-30, 30, -25, 25, 3)
        self.assertEqual(len(route) % 2, 0)
        self.assertTrue({(-30, -25), (30, -25), (-30, 25), (30, 25)} <= set(route))
        rows = sorted({y for _, y in route})
        self.assertTrue(all(b-a <= 3+1e-9 for a, b in zip(rows, rows[1:])))

    def test_sweep_longer_axis_and_nearest_start(self):
        route = plan_lawnmower(-10, 10, -50, 50, 5, start=(-10, 48))
        self.assertEqual({x for x, _ in route}, set(range(-10, 11, 5)))
        self.assertEqual(route[0], (-10, 50))
        self.assertGreater(route_stats(route)['distance_m'], 0)

    def test_finite_and_validated(self):
        cases = [(0, 0, 0, 10, 2), (0, 10, 5, -5, 2),
                 (0, 10, 0, 10, float('nan')),
                 (0, 10, 0, 10, 0.1), (0, 400, 0, 10, 5)]
        for bounds in cases:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                plan_lawnmower(*bounds)

    def test_estimated_distance_includes_transit(self):
        route = plan_lawnmower(-5, 5, -5, 5, 2)
        self.assertGreater(route_length(route, (-50, 50)), route_length(route))

    def test_full_area_is_slow(self):
        route = plan_lawnmower(-70, 70, -70, 70, 2)
        stats = route_stats(route, (-10, 7))
        self.assertGreater(stats['distance_m'], 9000)
        self.assertGreater(stats['optimistic_sim_minutes'], 120)


if __name__ == '__main__':
    unittest.main()
