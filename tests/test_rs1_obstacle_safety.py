import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rs1_mission_logic import manual_axis_speed, OverflightAssist, goto_step


class ManualLimiterTests(unittest.TestCase):
    def test_clear_forward(self):
        self.assertEqual(manual_axis_speed(0.5, 5.0, True), 0.5)

    def test_slowdown(self):
        speed = manual_axis_speed(0.5, 1.0, True)
        self.assertGreater(speed, 0.0)
        self.assertLess(speed, 0.5)

    def test_stop_near_wall(self):
        self.assertEqual(manual_axis_speed(0.5, 0.65, True), 0.0)
        self.assertEqual(manual_axis_speed(-0.5, 0.48, True), 0.0)

    def test_invalid_command_is_rejected(self):
        self.assertEqual(manual_axis_speed(float('nan'), 5, True), 0.0)

    def test_reverse_uses_same_limit(self):
        speed = manual_axis_speed(-0.5, 0.95, True)
        self.assertLess(speed, 0.0)
        self.assertGreater(speed, -0.5)

    def test_missing_or_stale_lidar_stops_motion(self):
        for d, fresh in ((float('nan'), True), (0.0, True), (5.0, False)):
            with self.subTest(distance=d, fresh=fresh):
                self.assertEqual(manual_axis_speed(0.5, d, fresh), 0.0)


class OverflightTests(unittest.TestCase):
    def test_climbs_without_horizontal_motion_then_resumes(self):
        assist = OverflightAssist(maximum_z=25.0)
        self.assertTrue(assist.begin(12.0))
        v, done, err = assist.step(12.0, 1.0)
        self.assertGreater(v, 0.0)
        self.assertFalse(done)
        self.assertFalse(err)
        for i in range(3):
            v, done, err = assist.step(13.5, 4.0)
        self.assertFalse(done)
        self.assertEqual(assist.phase, 'EXTRA_LIFT')
        self.assertGreater(v, 0.0)
        v, done, err = assist.step(14.51, 4.0)
        self.assertEqual(v, 0.0)
        self.assertTrue(done)
        self.assertFalse(err)
        self.assertFalse(assist.active)

    def test_missing_sensor_causes_error(self):
        assist = OverflightAssist()
        self.assertTrue(assist.begin(12.0))
        v, done, err = assist.step(12.0, float('nan'))
        self.assertEqual(v, 0.0)
        self.assertFalse(done)
        self.assertTrue(err)

    def test_altitude_cap_blocks_climb(self):
        assist = OverflightAssist(maximum_z=15.0)
        self.assertFalse(assist.begin(14.0))
        self.assertTrue(assist.begin(13.0))
        v, done, err = assist.step(14.95, 1.0)
        self.assertEqual(v, 0.0)
        self.assertTrue(err)

    def test_limit_each_climb(self):
        assist = OverflightAssist(maximum_z=25.0, maximum_gain_m=8.0)
        self.assertTrue(assist.begin(10.0))
        v, done, err = assist.step(17.95, 1.0)
        self.assertTrue(err)

    def test_reblocked_at_limit_never_descends(self):
        assist = OverflightAssist(maximum_z=25.0, maximum_gain_m=8.0)
        self.assertTrue(assist.begin(12.0))
        for _ in range(3):
            assist.step(13.0, 10.0)
        self.assertEqual(assist.phase, 'EXTRA_LIFT')
        vz, done, err = assist.step(19.95, 1.0)
        self.assertEqual(vz, 0.0)
        self.assertTrue(err)

    def test_goto_exact_threshold_is_blocked(self):
        vx, wz, vz, done, err = goto_step(0, 0, 12, 0, 10, 0, 2.5)
        self.assertEqual(vx, 0.0)
        self.assertTrue(err)

    def test_goto_accepts_elevated_cruise_altitude(self):
        vx, wz, vz, done, err = goto_step(
            0, 0, 15, 0, 10, 0, 9, target_z=15, maximum_z=25)
        self.assertGreater(vx, 0)
        self.assertFalse(done)
        self.assertFalse(err)


if __name__ == '__main__':
    unittest.main()
