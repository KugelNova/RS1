import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rs1_mission_logic import (finite_xy, near_launch, goto_step, descent_speed,
                               LAND_Z_M)


class MissionLogicTests(unittest.TestCase):
    def test_reject_invalid_destination(self):
        for x, y in [('nan', 0), (46, 0), (1, 'inf'), ('wat', 0)]:
            with self.subTest(x=x,y=y), self.assertRaises(ValueError):
                finite_xy(x,y)

    def test_accept_far_test_person(self):
        self.assertEqual(finite_xy(36, -2), (36., -2.))

    def test_land_only_near_launch(self):
        self.assertTrue(near_launch(2, 0))
        self.assertTrue(near_launch(4, 0))
        self.assertFalse(near_launch(25, -8))

    def test_goto_climb_before_translate(self):
        vx,wz,vz,done,err = goto_step(2, 0, 1, 0, 25, -8, 10)
        self.assertEqual(vx, 0)
        self.assertEqual(wz, 0)
        self.assertGreater(vz, 0)
        self.assertFalse(done)
        self.assertFalse(err)

    def test_goto_blocked_by_forward_obstacle(self):
        vx,wz,vz,done,err = goto_step(2, 0, 12, 0, 25, 0, 1.5)
        self.assertEqual(vx, 0)
        self.assertTrue(err)

    def test_goto_reaches_coordinate(self):
        vx,wz,vz,done,err = goto_step(25.1, -8.2, 12.1, 0, 25, -8, 20)
        self.assertTrue(done)

    def test_descent_rate_bound(self):
        v,done=descent_speed(12)
        self.assertGreaterEqual(v,-0.2)
        self.assertLess(v,0)
        self.assertFalse(done)

    def test_descent_stops_at_estimated_touchdown(self):
        v,done=descent_speed(LAND_Z_M)
        self.assertTrue(done)
        self.assertEqual(v,0)


if __name__=='__main__':
    unittest.main()
