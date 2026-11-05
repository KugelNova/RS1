import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pyqt_rescue_records import list_missions, mission_path, read_mission


class RescueRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.one = 'targets_20261031_034641_800291.csv'
        with (self.root / self.one).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'sim_time_sec', 'frame_id', 'x_m', 'y_m', 'z_m',
                             'max_temp_c', 'foot_pixel_u', 'foot_pixel_v', 'hot_pixels',
                             'edge_clipped'])
            writer.writerow([1, '14.500', 'parrot1_odom', '4.029', '-2.852', '0.0',
                             '32.68', '303.5', '170', '1392', 'True'])
            writer.writerow([2, 'garbled'])

    def tearDown(self):
        self.temp.cleanup()

    def test_read_valid_rows(self):
        records = read_mission(self.one, self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['id'], 1)
        self.assertEqual(records[0]['frame_id'], 'parrot1_odom')
        self.assertAlmostEqual(records[0]['x_m'], 4.029)
        self.assertTrue(records[0]['edge_clipped'])
        self.assertNotIn('waypoint_frame_id', records[0])

    def test_read_waypoint_and_keep_body_estimate_when_waypoint_invalid(self):
        path = self.root / self.one
        fields = ['id', 'sim_time_sec', 'frame_id', 'x_m', 'y_m', 'z_m', 'max_temp_c',
                  'foot_pixel_u', 'foot_pixel_v', 'hot_pixels', 'edge_clipped',
                  'waypoint_x_m', 'waypoint_y_m', 'waypoint_z_m', 'waypoint_frame_id']
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerow([1, 14.5, 'parrot1_odom', 4.029, -2.852, 2.1, 32.68, 303.5,
                             170, 1392, False, 4.02, -2.85, -0.7, 'terrain_world'])
            writer.writerow([2, 15.2, 'parrot1_odom', 6.0, 1.0, 1.3, 36, 305, 171, 1580,
                             False, 6.0, 1.0, 'nan', 'terrain_world'])

        first, second = read_mission(self.one, self.root)
        self.assertEqual(first['z_m'], 2.1)
        self.assertEqual(first['waypoint_z_m'], -0.7)
        self.assertEqual(first['waypoint_frame_id'], 'terrain_world')
        self.assertEqual(second['z_m'], 1.3)
        self.assertNotIn('waypoint_frame_id', second)

    def test_list_sorted_and_filtered(self):
        self.assertEqual(list_missions(self.root), [self.one])
        (self.root / 'other.csv').write_text('anything')
        self.assertEqual(list_missions(self.root), [self.one])

    def test_reject_traversal(self):
        with self.assertRaises(ValueError):
            mission_path('../outside.csv', self.root)

    def test_missing_directory(self):
        self.assertEqual(list_missions(self.root / 'missing'), [])

    def test_unknown_mission(self):
        with self.assertRaises(FileNotFoundError):
            mission_path('targets_20261031_034641_111111.csv', self.root)


if __name__ == '__main__':
    unittest.main()
