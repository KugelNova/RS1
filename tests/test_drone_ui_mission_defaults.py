"""Regression tests for RS1 PyQt mission settings without requiring Qt or ROS.

Execute with: python3 -m pytest -q tests/test_drone_ui_mission_defaults.py
Tests the source of the real UI, extracting standalone methods with ast.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / 'scripts/drone_ui.py').read_text()
MODULE = ast.parse(SOURCE)
UI_CLASS = next(n for n in MODULE.body if isinstance(n, ast.ClassDef) and n.name == 'DroneUI')


class FakeMessageBox:
    Yes = 1
    No = 2
    warnings = []
    answer = No

    @classmethod
    def warning(cls, *args):
        cls.warnings.append(args)

    @classmethod
    def question(cls, *args):
        return cls.answer


def method(name):
    """Compile just a DroneUI method, avoiding heavyweight ROS/Qt imports."""
    node = next(n for n in UI_CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name)
    clone = copy.deepcopy(node)
    clone.decorator_list = []
    program = ast.fix_missing_locations(ast.Module(body=[clone], type_ignores=[]))
    ns = {'QMessageBox': FakeMessageBox}
    exec(compile(program, '<RS1 drone_ui method>', 'exec'), ns)
    return ns[name]


class TestMissionUiDefaults(unittest.TestCase):
    def setUp(self):
        FakeMessageBox.warnings = []
        FakeMessageBox.answer = FakeMessageBox.No
        self.calls = []
        self.values = {'min_x': -22., 'max_x': 22., 'min_y': -22.,
                       'max_y': 22., 'lane_spacing': 3., 'altitude': 12.}
        self.stats = {'optimistic_sim_minutes': 12.0}
        self.ui = SimpleNamespace(
            node=SimpleNamespace(request=lambda *args, **kwargs: self.calls.append((args, kwargs))),
            route_clearance_box=SimpleNamespace(isChecked=lambda: True),
            _terrain_request=lambda: (self.values, self.stats),
        )

    def test_no_manual_trial_or_legacy_route_picker(self):
        self.assertNotIn('self.trial_selector', SOURCE)
        self.assertNotIn('self.search_mode_selector', SOURCE)
        self.assertNotIn('Legacy far-pair', SOURCE)
        self.assertNotIn('def _route_mode_changed(', SOURCE)

    def test_bounds_default_to_plus_minus_22(self):
        method_body = next(n for n in UI_CLASS.body if isinstance(n, ast.FunctionDef)
                           and n.name == '_mission_dashboard_panel')
        loop = next(n for n in ast.walk(method_body)
                    if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Tuple)
                    and [part.id for part in n.target.elts if isinstance(part, ast.Name)] == ['key', 'label', 'default'])
        defaults = {k: default for k, _label, default in ast.literal_eval(loop.iter)}
        self.assertEqual(defaults, {'min_x': '-22', 'max_x': '22',
                                    'min_y': '-22', 'max_y': '22',
                                    'lane_spacing': '3', 'altitude': '12'})

    def test_sensor_start_uses_automatic_terrain_trial_name(self):
        method('_start_sensors')(self.ui)
        self.assertEqual(self.calls, [(('start_sensors',), {'trial': 'terrain'})])

    def test_search_always_requests_terrain(self):
        method('_start_search')(self.ui)
        self.assertEqual(self.calls, [(('start_search',),
                        {'trial': 'terrain', 'search_mode': 'terrain',
                         'clearance_verified': True, **self.values})])

    def test_search_requires_clearance(self):
        self.ui.route_clearance_box.isChecked = lambda: False
        method('_start_search')(self.ui)
        self.assertFalse(self.calls)
        self.assertEqual(len(FakeMessageBox.warnings), 1)

    def test_invalid_route_blocks_search(self):
        def fail():
            raise ValueError('Invalid bounds')
        self.ui._terrain_request = fail
        method('_start_search')(self.ui)
        self.assertFalse(self.calls)
        self.assertEqual(len(FakeMessageBox.warnings), 1)

    def test_long_route_requires_standard_clearance_only(self):
        # The old duration-based confirmation was deliberately removed from
        # the UI; the normal route-clearance checkbox is still mandatory.
        self.stats['optimistic_sim_minutes'] = 25
        self.ui.route_clearance_box.isChecked = lambda: False
        method('_start_search')(self.ui)
        self.assertFalse(self.calls)
        self.assertEqual(len(FakeMessageBox.warnings), 1)
        self.ui.route_clearance_box.isChecked = lambda: True
        method('_start_search')(self.ui)
        self.assertEqual(len(self.calls), 1)


if __name__ == '__main__':
    unittest.main()
