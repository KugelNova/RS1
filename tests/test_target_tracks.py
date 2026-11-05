from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from target_tracks import StableTargetTracker


def observation(classification, x=0.0):
    return dict(classification=classification, xyz=(x, 0.0, 1.0), pixels=110,
                bbox_height=28, edge_clipped=False)


def make_tracker():
    return StableTargetTracker(required_classification='human', confirm_frames=3,
                               confirm_span_s=0.15)


def test_only_repeated_human_evidence_can_confirm():
    tracker = make_tracker()
    for time in (1.0, 1.1, 1.2):
        assert tracker.observe([observation('animal'), observation('unknown', 3.0)], time, []) == []
    for time in (2.0, 2.1):
        assert tracker.observe([observation('human')], time, []) == []
    confirmed = tracker.observe([observation('human')], 2.2, [])
    assert [target['classification'] for target in confirmed] == ['human']
    assert tracker.observe([observation('human')], 2.3, [(0.0, 0.0, 1.0)]) == []


def test_animal_evidence_clears_only_nearby_pending_human():
    tracker = make_tracker()
    humans = [observation('human'), observation('human', 4.0)]
    assert tracker.observe(humans, 1.0, []) == []
    assert tracker.observe(humans, 1.1, []) == []
    confirmed = tracker.observe([observation('animal'), *humans], 1.2, [])
    assert [target['xyz'][0] for target in confirmed] == [4.0]
    assert tracker.observe(humans, 1.3, [(4.0, 0.0, 1.0)]) == []
    assert tracker.observe(humans, 1.4, [(4.0, 0.0, 1.0)]) == []
    assert len(tracker.observe(humans, 1.5, [(4.0, 0.0, 1.0)])) == 1


def test_reports_distinct_lidar_scans_used_for_a_confirmed_track():
    tracker = make_tracker()
    human = observation('human')
    scan_one = dict(human, lidar_stamp_ns=1_000_000_000)
    scan_two = dict(human, lidar_stamp_ns=1_200_000_000)
    assert tracker.observe([scan_one], 1.0, []) == []
    assert tracker.observe([scan_one], 1.1, []) == []
    confirmed = tracker.observe([scan_two], 1.2, [])
    assert [target['lidar_frames'] for target in confirmed] == [2]
