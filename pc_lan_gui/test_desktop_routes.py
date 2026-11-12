import unittest
import time

from desktop_routes import DesktopRoutes
from lan_protocol import BoardState


class FakeControl:
    def __init__(self):
        self.changed = []
        self.modes = []
        self.moves = []
        self.policies = []

    def ping(self):
        return BoardState(mode=0, target=0, gimbal=0, error=0, selected_id=0)

    def position(self):
        return dict(pan=52.5, tilt=40.0)

    def catalog(self, kind, refresh=False):
        name = "YOLO26m" if kind == "model" else "BoT-SORT"
        return [dict(id=1, name=name, available=True, active=True)]

    def tracker_policy(self):
        return dict(tracker_id=1, warmth_min=8000, warmth_max=64000,
                    edge_birth=0.5, confirmation_frames=3, confirmation_ms=130,
                    class_gate=True, gmc=True, post_reid=False)

    def change(self, kind, ident):
        self.changed.append((kind, ident))
        return 0

    def change_policy(self, **values):
        self.policies.append(values)
        return 0

    def mode(self, ident):
        self.modes.append(ident)
        return 0

    def move(self, x, y):
        self.moves.append((x, y))


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.control = FakeControl()
        self.routes = DesktopRoutes(self.control, lambda: dict(
            received=7, tracks=[dict(id=42, x=100, y=100, width=80, height=100)]))

    def test_catalog_and_tracker_policy_use_runtime_values(self):
        self.assertEqual(self.routes.handle("/model")["catalog"][0]["name"], "YOLO26m")
        rules = self.routes.handle("/rules")
        self.assertEqual(rules["presets"][0]["name"], "BoT-SORT")
        self.assertEqual(rules["confirm"]["ms"], 130)
        self.assertEqual(rules["edge_birth"], 0.5)

    def test_changes_wait_for_completed_result(self):
        self.assertEqual(self.routes.handle("/model/set?id=1", "POST")["current"], 1)
        self.assertEqual(self.routes.handle("/rules/set?tracker=1", "POST")["tracker"], "1")
        self.assertEqual(self.control.changed, [("model", 1), ("tracker", 1)])

    def test_filter_update_uses_completed_runtime_result(self):
        self.routes.handle("/rules/set?warm_min=9000&warm_max=60000&edge_birth=0.4&confirm_ms=150", "POST")
        self.assertEqual(self.control.policies, [dict(warmth_min=9000,
            warmth_max=60000, edge_birth=0.4, confirmation_ms=150)])

    def test_unsupported_setting_cannot_report_success(self):
        answer = self.routes.handle("/rules/set?predictor=gru", "POST")
        self.assertFalse(answer["ok"])
        self.assertEqual(self.control.changed, [])

    def test_filter_tiles_restore_applied_thresholds(self):
        self.routes.handle("/rules")
        self.routes.handle("/rules/set?warm_on=0", "POST")
        self.routes.handle("/rules/set?warm_on=1", "POST")
        self.routes.handle("/rules/set?edge_on=0", "POST")
        self.routes.handle("/rules/set?edge_on=1", "POST")
        self.assertEqual(self.control.policies, [
            dict(warmth_min=0, warmth_max=0),
            dict(warmth_min=8000, warmth_max=64000),
            dict(edge_birth=0.0), dict(edge_birth=0.5)])

    def test_filter_tiles_require_an_initial_value(self):
        self.assertFalse(self.routes.handle("/rules/set?edge_on=1", "POST")["ok"])
        self.assertFalse(self.routes.handle("/rules/set?warm_on=1", "POST")["ok"])

    def test_gmc_and_keepid_use_existing_runtime_changes(self):
        self.assertTrue(self.routes.handle("/rules/set?stab=1", "POST")["stab"])
        self.assertEqual(self.control.policies, [dict(gmc=True)])
        self.assertEqual(self.routes.handle("/rules/set?relabel=1", "POST")["tracker"], "1")
        self.assertEqual(self.control.changed, [("tracker", 3)])

    def test_unavailable_actions_have_explicit_page_state(self):
        self.assertFalse(self.routes.handle("/measure")["available"])
        self.assertFalse(self.routes.handle("/replay")["available"])

    def test_performance_graph_uses_real_receive_counts_only(self):
        self.routes.latest = lambda: dict(received=12, video_frames=20)
        self.routes.perf_last = (time.monotonic() - 1.0, (7, 10))
        answer = self.routes.handle("/perf?since=0")
        self.assertTrue(answer["rows"])
        row = answer["rows"][-1]
        self.assertGreater(row[1], 0)
        self.assertGreater(row[2], 0)
        self.assertEqual(row[3:], [None, None, None])
        self.assertFalse(answer["latency_available"])

    def test_click_uses_latest_metadata_id(self):
        self.routes.handle("/gimbal/click?x=0.2&y=0.3", "POST")
        self.assertEqual(self.control.modes, [42])
        self.routes.handle("/gimbal/click?x=0.9&y=0.9", "POST")
        self.assertEqual(self.control.modes, [42, None])
        self.assertEqual(self.control.moves, [(0, 0)])

    def test_manual_stop(self):
        self.routes.handle("/gimbal/move?x=-1&y=0", "POST")
        self.routes.handle("/gimbal/stop", "POST")
        self.assertEqual(self.control.moves, [(-1, 0), (0, 0)])

    def test_position_only_with_live_video_and_metadata(self):
        self.assertIsNone(self.routes.handle("/gimbal/ping", "POST")["position_estimate"])
        now = time.monotonic()
        self.routes.latest = lambda: dict(received_at=now, video_received_at=now)
        self.assertEqual(self.routes.handle("/gimbal/ping", "POST")["position_estimate"],
                         dict(pan=52.5, tilt=40.0))


if __name__ == "__main__":
    unittest.main()
