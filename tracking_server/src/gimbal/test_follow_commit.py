"""Bridge confirmations must precede committed follow handoffs."""
from collections import deque
from threading import Condition
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..common.state import APP
from . import live_tracks


class FollowCommitTests(unittest.TestCase):
    def test_ack_then_matching_state_commits_once(self):
        guard = Mock(sel=7)
        proposal = (7, 4, 8, 55, 'confirmed', 'guard_n', (guard, 'decision'))
        gb = NS(handoff_pending=(proposal, False), lead_pending=1, selection_reset=False,
                fguard=guard, guard_n=0)
        live_tracks.confirm_handoff(gb, True, 8)
        guard.accept_retarget.assert_not_called()
        gb.handoff_pending = (proposal, True)
        live_tracks.confirm_handoff(gb, True, 8)
        guard.accept_retarget.assert_called_once_with('decision')
        self.assertEqual((gb.guard_n, gb.lead_pending), (1, 0))
        live_tracks.confirm_handoff(gb, True, 8)
        self.assertEqual(gb.guard_n, 1)

    def test_wrong_state_or_cancelled_selection_never_commits(self):
        for auto, target, reset in [(False, 8, False), (True, 7, False), (True, 8, True)]:
            guard = Mock(sel=7)
            proposal = (7, 4, 8, 55, 'unused', 'guard_n', (guard, 'decision'))
            gb = NS(handoff_pending=(proposal, True), lead_pending=1, selection_reset=reset,
                    fguard=guard, guard_n=0)
            live_tracks.confirm_handoff(gb, auto, target)
            guard.accept_retarget.assert_not_called()
            self.assertEqual(gb.guard_n, 0)

    def test_stop_discards_queued_home_and_follow_work_even_when_full(self):
        gb = NS(wq=deque([b'P\n', b'A 1\n', b'W 2\n', b'J 1 4 2 5\n', b'C\n', b'V 1 0\n']),
                wframe=b'F old\n', tracks_ticket=0, wcond=Condition())
        with patch.dict(vars(APP), GB=gb, CTL={}), patch.object(live_tracks, 'current_source', return_value=('fixture',)), gb.wcond:
            live_tracks.stop_with_ack_locked(gb)
        self.assertTrue(any(line.startswith(b'H ') for line in gb.wq))
        self.assertFalse(any(line.split()[0] in (b'A', b'W', b'J', b'C', b'V') for line in gb.wq))


class PersistentChoiceTests(unittest.TestCase):
    def test_web_lifetime_requires_opt_in_and_capable_bridge(self):
        from . import test_live_tracks as fixtures
        for enabled, capable, expected in [('0', True, b'A 17\n'), ('1', False, b'A 17\n'), ('1', True, b'W 17\n')]:
            with self.subTest(enabled=enabled, capable=capable):
                fixture = fixtures.LiveTracksTests()
                fixture.setUp()
                try:
                    fixture.publish()
                    APP.GB.web_follow_control_supported = capable
                    with patch.dict('os.environ', {'LWIR_WEB_FOLLOW_PERSISTENT': enabled}):
                        self.assertTrue(live_tracks.click(120, 100)[0])
                    self.assertEqual(APP.GB.wq[-1], expected)
                finally:
                    fixture.tearDown()
