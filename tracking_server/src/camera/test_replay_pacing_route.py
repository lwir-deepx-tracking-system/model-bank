"""Freeze replay timing per run, including restart, without changing camera input."""
import os
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..common.state import APP
from . import replay_link


class ReplayPacingRouteTests(unittest.TestCase):
    def setUp(self):
        self.live = NS(replay=None, stopped=False, cam_hold=False, replay_busy=False,
                       sys_lock=threading.Lock(), cam_lock=threading.Lock())
        self.thread = Mock()
        patches = [patch.dict(vars(APP), LIVE=self.live, args=NS(live_wait=0, live_cam='camera'),
                              lv_th=NS(Thread=self.thread)),
                   patch.object(replay_link, 'find', return_value=(dict(name='test',bits=16,frames=2), 'images')),
                   patch.object(replay_link, 'set_preview_producer'),
                   patch.object(replay_link, '_gimbal_stop'),
                   patch.dict(os.environ, LWIR_REPLAY_PACING='deadline')]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_requested_baseline_is_frozen_and_passed_to_process(self):
        status, _ = replay_link.start_request({'seq':['test'], 'pacing':['legacy']})
        self.assertEqual(status, 200)
        target = self.thread.call_args.kwargs['args'][0]
        self.assertEqual(target['pacing'], 'legacy')
        self.live.replay = target
        with patch.object(replay_link.replay_eval, 'lossless', return_value=True):
            argv = replay_link.cam_argv()
        self.assertEqual(argv[argv.index('--pacing') + 1], 'legacy')
        self.assertIn('--no-skip', argv)

    def test_environment_default_is_frozen_for_new_replay(self):
        self.assertEqual(replay_link.start_request({'seq':['test']})[0], 200)
        self.assertEqual(self.thread.call_args.kwargs['args'][0]['pacing'], 'deadline')

    def test_invalid_or_multiple_modes_do_not_change_source(self):
        for values in (['typo'], ['legacy','deadline'], []):
            self.assertEqual(replay_link.start_request({'pacing':values})[0], 400)
        self.assertFalse(self.live.cam_hold)
        self.thread.assert_not_called()

    def test_camera_command_is_unaffected(self):
        self.assertEqual(replay_link.cam_argv(), ['camera'])

    def test_restart_preserves_pacing_against_new_default(self):
        self.live.replay = dict(name='test',fps=30,backend='python',pacing='legacy',preview='async')
        with patch.object(replay_link, 'start_request', return_value=(200,'ok')) as start:
            replay_link.restart_request()
        self.assertEqual(start.call_args.args[0]['pacing'], ['legacy'])


if __name__ == '__main__':
    unittest.main()
