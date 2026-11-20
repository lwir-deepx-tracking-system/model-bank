"""The server delay window starts again when the input changes between the camera and a recording (2026-11-20)."""
import inspect
import types
import unittest

from . import replay_link


class Window:
    def __init__(self): self.n = 0
    def new_profile(self): self.n += 1


class ReplayDelayWindowTest(unittest.TestCase):
    def test_window_that_can_start_again_is_started_again(self):
        live = types.SimpleNamespace(pipeline_perf=Window())
        self.assertTrue(replay_link._new_delay_window(live))
        self.assertEqual(live.pipeline_perf.n, 1)

    def test_no_window_or_old_window_is_left_alone(self):
        self.assertFalse(replay_link._new_delay_window(types.SimpleNamespace()))
        self.assertFalse(replay_link._new_delay_window(types.SimpleNamespace(pipeline_perf=object())))

    def test_failure_does_not_stop_the_switch(self):
        class Bad:
            def new_profile(self): raise RuntimeError('x')
        said = []
        old = replay_link.camera_extra.say
        replay_link.camera_extra.say = said.append
        try: self.assertFalse(replay_link._new_delay_window(types.SimpleNamespace(pipeline_perf=Bad())))
        finally: replay_link.camera_extra.say = old
        self.assertEqual(len(said), 1)

    def test_both_input_changes_call_it_after_the_source_is_set(self):
        for fn, mark in ((replay_link._switch, 'L_.replay = tgt_'), (replay_link.on_cam_lost, 'L_.replay = None')):
            src = inspect.getsource(fn)
            self.assertIn('_new_delay_window(L_)', src)
            self.assertLess(src.index(mark), src.index('_new_delay_window(L_)'))


if __name__ == '__main__':
    unittest.main()
