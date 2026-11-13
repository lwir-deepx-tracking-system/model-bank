"""11-13: a detector model change during a replay plays the recording again from the first frame (same settings)."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import replay_link


class ReplayModelRestartTest(unittest.TestCase):
    def run_restart(self, replay, answers):
        calls = []

        def start(query):
            calls.append(query)
            return answers[min(len(calls) - 1, len(answers) - 1)]
        app = SimpleNamespace(LIVE=SimpleNamespace(replay=replay))
        with patch.object(replay_link, 'APP', app), patch('builtins.print'):
            th = replay_link.restart_after_change('test', tries=5, wait_s=0, start=start)
            if th is not None:
                th.join(5)
        return th, calls

    def test_no_replay_does_nothing(self):
        th, calls = self.run_restart(None, [(200, 'ok')])
        self.assertIsNone(th)
        self.assertEqual(calls, [])

    def test_same_settings_from_start(self):
        rp = dict(name='seq_a', fps=25.0, loop=True, backend='cpp_bytetrack', preview='async', pacing='deadline', paused=True)
        th, calls = self.run_restart(rp, [(200, 'ok')])
        self.assertEqual(calls, [{'seq': ['seq_a'], 'fps': ['25.0'], 'loop': ['1'], 'backend': ['cpp_bytetrack'],
                                  'preview': ['async'], 'pacing': ['deadline']}])

    def test_waits_while_another_switch_runs(self):
        rp = dict(name='seq_b', fps=10, loop=False)
        th, calls = self.run_restart(rp, [(409, 'busy'), (409, 'busy'), (200, 'ok')])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1]['loop'], ['0'])
        self.assertEqual(calls[-1]['backend'], ['python'])

    def test_gives_up(self):
        th, calls = self.run_restart(dict(name='seq_c', fps=10), [(409, 'busy')])
        self.assertEqual(len(calls), 5)


if __name__ == '__main__':
    unittest.main()
