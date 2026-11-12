import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import replay_eval


class ReplayPrefixTest(unittest.TestCase):
    def make_replay(self, folder):
        images = folder / 'images'
        images.mkdir(exist_ok=True)
        for slot in (1, 2):
            (images / f'{slot:06d}.png').touch()
        (folder / 'tracks_edited_mot.txt').write_text(
            '1,10,100,100,20,80,1,0,-1,-1\n2,10,100,100,20,80,1,0,-1,-1\n', encoding='utf-8')
        return replay_eval.Eval(dict(name='fixture', fps=25, loop=False, dir=str(images), frames=2))

    def test_prefix_and_final_stay_distinct_after_seek_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(replay_eval, EV=None, LAST=None, FINAL=replay_eval.collections.OrderedDict()):
            ev = self.make_replay(Path(tmp))
            replay_eval.EV = ev
            ev.tag(1, 1, 0)
            ev.frame(1, [[100, 116, 120, 196, 1, 0.9, 0, 0]], None, 'one', [])
            self.assertTrue(ev.prefix(1)['prefix_unavailable'])
            ev.tag(2, 2, 1)
            ev.frame(2, [], None, 'two', [])
            ev.metric_worker.join(5)
            self.assertFalse(ev.metric_worker.is_alive())
            ev.want_seek(0, 3)
            self.assertTrue(ev.prefix(0)['prefix_pending'])
            worker = ev.prefix_worker
            if worker is not None:
                worker.join(5)
            prefix = replay_eval.results({'pos': ['0']})['eval']
            final = replay_eval.results({'view': ['final']})['eval']
            self.assertEqual((prefix['pass_frames'], prefix['new']['missed'], prefix['prefix_pending']), (1, 0, False))
            self.assertEqual((final['pass_frames'], final['new']['missed'], final['final_previous']), (2, 1, False))
            replay_eval.EV = self.make_replay(Path(tmp))
            earlier = replay_eval.results({'view': ['final']})['eval']
            self.assertTrue(earlier['final_previous'])
            self.assertEqual(earlier['run_id'], final['run_id'])


if __name__ == '__main__':
    unittest.main()
