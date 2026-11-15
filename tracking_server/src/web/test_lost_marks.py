"""PC tests of lost_marks.py: how long the prediction marks of a lost person stay (11-16).

python -m unittest src.web.test_lost_marks   (from tracking_server/)
"""
import unittest
from unittest.mock import patch

import numpy as np

from . import lost_marks as lm

FPS = 30.0


def entry(tid, n, cx=300.0, cy=216.0, last=(280.0, 216.0, 40.0, 160.0)):
    return (tid, np.array([cx, cy, 40.0, 160.0]), dict(src='gru_kpt', last=last, pts=[(cx, cy)], n=n, r=160.0 * min(0.5 + 0.05 * n, 4.0)))


class LostMarksTests(unittest.TestCase):
    def setUp(self):
        lm.reset(); lm.LM['tracker'] = None
        self.posp = patch.dict(lm.POSP, show=True, boxes=[])
        self.posp.start(); self.addCleanup(self.posp.stop)
        self.tracker = object()

    def run_loss(self, seconds, *, tracker_frames=30, follow=0, keepid=True, id_map=None, shown=lambda k: (), motion=None,
                 follow_s=10.0):
        """One person (tracker id 5) lost at frame 0. -> [(frame, entries)]"""
        out = []
        for k in range(1, int(seconds * FPS) + 1):
            lm.POSP['boxes'] = [entry(5, k, cx=300.0 + k)] if k <= tracker_frames else []
            out.append((k, lm.update(100.0 + k / FPS, set(shown(k)), follow(k) if callable(follow) else follow,
                                     tracker=self.tracker, id_map=id_map, keepid=keepid, motion=motion, follow_s=follow_s, fps=FPS)))
        return out

    def last_frame(self, frames):
        return max((k for k, e in frames if e), default=0)

    def test_without_keepid_the_marks_end_with_the_tracker_as_before(self):
        self.assertEqual(self.last_frame(self.run_loss(5, keepid=False)), 30)

    def test_with_keepid_they_stay_for_its_window_of_90_frames(self):
        frames = self.run_loss(5)
        self.assertEqual(self.last_frame(frames), 90)
        k, e = frames[59]   # frame 60: the tracker's memory ended 30 frames ago
        oid, box, info = e[0]
        self.assertEqual((oid, info['n'], info['mem']), (5, 60, True))
        self.assertAlmostEqual(box[0], 330.0)                         # the place of the tracker's last prediction
        self.assertAlmostEqual(info['r'], 160.0 * (0.5 + 0.05 * 60))   # the circle keeps growing by the KeepID rule
        self.assertEqual(info['last'], (280.0, 216.0, 40.0, 160.0))
        self.assertAlmostEqual(frames[89][1][0][2]['r'], 160.0 * 4.0)  # ... up to 4 heights

    def test_the_followed_person_stays_until_the_gimbal_stops_waiting(self):
        frames = self.run_loss(12, follow=5, follow_s=10.0)
        self.assertEqual(self.last_frame(frames), 300)                 # 10 s = the bridge's --reacquire-ms
        frames = self.run_loss(12, follow=lambda k: 5 if k <= 200 else 0, follow_s=10.0)
        self.assertEqual(self.last_frame(frames), 200)                 # the bridge dropped the selection: gone in that frame

    def test_they_disappear_in_the_frame_the_number_is_seen_again(self):
        frames = self.run_loss(4, follow=5, shown=lambda k: (5,) if k >= 70 else ())
        self.assertEqual(self.last_frame(frames), 69)
        self.assertEqual(lm.LM['items'], {})

    def test_keepid_moving_the_number_to_a_new_tracker_id_ends_them_too(self):
        id_map = {5: 12}                                               # tracker id 5 is shown as number 12
        def shown(k):
            if k == 50: id_map.pop(5); id_map[9] = 12                  # KeepID relink: tracker id 9 now carries number 12
            return (12,) if k >= 50 else ()
        frames = self.run_loss(4, follow=12, id_map=id_map, shown=shown)
        self.assertEqual(frames[10][1][0][0], 12)                      # drawn under the output number
        self.assertEqual(self.last_frame(frames), 49)

    def test_marks_fade_and_never_go_below_the_floor(self):
        frames = self.run_loss(10, follow=5, follow_s=10.0)
        a = [e[0][2]['alpha'] for k, e in frames if e]
        self.assertAlmostEqual(a[0], 1.0, delta=0.02)
        self.assertTrue(all(x >= y for x, y in zip(a, a[1:])))
        self.assertAlmostEqual(a[-1], lm.FADE_MIN, delta=0.01)
        self.assertAlmostEqual(a[149], 0.7, delta=0.01)                # half of the life

    def test_kept_marks_follow_the_camera_motion(self):
        shift = np.array([[1.0, 0.0, -4.0], [0.0, 1.0, 0.0]])          # the camera turns right: the scene moves 4 px left per frame
        calls = []
        def motion(t0, t1):
            calls.append(round((t1 - t0) * FPS)); return shift
        frames = self.run_loss(2, follow=5, motion=motion)
        oid, box, info = frames[59][1][0]                              # frame 60
        self.assertAlmostEqual(box[0], 330.0 - 4.0 * 30)               # carried for the 30 frames after the tracker's last prediction
        self.assertAlmostEqual(info['last'][0], 280.0 - 4.0 * 59)      # the last-seen box is carried from the start
        self.assertAlmostEqual(info['pts'][0][0], 301.0 - 4.0 * 59)
        self.assertEqual(set(calls), {1})
        lm.reset()
        frames = self.run_loss(2, follow=5, motion=lambda t0, t1: None)   # no camera motion known: the marks stay where they are
        self.assertAlmostEqual(frames[59][1][0][1][0], 330.0)

    def test_switching_the_display_off_or_a_new_tracker_forgets_everything(self):
        self.run_loss(1.5, follow=5)
        self.assertTrue(lm.LM['items'])
        lm.POSP['boxes'] = []
        self.assertEqual(lm.update(200.0, set(), 5, tracker=object(), keepid=True, fps=FPS), [])
        self.run_loss(1.5, follow=5)
        with patch.dict(lm.POSP, show=False):
            self.assertEqual(lm.update(300.0, set(), 5, tracker=self.tracker, keepid=True, fps=FPS), [])
        self.assertEqual(lm.LM['items'], {})

    def test_a_stale_tracker_list_is_not_read(self):
        lm.POSP['boxes'] = [entry(5, 3)]
        self.assertEqual(lm.update(100.0, set(), 0, tracker=self.tracker, keepid=True, fresh=False, fps=FPS), [])


if __name__ == '__main__':
    unittest.main()
