"""--confirm-ms 시험 (PC 에서, 카메라·NPU 없이):  tracking_server 폴더에서  python -m unittest src.track.test_confirm_time

1) 규칙 계산: 130 ms 가 초당 장 수별로 몇 장이 되는지, 경계에서 오가지 않는지, 끊김을 세지 않는지
2) 진짜 추적기(Tracker)에 붙였을 때: 새 사람이 몇 번째 장에 화면에 나오는지 (3 장 / 4 장), 이미 나온 사람은 그대로인지
"""
import unittest

import numpy as np

from .confirm_time import ConfirmByTime, apply
from .trackers import Tracker


def run_fps(rule, fps, seconds=4.0, t0=0.0):
    n = None; t = t0
    for _ in range(int(fps * seconds)):
        t += 1.0 / fps; n = rule.tick(t)
    return n, t


class RuleNumbers(unittest.TestCase):
    def test_frames_by_speed(self):
        for fps, want in ((15, 2), (20, 3), (22, 3), (24, 3), (25, 3), (27, 3), (28, 3), (29, 4), (30, 4), (40, 5)):
            self.assertEqual(run_fps(ConfirmByTime(130), fps)[0], want, fps)

    def test_never_below_tracker_count(self):
        r = ConfirmByTime(130); run_fps(r, 15)
        self.assertEqual(r.n, 2); self.assertEqual(r.frames_for(3), 3)

    def test_start_value_before_measuring(self):
        r = ConfirmByTime(130); self.assertEqual(r.tick(0.0), 3); self.assertIsNone(r.fps)

    def test_no_flicker_at_the_border(self):
        # x = 130 x fps / 1000 = 3.5 at 26.9 fps: the count must not move while the speed wanders around that point
        r = ConfirmByTime(130); t = 0.0
        for k in range(40):
            _, t = run_fps(r, 26.0 if k % 2 else 27.9, seconds=3.0, t0=t)
        self.assertEqual(r.n, 3); self.assertEqual(r.changes, 0)
        _, t = run_fps(r, 30, t0=t); self.assertEqual(r.n, 4)
        for k in range(40):
            _, t = run_fps(r, 26.0 if k % 2 else 27.9, seconds=3.0, t0=t)
        self.assertEqual(r.n, 4); self.assertEqual(r.changes, 1)
        _, t = run_fps(r, 24, t0=t); self.assertEqual(r.n, 3)

    def test_stall_is_not_counted(self):
        r = ConfirmByTime(130); _, t = run_fps(r, 24)
        for _ in range(30):              # 30 camera stalls of 5 s each, one frame in between
            t += 5.0; r.tick(t)
        self.assertEqual(r.n, 3); self.assertAlmostEqual(r.fps, 24.0, places=3)


def person(conf=0.9):
    return np.array([[300.0, 180.0, 340.0, 300.0, conf, 0.0]], np.float32)


def first_output_frame(n_init, rule=None, fps=25.0, warm=0):
    """frames a NEW person needs before the tracker outputs it. warm = empty frames fed first (so the rule can measure the speed)."""
    trk = Tracker('bytetrack', n_init=n_init, class_gate=True); t = 0.0
    for _ in range(max(1, warm)):       # the first frame of a video outputs births at once, so the person never appears on it
        t += 1.0 / fps
        if rule is not None: apply(rule, trk, t)
        trk.update(np.zeros((0, 6), np.float32))
    for k in range(1, 12):
        t += 1.0 / fps
        if rule is not None: apply(rule, trk, t)
        if len(trk.update(person())): return k, trk
    return None, trk


class OnTheTracker(unittest.TestCase):
    def test_off_is_the_old_rule(self):
        self.assertEqual(first_output_frame(3)[0], 3)

    def test_25fps_same_as_before(self):
        k, trk = first_output_frame(3, ConfirmByTime(130), fps=25.0, warm=100)
        self.assertEqual(k, 3); self.assertEqual(trk.n_init, 3)

    def test_30fps_needs_4_frames(self):
        k, trk = first_output_frame(3, ConfirmByTime(130), fps=30.0, warm=100)
        self.assertEqual(k, 4); self.assertEqual(trk.n_init, 4); self.assertEqual(trk.n_init_base, 3)

    def test_slow_camera_keeps_3_frames(self):
        k, trk = first_output_frame(3, ConfirmByTime(130), fps=12.0, warm=100)
        self.assertEqual(k, 3)

    def test_shown_person_stays_when_the_count_rises(self):
        rule = ConfirmByTime(130); trk = Tracker('bytetrack', n_init=3, class_gate=True); t = 0.0
        t += 0.04; apply(rule, trk, t); trk.update(np.zeros((0, 6), np.float32))
        ids = []
        for _ in range(5):
            t += 0.04; apply(rule, trk, t); rows = trk.update(person())
            if len(rows): ids.append(int(rows[0][4]))
        self.assertEqual(len(ids), 3)                      # shown from the 3rd frame on
        for _ in range(200):                               # the stream speeds up to 30 fps: the count becomes 4
            t += 1.0 / 30; apply(rule, trk, t); rows = trk.update(person())
            self.assertEqual(len(rows), 1); self.assertEqual(int(rows[0][4]), ids[0])
        self.assertEqual(trk.n_init, 4)


if __name__ == '__main__':
    unittest.main()
