"""검출 모델 자동 선택 (auto_det.py) 시험. NPU 없이 돈다. 상자 높이와 시각은 지어낸 값이다 (장치에서 잰 값이 아니다).

- Chooser: 작아지는 사람은 1초 뒤에야 FAR, 문턱 둘레 오락가락은 안 바뀜, 바꾼 뒤 5초 유지,
  (11-15 조사 권고) 사람을 잃으면 0.5초 뒤 FAR(머무름 무시), 가려졌던 사람은 3초, 가려지거나 잘린 작은 상자는 FAR 안 감,
  가려진 중간 크기는 NEAR, 1분에 4번까지, 빠르게 움직이면 3초까지 미룸, 바꾸는 중 빈 프레임은 사람 잃음이 아님
- size_signal / observe / Smooth: 따라가는 사람이 화면 가운데값보다 먼저, 화면 끝 프레임은 가운뎃값에서 뺌, 관절 수로 가림
- on_frame: 녹화 재생이면 기록(dry run)만 하고 요청하지 않음, 요청은 화면과 같은 길(model_request -> MODEL['req'] -> do_model_swap)
"""
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from ..common.state import APP
from ..common.shared_values import CORE, DET
from . import auto_det
from ..web import page_settings

FPS = 30.0
NEAR_DET, FAR_DET = 'bank:' + auto_det.NEAR_KEY, 'bank:' + auto_det.FAR_KEY


def run(ch, seq, current, t0=0.0, occl=None, fast=None):
    """seq: list of px (None = nobody), one per frame at FPS (occl, fast: lists of the same length, default all False).
    -> [(t, wanted)] for the frames where step() asked for a model.
    Like the server: an asked model is taken at once (current changes, switched(t) starts the dwell)."""
    out = []
    for k, px in enumerate(seq):
        t = t0 + k / FPS
        w = ch.step(t, px, current, bool(occl[k]) if occl else False, bool(fast[k]) if fast else False)
        if w is not None: out.append((t, w)); current = w; ch.switched(t)
    return out


def frames(seconds, px):
    return [px] * int(round(seconds * FPS))


def person(tid, h, score=0.9, cls=0, x=100.0, top=50.0):
    return [x, top, x + h * 0.4, top + h, tid, score, cls, 0]


class ChooserTest(unittest.TestCase):
    def test_shrinking_target_goes_far_only_after_one_second(self):
        ch = auto_det.Chooser()
        seq = [160.0 - 30.0 * (k / FPS) for k in range(int(4 * FPS))]   # 160 px -> 40 px in 4 s, 30 px a second
        first_below = next(k / FPS for k, px in enumerate(seq) if px < 100.0)
        got = run(ch, seq, 'near')
        self.assertEqual([w for _, w in got], ['far'])
        self.assertGreaterEqual(got[0][0], first_below + 1.0 - 1e-9)
        self.assertLessEqual(got[0][0], first_below + 1.0 + 1.0 / FPS + 1e-9)

    def test_no_switch_when_already_on_the_wanted_model(self):
        self.assertEqual(run(auto_det.Chooser(), frames(5, 60.0), 'far'), [])
        self.assertEqual(run(auto_det.Chooser(), frames(5, 300.0), 'near'), [])

    def test_flicker_around_the_far_threshold_does_not_switch(self):
        ch = auto_det.Chooser()
        seq = [95.0 if k % 2 else 105.0 for k in range(int(10 * FPS))]   # every other frame over the line
        self.assertEqual(run(ch, seq, 'near'), [])
        seq = (frames(0.9, 90.0) + frames(0.1, 110.0)) * 10                 # 0.9 s under, a short look over, again
        self.assertEqual(run(auto_det.Chooser(), seq, 'near'), [])

    def test_flicker_around_the_near_threshold_does_not_switch(self):
        seq = [135.0 if k % 2 else 150.0 for k in range(int(10 * FPS))]
        self.assertEqual(run(auto_det.Chooser(), seq, 'far'), [])

    def test_between_the_two_thresholds_keeps_the_model(self):
        for cur in ('near', 'far'):
            self.assertEqual(run(auto_det.Chooser(), frames(10, 120.0), cur), [])

    def test_large_target_goes_near_after_one_second(self):
        got = run(auto_det.Chooser(), frames(3, 200.0), 'far')
        self.assertEqual([w for _, w in got], ['near'])
        self.assertAlmostEqual(got[0][0], 1.0, delta=1.0 / FPS + 1e-9)

    def test_dwell_after_a_switch(self):
        ch = auto_det.Chooser()
        ch.switched(0.0)   # e.g. just went to FAR
        got = run(ch, frames(8, 200.0), 'far')
        self.assertEqual([w for _, w in got], ['near'])
        self.assertGreaterEqual(got[0][0], 5.0 - 1e-9)
        self.assertLessEqual(got[0][0], 5.0 + 1.0 / FPS + 1e-9)
        ch = auto_det.Chooser(); ch.switched(0.0); ch.step(1.0, 200.0, 'far')
        self.assertIsNone(ch.step(2.5, 200.0, 'far'))
        self.assertIn('남음', ch.reason)   # the page says why it waits

    def test_dwell_counts_for_a_switch_by_hand_too(self):
        ch = auto_det.Chooser(); ch.switched(2.0)
        got = run(ch, frames(10, 60.0), 'near')
        self.assertGreaterEqual(got[0][0], 7.0 - 1e-9)

    def test_person_lost_goes_far_at_once_without_dwell(self):
        # 11-15 research fix 1: the near model does not see small people at all, so a lost person means far now (0.5 s), dwell ignored
        ch = auto_det.Chooser(); ch.switched(0.0)   # just switched: the dwell would block a size switch
        got = run(ch, frames(1.0, 200.0) + frames(2.0, None), 'near')
        self.assertEqual([w for _, w in got], ['far'])
        self.assertAlmostEqual(got[0][0], 1.0 + 0.5, delta=1.0 / FPS + 1e-9)
        ch = auto_det.Chooser(); ch.switched(0.0); ch.step(0.1, None, 'near')
        self.assertEqual(ch.step(0.65, None, 'near'), 'far')
        self.assertTrue(ch.forced); self.assertIn('제한 없이', ch.reason)
        self.assertEqual(run(auto_det.Chooser(), frames(30, None), 'far'), [])   # already far: nothing to ask
        self.assertEqual([w for _, w in run(auto_det.Chooser(), frames(1, None), 'near')], ['far'])   # empty picture from the start

    def test_lost_while_hidden_waits_longer(self):
        # last seen partly hidden (behind a desk) or cut by the edge: probably behind something, the near model is the one for it -> 3 s
        ch = auto_det.Chooser()
        got = run(ch, frames(1.0, 120.0) + frames(5.0, None), 'near', occl=frames(1.0, True) + frames(5.0, False))
        self.assertEqual([w for _, w in got], ['far'])
        self.assertAlmostEqual(got[0][0], 1.0 + 3.0, delta=1.0 / FPS + 1e-9)
        self.assertEqual(run(auto_det.Chooser(), frames(1.0, 120.0) + frames(2.5, None), 'near', occl=frames(3.5, True)), [])   # back within 3 s: no switch

    def test_change_frames_are_not_a_lost_person(self):
        ch = auto_det.Chooser()
        self.assertIsNone(ch.step(0.0, None, 'far'))   # on far: nobody seen (no switch); the clock of "nobody" runs
        for k in range(12): ch.step(0.1 * k, None, 'far')
        ch.switched(1.2)                                 # e.g. a switch to near just made by the frame loop
        self.assertIsNone(ch.step(1.3, None, 'near'))   # 1.3 s without a person, but only 0.1 s since the change
        self.assertIsNone(ch.step(1.6, None, 'near'))
        self.assertEqual(ch.step(1.85, None, 'near'), 'far')   # 0.55 s of frames from the new model and still nobody: it does not see them

    def test_short_gap_keeps_the_collected_time_long_gap_starts_again(self):
        # on far: 0.6 s big, 2 s nobody (<= 3 s), big again: the time counts through the gap -> NEAR at the first big frame after it
        seq = frames(0.6, 200.0) + frames(2.0, None) + frames(2.0, 200.0)
        got = run(auto_det.Chooser(), seq, 'far')
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0][0], 2.6, delta=1.0 / FPS + 1e-9)
        # 0.6 s big, 4 s nobody (> 3 s): collected time forgotten -> NEAR only 1 s after the person is back
        seq = frames(0.6, 200.0) + frames(4.0, None) + frames(2.0, 200.0)
        got = run(auto_det.Chooser(), seq, 'far')
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0][0], 4.6 + 1.0, delta=1.0 / FPS + 1e-9)

    def test_hidden_or_cut_small_box_does_not_go_far(self):
        # 11-15 research fix 2: indoors a box is small mostly because a desk hides the body or the edge cuts it, not because it is far
        self.assertEqual(run(auto_det.Chooser(), frames(10, 70.0), 'near', occl=frames(10, True)), [])
        ch = auto_det.Chooser(); run(ch, frames(2, 70.0), 'near', occl=frames(2, True))
        self.assertIn('안 바꿈', ch.reason)

    def test_hidden_person_of_middle_size_goes_near(self):
        got = run(auto_det.Chooser(), frames(3, 120.0), 'far', occl=frames(3, True))   # 120 px is in the middle band, but hidden
        self.assertEqual([w for _, w in got], ['near'])
        self.assertAlmostEqual(got[0][0], 1.0, delta=1.0 / FPS + 1e-9)
        self.assertEqual(run(auto_det.Chooser(), frames(3, 120.0), 'far'), [])   # not hidden: the middle band keeps the model

    def test_at_most_four_size_switches_a_minute(self):
        ch = auto_det.Chooser(dict(dwell_s=0.0))
        seq = (frames(1.5, 200.0) + frames(1.5, 60.0)) * 6 + frames(1.5, 200.0)   # big / small every 1.5 s: would switch 13 times in 19.5 s
        got = run(ch, seq, 'far')
        self.assertEqual(len(got), 4)
        self.assertIn('1분에', ch.reason)
        got = run(ch, frames(60, 200.0), got[-1][1], t0=len(seq) / FPS)   # a minute after the first switch the next one may come
        self.assertEqual(len(got), 1)

    def test_fast_moving_person_defers_a_size_switch(self):
        # 11-15 research: switch when the person moves slowly (the gap of a change costs more when they move fast); at most defer_s
        got = run(auto_det.Chooser(), frames(6, 60.0), 'near', fast=frames(6, True))
        self.assertEqual([w for _, w in got], ['far'])
        self.assertAlmostEqual(got[0][0], 1.0 + 3.0, delta=1.0 / FPS + 1e-9)
        got = run(auto_det.Chooser(), frames(6, 60.0), 'near', fast=frames(2.0, True) + frames(4, False))   # slows down at 2 s
        self.assertAlmostEqual(got[0][0], 2.0, delta=1.0 / FPS + 1e-9)
        got = run(auto_det.Chooser(), frames(3, None), 'near', fast=frames(3, True))   # a lost person is never deferred
        self.assertAlmostEqual(got[0][0], 0.5, delta=1.0 / FPS + 1e-9)

    def run_models(self, ch, seconds, far_px, near_px, current, t0=0.0, occl=True):
        """Like the device on 11-15: what each model sees differs. far_px / near_px: px (or None) the far / near model gives."""
        out = []
        for k in range(int(round(seconds * FPS))):
            t = t0 + k / FPS
            px = far_px if current == 'far' else near_px
            w = ch.step(t, px, current, occl and px is not None)
            if w is not None: out.append((round(t, 2), w)); current = w; ch.switched(t)
        return out, current

    def test_near_that_sees_nobody_backs_off(self):
        # 11-15 device: the far model saw a giant cut box on a blank picture, the near model saw nothing -> near / far every few seconds.
        # Now: near once, its model sees nobody -> far, and the next near waits 30 s, then 60 s, ... (at most near_backoff_max_s)
        ch = auto_det.Chooser()
        got, cur = self.run_models(ch, 120, 300.0, None, 'far')
        times = [tt for tt, w in got if w == 'near']
        self.assertAlmostEqual(times[0], 1.0, delta=0.1)
        self.assertGreaterEqual(times[1] - times[0], 30.0)
        self.assertGreaterEqual(times[2] - times[1], 60.0)
        self.assertEqual(len(times), 3)                      # 120 s: near at about 1, 34, 97 s instead of every 8 s
        self.assertIn('쉼', ch.reason)
        self.assertEqual(ch.backoff, 120.0)                 # grew 30 -> 60 -> 120 s
        self.assertEqual([w for _, w in got][:2], ['near', 'far'])

    def test_near_that_sees_the_person_clears_the_back_off(self):
        ch = auto_det.Chooser()
        got, cur = self.run_models(ch, 6, 300.0, None, 'far')         # near, nobody -> far, pause 30 s
        self.assertEqual(ch.backoff, 30.0)
        got, cur = self.run_models(ch, 40, 300.0, 300.0, cur, t0=6.0)  # near again after the pause, and now the near model sees the person
        self.assertEqual(cur, 'near'); self.assertEqual(ch.backoff, 0.0)
        got, cur = self.run_models(ch, 5, 300.0, None, cur, t0=46.0)  # the person leaves: far as usual, no pause added
        self.assertEqual(cur, 'far'); self.assertEqual(ch.backoff, 0.0)

    def test_one_switch_told_twice_counts_once(self):
        ch = auto_det.Chooser()
        ch.switched(1.0); ch.switched(1.4, count=False)
        self.assertEqual(ch.recent(2.0), 1)
        self.assertAlmostEqual(ch.dwell_left(2.0), 5.0 - 0.6)

    def test_other_model_goes_to_one_of_the_two(self):
        self.assertEqual([w for _, w in run(auto_det.Chooser(), frames(2, 120.0), 'other')], ['near'])   # middle band from 'other' = default side
        self.assertEqual([w for _, w in run(auto_det.Chooser(), frames(2, 60.0), 'other')], ['far'])

    def test_parameters(self):
        ch = auto_det.Chooser(dict(far_px=50.0, hold_s=0.5))
        self.assertEqual(run(ch, frames(3, 60.0), 'near'), [])
        got = run(auto_det.Chooser(dict(far_px=50.0, hold_s=0.5)), frames(3, 40.0), 'near')
        self.assertAlmostEqual(got[0][0], 0.5, delta=1.0 / FPS + 1e-9)


class SignalTest(unittest.TestCase):
    def test_followed_target_wins_over_scene_median(self):
        rows = [person(1, 200), person(2, 210), person(7, 80), person(3, 220)]
        self.assertEqual(auto_det.size_signal(rows, 7)[:3], (80.0, 'followed', 7))
        px, src, _, n = auto_det.size_signal(rows, 0)
        self.assertEqual((px, src, n), (205.0, 'scene', 4))

    def test_followed_but_not_in_the_frame_is_no_signal(self):
        self.assertEqual(auto_det.size_signal([person(1, 200)], 7)[:2], (None, 'followed_missing'))

    def test_low_score_and_other_classes_are_left_out_of_the_median(self):
        rows = [person(1, 60, score=0.1), person(2, 150), person(3, 30, cls=2)]
        self.assertEqual(auto_det.size_signal(rows, 0)[:2], (150.0, 'scene'))
        self.assertEqual(auto_det.size_signal([person(1, 60, score=0.1)], 0)[:2], (None, 'none'))
        self.assertEqual(auto_det.size_signal([person(9, 60, score=0.05)], 9)[:2], (60.0, 'followed'))   # the followed one counts at any score
        self.assertEqual(auto_det.size_signal([], 0)[:2], (None, 'none'))


def kp_set(box, n_visible):
    """A detect_frame.KP-like set: one detection box with 17 keypoints, the first n_visible of them visible (0.9), the rest 0.1."""
    return dict(boxes=[list(box)], kpts=[[[0.0, 0.0, 0.9 if j < n_visible else 0.1] for j in range(17)]])


class SmoothObserveTest(unittest.TestCase):
    def obs(self, px, edge=False, occl=False, cx=None, cy=None, src='followed', tid=7):
        return dict(px=px, src=src, id=tid, n=1, edge=edge, occl=occl, kpts=None, cx=cx, cy=cy, score=0.9)

    def test_edge_frames_are_left_out_of_the_median(self):
        sm = auto_det.Smooth()
        for k in range(30): s = sm.add(k / FPS, self.obs(60.0 if k % 2 else 200.0, edge=bool(k % 2)), 1.0)   # the cut frames read small
        self.assertEqual((s['px'], s['cut']), (200.0, False))
        sm = auto_det.Smooth()
        for k in range(30): s = sm.add(k / FPS, self.obs(60.0, edge=True), 1.0)
        self.assertEqual((s['px'], s['cut']), (60.0, True))   # every frame touches the edge: still a size, but marked cut

    def test_window_median_occlusion_share_and_speed(self):
        sm = auto_det.Smooth()
        for k in range(60): s = sm.add(k / FPS, self.obs(100.0 + k, occl=k >= 40, cx=100.0 + 4.0 * k, cy=200.0), 1.0)
        self.assertAlmostEqual(s['px'], 100.0 + 44.5, delta=0.6)        # median of the last second (frames 30..59)
        self.assertTrue(s['occl'])                                       # 20 of 30 frames hidden
        self.assertAlmostEqual(s['speed'], 4.0 * FPS / s['px'], delta=0.05)   # 120 px/s over a 144 px box = 0.83 box heights a second
        s = sm.add(2.1, self.obs(150.0, tid=8), 1.0)                     # another person: starts again
        self.assertEqual((s['px'], s['speed']), (150.0, None))

    def test_observe_edge_uses_the_real_camera_position(self):
        p = auto_det.PARAMS
        self.assertTrue(auto_det.observe([person(7, 100, x=1.0)], 7, p)['edge'])
        self.assertFalse(auto_det.observe([person(7, 100, x=12.0)], 7, p)['edge'])
        self.assertTrue(auto_det.observe([person(7, 100, x=12.0)], 7, p, stab=(10.0, 0.0))['edge'])   # stabilizer shifted it 10 px
        self.assertTrue(auto_det.observe([person(7, 100, top=17.0)], 7, p)['edge'])                   # canvas y 17 = picture y 1
        self.assertTrue(auto_det.observe([person(7, 470, top=25.0)], 7, p)['edge'])                   # bottom at picture y 479
        self.assertFalse(auto_det.observe([person(7, 470, top=20.0)], 7, p)['edge'])                  # bottom at picture y 474
        o = auto_det.observe([person(1, 60, x=0.0), person(2, 200)], 0, p)
        self.assertEqual((o['px'], o['edge'], o['n']), (200.0, False, 2))   # scene: the cut person is left out of the median

    def test_detector_output_drops_impossible_person_boxes(self):
        import numpy as np
        from . import detect_frame
        n0 = detect_frame.GIANT[0]
        arr = np.array([[0, -2000, 600, 3000, 0.45, 0], [100, 50, 160, 250, 0.9, 0], [0, -2000, 600, 3000, 0.8, 2]], np.float32)
        out = detect_frame.drop_giant(arr, 0)
        self.assertEqual(out.shape[0], 2); self.assertEqual(detect_frame.GIANT[0], n0 + 1)   # the giant person goes; a normal person and other classes stay
        self.assertEqual(detect_frame.drop_giant(arr[1:2], 0).shape[0], 1)
        self.assertEqual(len(detect_frame.drop_giant(arr[:0], 0)), 0)

    def test_observe_drops_impossible_boxes(self):
        p = auto_det.PARAMS
        self.assertEqual(auto_det.observe([person(1, 5000, top=-2000)], 0, p)['src'], 'none')   # 11-15 device: ~5000 px boxes on a blank picture
        o = auto_det.observe([person(1, 5000, top=-2000), person(2, 200)], 0, p)
        self.assertEqual((o['px'], o['n']), (200.0, 1))
        self.assertEqual(auto_det.observe([person(1, 1400, top=-500)], 0, p)['px'], 1400.0)    # a very close person is still a person

    def test_observe_keypoints_say_hidden(self):
        p = auto_det.PARAMS
        r = person(7, 150)
        box = r[:4]
        for n, hidden in ((15, False), (12, False), (11, True), (5, True), (3, True), (2, False), (0, False)):
            o = auto_det.observe([r], 7, p, kp_sets=(kp_set(box, n), None))
            self.assertEqual((o['kpts'], o['occl']), (n, hidden), n)
        o = auto_det.observe([r], 7, p, kp_sets=(kp_set([400, 50, 460, 200], 5),))   # keypoints of somebody else (no overlap)
        self.assertEqual((o['kpts'], o['occl']), (None, False))
        self.assertEqual(auto_det.observe([r], 7, p)['kpts'], None)   # a detector without keypoints


class GlueTest(unittest.TestCase):
    """on_frame / state / parse_request with a stand-in APP (real APP object of common.state, saved and restored around each test)."""
    def setUp(self):
        self.saved = dict(vars(APP))
        self.core = dict(CORE); self.det = dict(DET)
        auto_det.reset()
        self.sel = []
        cat = [dict(id=1, det='w480', key='NEWMRW480', name='YOLO26m_(FLIR) 640x480'),
               dict(id=2, det=NEAR_DET, key=auto_det.NEAR_KEY, name='near'),
               dict(id=3, det=FAR_DET, key=auto_det.FAR_KEY, name='far')]
        APP.MODEL = dict(lock=threading.Lock(), req=None, busy=False, last=None, history=[], catalog=cat)
        APP.MODEL_TEXT = {0: '바뀜', 1: '없는 모델 번호', 2: '다른 변경이 진행 중', 3: '모델을 만들지 못함', 4: '실패해서 이전 모델로 되돌림', 6: '준비 안 됨'}
        APP.args = SimpleNamespace(det=NEAR_DET, mock=True, npu_depth=1)
        APP.man = {}; APP.SEL = {}; APP.SEL_ON = False
        APP.sel_save = lambda **kw: self.sel.append(kw)
        APP.model_request = MagicMock(side_effect=page_settings.model_request)   # the page's own function, watched
        APP.GB = None; APP.LIVE = None
        for d_, k_ in ((NEAR_DET, auto_det.NEAR_KEY), (FAR_DET, auto_det.FAR_KEY)):
            CORE[k_] = 0; DET[d_] = dict(names=[k_], yoff=16.0)
        self.p = patch('builtins.print'); self.p.start()

    def tearDown(self):
        self.p.stop()
        APP.__dict__.clear(); APP.__dict__.update(self.saved)
        CORE.clear(); CORE.update(self.core); DET.clear(); DET.update(self.det)
        auto_det.reset()

    def turn_on(self, **q):
        ok, text, apply = auto_det.parse_request(dict({'auto_det': ['1']}, **{k: [str(v)] for k, v in q.items()}))
        self.assertTrue(ok, text); apply()

    def feed(self, seconds, rows, t0=0.0, **kw):
        asked = []
        for k in range(int(round(seconds * FPS))):
            t = t0 + k / FPS
            r = auto_det.on_frame(t, rows, frame='live_%09d.png' % k, **kw)
            if r is not None: asked.append((t, r))
        return asked

    def frame_loop_boundary(self):
        """Stand-in for the top of the frame loop in app.py (v23 block): take MODEL['req'] and swap with page_settings.do_model_swap."""
        rq = APP.MODEL['req']
        if rq is None: return None
        APP.MODEL['busy'] = True
        code = page_settings.do_model_swap(rq['det'])
        with APP.MODEL['lock']:
            APP.MODEL['req'] = None; APP.MODEL['busy'] = False
            APP.MODEL['last'] = dict(id=rq['id'], result=code, text=APP.MODEL_TEXT.get(code, '?'))
        return code

    def test_off_by_default_does_nothing(self):
        self.assertEqual(self.feed(3, [person(1, 60)]), [])
        APP.model_request.assert_not_called()
        self.assertEqual(auto_det.state()['mode'], 'off')

    def test_request_goes_through_the_page_model_change_path(self):
        self.turn_on()
        self.assertEqual(self.sel[-1], dict(auto_det=True, auto_det_p={}))
        asked = self.feed(2, [person(4, 70)])
        self.assertEqual([w for _, w in asked], ['far'])
        APP.model_request.assert_called_once_with(3)                       # far model id, through model_request
        self.assertEqual(APP.MODEL['req']['det'], FAR_DET)                   # queued for the frame loop, nothing swapped here
        self.assertEqual(APP.args.det, NEAR_DET)
        self.assertEqual(self.feed(1, [person(4, 70)], t0=2.0), [])          # pending: never asks twice
        self.assertEqual(APP.model_request.call_count, 1)
        APP.eng = {auto_det.NEAR_KEY: object()}; APP.det_names = [auto_det.NEAR_KEY]; APP.a = None; APP.b = None
        APP.Engine = MagicMock(return_value='engine')
        self.assertEqual(self.frame_loop_boundary(), 0)                      # the frame loop swaps between two frames
        APP.Engine.assert_called_once_with(auto_det.FAR_KEY, 0, 1)
        self.assertEqual(APP.args.det, FAR_DET)
        self.feed(0.1, [person(4, 70)], t0=3.0)
        st = auto_det.state()
        self.assertEqual((st['current'], st['pending'], st['switches'], st['mode']), ('far', None, 1, 'live'))
        self.assertEqual(st['last_switch']['from'], 'near'); self.assertEqual(st['last_switch']['to'], 'far')
        json.dumps(st)   # the /rules answer must be JSON

    def test_never_requests_while_another_change_is_pending_or_busy(self):
        self.turn_on()
        APP.MODEL['busy'] = True
        self.assertEqual(self.feed(3, [person(4, 70)]), [])
        APP.MODEL['busy'] = False; APP.MODEL['req'] = dict(det='w480', id=1, req=None, t=0.0)
        self.assertEqual(self.feed(3, [person(4, 70)], t0=3.0), [])
        APP.model_request.assert_not_called()
        self.assertIn('기다림', auto_det.state()['reason'])

    def test_followed_target_preferred_over_scene(self):
        self.turn_on()
        APP.GB = SimpleNamespace(state=dict(auto=1, id=7), selection_reset=False)
        rows = [person(1, 220), person(2, 230), person(7, 80)]
        self.assertEqual([w for _, w in self.feed(2, rows)], ['far'])
        self.assertEqual(auto_det.state()['signal']['source'], 'followed')

    def test_followed_person_lost_asks_far_fast(self):
        self.turn_on()
        APP.GB = SimpleNamespace(state=dict(auto=1, id=7), selection_reset=False)
        self.assertEqual(self.feed(1, [person(7, 200)]), [])
        asked = self.feed(2, [person(1, 200)], t0=1.0)   # somebody else is there, the followed one is gone
        self.assertEqual([w for _, w in asked], ['far'])
        self.assertAlmostEqual(asked[0][0], 1.5, delta=1.0 / FPS + 1e-9)
        st = auto_det.state()
        self.assertTrue(st['last_switch']['lost']); self.assertEqual(st['last_switch']['source'], 'followed_missing')

    def test_frames_of_a_change_do_not_bounce_back(self):
        APP.args.det = FAR_DET
        self.turn_on()
        asked = self.feed(1.2, [person(4, 200)])
        self.assertEqual([w for _, w in asked], ['near'])
        self.assertEqual(self.feed(0.4, [], t0=1.2), [])   # the change is queued: no detector rows for a moment
        APP.eng = {auto_det.FAR_KEY: object()}; APP.det_names = [auto_det.FAR_KEY]; APP.a = None; APP.b = None
        APP.Engine = MagicMock(return_value='engine')
        self.assertEqual(self.frame_loop_boundary(), 0)
        self.assertEqual(APP.args.det, NEAR_DET)
        self.assertEqual(self.feed(0.4, [], t0=1.6), [])   # first frames of the new model, nobody yet (0.4 s < 0.5 s)
        self.assertEqual(self.feed(3, [person(4, 200)], t0=2.0), [])
        self.assertEqual(auto_det.state()['current'], 'near')
        self.assertEqual(auto_det.AUTO.ch.recent(5.0), 1)   # one switch, counted once

    def test_edge_and_keypoints_reach_the_decision(self):
        self.turn_on()
        r = person(7, 70)
        asked = self.feed(3, [r], kp_sets=(kp_set(r[:4], 8), None))   # small but hidden (8 keypoints): stays near
        self.assertEqual(asked, [])
        sg = auto_det.state()['signal']
        self.assertEqual((sg['kpts'], sg['hidden'], sg['occl']), (8, True, True))
        r = person(7, 70, x=0.0)
        self.assertEqual(self.feed(3, [r], t0=3.0), [])   # small but cut by the edge: stays near
        self.assertTrue(auto_det.state()['signal']['cut'])
        self.assertEqual([w for _, w in self.feed(3, [person(7, 70)], t0=6.0)], ['far'])   # whole and small: far
        json.dumps(auto_det.state()); json.dumps(auto_det.dry_log())

    def test_scene_median_when_nobody_is_followed(self):
        self.turn_on()
        APP.GB = SimpleNamespace(state=dict(auto=0, id=7), selection_reset=False)
        self.assertEqual(self.feed(3, [person(1, 220), person(2, 230), person(7, 80)]), [])
        self.assertEqual(auto_det.state()['signal']['source'], 'scene')

    def test_replay_is_dry_run_only(self):
        self.turn_on()
        APP.LIVE = SimpleNamespace(replay=dict(name='rec'), replay_busy=False)
        with patch.object(auto_det, '_replay_pos', side_effect=lambda: 123):
            asked = self.feed(8, [person(1, 70)])
        self.assertEqual(asked, [])
        APP.model_request.assert_not_called()
        self.assertIsNone(APP.MODEL['req'])
        st = auto_det.state()
        self.assertEqual((st['mode'], st['dry_events'], st['dry_choice']), ('dry_run', 1, 'far'))
        ev = st['dry_recent'][0]
        self.assertEqual((ev['frame'], ev['from'], ev['to'], ev['signal']), (123, 'near', 'far', 70.0))
        self.assertAlmostEqual(ev['t'], 1.0, delta=1.0 / FPS + 0.01)
        with patch.object(auto_det, '_replay_pos', side_effect=lambda: 400):   # the recording's person comes close: dwell from the dry switch, then near
            self.feed(8, [person(1, 300)], t0=8.0)
        self.assertEqual(auto_det.state()['dry_events'], 2)
        self.assertEqual(auto_det.state()['dry_recent'][-1]['to'], 'near')
        APP.model_request.assert_not_called()
        json.dumps(auto_det.state())

    def test_paused_replay_frames_do_not_count(self):
        self.turn_on()
        APP.LIVE = SimpleNamespace(replay=dict(name='rec'), replay_busy=False)
        self.feed(5, [person(1, 70)], repeat=True)
        self.assertEqual(auto_det.state()['dry_events'], 0)
        # 0.6 s small, paused 30 s (the same frame repeated), 0.3 s small: 0.9 s of recording, no switch yet
        self.feed(0.6, [person(1, 70)], t0=10.0); self.feed(30, [person(1, 70)], t0=11.0, repeat=True); self.feed(0.3, [person(1, 70)], t0=41.0)
        self.assertEqual(auto_det.state()['dry_events'], 0)
        self.feed(0.3, [person(1, 70)], t0=41.3)
        self.assertEqual(auto_det.state()['dry_events'], 1)
        # a seek (a jump in time without repeats) adds at most DRY_STEP_MAX
        self.feed(0.1, [person(1, 300)], t0=50.0)
        self.assertLess(auto_det.AUTO.dry_clock, 0.6 + 0.6 + 2 * auto_det.DRY_STEP_MAX + 0.2)

    def test_far_model_missing_stays_off(self):
        APP.MODEL['catalog'] = APP.MODEL['catalog'][:2]
        self.turn_on()   # accepted (a saved choice must survive a start before the model scan lists the files) but it waits
        self.assertEqual(self.feed(3, [person(1, 70)]), [])
        st = auto_det.state()
        self.assertEqual((st['mode'], st['active'], st['far']['present'], st['near']['present']), ('waiting', False, False, True))
        self.assertIn('먼 거리 모델 없음', st['reason'])
        APP.model_request.assert_not_called()

    def test_end_to_end_model_pauses(self):
        self.turn_on()
        with patch.object(auto_det, '_e2e_on', return_value=True):
            self.assertEqual(self.feed(3, [person(1, 70)]), [])
        self.assertEqual(auto_det.state()['mode'], 'paused')

    def test_failed_swap_waits_before_the_next_try(self):
        self.turn_on()
        self.feed(2, [person(4, 70)])
        APP.MODEL['req'] = None; APP.MODEL['last'] = dict(id=3, result=4, text='실패해서 이전 모델로 되돌림')   # the frame loop took it, the model stayed
        self.assertEqual(self.feed(10, [person(4, 70)], t0=2.0), [])
        st = auto_det.state()
        self.assertEqual((st['failures'], st['current']), (1, 'near'))
        self.assertIn('다시 시도', st['reason'])

    def test_manual_choice_turns_it_off(self):
        self.turn_on()
        self.assertTrue(auto_det.manual_choice('page'))
        self.assertFalse(auto_det.state()['on'])
        self.assertEqual(self.sel[-1], dict(auto_det=False))
        self.feed(0.5, [person(4, 70)])
        self.assertIn('손으로', auto_det.state()['reason'])   # the next frames keep why it went off (the page shows it)
        self.assertFalse(auto_det.manual_choice('page'))   # already off: nothing

    def test_parse_request_checks_everything_first(self):
        for q in ({'auto_det': ['2']}, {'auto_det_far_px': ['x']}, {'auto_det_far_px': ['150']}, {'auto_det_hold_s': ['-1']},
                  {'auto_det_nope': ['1']}, {'auto_det_far': ['a b']}, {'auto_det_far': [auto_det.NEAR_KEY]}):
            ok, text, apply = auto_det.parse_request(q)
            self.assertFalse(ok, q); self.assertIsNone(apply)
        self.assertEqual(auto_det.AUTO.params, auto_det.PARAMS)
        APP.MODEL['catalog'] = []
        self.assertFalse(auto_det.parse_request({'auto_det': ['1']})[0])   # this run cannot change its detector
        APP.MODEL['catalog'] = [dict(id=1, det='w480', key='NEWMRW480', name='x')]
        ok, _, apply = auto_det.parse_request({'auto_det': ['1'], 'auto_det_far_px': ['90'], 'auto_det_dwell_s': ['8']})
        self.assertTrue(ok); apply()
        self.assertEqual((auto_det.AUTO.params['far_px'], auto_det.AUTO.params['dwell_s'], auto_det.AUTO.ch.p['far_px']), (90.0, 8.0, 90.0))
        self.assertEqual(self.sel[-1], dict(auto_det=True, auto_det_p=dict(far_px=90.0, dwell_s=8.0)))
        q = auto_det.restore_query(True, self.sel[-1]['auto_det_p'])
        self.assertEqual(q, {'auto_det': ['1'], 'auto_det_far_px': ['90.0'], 'auto_det_dwell_s': ['8.0']})

    def test_rules_request_takes_auto_det_only_with_the_rest(self):
        APP.rules_request0_ = MagicMock(return_value=(False, 'bad'))
        ok, _ = page_settings.rules_request_2({'auto_det': ['1'], 'warm_min': ['x']})
        self.assertFalse(ok)
        self.assertFalse(auto_det.AUTO.on)                                      # the rest failed: nothing changed
        APP.rules_request0_ = MagicMock(return_value=(True, '적용됨'))
        ok, _ = page_settings.rules_request_2({'auto_det': ['1'], 'warm_min': ['0']})
        self.assertTrue(ok); self.assertTrue(auto_det.AUTO.on)
        APP.rules_request0_.assert_called_once_with({'warm_min': ['0']})
        ok, _ = page_settings.rules_request_2({'auto_det': ['0']})               # alone: the old parser is not called
        self.assertTrue(ok); self.assertFalse(auto_det.AUTO.on)
        self.assertEqual(APP.rules_request0_.call_count, 1)


if __name__ == '__main__':
    unittest.main()
