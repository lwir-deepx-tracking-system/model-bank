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


# ------------------------------------------------------------------------------------------------ 11-16 lost_same (box_refine switch, judge A = U0)
def person_frames(n_seen=8, n_lost=6, car=False):
    """detections per frame: one person (40 x 160) walking right, seen n_seen frames, then gone; optionally a car that goes at the same time"""
    for k in range(n_seen + n_lost):
        rows = [[280. + 2 * k, 136., 320. + 2 * k, 296., 0.9, 0.]] if k < n_seen else []
        if car and k < n_seen: rows.append([500., 300., 580., 340., 0.9, 2.])
        yield k, np.array(rows, np.float32).reshape(-1, 6)


def snapshot(tr):
    """everything of a tracker that association reads"""
    return (tr.frame_id, tr.next_id, [(t.track_id, t.state, t.cls, t.end_frame, np.array(t.mean).tobytes(), np.array(t.cov).tobytes()) for t in tr.tracked + tr.lost])


class LostSameMarksTests(unittest.TestCase):
    def setUp(self):
        lm.reset(); lm.LM['tracker'] = None
        self.posp = patch.dict(lm.POSP, show=True, boxes=[], boxes_t=0.0, key='kalman', model=None)
        self.posp.start(); self.addCleanup(self.posp.stop)

    def test_switch_off_is_the_same_call_as_before(self):
        out = {}
        for kw in ({}, dict(lost_now=None, convert=None)):
            lm.reset(); lm.LM['tracker'] = None; got = []
            for k in range(1, 100):
                lm.POSP['boxes'] = [entry(5, k, cx=300.0 + k)] if k <= 30 else []
                got.append(lm.update(100.0 + k / FPS, set(), 5, tracker=self, keepid=True, motion=lambda a, b: np.array([[1., 0, -2.], [0, 1., 0]]), fps=FPS, **kw))
            out[len(kw)] = got
        self.assertEqual(out[0], out[2])

    def test_pool_now_reads_the_trackers_without_touching_them(self):
        from ..track.botsort import BotSortTracker
        from ..track.bytetrack import ByteTrackTracker
        from ..track.kalman import GKF
        for show in (False, True):
            for mk in (lambda: ByteTrackTracker('bytetrack'), lambda: BotSortTracker('bytetrack'), lambda: ByteTrackTracker('rebind')):
                lm.POSP['show'] = show; tr = mk(); tr.kf = GKF(); steps = []
                for k, arr in person_frames(car=True):
                    tr.kf.warp = np.array([[1., 0, 1.], [0, 1., 0]]); tr.update(arr); before = snapshot(tr)
                    got = lm.pool_now(tr); lm.pool_now(tr)
                    self.assertEqual(snapshot(tr), before)                                         # read only
                    for e in got: e[1][:] = -1.0                                                  # the boxes are copies
                    self.assertEqual(snapshot(tr), before)
                    got = lm.pool_now(tr); steps.append([(e[0], e[2]['j']) for e in got])
                    for e in got:
                        t = next(x for x in tr.lost if x.track_id == e[0]); np.testing.assert_array_equal(e[1], t.mean[:4])
                        self.assertEqual(e[2]['n'], e[2]['j']); self.assertEqual(e[2]['src'], 'kalman'); self.assertEqual(e[2]['last'] is not None, show)
                self.assertEqual(steps[7], []); self.assertEqual(steps[8:], [[(1, j)] for j in range(1, 7)])   # the person from lost step 1; never the car
        self.assertIsNone(lm.pool_now(None)); self.assertIsNone(lm.pool_now(object()))

    def run_tracker(self, use_pool, convert=None, id_map=None):
        from ..track.bytetrack import ByteTrackTracker
        lm.reset(); lm.LM['tracker'] = None; tr = ByteTrackTracker('bytetrack'); out = []
        for k, arr in person_frames():
            rows = tr.update(arr)
            kw = dict(lost_now=lm.pool_now(tr), convert=convert) if use_pool else {}
            out.append(lm.update(100.0 + k / FPS, {int(r[4]) if id_map is None else id_map[int(r[4])] for r in rows}, 0, tracker=tr, id_map=id_map,
                                 keepid=id_map is not None, fresh=True, fps=FPS, **kw))
        return out

    def test_the_mark_is_drawn_from_the_first_lost_frame(self):
        """today the mark comes from POSP['boxes'] (made in the next frame's predict): nothing on the first lost frame. With the pool read after
        the update it is there on lost step 1, and from step 2 on it is the very same tracker box as today's."""
        old = self.run_tracker(False); new = self.run_tracker(True)
        self.assertEqual([k for k, e in enumerate(old) if e][:1], [9]); self.assertEqual([k for k, e in enumerate(new) if e][:1], [8])   # lost from frame 8
        for k in range(9, 14):
            self.assertEqual(old[k][0][1], new[k][0][1], k)
        self.assertEqual(new[8][0][2]['n'], 1)

    def test_convert_gets_output_ids_once_per_frame_and_its_boxes_are_drawn(self):
        calls = []
        def convert(items):
            calls.append([(oid, j) for oid, _, j in items]); return [np.asarray(b, float) + [3., -5., 0., 0.] for _, b, _ in items]
        new = self.run_tracker(True, convert=convert, id_map={1: 12})
        plain = self.run_tracker(True)
        self.assertEqual(calls, [[(12, j)] for j in range(1, 7)])
        for k in range(8, 14):
            oid, box, info = new[k][0]; self.assertEqual(oid, 12)
            np.testing.assert_allclose(box, np.asarray(plain[k][0][1]) + [3., -5., 0., 0.])
            self.assertEqual(info['pts'][-1], (box[0], box[1]))

    def test_a_failing_or_short_convert_draws_the_tracker_boxes(self):
        plain = self.run_tracker(True)
        def boom(items): raise RuntimeError('x')
        for conv in (boom, lambda items: [], lambda items: None):
            got = self.run_tracker(True, convert=conv)
            self.assertEqual([e[0][1] if e else None for e in got], [e[0][1] if e else None for e in plain])

    def test_page_handler_reads_the_pool_only_when_it_is_the_servers(self):
        """page_handler._lost_marks with lost_same on: the pool of APP.tracker (mark on lost step 1); the camera C++ ByteTrack's rows or the
        end-to-end model: today's path, with the reason for the page; switch off: today's path."""
        from ..common.state import APP
        from ..track import box_refine
        from ..track.bytetrack import ByteTrackTracker
        from . import page_handler as page
        had = {k: vars(APP)[k] for k in ('GB', 'tracker', 'relabeler', 'RULES') if k in vars(APP)}
        self.addCleanup(lambda: ([vars(APP).pop(k, None) for k in ('GB', 'tracker', 'relabeler', 'RULES')], vars(APP).update(had)))
        saved = dict(box_refine.BR); self.addCleanup(lambda: (box_refine.BR.clear(), box_refine.BR.update(saved)))
        for lost, own, e2e, want in ((True, True, False, True), (True, False, False, False), (True, True, True, False), (False, True, False, False)):
            box_refine.BR.update(lost_same=lost, lost_why=''); lm.reset(); lm.LM['tracker'] = None
            tr = ByteTrackTracker('bytetrack'); APP.tracker = tr; APP.GB = None; APP.relabeler = None; APP.RULES = {}; lm.POSP['boxes'] = []
            with patch.object(box_refine, '_e2e_on', return_value=e2e), patch.object(page, '_canvas_motion', return_value=None):
                for k, arr in person_frames(n_lost=1):
                    rows = tr.update(arr); marks = page._lost_marks(100.0 + k / FPS, rows, 0, FPS, own_pool=own)
            self.assertEqual(bool(marks), want, (lost, own, e2e))
            self.assertEqual(bool(box_refine.BR['lost_why']), lost and not want, (lost, own, e2e))

    def test_the_page_request_takes_the_switch(self):
        from ..common.state import APP
        from ..track import box_refine
        from . import page_settings
        saved = dict(box_refine.BR); self.addCleanup(lambda: (box_refine.BR.clear(), box_refine.BR.update(saved)))
        had = vars(APP).get('sel_save'); got = []; APP.sel_save = lambda **kw: got.append(kw)
        self.addCleanup(lambda: vars(APP).pop('sel_save', None) if had is None else setattr(APP, 'sel_save', had))
        with patch('builtins.print'):
            self.assertEqual(page_settings.rules_request_2({'lost_same': ['1']}), (True, '적용됨'))
            self.assertTrue(box_refine.BR['lost_same']); self.assertEqual(got[-1]['lost_same'], 1)
            self.assertFalse(page_settings.rules_request_2({'lost_same': ['maybe']})[0]); self.assertTrue(box_refine.BR['lost_same'])
            page_settings.rules_request_2({'lost_same': ['0']}); self.assertFalse(box_refine.BR['lost_same'])



if __name__ == '__main__':
    unittest.main()
