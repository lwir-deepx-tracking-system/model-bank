"""11-14: skin-temperature cue per person box (src/detect/skin_temp.py): synthetic temperature map, switch, server hook and label drawing."""
import contextlib
import io
import json
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from src.camera import temp_ring as tr
from src.common.state import APP
from src.detect import skin_temp as st
from src.web import page_handler as page
from src.web import preview_worker as pw
from src.web.test_preview_pipeline import PublishFixture

PERSON = (100, 100, 140, 260)    # real camera coordinates (x0, y0, x1, y1)
LAPTOP = (380, 280, 500, 370)
EMPTY = (520, 40, 620, 140)
FACE_C = 34.0


def _blur(c):
    """3x3 box blur twice (the camera's optics: hot edges pass through 30-38 C)"""
    for _ in range(2):
        p = np.pad(c, 1, mode='edge')
        c = sum(p[dy:dy + c.shape[0], dx:dx + c.shape[1]] for dy in range(3) for dx in range(3)) / 9.0
    return c.astype(np.float32)


def scene():
    """22 C room; a person (27 C clothes, 34 C face blob of ~300 px) and a 45 C laptop"""
    c = np.full((480, 640), 22.0, np.float32)
    c[130:260, 102:138] = 27.0
    yy, xx = np.mgrid[0:480, 0:640]
    c[((xx - 120) / 9.0) ** 2 + ((yy - 115) / 11.0) ** 2 <= 1.0] = FACE_C
    c[300:350, 400:480] = 45.0
    return _blur(c)


def temp_map(values, t_ns, frame=30, map_seq=10, ring_id=77):
    return tr.TempMap(frame, map_seq, ring_id, t_ns, 640, 480, 0.98, 1500, tr.FLAG_AGC_READ, 5000, 100, tr.STATE_ON, 3, values)


class FakeReader:
    """a new map (next map number) at every call, t_mono_ns age_s before now"""
    def __init__(self, values, age_s=0.05, ring_id=77):
        self.values, self.age_s, self.calls, self.ring_id = values, age_s, 0, ring_id

    def newest(self):
        self.calls += 1
        if self.values is None:
            return None
        return temp_map(self.values, time.monotonic_ns() - int(self.age_s * 1e9), frame=3 * self.calls, map_seq=self.calls, ring_id=self.ring_id)


class CueTests(unittest.TestCase):
    def setUp(self):
        self.map = tr.from_celsius(scene())

    def test_face_laptop_background(self):
        person, laptop, empty = st.cues(self.map, [PERSON, LAPTOP, EMPTY])
        self.assertTrue(person['skin'])
        self.assertGreaterEqual(person['spot_px'], 150)
        self.assertAlmostEqual(person['spot_c'], FACE_C, delta=0.3)
        self.assertGreater(person['share'], 0.02)
        self.assertGreaterEqual(person['score'], 0.5)
        self.assertGreater(person['top1'], 33.0)
        self.assertFalse(laptop['skin'])   # its blurred 30-38 C border is thin: opened away
        self.assertGreater(laptop['top1'], 44.0); self.assertGreater(laptop['p95'], 40.0)
        self.assertLess(laptop['score'], 0.5)
        self.assertFalse(empty['skin']); self.assertEqual(empty['share'], 0.0); self.assertEqual(empty['score'], 0.0)
        self.assertAlmostEqual(empty['top1'], 22.0, delta=0.05)

    def test_small_spot_is_not_skin_and_float_map(self):
        c = np.full((60, 60), 22.0, np.float32)
        c[20:24, 20:24] = 34.0   # 16 px: under SPOT_MIN_PX
        self.assertFalse(st.cues(c, [(0, 0, 60, 60)])[0]['skin'])
        c[20:25, 20:25] = 34.0   # 25 px
        cue = st.cues(c, [(0, 0, 60, 60)])[0]
        self.assertTrue(cue['skin']); self.assertEqual(cue['spot_px'], 25)

    def test_box_off_the_map_and_scaling(self):
        self.assertEqual(st.cues(self.map, [(700, 10, 760, 50), (10, 10, 11, 11)]), [None, None])
        half = self.map[::2, ::2]   # a map of half the picture size: boxes are scaled
        cue = st.cues(half, [PERSON], frame_wh=(640, 480))[0]
        self.assertTrue(cue['skin'])

    def test_numpy_opening_equals_opencv(self):
        rng = np.random.default_rng(1)
        masks = [rng.random((int(rng.integers(3, 40)), int(rng.integers(3, 40)))) < rng.random() for _ in range(100)]
        with_cv = [st._open3(m).copy() for m in masks]
        with patch.object(st, '_cv2', None):
            for m, a in zip(masks, with_cv):
                np.testing.assert_array_equal(st._open3(m), a)
            person, laptop = st.cues(self.map, [PERSON, LAPTOP])   # without OpenCV the whole opened mask is the spot
        self.assertTrue(person['skin']); self.assertFalse(laptop['skin'])

    def test_label(self):
        person, laptop = st.cues(self.map, [PERSON, LAPTOP])
        self.assertEqual(st.label(person), ('%.1fC' % person['spot_c'], st.SKIN_COLOR))
        self.assertEqual(st.label(laptop), ('%.1fC' % laptop['top1'], st.OTHER_COLOR))


class SwitchTests(unittest.TestCase):
    def setUp(self):
        saved = dict(st.TEMP); self.addCleanup(lambda: (st.TEMP.clear(), st.TEMP.update(saved)))
        self.addCleanup(st.set_show, False)
        self.reader = FakeReader(tr.from_celsius(scene()))

    def test_off_does_nothing(self):
        st.set_show(False)
        self.assertEqual(st.on_frame(1.0, [(7, *PERSON)], reader=self.reader), [])
        self.assertEqual(self.reader.calls, 0)
        self.assertEqual(st.preview_labels(1.0, 1.5), ())

    def test_on_fresh_stale_missing_and_blocked(self):
        st.set_show(True)
        out = st.on_frame(1.0, [(7, *PERSON), (8, *LAPTOP)], reader=self.reader)
        self.assertEqual([o[0] for o in out], [7, 8])
        self.assertTrue(out[0][2]['skin']); self.assertFalse(out[1][2]['skin'])
        labels = st.preview_labels(1.0, 1.5, (2.0, -4.0))   # picture = (real + c) * scale
        self.assertEqual(labels[0][:2], ((153, 144), (213, 384)))
        self.assertEqual(labels[0][3], st.SKIN_COLOR); self.assertTrue(labels[0][2].endswith('C'))
        self.assertEqual(len(st.preview_labels(1.0, 1.0, None, {8})), 1)   # only the drawn boxes
        self.assertEqual(st.preview_labels(2.0, 1.0), ())   # labels belong to their frame
        s = st.state()
        self.assertTrue(s['show']); self.assertTrue(s['fresh']); self.assertEqual(s['calls'], 1); self.assertEqual(s['skin_boxes'], 1)
        json.dumps(s)   # the page gets it as JSON (rules_state)
        self.assertIsNotNone(s['ms']); self.assertEqual((s['map_seq'], s['frame'], s['agc_read'], s['every']), (1, 3, True, 3)); self.assertAlmostEqual(s['median_c'], 22.0, delta=0.1); self.assertEqual(s['recent'][-1]['boxes'][0]['id'], 7)

        self.reader.age_s = 2.0   # old map (cam_pipe stopped or CalcTemp seldom): no labels
        self.assertEqual(st.on_frame(3.0, [(7, *PERSON)], reader=self.reader), [])
        self.assertEqual(st.preview_labels(3.0, 1.0), ()); self.assertEqual(st.state()['stale'], 1); self.assertFalse(st.state()['fresh'])
        self.reader.age_s = 0.1
        self.assertEqual(st.on_frame(4.0, [(7, *PERSON)], ok=False, reader=self.reader), [])   # recording / invalid frame
        self.assertEqual(st.state()['skipped'], 1)
        self.reader.values = None   # no ring (server started without --cam-temp)
        self.assertEqual(st.on_frame(5.0, [(7, *PERSON)], reader=self.reader), [])
        self.assertEqual(st.state()['no_map'], 1); self.assertIsNone(st.state()['age'])

    def test_implausible_map_is_not_used(self):   # e.g. CalcTemp after the AGC read returning picture values, not temperatures
        st.set_show(True)
        bad = FakeReader(np.full((480, 640), 30000, np.uint16), ring_id=5)   # (30000 - 5000) / 100 = 250 C everywhere
        self.assertEqual(st.on_frame(1.0, [(7, *PERSON)], reader=bad), [])
        s = st.state()
        self.assertEqual((s['implausible'], s['median_c'], s['maps']), (1, 250.0, 0))
        self.assertEqual(st.on_frame(2.0, [(7, *PERSON)], reader=self.reader)[0][0], 7)   # a plausible map is used again

    def test_errors_never_raise(self):
        st.set_show(True)
        bad = NS(newest=lambda: (_ for _ in ()).throw(OSError('boom')))
        self.assertEqual(st.on_frame(1.0, [(7, *PERSON)], reader=bad), [])
        self.assertEqual(st.state()['errors'], 1)

    def test_bad_label_and_summary_error_never_raise(self):   # 11-14 review: preview_labels runs in the frame loop, _summary in the worker's finally
        st.set_show(True)
        st.TEMP.update(labels=((7, (1.0, float('nan'), 3.0, 4.0), dict(skin=False, top1=30.0)),), labels_frame_t=1.0)
        errors = st.TEMP['errors']
        self.assertEqual(st.preview_labels(1.0, 1.5), ())
        self.assertEqual(st.TEMP['errors'], errors + 1)
        w = st._Worker.__new__(st._Worker); w._slot, w._ev, w._lock = None, threading.Event(), threading.Lock()
        with patch.object(st, '_pass', side_effect=[RuntimeError('summary'), None]) as p:
            t = threading.Thread(target=w._run, daemon=True); t.start()
            w.submit(([], True, None, (640, 480)))
            for _ in range(200):
                if p.call_count >= 1 and st.TEMP['errors'] == errors + 2: break
                time.sleep(0.01)
            w.submit(([], True, None, (640, 480)))
            for _ in range(200):
                if p.call_count >= 2: break
                time.sleep(0.01)
        self.assertEqual(p.call_count, 2)   # the thread survived the first error
        self.assertTrue(t.is_alive()); self.assertIn('summary', st.TEMP['last_error'])

    def test_summary_line(self):
        st.set_show(True)
        notes = []
        with patch('src.camera.camera_extra.note', side_effect=lambda who, t: notes.append(t)):
            st.on_frame(1.0, [(7, *PERSON)], reader=self.reader)
            st.TEMP['t_sum'] = time.monotonic() - st.SUMMARY_S - 1
            st.on_frame(2.0, [(7, *PERSON)], reader=self.reader)
        self.assertEqual(len(notes), 1); self.assertIn('TEMP 60 s: maps used 2, person boxes on them 2, with a skin spot 2', notes[0])

    def test_same_map_is_computed_once_per_id(self):
        st.set_show(True)
        fixed = FakeReader(tr.from_celsius(scene()))
        t_ns = time.monotonic_ns()
        fixed.newest = lambda: temp_map(fixed.values, t_ns)
        with patch.object(st, 'cues', wraps=st.cues) as spy:
            st.on_frame(1.0, [(7, *PERSON)], reader=fixed)
            moved = (7, PERSON[0] + 5, PERSON[1], PERSON[2] + 5, PERSON[3])
            out = st.on_frame(2.0, [moved, (8, *LAPTOP)], reader=fixed)   # same map: only id 8 is new
            self.assertEqual([len(c.args[1]) for c in spy.call_args_list], [1, 1])
        self.assertEqual(out[0][1], tuple(float(v) for v in moved[1:]))   # the label follows the box of this frame
        self.assertTrue(out[0][2]['skin']); self.assertEqual(st.state()['maps'], 1)
        t_ns += 1000   # a new map (same number here, another time): everybody again
        with patch.object(st, 'cues', wraps=st.cues) as spy:
            st.on_frame(3.0, [moved, (8, *LAPTOP)], reader=fixed)
            self.assertEqual([len(c.args[1]) for c in spy.call_args_list], [2])

    def test_background_thread(self):
        st.set_show(True)
        with patch.object(st, '_READER', [self.reader]):
            st.on_frame(1.0, [(7, *PERSON)])
            for _ in range(200):
                if st.TEMP['passes'] >= 1 and st.TEMP['map_t_ns'] is not None: break
                time.sleep(0.01)
            out = st.on_frame(2.0, [(7, *PERSON)])
        self.assertEqual(len(out), 1); self.assertTrue(out[0][2]['skin'])
        self.assertEqual(st.preview_labels(2.0, 1.0)[0][3], st.SKIN_COLOR)

    def test_rules_switch_is_not_saved(self):
        from src.web import page_settings
        saved_app = APP.__dict__.copy(); self.addCleanup(lambda: (APP.__dict__.clear(), APP.__dict__.update(saved_app)))
        kept = {}
        APP.RULES = dict(lock=threading.Lock(), pending=None, changes=0, last=None)
        APP.CTL = dict(app='off', lead=False, lead_ms=0, warm_prev=(0, 0))
        APP.sel_save = lambda **kw: kept.update(kw)
        APP.LIVE = None; APP.args = NS(live_port=0, relabel='')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(page_settings.rules_request({'temp_show': ['1']})[0], True)
            self.assertTrue(st.TEMP['show'])
            self.assertNotIn('temp_show', kept)   # not kept over a restart
            self.assertEqual(page_settings._temp_keys()['temp_show'], True)
            self.assertTrue(page_settings.rules_request({'temp_show': ['0']})[0])
        self.assertFalse(st.TEMP['show']); self.assertFalse(page_settings._temp_keys()['temp']['show'])


class ServerHookTests(PublishFixture):
    def setUp(self):
        super().setUp()
        saved = dict(st.TEMP); self.addCleanup(lambda: (st.TEMP.clear(), st.TEMP.update(saved)))
        self.addCleanup(st.set_show, False)
        self.stack.enter_context(patch.object(st, '_READER', [FakeReader(tr.from_celsius(scene()))]))
        self.stack.enter_context(patch.object(st, 'INLINE', [True]))   # the background pass in this thread (test_background_thread covers the thread)

    def frame_of(self, rows):
        with patch.object(page.preview_worker, 'render_preview', return_value=b'jpeg') as render:
            self.publish(rows)
        return render.call_args.args[1]

    def test_off_is_unchanged(self):
        frame = self.frame_of(np.array([[100, 116, 140, 276, 7, .9, 0]], np.float32))
        self.assertEqual(frame.temps, ())
        self.assertEqual(st._READER[0].calls, 0)

    def test_label_next_to_the_person_box(self):
        st.set_show(True)
        frame = self.frame_of(np.array([[100, 116, 140, 276, 7, .9, 0], [380, 296, 500, 386, 9, .8, 0], [0, 16, 30, 46, 3, .9, 2]], np.float32))
        self.assertEqual(len(frame.temps), 2)   # people only
        (p1, p2, text, col), lap = frame.temps
        self.assertEqual((p1, p2, col), ((100, 100), (140, 260), st.SKIN_COLOR))
        self.assertEqual(text, '%.1fC' % st.TEMP['labels'][0][2]['spot_c'])
        self.assertEqual(lap[3], st.OTHER_COLOR)

    def test_no_label_on_a_recording_frame(self):
        st.set_show(True)
        with patch.object(page.replay_link, 'frame_blocked', lambda _: True):
            frame = self.frame_of(np.array([[100, 116, 140, 276, 7, .9, 0]], np.float32))
        self.assertEqual(frame.temps, ()); self.assertEqual(st.TEMP['skipped'], 1)


class DrawTests(unittest.TestCase):
    def test_render_draws_label(self):
        import cv2
        base = dict(frame_t=0.0, gray=np.full((480, 640), 75, np.uint8), scale=1.0, boxes=(((100, 100), (140, 260), (255, 100, 30), 7),),
                    gt_boxes=(), predictions=(), top_text='', replay=False, gimbal_text=None, gimbal_color=(0, 0, 0))
        a = cv2.imdecode(np.frombuffer(pw.render_preview(cv2, pw.PreviewFrame(**base)), np.uint8), cv2.IMREAD_COLOR)
        b = cv2.imdecode(np.frombuffer(pw.render_preview(cv2, pw.PreviewFrame(**base, temps=(((100, 100), (140, 260), '34.2C', st.SKIN_COLOR),))), np.uint8), cv2.IMREAD_COLOR)
        right = b[98:120, 142:200].astype(int)   # right of the box's top corner
        self.assertTrue(((right[..., 1] - right[..., 2]) > 80).any())   # green text
        self.assertLess(right.mean(), a[98:120, 142:200].mean())   # dark backing
        np.testing.assert_array_equal(a[300:, :], b[300:, :])
        edge = pw.PreviewFrame(**dict(base, boxes=()), temps=(((600, 100), (639, 200), '30.0C', st.OTHER_COLOR),))   # no room on the right: left of the box
        c = cv2.imdecode(np.frombuffer(pw.render_preview(cv2, edge), np.uint8), cv2.IMREAD_COLOR)
        self.assertLess(c[98:120, 540:598].mean(), 60)


if __name__ == '__main__':
    unittest.main()
