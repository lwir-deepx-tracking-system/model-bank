"""PC tests of 'picture first' (picture_first.py): picture at frame arrival, boxes as /overlay events, clicks on the drawn boxes.

python -m unittest src.web.test_picture_first   (from tracking_server/)
"""
import json
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from . import page_handler as page
from . import picture_first as pf
from .test_preview_pipeline import PublishFixture


class PictureFirstTests(PublishFixture):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.dict(pf.PF, on=True))
        self.live.lo, self.live.hi = 1000.0, 2000.0
        self.live.replay = None

    def test_picture_is_made_at_frame_arrival_without_touching_the_click_boxes(self):
        self.enable(True)
        self.live.jpeg_people, self.live.jpeg_tracks_ticket = 'boxes', 3
        seq0 = self.live.seq
        raw = np.full((480, 640), 1500, np.uint16)
        raw[100:200, 100:200] = 2000
        t = time.perf_counter()
        self.assertTrue(self.live.picture_worker.offer(t, raw))
        with self.live.jcond:
            self.assertTrue(self.live.jcond.wait_for(lambda: self.live.seq > seq0, timeout=3))
        self.assertLess(time.perf_counter() - t, 0.5)
        image = cv2.imdecode(np.frombuffer(self.live.jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(image.shape, (480, 640))   # a gray picture of the frame itself, no boxes
        self.assertAlmostEqual(int(image[300, 300]), 127, delta=3)
        self.assertGreater(int(image[150, 150]), 250)
        self.assertEqual(self.live.jpeg_frame_t, t)
        self.assertEqual((self.live.jpeg_people, self.live.jpeg_tracks_ticket), ('boxes', 3))
        self.assertIs(self.live.jpeg_drawn[0], self.live.jpeg)   # the lighter picture is made from it too

    def test_no_picture_while_off_during_replay_or_with_drawn_extras(self):
        self.enable(True)
        raw = np.zeros((480, 640), np.uint16)
        for change in (dict(on=False),):
            with patch.dict(pf.PF, change):
                self.assertFalse(self.live.picture_worker.offer(time.perf_counter(), raw))
        self.live.replay = {'name': 'x'}
        self.assertFalse(self.live.picture_worker.offer(time.perf_counter(), raw))
        self.live.replay = None
        with patch.dict(page.POSP, show=True):   # predicted boxes go in the overlay: no reason to stop
            self.assertTrue(pf.active(self.live))
        for target in (pf.spec_proposals.PROP, pf.skin_temp.TEMP):
            with patch.dict(target, show=True):
                self.assertFalse(pf.active(self.live))
        with patch.dict(pf.STAB, on=True):
            self.assertFalse(pf.active(self.live))
        self.assertTrue(pf.active(self.live))

    def test_tracking_result_goes_out_as_boxes_not_as_a_picture(self):
        self.enable(True)
        seq0 = self.live.seq
        stamp = self.publish()
        time.sleep(0.2)
        self.assertEqual(self.live.seq, seq0)   # no drawn picture: the picture came at frame arrival
        msg = json.loads(self.live.overlay)
        self.assertEqual(msg['pf'], 1)
        self.assertEqual(msg['b'], [[100, 100, 140, 260, 7, 0]])   # picture pixels (the canvas rows are 16 lower)
        self.assertEqual(self.live.click_frame_t, stamp)

    def test_predicted_boxes_go_in_the_overlay(self):
        self.enable(True)
        info = dict(src='kalman', n=5, last=(120.0, 216.0, 40.0, 160.0), pts=((130.0, 220.0),), r=20.0)
        with patch.dict(page.POSP, show=True, boxes_t=time.monotonic(), boxes=[(42, (300.0, 216.0, 40.0, 160.0), info)]):
            APP_GB = page.APP.GB
            with patch.object(APP_GB, 'state', dict(APP_GB.state)):
                with patch.object(page.live_tracks, 'publish', return_value=42):
                    self.publish()
        msg = json.loads(self.live.overlay)
        self.assertEqual(len(msg['p']), 1)
        x1, y1, x2, y2, tid, lab, last, pts, rad, col, alpha = msg['p'][0]
        self.assertAlmostEqual(alpha, 0.9, delta=0.03)   # 11-16: fades while it stays lost (lost_marks): 5 frames of a 1 s wait (no --reacquire-ms here)
        self.assertEqual((x1, y1, x2, y2, tid, rad), (280, 120, 320, 280, 42, 20))
        self.assertEqual((last, pts, lab, col), ([120, 200, 40, 160], [[300, 200]], '42 Kalman 5f', 'rgb(255,220,0)'))   # 11-16: the path is kept by lost_marks from the frame the person is first drawn as lost

    def test_marks_of_the_followed_person_outlive_the_tracker_memory_and_end_when_seen_again(self):
        """11-16 (user: the marks vanish too quickly): the tracker forgets a lost track after 30 frames; the page keeps the
        marks while the gimbal still waits for the number, and drops them in the frame the number has a box again."""
        self.enable(True)
        page.lost_marks.reset()
        info = dict(src='kalman', n=30, last=(120.0, 216.0, 40.0, 160.0), pts=(), r=320.0)
        APP_GB = page.APP.GB
        with patch.object(APP_GB, 'state', dict(APP_GB.state)), patch.object(page.live_tracks, 'publish', return_value=42), \
                patch.object(page.live_tracks, '_bridge_option', return_value='10000'):
            with patch.dict(page.POSP, show=True, boxes_t=time.monotonic(), boxes=[(42, (300.0, 216.0, 40.0, 160.0), info)]):
                self.publish()
            with patch.dict(page.POSP, show=True, boxes_t=time.monotonic(), boxes=[]):   # the tracker dropped the lost track
                self.publish()
                msg = json.loads(self.live.overlay)
                self.assertEqual(len(msg['p']), 1)
                self.assertEqual(msg['p'][0][:5], [280, 120, 320, 280, 42])
                self.assertTrue(msg['p'][0][5].startswith('42 lost 1.'), msg['p'][0][5])
                self.assertEqual(msg['p'][0][8], int(160 * (0.5 + 0.05 * 31)))   # the circle keeps growing
                self.publish(np.array([[100, 116, 140, 276, 42, .9, 0]], np.float32))   # number 42 has a box again
                self.assertEqual(json.loads(self.live.overlay)['p'], [])
        page.lost_marks.reset()

    def test_the_gimbals_search_goes_in_the_overlay_only_while_following(self):
        """11-16 (bridge --seek 1): /gimbal/people state 'seek' (search_follow) is sent as 's' so the page can draw
        '찾는 중' and the span of the sweep; nothing without it or when nobody is followed."""
        self.enable(True)
        APP_GB = page.APP.GB
        with patch.object(APP_GB, 'state', dict(APP_GB.state, seek=[2, 400.4, 240.0, 612.0, 211.0, -80.0, 211.0])):
            with patch.object(page.live_tracks, 'publish', return_value=42):
                self.publish()
            self.assertEqual(json.loads(self.live.overlay)['s'], [2, 400, 240, 612, 211, -80, 211])
            with patch.object(page.live_tracks, 'publish', return_value=0):
                self.publish()
            self.assertNotIn('s', json.loads(self.live.overlay))
        with patch.object(page.live_tracks, 'publish', return_value=42):
            self.publish()
        self.assertNotIn('s', json.loads(self.live.overlay))

    def test_the_drawn_picture_shows_the_search_too(self):
        from .preview_worker import PreviewFrame, render_preview
        gray = np.full((480, 640), 75, np.uint8)
        base = PreviewFrame(time.perf_counter(), gray, 1., (), (), (), 'x', False, None, (255, 255, 255))
        plain = cv2.imdecode(np.frombuffer(render_preview(cv2, base), np.uint8), cv2.IMREAD_COLOR)
        import dataclasses
        drawn = cv2.imdecode(np.frombuffer(render_preview(cv2, dataclasses.replace(base, seek=(2, 400., 240., 150., 150.))), np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int(np.abs(drawn.astype(int) - plain.astype(int)).sum()), 10000)   # the span, its ends, the target dot, 'SEARCH'
        self.assertGreater(int(drawn[240, 548, 2]), 180)   # the yellow target dot at centre + 150 (a little left of it the end tick)
        waiting = render_preview(cv2, dataclasses.replace(base, seek=(4, 400., 240., 150., None)))
        self.assertEqual(waiting, render_preview(cv2, base))   # waiting: nothing drawn

    def test_boxes_are_the_smoothed_rows_sent_to_the_gimbal(self):
        """11-15 box_refine: with picture first the page draws the same (smoothed) boxes the gimbal gets, not the tracker rows."""
        self.enable(True)
        shifted = lambda rows, dets, t: [[float(r[0]) + 2, float(r[1]) + 4, float(r[2]) + 2, float(r[3]) - 4] + list(r[4:]) for r in rows]
        with patch.object(page.box_refine, 'apply', side_effect=shifted):
            self.publish()
        msg = json.loads(self.live.overlay)
        self.assertEqual(msg['b'], [[102, 104, 142, 256, 7, 0]])
        self.assertEqual([tuple(round(v, 1) for v in p[1:5]) for p in self.controls[-1]], [(102.0, 104.0, 142.0, 256.0)])

    def test_click_uses_the_people_of_the_drawn_boxes(self):
        self.enable(True)
        with self.live.jcond:
            pf.publish_overlay_locked(self.live, b'{"pf":1}', people=((7, 100., 100., 140., 260.),), ticket=4,
                                      frame_t=time.perf_counter())
        self.live.jpeg_people, self.live.jpeg_tracks_ticket = ((9, 0., 0., 10., 10.),), 1   # not these
        with patch.object(page.live_tracks, 'click', return_value=(True, '')) as click:
            self.assertTrue(page._click_live(110., 130.)[0])
            click.assert_called_once_with(110., 130., expected_id=7, expected_ticket=4)
        with self.live.jcond:
            pf.publish_overlay_locked(self.live, b'{"pf":1}', people=((7, 100., 100., 140., 260.),), ticket=4,
                                      frame_t=time.perf_counter() - 2)   # boxes 2 s old: no click
        with patch.object(page.live_tracks, 'click') as click:
            self.assertFalse(page._click_live(110., 130.)[0])
            click.assert_not_called()

    def test_overlay_events(self):
        out = []
        live = self.live
        with live.jcond:
            pf.publish_overlay_locked(live, pf.overlay_message(time.perf_counter(), [(1.4, 2.6, 30, 40, 5, 1)], 5,
                                                               time.perf_counter(), 1000.0))
        live.stopped = False

        def write(b):
            out.append(b)
            if len(out) >= 2:
                live.stopped = True
        with patch.object(pf, 'json', json):
            pf.serve_overlay(live, write, keepalive=0.05)
        self.assertTrue(out[0].startswith(b'data: {') and out[0].endswith(b'\n\n'))
        msg = json.loads(out[0][6:])
        self.assertEqual((msg['b'], msg['f'], msg['w'], msg['h']), ([[1, 3, 30, 40, 5, 1]], 5, 640, 480))
        self.assertAlmostEqual(msg['t'], 1000.0, delta=0.05)
        self.assertEqual(out[1], b': keepalive\n\n')
        live.stopped = False
        with live.jcond:
            pf.publish_overlay_locked(live, None)   # switched off: the page clears its boxes
        self.assertEqual(live.overlay, b'{"pf":0}')


if __name__ == '__main__':
    unittest.main()
