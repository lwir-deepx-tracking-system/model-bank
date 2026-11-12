"""No-device tests for the live-only observation contract and STOP barrier."""
from collections import deque
from threading import Condition
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from . import live_tracks, gimbal_link
from . import test_live_tracks as fixtures


class ObservationTransportTests(unittest.TestCase):
    def setUp(self):
        self.saved = APP.__dict__.copy()
        APP.tracker = object()
        APP.LIVE = NS(camera_bytetrack=None)
        APP.LIVE_W, APP.LIVE_H = 640, 480
        APP.CTL = dict(lv=None, lead_px=0, lead=False, lead_ms=150, lead_motion=None)
        APP.GB = NS(wcond=Condition(), wopen=True, wq=deque(), wframe=None,
                    people=[], frame=0, gui=False, skipped=0, track_control_supported=True,
                    observation_status_supported=True, state=dict(id=17, auto=1))
        APP.gb_send_frame = gimbal_link.gb_send_frame
        self.people = [(17, 100., 50., 150., 200., .9)]
        self.t = time.perf_counter() - .12
        self.send()
        live_tracks.halted(APP.GB.tracks_ticket)
        APP.GB.wq.clear()
        APP.GB.observation_barrier = None

    def tearDown(self):
        APP.__dict__.clear()
        APP.__dict__.update(self.saved)

    def send(self, *, people=None, stamp=None, healthy=True, blocked=False, raw=None):
        self.t += .001
        return live_tracks.publish(self.people if people is None else people, None,
                                  self.t if stamp is None else stamp, raw,
                                  live_tracks.current_source(), blocked,
                                  healthy_live=healthy)

    def test_valid_observed_and_empty_F_G_are_healthy(self):
        for gui in (False, True):
            APP.GB.gui = gui
            for rows in (self.people, []):
                self.send(people=rows, raw=np.zeros((480,640),np.uint16))
                header = APP.GB.wframe.split(b'\n')[0]
                self.assertTrue(header.startswith(b'G ' if gui else b'F '))
                self.assertTrue(header.endswith(b' OBS 1'))
                self.assertEqual(APP.GB.people, rows)

    def test_unhealthy_and_pending_generation_cannot_offer_rows(self):
        self.send(healthy=False)
        self.assertEqual(APP.GB.people, [])
        self.assertTrue(APP.GB.wframe.endswith(b' 0 OBS 0\n'))
        self.assertIsNotNone(APP.GB.observation_barrier)
        APP.LIVE.camera_bytetrack = NS(control_source=lambda: ('cpp', 2))
        self.send()
        self.assertTrue(APP.GB.selection_reset)
        self.assertTrue(APP.GB.wframe.endswith(b' 0 OBS 0\n'))

    def test_invalid_timestamp_cannot_crash_or_authorize_motion(self):
        for stamp in (float('nan'), float('inf'), -1, 0, 1e30,
                      time.perf_counter()+1, time.perf_counter()-.5):
            with self.subTest(stamp=stamp):
                self.send(stamp=stamp)
                self.assertEqual(APP.GB.people, [])
                self.assertTrue(APP.GB.wframe.endswith(b' 0 OBS 0\n'))
                self.assertFalse(live_tracks.click(120,100)[0])

    def test_duplicate_or_backwards_capture_is_not_healthy(self):
        self.send()
        stamp = self.t
        self.send(stamp=stamp)
        self.assertTrue(APP.GB.wframe.endswith(b' 0 OBS 0\n'))
        live_tracks.halted(APP.GB.tracks_ticket)
        self.send()
        self.send(stamp=self.t-.01)
        self.assertTrue(APP.GB.wframe.endswith(b' 0 OBS 0\n'))

    def test_stop_barrier_survives_newer_frame_and_is_written_first(self):
        self.send()
        self.send(healthy=False)
        unsafe = APP.GB.observation_barrier
        self.send()
        good = APP.GB.wframe
        self.assertTrue(good.endswith(b' OBS 1\n'))
        sent = []
        p = NS(stdin=NS(write=sent.append, flush=lambda: setattr(APP.GB, 'wopen', False)))
        APP.GB.proc = p
        gimbal_link.gb_writer(p)
        self.assertEqual(sent, [unsafe, good])
        self.assertIsNone(APP.GB.observation_barrier)

    def test_lead_removal_preserves_status_and_image(self):
        frame = b'G 8 123 640 480 0 OBS 0 LEAD 17 1 2 3\nimage'
        self.assertEqual(gimbal_link.without_lead(frame), b'G 8 123 640 480 0 OBS 0\nimage')

    def test_default_unknown_cannot_opt_into_grace(self):
        live_tracks.publish(self.people, None, self.t+.002, None, live_tracks.current_source())
        self.assertTrue(APP.GB.wframe.endswith(b' 0 OBS 0\n'))


class ObservationIngressTests(unittest.TestCase):
    setUp = fixtures.LiveTracksTests.setUp
    tearDown = fixtures.LiveTracksTests.tearDown
    publish = fixtures.LiveTracksTests.publish
    # Reuse the integration fixture, but run just the public live_publish path
    # with real-format raw metadata and a negotiated status-capable bridge.
    def test_ingress_status_for_pairing_frozen_replay_and_camera_loss(self):
        from ..web import page_handler
        original = live_tracks.publish
        observed = []
        def capture(*args, **kwargs):
            observed.append(kwargs['healthy_live'])
            return original(*args, **kwargs)
        # The inherited fixture explicitly has no rawq: this must stay OBS0.
        with patch.object(live_tracks, 'publish', side_effect=capture):
            fixtures.LiveTracksTests.test_live_publish_draws_and_transports_native_rows_not_python(self)
        self.assertEqual(observed, [False])
        def publish_case(*, raw=True, arrival=True, repeated=False, blocked=False,
                         detections=True, flag=None, raw_value=None, stamp=None):
            now = time.perf_counter()
            APP.LIVE.arr = deque([now if stamp is None else stamp] if arrival else [])
            APP.LIVE.rawq = deque([np.full((480,640),len(observed),np.uint16)
                                   if raw_value is None else raw_value] if raw else [])
            if flag: setattr(APP.LIVE, flag, True)
            with patch.object(live_tracks, 'publish', side_effect=capture), \
                 patch.object(page_handler.replay_link, 'frame_blocked', return_value=blocked), \
                 patch.object(page_handler.replay_eval, 'is_repeat', return_value=repeated), \
                 patch.object(page_handler.replay_eval, 'on_frame', return_value=(None,None)):
                page_handler.live_publish(np.zeros((512,640),np.uint8), [],
                                          dict(frame_total_ms=20),
                                          tracker_detections=[] if detections else None)
            if flag: setattr(APP.LIVE, flag, False)
            return observed[-1]
        self.assertTrue(publish_case())
        for options in (dict(raw=False), dict(arrival=False), dict(repeated=True),
                        dict(blocked=True), dict(detections=False), dict(flag='cam_lost'),
                        dict(flag='stopped'), dict(flag='restart_stop'), dict(flag='cam_hold'),
                        dict(flag='restart_pending'), dict(stamp=float('nan')),
                        dict(stamp=float('inf')), dict(stamp='broken'),
                        dict(raw_value=np.zeros(10,np.uint16)),
                        dict(raw_value=np.zeros((480,640),np.uint8))):
            with self.subTest(options=options):
                self.assertFalse(publish_case(**options))
        # Exact same RAW contents under a new arrival timestamp are frozen.
        frozen_raw = APP.LIVE.frozen_prev.copy()
        raw = np.repeat(np.repeat(frozen_raw,4,axis=0),4,axis=1)
        self.assertTrue(page_handler.frozen_check(raw))


if __name__ == '__main__':
    unittest.main()
