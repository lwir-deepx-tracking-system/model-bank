"""PC-only shared live-track tests. No camera or serial port is opened."""
from collections import deque
from threading import Condition, Lock
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from ..track import bytetrack
from . import live_tracks, gimbal_link


class LiveTracksTests(unittest.TestCase):
    def setUp(self):
        self.saved = APP.__dict__.copy()
        APP.tracker = object()
        APP.LIVE = NS(camera_bytetrack=None)
        APP.LIVE_W, APP.LIVE_H = 640, 480
        APP.CTL = dict(lv=None, lead_px=0, lead=False)
        APP.GB = NS(wcond=Condition(), wopen=True, wq=deque(), wframe=None,
                    people=[], frame=0, gui=False, skipped=0, track_control_supported=True,
                    state=dict(id=0, auto=0))
        APP.gb_send_frame = gimbal_link.gb_send_frame
        APP.gb_send = gimbal_link.gb_send
        APP.lead_shift = gimbal_link.lead_shift
        self.people = [(17, 100., 50., 150., 200., .9)]

    def tearDown(self):
        APP.__dict__.clear()
        APP.__dict__.update(self.saved)

    def publish(self, source=None, **kwargs):
        acknowledge = kwargs.pop('acknowledge', True)
        selected = live_tracks.publish(self.people, kwargs.pop('correction', None),
                                   kwargs.pop('frame_t', time.perf_counter()),
                                   kwargs.pop('raw16', None),
                                   source or live_tracks.current_source(), **kwargs)
        if acknowledge:
            live_tracks.halted(APP.GB.tracks_ticket)
        return selected

    def test_python_box_and_id_unchanged(self):
        self.publish()
        self.assertEqual(APP.GB.people, self.people)
        self.assertIn(b' 17 100.0 50.0 50.0 150.0 0.900', APP.GB.wframe)
        self.assertTrue(live_tracks.click(120, 100)[0])
        self.assertEqual(APP.GB.wq[-1], b'A 17\n')
        self.assertTrue(APP.GB.wq[-2].startswith(b'F '))

    def test_old_bridge_requires_rebuild_instead_of_accepting_unacknowledged_click(self):
        self.publish()
        APP.GB.track_control_supported = False
        ok, message = live_tracks.click(120, 100)
        self.assertFalse(ok)
        self.assertIn('빌드', message)
        self.assertFalse(any(command.startswith(b'A ') for command in APP.GB.wq))

    def test_g_picture_precedes_selection_without_extra_copy(self):
        APP.GB.gui = True
        raw = np.full((480, 640), 413, np.uint16)
        self.publish(raw16=raw)
        frame = APP.GB.wframe
        self.assertEqual(frame.split(b'\n', 1)[1], raw.tobytes())
        live_tracks.click(120, 100)
        self.assertIs(APP.GB.wq[-2], frame)
        self.assertEqual(APP.GB.wq[-1], b'A 17\n')
        self.assertIsNone(APP.GB.wframe)

    def test_camera_stabilization_keeps_click_in_display_coordinates(self):
        self.publish(correction=(10., 20.))
        self.assertEqual(APP.GB.people, self.people)
        self.assertIn(b' 17 90.0 30.0 50.0 150.0 0.900', APP.GB.wframe)
        self.assertTrue(live_tracks.click(149, 199)[0])

    def test_cpp_source_replaces_python_for_click(self):
        self.publish()
        APP.LIVE.camera_bytetrack = NS(control_source=lambda: ('cpp', 1))
        self.people = [(3, 300., 100., 360., 240., .8)]
        self.publish()
        self.assertEqual(APP.GB.people[0][0], 3)
        self.assertTrue(live_tracks.click(320, 180)[0])
        self.assertEqual(APP.GB.wq[-1], b'A 3\n')
        self.assertFalse(any(b' 17 ' in msg for msg in APP.GB.wq))

    def test_switch_discards_old_queued_click_and_frame_even_if_full(self):
        self.publish()
        live_tracks.click(120, 100)
        APP.GB.wq.extend([b'A 17\n'] * 64)
        APP.GB.state.update(id=17, auto=1)
        APP.CTL.update(lv=('old',), lead_px=10)
        APP.LIVE.camera_bytetrack = NS(control_source=lambda: ('cpp', 2))
        self.people = [(17, 300., 100., 360., 240., .8)]
        self.assertEqual(self.publish(acknowledge=False), 0)
        self.assertEqual(list(APP.GB.wq), [('H %d\n' % APP.GB.tracks_ticket).encode()])
        self.assertIsNone(APP.CTL['lv'])
        self.assertTrue(APP.GB.selection_reset)

    def test_reset_same_id_new_generation_cannot_reuse_selection(self):
        native = NS(control_source=lambda: ('cpp', 1))
        APP.LIVE.camera_bytetrack = native
        self.publish()
        live_tracks.click(120, 100)
        native.control_source = lambda: ('cpp', 2)
        self.assertFalse(live_tracks.click(120, 100)[0])
        self.publish()
        self.assertEqual(list(APP.GB.wq), [('H %d\n' % APP.GB.tracks_ticket).encode()])

    def test_cpp_pending_blocks_old_python_snapshot(self):
        self.publish()
        APP.LIVE.camera_bytetrack = NS(control_source=lambda: None)
        self.assertFalse(live_tracks.click(120, 100)[0])

    def test_stale_and_invalid_clicks_refused(self):
        self.publish(frame_t=time.perf_counter()-0.6)
        self.assertFalse(live_tracks.click(120, 100)[0])
        self.publish()
        for x,y in ((float('nan'), 0), (0, float('inf')), (-1, 0), (641, 100)):
            self.assertFalse(live_tracks.click(x,y)[0])

    def test_empty_and_frozen_frames_keep_no_click_candidates(self):
        self.publish(blocked=True)
        self.assertEqual(APP.GB.people, [])
        self.assertEqual(len(APP.GB.wframe.split()), 4)
        live_tracks.click(120, 100)
        self.assertFalse(any(m.startswith(b'A ') for m in APP.GB.wq))

    def test_replay_source_cannot_be_selected_as_live(self):
        self.publish(source=('replay',), blocked=True)
        self.assertFalse(live_tracks.click(120, 100)[0])

    def test_smallest_overlapping_box_and_empty_click_stop(self):
        self.people += [(23, 110., 80., 140., 150., .8)]
        self.publish()
        live_tracks.click(120, 100)
        self.assertEqual(APP.GB.wq[-1], b'A 23\n')
        live_tracks.click(600, 450)
        self.assertTrue(APP.GB.wq[0].startswith(b'H '))
        self.assertFalse(any(m.startswith(b'A ') for m in APP.GB.wq))

    def test_input_gap_clears_pending_selection(self):
        self.publish()
        live_tracks.click(120, 100)
        self.publish(frame_t=time.perf_counter()+1.1)
        self.assertEqual(list(APP.GB.wq), [('H %d\n' % APP.GB.tracks_ticket).encode()])

    def test_queue_full_refuses_new_selection(self):
        self.publish()
        APP.GB.wq.extend([b'P\n']*64)
        self.assertFalse(live_tracks.click(120, 100)[0])
        self.assertIsNotNone(APP.GB.wframe)

    def test_stop_ack_required_before_new_selection(self):
        self.publish(acknowledge=False)
        APP.GB.state.update(id=17, auto=1)
        self.assertFalse(live_tracks.click(120,100)[0])
        live_tracks.halted(APP.GB.tracks_ticket-1)
        self.assertTrue(APP.GB.selection_reset)
        live_tracks.halted(APP.GB.tracks_ticket)
        self.assertEqual(APP.GB.state['id'],0)
        self.assertTrue(live_tracks.click(120,100)[0])

    def test_first_pending_source_initializes_without_crashing(self):
        APP.LIVE.camera_bytetrack = NS(control_source=lambda: None)
        self.publish(acknowledge=False)
        self.assertEqual(APP.GB.people,[])
        self.assertTrue(APP.GB.selection_reset)

    def test_pc_gui_selection_after_ack_uses_actual_bridge_id(self):
        self.publish()
        APP.GB.state.update(id=17,auto=1)
        self.assertEqual(self.publish(acknowledge=False),17)

    def test_repeated_empty_clicks_do_not_accumulate_stop(self):
        self.publish()
        for _ in range(100):
            live_tracks.click(600,450)
        self.assertEqual(sum(msg.startswith(b'H ') for msg in APP.GB.wq),1)

    def test_live_publish_draws_and_transports_native_rows_not_python(self):
        from ..web import page_handler
        now = time.perf_counter()
        native_rows = [[300., 116., 360., 256., 3, .8, 0, 0]]
        APP.LIVE.camera_bytetrack = NS(step=lambda *args: native_rows,
                                      control_source=lambda: ('cpp', 1))
        for key, value in dict(arr=deque([now]), raws=deque(), rawq=deque(), stabq=deque(),
                               done_ts=deque(), meas=None, cam_fps=30.,
                               perf_acc=[now, 0, 0, 0, 0], viewers=1, pic_t=0.,
                               snap_t=0., pic_n=0, jcond=Condition(), seq=0,
                               dlock=Lock(), delays=deque(), t0=time.monotonic(),
                               cam_lost=False, stopped=False).items():
            setattr(APP.LIVE, key, value)
        APP.args = NS(live_scale=1, live_seconds=1000, live_wait=0, mamba=False)
        APP.relabeler = None
        APP.rows = []
        APP.GB.state.update(running=True, error=0, lost=0, moving=0)
        drawn = []
        APP.cv2 = NS(COLOR_GRAY2BGR=0, FONT_HERSHEY_SIMPLEX=0, IMWRITE_JPEG_QUALITY=0,
                     cvtColor=lambda img, _: np.zeros((480,640,3),np.uint8),
                     rectangle=lambda *args: None,
                     putText=lambda img, text, *args: drawn.append(text),
                     imencode=lambda *args: (True,np.array([1],np.uint8)))
        with patch.object(page_handler.replay_link, 'frame_blocked', return_value=False), \
             patch.object(page_handler.replay_eval, 'is_repeat', return_value=False), \
             patch.object(page_handler.replay_eval, 'on_frame', return_value=(None,None)):
            page_handler.live_publish(np.zeros((512,640),np.uint8),
                                      [[100.,66.,150.,216.,17,.9,0,0]],
                                      dict(frame_total_ms=20), tracker_detections=[])
        self.assertIn('3', drawn)
        self.assertNotIn('17', drawn)
        self.assertEqual(APP.GB.people, [(3,300.,100.,360.,240.,.8)])
        self.assertIn(b' 3 300.0 100.0 60.0 140.0 0.800',APP.GB.wframe)
        live_tracks.halted(APP.GB.tracks_ticket)
        self.assertTrue(live_tracks.click(320,180)[0])
        self.assertEqual(APP.GB.wq[-1],b'A 3\n')


class NativeGenerationTests(unittest.TestCase):
    def test_normal_empty_and_reset_or_failure_are_distinct(self):
        native = NS(update=lambda rows: [], close=lambda: None)
        with patch.object(bytetrack, 'available_bytetrack_library', return_value='fake'), \
             patch.object(bytetrack, 'NativeByteTrack', return_value=native):
            tracker = bytetrack.CameraByteTrack(NS())
            self.assertTrue(tracker.select(True))
            self.assertTrue(tracker.blocks_selection())
            self.assertEqual(tracker.step([], 1., True), [])
            self.assertEqual(tracker.control_source(), ('cpp', 1))
            self.assertFalse(tracker.blocks_selection())
            self.assertEqual(tracker.step([], 2.1, True), [])
            self.assertEqual(tracker.control_source(), ('cpp', 2))
            native.update = lambda rows: (_ for _ in ()).throw(RuntimeError('bad ABI'))
            self.assertIsNone(tracker.step([], 2.2, True))
            self.assertEqual(tracker.control_source(), ('python',))
            self.assertIn('bad ABI', tracker.state()['error'])


if __name__ == '__main__':
    unittest.main()
