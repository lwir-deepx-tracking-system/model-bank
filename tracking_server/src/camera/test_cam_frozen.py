"""11-15: the server's side of a frozen camera (cam_pipe v2.12 exit 8): cam_frozen.py and its hooks in camera_link.py / camera_extra.py. No camera, no board.

  - USB place: behind the board's switchable hub (bus 1 / 2 behind a hub port) or not, read from a folder laid out like /sys/bus/usb/devices;
  - lv_cam_wait: after exit 8 no new cam_pipe until the camera's USB entry changes (replug) or 60 s pass; later failed tries keep the state; other exits as before;
  - the frame loop waits 2 s instead of 10 s while frozen (so a replug is seen within a few seconds);
  - a camera frame clears the state (both readers), a replay frame does not;
  - the reset button: never usbreset; the 5 V step only behind the hub (or, off the bus, when last seen there); else the replug notice.
Run from the tracking_server folder: python -m unittest src.camera.test_cam_frozen
"""
from collections import deque
import io
import os
import shutil
import struct
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2

from ..common.state import APP
from . import cam_frozen, camera_extra, camera_link

FROZEN_LINE = ("CAMERA_FROZEN at a stall after 14964 frames (control libusb_timeout, ResetMainBoard refused): the camera's USB chip does not answer; "
               "only a power cut (unplug ~10 s, or the 5 V step on a switchable port) clears it, at 20:33:22\n")


class SysfsCase(unittest.TestCase):
    """A temporary folder laid out like /sys/bus/usb/devices, with cam_frozen.SYSFS pointed at it."""
    def setUp(self):
        self.sys = tempfile.mkdtemp(prefix='usb_sysfs_')
        self.addCleanup(shutil.rmtree, self.sys, True)
        p_ = patch.object(cam_frozen, 'SYSFS', self.sys); p_.start(); self.addCleanup(p_.stop)
        for name_, vid_, pid_, bus_, dev_ in (('usb1', '1d6b', '0002', 1, 1), ('usb3', '1d6b', '0002', 3, 1), ('1-1', '05e3', '0610', 1, 8), ('1-1.3', '10c4', 'ea60', 1, 9)):
            self.dev(name_, vid_, pid_, bus_, dev_)
        os.makedirs(os.path.join(self.sys, 'iface_1-1.3_1.0'))   # an interface folder: no idVendor (the real name 1-1.3:1.0 is not allowed on Windows)

    def dev(self, name_, vid_, pid_, bus_, dev_):
        d_ = os.path.join(self.sys, name_); os.makedirs(d_, exist_ok=True)
        for k_, v_ in (('idVendor', vid_), ('idProduct', pid_), ('busnum', bus_), ('devnum', dev_)):
            with open(os.path.join(d_, k_), 'w') as f_: f_.write('%s\n' % v_)

    def cam(self, name_='3-1', bus_=3, dev_=6): self.dev(name_, '0547', '0080', bus_, dev_)

    def unplug(self, name_='3-1'): shutil.rmtree(os.path.join(self.sys, name_))


class Place(SysfsCase):
    def test_root_port_hub_port_and_absent(self):
        self.assertEqual(cam_frozen.usb_cams(), [])
        self.assertIsNone(cam_frozen.where(NS()))
        self.cam('3-1', 3, 6)
        live_ = NS()
        self.assertEqual(cam_frozen.usb_cams(), [('3-1', 3, 6)])
        self.assertEqual(cam_frozen.where(live_), 'root'); self.assertEqual(cam_frozen.last_place(live_), 'root')
        self.unplug('3-1'); self.cam('1-1.2', 1, 12)
        self.assertEqual(cam_frozen.where(live_), 'hub')
        self.unplug('1-1.2')
        self.assertIsNone(cam_frozen.where(live_)); self.assertEqual(cam_frozen.last_place(live_), 'hub')   # off the bus: the last place is kept

    def test_place_rule(self):
        for name_, bus_, want_ in (('3-1', 3, 'root'), ('4-1', 4, 'root'), ('1-1.2', 1, 'hub'), ('2-1.4', 2, 'hub'), ('1-1', 1, 'root'), ('5-1.1', 5, 'root')):
            self.assertEqual(cam_frozen.place(name_, bus_), want_, name_)

    def test_bus_that_cannot_be_read(self):
        with patch.object(cam_frozen, 'SYSFS', os.path.join(self.sys, 'missing')):
            self.assertIsNone(cam_frozen.usb_cams())
        empty_ = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, empty_, True)
        self.assertIsNone(cam_frozen.usb_cams(empty_))   # no readable device at all = cannot tell (not "camera gone")

    def test_reason_from_log(self):
        d_ = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d_, True); p_ = os.path.join(d_, 'cam_pipe.log')
        with open(p_, 'w') as f_: f_.write('CAMERA 640 480\n' + FROZEN_LINE + 'FRAMES 14964 in 547.98 s\n')
        self.assertEqual(cam_frozen.reason_from_log(p_), 'at a stall after 14964 frames (control libusb_timeout, ResetMainBoard refused)')
        self.assertEqual(cam_frozen.reason_from_log(os.path.join(d_, 'none.log')), '')
        self.assertEqual(cam_frozen.reason_from_log(None), '')

    def test_texts(self):
        live_ = NS(cam_frozen=dict(t=1.0, at='20:33', reason='r', ref=[], hub=False, tries=0))
        s_ = cam_frozen.state(live_)
        self.assertEqual(s_['title'] + '. ' + s_['text'].split(' (')[0], '카메라가 멈췄습니다 (USB 칩 응답 없음). 카메라 USB 선을 10초 뽑았다 다시 꽂아 주세요.')
        self.assertNotIn('초기화 버튼', s_['text'])
        live_.cam_frozen['hub'] = True
        self.assertIn('또는 카메라 초기화 버튼', cam_frozen.state(live_)['text'])
        self.assertIsNone(cam_frozen.state(NS())); self.assertIsNone(cam_frozen.state(None))


class LiveCase(SysfsCase):
    """APP.LIVE as lv_cam_wait sees it after cam_pipe ended. APP.lv_spawn records the starts (and notes the USB entry, as lv_spawn does)."""
    def setUp(self):
        super().setUp()
        saved_ = APP.__dict__.copy()
        self.addCleanup(lambda: (APP.__dict__.clear(), APP.__dict__.update(saved_)))
        self.logdir = tempfile.mkdtemp(prefix='cam_log_'); self.addCleanup(shutil.rmtree, self.logdir, True)
        self.log = open(os.path.join(self.logdir, 'cam_pipe.log'), 'w'); self.addCleanup(self.log.close)
        now_ = time.monotonic()
        self.live = NS(t0=now_ - 1000, last_frame_t=now_ - 300, proc_t0=now_ - 100, cam_hold=False, cam_lost=True, replay=None, cam_next_spawn=None,
                       cam_fail_n=0, cam_unrec_n=0, log=self.log, stopped=False, jcond=threading.Condition(), cam_lock=threading.Lock())
        APP.LIVE = self.live; APP.args = NS(live_seconds=1e9, live_wait=0, live_cam=''); APP.GB = None; APP.cv2 = cv2
        self.spawns = []; self.said = []
        APP.lv_spawn = self.spawn
        for target_, name_, value_ in ((camera_link, 'publish_jpeg_locked', lambda *a, **k: None), (camera_extra, 'say', self.said.append)):
            p_ = patch.object(target_, name_, value_); p_.start(); self.addCleanup(p_.stop)

    def spawn(self):
        self.spawns.append(time.monotonic()); cam_frozen.on_spawn(APP.LIVE)
        self.live.proc = NS(poll=lambda: None); self.live.cam_lost = False; self.live.proc_t0 = time.monotonic()

    def ended(self, rc_, ago_=100):   # the running cam_pipe ended with rc_, ago_ s after its start
        self.live.proc = NS(poll=lambda: rc_); self.live.cam_lost = True; self.live.proc_t0 = time.monotonic() - ago_

    def freeze(self):
        self.cam('3-1', 3, 6)
        self.log.write(FROZEN_LINE); self.log.flush()
        self.ended(8); camera_link.lv_cam_wait()


class CamWait(LiveCase):
    def test_exit_8_waits_for_a_replug(self):
        self.freeze()
        fz_ = self.live.cam_frozen
        self.assertEqual(fz_['reason'], 'at a stall after 14964 frames (control libusb_timeout, ResetMainBoard refused)')
        self.assertEqual(fz_['ref'], [('3-1', 3, 6)]); self.assertFalse(fz_['hub'])
        self.assertEqual(self.spawns, [])
        self.assertAlmostEqual(self.live.cam_next_spawn - time.monotonic(), 60, delta=2)
        self.assertIn('ended with code 8', self.said[-1]); self.assertIn('CAMERA FROZEN', self.said[-1]); self.assertIn('without a software power switch', self.said[-1])
        for _ in range(3): camera_link.lv_cam_wait()          # nothing changed on USB: still waiting
        self.assertEqual(self.spawns, [])
        self.unplug('3-1'); camera_link.lv_cam_wait()           # cable out: nothing to open
        self.assertEqual(self.spawns, [])
        self.cam('3-1', 3, 7); camera_link.lv_cam_wait()        # back with a new device number: start at once
        self.assertEqual(len(self.spawns), 1)
        self.assertIn('replugged', self.said[-2])
        self.assertEqual(self.live.cam_frozen['ref'], [('3-1', 3, 7)])   # the state stays until a frame comes

    def test_failed_try_keeps_the_state_and_the_next_comes_after_60_s(self):
        self.freeze()
        self.live.cam_next_spawn = time.monotonic() - 1; camera_link.lv_cam_wait()   # 60 s passed
        self.assertEqual(len(self.spawns), 1)
        self.ended(2, ago_=30); camera_link.lv_cam_wait()                             # that try: OPEN_FAILED (send timeout), exit 2
        self.assertEqual(len(self.spawns), 1)
        self.assertEqual(self.live.cam_frozen['tries'], 1)
        self.assertAlmostEqual(self.live.cam_next_spawn - time.monotonic(), 60, delta=2)   # not the 15 s of a plain exit 2
        self.cam('3-1', 3, 9); camera_link.lv_cam_wait()                              # replug while waiting
        self.assertEqual(len(self.spawns), 2)

    def test_failed_try_after_a_replug_is_followed_sooner(self):   # 11-15 review: a camera just powered on may not answer the first open; 15 s (the old exit-2 pause), not 60 s
        self.freeze()
        self.unplug('3-1'); self.cam('3-1', 3, 7); camera_link.lv_cam_wait()          # replug: a try starts at once
        self.assertEqual(len(self.spawns), 1)
        self.ended(2, ago_=30); camera_link.lv_cam_wait()                             # that try failed
        self.assertAlmostEqual(self.live.cam_next_spawn - time.monotonic(), cam_frozen.AFTER_REPLUG_S, delta=2)
        self.assertIn('or 15 s pass', self.said[-1])
        self.live.cam_next_spawn = time.monotonic() - 1; camera_link.lv_cam_wait()   # the next try (by the timer, same USB entry)
        self.assertEqual(len(self.spawns), 2)
        self.ended(2, ago_=30); camera_link.lv_cam_wait()                             # failed again: back to 60 s
        self.assertAlmostEqual(self.live.cam_next_spawn - time.monotonic(), 60, delta=2)
        self.assertEqual(self.live.cam_frozen['tries'], 2)

    def test_replug_during_a_try_is_seen_after_it(self):
        self.freeze()
        self.live.cam_next_spawn = time.monotonic() - 1; camera_link.lv_cam_wait()   # a try starts with device 6
        self.unplug('3-1'); self.cam('3-1', 3, 8)                                     # replugged while that try was opening
        self.ended(2, ago_=30); camera_link.lv_cam_wait()                             # it failed; the exit is counted first
        self.assertEqual(len(self.spawns), 1)
        camera_link.lv_cam_wait()                                                     # next look: device 8 is not the one the try started with
        self.assertEqual(len(self.spawns), 2)

    def test_other_exits_unchanged_without_the_state(self):
        self.cam('3-1', 3, 6)
        self.ended(2); camera_link.lv_cam_wait()
        self.assertIsNone(getattr(self.live, 'cam_frozen', None))
        self.assertAlmostEqual(self.live.cam_next_spawn - time.monotonic(), 15, delta=2)
        self.assertNotIn('FROZEN', self.said[-1])
        self.live.cam_next_spawn = None; self.live.cam_fail_n = 0
        self.ended(0); camera_link.lv_cam_wait()
        self.assertEqual(len(self.spawns), 1)                                         # exit 0: at once, as before

    def test_exit_7_while_frozen_keeps_its_quick_retry(self):
        self.freeze()
        self.live.cam_next_spawn = None
        self.ended(7, ago_=30); camera_link.lv_cam_wait()
        self.assertAlmostEqual(self.live.cam_next_spawn - time.monotonic(), 5, delta=2)

    def test_lv_spawn_notes_the_usb_entry(self):
        self.cam('1-1.2', 1, 12)
        self.live.cam_frozen = dict(t=0, at='x', reason='r', ref=None, hub=False, tries=0)
        with patch.object(camera_extra, 'wanted', return_value=False), patch.object(camera_extra, 'install_producer'), patch.object(camera_extra, 'log_start'):
            APP.lv_sp = NS(Popen=lambda *a, **k: NS(), PIPE=-1); APP.lv_th = NS(Thread=lambda **k: NS(start=lambda: None)); APP.lv_reader = None
            camera_link.lv_spawn()
        self.assertEqual(self.live.cam_frozen['ref'], [('1-1.2', 1, 12)])
        self.assertEqual(self.live.cam_usb_place, 'hub')


class FrameLoopWait(LiveCase):
    class Cond:
        def __init__(self): self.waits = []
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def wait(self, timeout=None): self.waits.append(timeout); return False

    def wait_used(self):
        class Stop(Exception): pass
        def stop(): raise Stop()
        self.live.cond = self.Cond(); self.live.q = deque(); self.live.last = object(); self.live.cam_lost = False
        APP.lv_cam_wait = stop
        with self.assertRaises(Stop): camera_link.live_next()
        return self.live.cond.waits

    def test_two_seconds_while_frozen_ten_otherwise(self):
        self.assertEqual(self.wait_used(), [10])
        self.live.cam_frozen = dict(t=0, at='x', reason='r', ref=[], hub=False, tries=0)
        self.assertEqual(self.wait_used(), [cam_frozen.POLL_S])
        self.live.replay = dict(name='rec')                   # a replay plays: its own timing, as before
        self.assertEqual(self.wait_used(), [10])


class Readers(LiveCase):
    def setUp(self):
        super().setUp()
        APP.LIVE_W, APP.LIVE_H = 4, 2
        self.live.q = deque(maxlen=3); self.live.cond = threading.Condition(); self.live.q_drop = 0; self.live.frames_in = 0; self.live.cam_fps = 0.0; self.live.rec = None
        self.live.cam_frozen = dict(t=0, at='20:33:22', reason='r', ref=[], hub=False, tries=2)
        self.notes = []
        p_ = patch.object(camera_extra, 'note', lambda who_, t_: self.notes.append(t_)); p_.start(); self.addCleanup(p_.stop)

    def stamped(self, flags_):   # one frame with cam_pipe's 48-byte header (pictures through the pipe) and its 16 bytes
        h_ = bytearray(camera_extra.HDR.pack(camera_extra.MAGIC, 1, flags_, 1, time.monotonic_ns(), 0, 0xFFFFFFFF, 16, 0, 4, 2, 0))
        x_ = 0x5A17C0DE
        for v_ in camera_extra._W12.unpack(bytes(h_))[:11]: x_ ^= v_
        struct.pack_into('<I', h_, 44, x_)
        return bytes(h_) + bytes(range(16))

    def test_plain_reader_clears_on_a_camera_frame(self):
        p_ = NS(stdout=io.BytesIO(bytes(16)), lwir_stamp=False); self.live.proc = p_
        camera_link.lv_reader(p_)
        self.assertIsNone(self.live.cam_frozen)
        self.assertEqual(len(self.notes), 1); self.assertIn('frozen-camera state (since 20:33:22, r, 2 failed tries since) is cleared', self.notes[0])

    def test_stamped_reader_clears_on_a_camera_frame_not_on_a_replay_frame(self):
        self.live.cam_stamp = camera_extra._new_stats(True); self.live.cam_stamp_recent = deque(maxlen=64)
        self.live.proc_replay_src_t = None; self.live.replay_src_t = None
        p_ = NS(stdout=io.BytesIO(self.stamped(2)), lwir_stamp=True, lwir_eval=None); self.live.proc = p_   # flags bit 1: a replay program's frame
        camera_extra._reader(p_, [None])
        self.assertEqual(self.live.frames_in, 1)
        self.assertIsNotNone(self.live.cam_frozen)
        p_ = NS(stdout=io.BytesIO(self.stamped(0)), lwir_stamp=True, lwir_eval=None); self.live.proc = p_   # cam_pipe's frame
        camera_extra._reader(p_, [None])
        self.assertEqual(self.live.frames_in, 2)
        self.assertIsNone(self.live.cam_frozen)


class ResetButton(LiveCase):
    def setUp(self):
        super().setUp()
        self.calls = []; self.seen = [True]
        self.live.proc = NS(poll=lambda: 8, terminate=lambda: self.calls.append('terminate'))
        self.live.last_frame_t = time.monotonic() - 300
        APP.cam_usb_seen = lambda: self.seen[0]
        APP.cam_usb_gone = lambda: self.calls.append('gone-check') or self.seen[0] is False
        APP.cam_usb_power = lambda: (self.calls.append('power'), ('cut', 'USB 전원 껐다 켬, 카메라 다시 보임. '))[1]
        APP.lv_sp = NS(run=lambda *a, **k: self.calls.append(('run', a[0])))   # usbreset would come through here
        for target_, name_ in ((camera_link.time, 'sleep'),):
            p_ = patch.object(target_, name_, lambda s_: None); p_.start(); self.addCleanup(p_.stop)

    def press(self):
        camera_link.cam_reset_work()
        self.assertFalse(any(isinstance(c_, tuple) for c_ in self.calls), self.calls)   # never usbreset (nor any other command)
        self.assertNotIn('초기화 오류', self.live.cam_reset['text'])                       # no exception swallowed by cam_reset_work
        return self.live.cam_reset['text']

    def test_frozen_on_a_root_port_answers_with_the_replug_notice_only(self):
        self.freeze(); self.live.proc = NS(poll=lambda: 8, terminate=lambda: self.calls.append('terminate'))
        text_ = self.press()
        self.assertTrue(text_.startswith('카메라가 멈췄습니다 (USB 칩 응답 없음). 카메라 USB 선을 10초 뽑았다 다시 꽂아 주세요.'), text_)
        self.assertEqual(self.calls, []); self.assertEqual(self.spawns, [])
        self.assertFalse(self.live.cam_hold)

    def test_frozen_behind_the_hub_cuts_the_power(self):
        self.cam('1-1.2', 1, 12); self.log.write(FROZEN_LINE); self.log.flush(); self.ended(8); camera_link.lv_cam_wait()
        self.assertTrue(self.live.cam_frozen['hub'])
        self.assertIn('또는 카메라 초기화 버튼', cam_frozen.state(self.live)['text'])
        self.live.proc = NS(poll=lambda: 8)
        self.press()
        self.assertEqual(self.calls, ['power']); self.assertEqual(len(self.spawns), 1)

    def test_dead_on_a_root_port_restarts_cam_pipe_without_touching_usb(self):
        self.cam('3-1', 3, 6); self.live.proc = NS(poll=lambda: None, terminate=lambda: self.calls.append('terminate'), wait=lambda t_: 0)
        text_ = self.press()
        self.assertEqual(self.calls, ['terminate']); self.assertEqual(len(self.spawns), 1)
        self.assertIn(cam_frozen.NO_SWITCH, text_)

    def test_dead_behind_the_hub_cuts_the_power(self):
        self.cam('2-1.4', 2, 5)
        self.press()
        self.assertEqual(self.calls, ['power']); self.assertEqual(len(self.spawns), 1)

    def test_off_the_bus_uses_the_last_place(self):
        for last_, want_, text_part_ in (('hub', ['gone-check', 'power'], 'USB 전원 껐다 켬'), ('root', ['gone-check'], cam_frozen.NO_SWITCH), (None, ['gone-check'], cam_frozen.UNKNOWN_PORT)):
            with self.subTest(last=last_):
                del self.calls[:]; self.seen[0] = False; self.live.cam_reset_t = None; self.live.proc = NS(poll=lambda: 3)   # the last cam_pipe has ended
                if last_ is None: self.live.__dict__.pop('cam_usb_place', None)
                else: self.live.cam_usb_place = last_
                text_ = self.press()
                self.assertEqual(self.calls, want_); self.assertIn(text_part_, text_)

    def test_alive_camera_only_restarts(self):
        self.cam('3-1', 3, 6); self.live.last_frame_t = time.monotonic(); self.live.proc = NS(poll=lambda: None, terminate=lambda: self.calls.append('terminate'), wait=lambda t_: 0)
        text_ = self.press()
        self.assertEqual(self.calls, ['terminate']); self.assertNotIn('USB', text_.split('cam_pipe')[0])


if __name__ == '__main__':
    unittest.main()
