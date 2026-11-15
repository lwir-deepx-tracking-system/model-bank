"""11-16: keeping the camera period (feed_auto.py): the period monitor and the automatic choice between NPU core 0 only and cores 0 and 1
(LWIR_NPU_FEED=auto). No NPU: the engines are imitations.
RuleTests: the rule alone, on a made-up clock. PeriodTests: the statement whether the period is kept. FeedTests: the feed's part (switch path,
what holds it back), no threads. LiveTests: the real feeding thread, a camera thread and a frame loop in real time (about 12 s), detection 20 -> 42 -> 20 ms
(20 and not 25 ms: the sleeps of this PC are a few ms late, which the limits 26.7 / 30.0 ms would see; RuleTests has 25 -> 42 -> 25 exactly).
Run from tracking_server/: python -m unittest src.detect.test_frame_feed_auto"""
from collections import deque
import http.server
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ..common.state import APP
from ..track import e2e_live
from ..web import page_handler, system_buttons
from . import feed_auto, frame_feed, resident_det

P = 1.0 / 30.0   # camera period (s)


def monitor(**env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith('LWIR_NPU_FEED_AUTO_')}
    clean.update({'LWIR_NPU_FEED_AUTO_' + k: str(v) for k, v in env.items()})
    with patch.dict(os.environ, clean, clear=True):
        return feed_auto.Monitor()


class Sim:
    """Camera frames at the camera period on a made-up clock. frame(): one arrives, is detected in service ms (4 ms after arrival), the loop
    works on it for post ms and comes for the next one (the rule is asked once). drop(): one arrives and is thrown away by the feeding thread."""
    def __init__(self, mon, mode='normal', on=True, hold=None, cores=2, judge=True):
        self.mon, self.mode, self.on, self.hold, self.cores, self.judge = mon, mode, on, hold, cores, judge
        self.t = 100.0; self.now = self.t; self.k = 0; self.log = []; self.notes = []   # log: (time, to, server log text)

    def drop(self, busy):
        self.t += P; self.mon.drop(self.t, busy)

    def frame(self, service, exact=True, core1=None, ask=True, post=0.0):
        self.t += P; self.k += 1; m = self.mon
        m.arrive(self.t)
        core = self.k % 2 if self.mode == 'arrival' else 0
        ms = core1 if (core == 1 and core1 is not None) else (service(self) if callable(service) else service)
        sub = int(self.t * 1e9) + 4_000_000; done = sub + int(ms * 1e6)
        m.result(core, self.t, sub, sub if exact else done - 500_000, done)
        self.now = done / 1e9 + post / 1e3
        if ask: m.close(self.now)
        if not ask: return None
        to = m.step(self.now, self.mode, self.on, self.hold, cores=self.cores, judge=self.judge)
        if m.note: self.notes.append(m.note); m.note = ''
        if to is not None: self.log.append((self.now, to, m.switched(to, self.now))); self.mode = to
        return to

    def run(self, seconds, service, **kw):
        end = self.t + seconds
        while self.t < end: self.frame(service, **kw)


class RuleTests(unittest.TestCase):
    def test_slow_detection_switches_to_two_cores_within_the_hold_and_recovery_returns_after_the_longer_hold(self):
        m = monitor(); s = Sim(m)
        s.run(20.0, 25.0)                       # 25 ms: one core keeps the 33.3 ms period
        self.assertEqual(s.log, [])
        self.assertIn('충분', m.state(True)['why'])
        t_slow = s.t
        s.run(6.0, 42.0)                        # 1000 -> 600 MHz
        self.assertEqual([x[1] for x in s.log], ['arrival'])
        took = s.log[0][0] - t_slow
        self.assertGreaterEqual(took, m.up_hold_s)
        self.assertLess(took, m.up_hold_s + 0.8)      # the hold (3 s), the 6 slow frames that move the p90 of the 2 s window, and two looks 0.2 s apart
        self.assertRegex(s.log[0][2], r'^core 0 only -> cores 0 and 1: detection p90 42\.0 ms > 30\.0 ms \(0\.90 x camera period 33\.3 ms\) for 3\.[0-2] s$')
        s.run(30.0, 42.0)                       # still slow on two cores: it stays
        self.assertEqual(len(s.log), 1)
        self.assertIn('아직 느림', m.state(True)['why'])
        t_fast = s.t
        s.run(9.5, 25.0)
        self.assertEqual(len(s.log), 1)         # not before the longer hold
        s.run(4.0, 25.0)
        self.assertEqual([x[1] for x in s.log], ['arrival', 'normal'])
        took = s.log[1][0] - t_fast
        self.assertGreaterEqual(took, m.down_hold_s)
        self.assertLess(took, m.down_hold_s + m.window_s + 0.8)   # the slow frames leave the window, then 10 s
        self.assertIn('cores 0 and 1 -> core 0 only: detection p90 on core 0 25.0 ms < 26.7 ms (0.80 x camera period 33.3 ms)', s.log[1][2])
        s.run(120.0, 25.0)
        self.assertEqual(len(s.log), 2)
        st = m.state(True)
        self.assertEqual((st['switches']['up'], st['switches']['down']), (1, 1))
        self.assertEqual(st['switches']['last']['to'], 'normal')
        self.assertEqual(st['period_ms'], 33.3)
        self.assertEqual((st['service_ms']['p50'], st['service_ms']['p90']), (25.0, 25.0))

    def test_no_flapping_between_and_around_the_limits(self):
        for mode in ('normal', 'arrival'):      # 28 ms = 0.84 x period, between 0.80 and 0.90: it stays where it is
            s = Sim(monitor(), mode); s.run(300.0, 28.0)
            self.assertEqual(s.log, [], mode)
        s = Sim(monitor())                      # a stall of 1 s every 10 s: shorter than the hold can see
        s.run(300.0, lambda x: 45.0 if x.t % 10.0 < 1.0 else 25.0)
        self.assertEqual(s.log, [])
        s = Sim(monitor())                      # one frame in twenty is slow (p90 stays fast): nothing
        s.run(300.0, lambda x: 45.0 if x.k % 20 == 0 else 25.0)
        self.assertEqual(s.log, [])
        s = Sim(monitor(), 'arrival')           # fast for 9 s, slow for 1 s: never fast for the whole longer hold
        s.run(300.0, lambda x: 31.0 if x.t % 10.0 < 1.0 else 25.0)
        self.assertEqual(s.log, [])
        s = Sim(monitor())                      # just under the limit of 30.0 ms
        s.run(120.0, 29.9)
        self.assertEqual(s.log, [])

    def test_a_return_that_fails_doubles_the_wait_and_a_good_one_resets_it(self):
        m = monitor(); s = Sim(m)
        # the worst case for the rule: one core measures slow (31 ms), on two cores core 0 measures fast (25 ms)
        s.run(900.0, lambda x: 31.0 if x.mode == 'normal' else 25.0)
        ups = [x for x in s.log if x[1] == 'arrival']; downs = [x for x in s.log if x[1] == 'normal']
        self.assertLessEqual(len(ups), 7)       # waits 10, 20, 40, 80, 160, 300, 300 s: without the doubling it would be about 55 switches each way
        self.assertGreaterEqual(len(ups), 5)
        gaps = [b[0] - a[0] for a, b in zip(ups, downs)]   # time on two cores before each return
        self.assertTrue(all(b > a * 1.7 for a, b in zip(gaps[:4], gaps[1:5])), gaps)
        self.assertEqual(m.hold_s, feed_auto.HOLD_MAX_S)
        self.assertIn('the next return waits', ups[2][2])
        for a, b in zip(s.log, s.log[1:]): self.assertGreaterEqual(b[0] - a[0], m.dwell_s)   # least time between two switches
        s.run(400.0, 25.0)                      # really fast again: it returns and stays
        self.assertEqual(s.mode, 'normal')
        n = len(s.log)
        s.run(100.0, 42.0)                      # slow much later than 60 s after the return: the return was good, the wait is the first value again
        self.assertEqual(len(s.log), n + 1)
        self.assertEqual(m.hold_s, m.down_hold_s)

    def test_dropped_frames_switch_only_when_the_core_was_busy(self):
        m = monitor(); s = Sim(m)
        for k in range(200):                    # every fifth camera frame dropped behind the busy core, detection itself measured fast
            s.drop(True) if k % 5 == 4 else s.frame(25.0)
        self.assertEqual([x[1] for x in s.log], ['arrival'])
        self.assertIn('frames dropped in 2.0 s while core 0 was busy', s.log[0][2])
        m = monitor(); s = Sim(m)
        for k in range(600):                    # every second frame dropped because the frame loop was behind: a second core would not help
            s.drop(False) if k % 2 else s.frame(25.0)
        self.assertEqual(s.log, [])
        st = m.state(True)
        self.assertEqual((st['misses']['dropped'], st['misses']['dropped_busy'], st['period_ms']), (300, 0, 33.3))
        self.assertEqual((st['period']['state'], st['period']['dropped_per_s']), ('drop', 15.0))
        m = monitor(); s = Sim(m, 'arrival')    # on two cores: no return while frames are still dropped behind busy cores
        for k in range(900):
            s.drop(True) if k % 30 == 29 else s.frame(25.0)
        self.assertEqual(s.log, [])

    def test_only_frames_the_loop_waited_for_are_measured(self):
        m = monitor(); s = Sim(m)
        s.run(30.0, 42.0, exact=False)          # the results were ready before the loop came: 42 ms is the loop's lateness, not the detector's
        self.assertEqual(s.log, [])
        st = m.state(True)
        self.assertEqual(st['service_ms']['n'], 0)
        self.assertEqual(st['late_collect'], st['misses']['frames'])
        self.assertIn('재는 중', st['why'])
        self.assertEqual(st['period']['state'], 'late')   # the period is still judged: the results were late

    def test_two_cores_return_by_core_0_alone(self):
        s = Sim(monitor(), 'arrival')           # core 1 slower than core 0 must not keep it on two cores
        s.run(15.0, 25.0, core1=29.0)
        self.assertEqual([x[1] for x in s.log], ['normal'])

    def test_counters(self):
        m = monitor(); s = Sim(m)
        s.run(2.0, 25.0)                        # arrival -> result 4 + 25 = 29 ms: in time
        a = m.state(True)['misses']
        self.assertEqual((a['late'], a['dropped']), (0, 0))
        n0 = a['frames']
        for _ in range(10): s.frame(31.0)       # 4 + 31 = 35 ms > 33.3 ms
        m.drop(s.t, True); m.drop(s.t, False)
        b = m.state(True)['misses']
        self.assertEqual((b['frames'] - n0, b['late'], b['dropped'], b['dropped_busy']), (10, 10, 2, 1))
        m = monitor(); s = Sim(m)               # the deadline is the end of the frame's work: 4 + 25 + 6 ms of tracker and publishing = 35 ms is late
        s.run(4.0, 25.0, post=6.0)
        st = m.state(True)
        self.assertEqual((st['period']['state'], st['period']['late_pct'], st['service_ms']['p90']), ('late', 100.0, 25.0))
        self.assertEqual(s.log, [])             # late by the loop's own work is no reason for two cores

    def test_off_measures_and_never_switches_and_a_hold_keeps_it_where_it_is(self):
        m = monitor(); s = Sim(m, on=False)
        s.run(30.0, 42.0)
        self.assertEqual(s.log, [])
        st = m.state(False)
        self.assertFalse(st['on']); self.assertEqual(st['service_ms']['p90'], 42.0); self.assertEqual(st['why'], '')
        why = ['검출 모델을 바꾸고 있습니다']
        m = monitor(); s = Sim(m, hold=lambda: why[0])
        s.run(30.0, 42.0)
        self.assertEqual(s.log, [])
        self.assertEqual(m.state(True)['why'], '머무름 · 검출 모델을 바꾸고 있습니다')
        why[0] = None; t0 = s.t
        s.run(5.0, 42.0)
        self.assertEqual(len(s.log), 1)
        self.assertGreaterEqual(s.log[0][0] - t0, m.up_hold_s)   # the hold time starts again after it

    def test_limits_from_the_environment(self):
        m = monitor(UP=1.1, DOWN=0.7, UP_HOLD_S=0.2, DOWN_HOLD_S=3, DWELL_S=1, WINDOW_S=1, DROPS=5, GIMBAL=0)
        self.assertEqual((m.up, m.down, m.up_hold_s, m.down_hold_s, m.dwell_s, m.window_s, m.drops_up, m.gimbal_ok), (1.1, 0.7, 0.2, 3.0, 1.0, 1.0, 5, False))
        s = Sim(m); s.run(20.0, 35.0)           # 35 ms < 1.1 x 33.3
        self.assertEqual(s.log, [])
        with patch('builtins.print') as said:
            m = monitor(UP='fast', DOWN=5, DOWN_HOLD_S=-1)
            self.assertEqual((m.up, m.down, m.up_hold_s, m.down_hold_s, m.gimbal_ok), (0.9, 0.8, 3.0, 10.0, True))
            m = monitor(UP=0.6, DOWN=0.7)
            self.assertEqual((m.up, m.down), (0.9, 0.8))
        self.assertEqual(said.call_count, 4)


class PeriodTests(unittest.TestCase):
    """The plain statement whether the camera period is kept (npu_feed.auto.period), in every feed setting."""
    def state(self, service, mode='arrival', seconds=8.0, **kw):
        m = monitor(); s = Sim(m, mode, on=False, **kw); s.run(seconds, service)
        return m.state(False)['period'], s

    def test_kept_late_and_cannot_be_kept(self):
        p, s = self.state(21.0)                 # device 11-16, general 26m pose on two cores: 30 frames a second, none dropped
        self.assertEqual((p['state'], p['late_pct'], p['dropped_per_s'], p['one_core_ms'], p['two_cores_ms']), ('ok', 0.0, 0.0, 30.0, 60.0))
        self.assertEqual(p['text'], '주기 지킴 (늦은 프레임 0%, 버림 없음)'); self.assertEqual(s.notes, [])
        p, s = self.state(42.0)                 # a throttled NPU on two cores: every frame is processed, each one later than one period
        self.assertEqual((p['state'], p['late_pct'], p['dropped_per_s']), ('late', 100.0, 0.0))
        self.assertIn('결과가 주기보다 늦음', p['text']); self.assertEqual(s.notes, [])
        p, s = self.state(66.0)                 # device 11-16, YOLO26x: 66 ms a frame, 27 frames a second with two cores
        self.assertEqual(p['state'], 'over')
        self.assertEqual(p['text'], '이 검출 모델로는 카메라 주기를 지킬 수 없음 · 검출 한 장 66 ms, NPU 두 개로도 60 ms 까지 (주기 33.3 ms) · 더 가벼운 모델을 고르세요')
        self.assertEqual(len(s.notes), 1)       # one server log line, not one a frame
        self.assertTrue(s.notes[0].startswith('FEED period: cannot be kept with this detector model: detection p90 66.0 ms > 60.0 ms (2 x 0.90 x camera period 33.3 ms)'), s.notes)
        self.assertIn('not changed automatically', s.notes[0])
        s.run(5.0, 21.0); self.assertEqual(s.notes[1:], [])     # a lighter model was chosen: said once it has lasted 10 s
        s.run(10.0, 21.0)
        self.assertEqual(s.notes[1:], ['FEED period: the detector model is no longer too slow for the camera period'])
        self.assertEqual(s.mon.state(False)['period']['state'], 'ok')

    def test_one_core_two_engines_and_one_engine(self):
        p, _ = self.state(42.0, mode='normal')  # core 0 only by hand, a second engine is there: not 'cannot', two cores would do
        self.assertEqual(p['state'], 'late')
        m = monitor(); s = Sim(m, 'normal', on=False)
        for k in range(300):
            s.drop(True) if k % 4 == 3 else s.frame(42.0)
        p = m.state(False)['period']
        self.assertEqual(p['state'], 'drop'); self.assertTrue(p['text'].endswith('NPU 0·1 교대면 지킬 수 있음'), p['text'])
        p, s = self.state(42.0, mode='normal', cores=1)   # no second engine: one core is all there is
        self.assertEqual(p['state'], 'over'); self.assertIn('NPU 하나로는 (둘째 엔진 없음) 30 ms 까지', p['text'])
        self.assertIn('(1 x 0.90 x camera period 33.3 ms)', s.notes[0])
        m = monitor(); s = Sim(m, 'arrival', on=False)    # both cores busy and frames dropped, detection measured under the limit
        for k in range(300):
            s.drop(True) if k % 6 == 5 else s.frame(55.0)
        p = m.state(False)['period']
        self.assertEqual(p['state'], 'over'); self.assertIn('NPU 두 개가 다 차서', p['text'])

    def test_a_replay_is_not_judged_and_a_short_stall_is_not_announced(self):
        p, s = self.state(66.0, judge=False)
        self.assertEqual((p['state'], p['text']), (None, '녹화 재생 중에는 판정하지 않음')); self.assertEqual(s.notes, [])
        m = monitor(); s = Sim(m, 'arrival', on=False)
        s.run(5.0, 21.0); s.run(0.5, 80.0); s.run(10.0, 21.0)   # 0.5 s of very slow frames: 'over' for under 3 s, no log line
        self.assertEqual(s.notes, [])

    def test_npu_clock_and_temperature_from_the_status_the_server_already_asks_for(self):
        status = SimpleNamespace(get_npu_clock=lambda k: (1000, 800, 600)[k], get_temperature=lambda k: 90 + k)
        with patch.dict(feed_auto.NPU, clear=True), patch.object(feed_auto, '_NPU_OK', [True]):
            m = monitor()
            self.assertIsNone(m.state(False)['npu'])
            feed_auto.npu_note(status)
            npu = m.state(False)['npu']
            self.assertEqual((npu['clock_mhz'], npu['temp_c']), ([1000, 800, 600], [90, 91, 92])); self.assertLess(npu['age_s'], 5)
            with patch('builtins.print') as said:
                feed_auto.npu_note(object()); feed_auto.npu_note(object())   # a runtime without these readings: one line, then left out
            self.assertEqual(said.call_count, 1)
            self.assertIsNone(m.state(False)['npu'])
            feed_auto.npu_note(status); self.assertIsNone(m.state(False)['npu'])


class FeedTests(unittest.TestCase):
    """As test_frame_feed_controls: a Feed without its thread. The made-up clock drives the rule through Feed._auto, the switch goes through Feed._apply_mode."""
    def setUp(self):
        self.live = SimpleNamespace(
            restart_supported=True, restart_pending=False, restart_stop=False, stopped=False, cam_hold=False, cam_lost=False, meas=None, replay=None,
            instance='test-instance', sys_lock=threading.Lock(), cam_lock=threading.Lock(), meas_lock=threading.Lock(), cond=threading.Condition(),
            arr=deque(), last=None, q=deque(maxlen=3), q_drop=0)
        app = patch.dict(vars(APP), dict(LIVE=self.live, GB=None, FEED=None, FEED_MODE='auto', FEED_READY=True, FEED_INCOMPATIBLE='',
                                         args=SimpleNamespace(live_wait=0), lv_http=http.server, MODEL=dict(req=None, busy=False, lock=threading.Lock())), clear=True)
        app.start(); self.addCleanup(app.stop)
        for target, name, value in ((frame_feed.camera_extra.replay_eval, 'EV', None), (resident_det, 'RES', resident_det._State())):
            p_ = patch.object(target, name, value); p_.start(); self.addCleanup(p_.stop)
        resident_det.RES.want = False
        p_ = patch.object(e2e_live, 'skip_detector', return_value=False); self.e2e = p_.start(); self.addCleanup(p_.stop)
        p_ = patch.object(frame_feed.camera_extra, 'say'); self.said = p_.start(); self.addCleanup(p_.stop)
        p_ = patch.object(frame_feed, 'save_mode'); self.saved = p_.start(); self.addCleanup(p_.stop)
        feed = self.feed = object.__new__(frame_feed.Feed)
        feed.hot_switch = True; feed.mode = 'normal'; feed.pending = None; feed.out = deque(); feed.err = feed.reason = None
        feed.direct = feed.closed = feed.paused = feed.working = False
        feed.eng = [object(), object()]; feed.busy = [False, False]; feed.last_e = 1
        feed.st = dict(core=[0, 0], submitted=0, taken=0, order_bad=0, drop_wait=0, one_core=0)
        feed.mon = monitor(); self.sim = Sim(feed.mon); feed._clock = lambda: self.sim.now
        APP.FEED = feed

    def frames(self, seconds, service):
        """Frames on the made-up clock; the feed's own per-frame step asks the rule and the switch is applied between two frames."""
        s = self.sim; end = s.t + seconds
        while s.t < end:
            s.mode = self.feed.mode
            s.frame(service, ask=False)
            with self.live.cond:
                self.feed._auto(); self.feed._apply_mode()

    def lines(self):
        return [c.args[0] for c in self.said.call_args_list]

    def test_the_switch_keeps_the_setting_and_saves_nothing(self):
        feed = self.feed
        self.frames(3.0, 25.0)
        self.assertEqual((feed.mode, feed.pending), ('normal', None))
        self.frames(5.0, 42.0)
        self.assertEqual((feed.mode, APP.FEED_MODE, feed.pending, feed.auto_to), ('arrival', 'auto', None, None))
        self.saved.assert_not_called()
        self.assertEqual(len(self.lines()), 1)
        self.assertTrue(self.lines()[0].startswith('FEED auto: core 0 only -> cores 0 and 1: detection p90 42.0 ms > 30.0 ms'), self.lines())
        st = system_buttons.npu_feed_state()
        self.assertEqual((st['requested'], st['mode'], st['active'], st['pending']), ('auto', 'arrival', True, False))
        self.assertTrue(st['auto']['on']); self.assertEqual(st['auto']['switches']['up'], 1)
        self.assertTrue(st['can_toggle'])
        self.frames(15.0, 25.0)
        self.assertEqual((feed.mode, APP.FEED_MODE), ('normal', 'auto'))
        self.assertEqual(len(self.lines()), 2)
        self.saved.assert_not_called()
        self.assertEqual(system_buttons.npu_feed_state()['auto']['period']['state'], 'ok')

    def test_the_switch_waits_for_every_frame_in_the_detector(self):
        feed = self.feed
        with patch.object(feed, '_apply_mode'): self.frames(5.0, 42.0)   # the rule asks; frames are still in the detector
        self.assertEqual((feed.pending, feed.auto_to, feed.mode), ('auto', 'arrival', 'normal'))
        self.assertTrue(system_buttons.npu_feed_state()['pending'])
        feed.out.append(object()); feed.busy[0] = True
        with self.live.cond:
            feed._apply_mode(); self.assertEqual(feed.mode, 'normal')
            feed.out.popleft(); feed._apply_mode(); self.assertEqual(feed.mode, 'normal')
            feed.busy[0] = False; feed._apply_mode()
        self.assertEqual((feed.mode, feed.pending, feed.auto_to), ('arrival', None, None))

    def test_what_holds_it_where_it_is(self):
        feed = self.feed; mon = feed.mon
        gimbal = SimpleNamespace(state=dict(auto=1, moving=0))
        cases = [
            ('model change asked', lambda: APP.MODEL.update(req=dict(det='x')), lambda: APP.MODEL.update(req=None), '검출 모델'),
            ('model change running', lambda: APP.MODEL.update(busy=True), lambda: APP.MODEL.update(busy=False), '검출 모델'),
            ('replay', lambda: setattr(self.live, 'replay', {'name': 'rec'}), lambda: setattr(self.live, 'replay', None), '녹화 재생'),
            ('camera work', lambda: setattr(self.live, 'cam_hold', True), lambda: setattr(self.live, 'cam_hold', False), '카메라'),
            ('resident pair wanted', lambda: setattr(resident_det.RES, 'want', True), lambda: setattr(resident_det.RES, 'want', False), '상주'),
            ('resident pair there', lambda: setattr(resident_det.RES, 'pairs', {}), lambda: setattr(resident_det.RES, 'pairs', None), '상주'),
            ('no second engine', lambda: feed.eng.pop(), lambda: feed.eng.append(object()), 'NPU 1'),
            ('end-to-end model', lambda: setattr(self.e2e, 'return_value', True), lambda: setattr(self.e2e, 'return_value', False), '엔드투엔드'),
            ('gimbal following, LWIR_NPU_FEED_AUTO_GIMBAL=0', lambda: (setattr(APP, 'GB', gimbal), setattr(mon, 'gimbal_ok', False)),
             lambda: (setattr(APP, 'GB', None), setattr(mon, 'gimbal_ok', True)), '짐벌'),
        ]
        for name, on, off, word in cases:
            on()
            self.frames(4.5, 42.0)
            self.assertEqual((feed.mode, feed.pending), ('normal', None), name)
            why = system_buttons.npu_feed_state()['auto']['why']
            self.assertTrue(why.startswith('머무름') and word in why, (name, why))
            off()
        APP.GB = gimbal                           # the gimbal following alone does not hold it (default)
        self.frames(2.0, 42.0); self.assertEqual(feed.mode, 'normal')   # the hold time started again when the last reason went away
        self.frames(2.0, 42.0); self.assertEqual(feed.mode, 'arrival')
        self.saved.assert_not_called()

    def test_no_switch_while_the_model_is_changed_and_the_new_model_is_measured_from_the_start(self):
        feed = self.feed
        self.frames(2.9, 42.0)                        # slow, the hold has almost passed
        feed.paused = True                            # Feed.pause(): the frame loop is changing the detector model
        self.frames(3.0, 42.0)
        self.assertEqual((feed.mode, feed.pending), ('normal', None))
        feed.eng = []; feed.busy = []
        with patch.dict(vars(APP), a=object()), patch.object(frame_feed, 'make_second', return_value=object()):
            feed.resume(0)
        self.assertEqual((feed.mode, feed.paused, len(feed.eng)), ('normal', False, 2))
        self.assertEqual(len(feed.mon.samp), 0)       # the old model's times are gone
        self.frames(1.0, 25.0)                        # the new model is fast: nothing
        self.assertEqual(feed.mode, 'normal')
        self.frames(5.0, 42.0); self.assertEqual(feed.mode, 'arrival')
        feed.paused = True; feed.eng = []; feed.busy = []
        with patch.dict(vars(APP), a=object()), patch.object(frame_feed, 'make_second', return_value=object()):
            feed.resume(0)                            # on two cores when the model changed: it stays on two cores, measured again
        self.assertEqual((feed.mode, APP.FEED_MODE, feed.paused), ('arrival', 'auto', False))
        feed.paused = True; feed.eng = []; feed.busy = []
        with patch.dict(vars(APP), a=object(), det_names=['m']), patch.object(frame_feed, 'make_second', side_effect=RuntimeError('npu memory')):
            feed.resume(0)                            # the second engine could not be made: core 0 only, and it stays there
        self.assertEqual((feed.mode, len(feed.eng)), ('normal', 1))
        self.frames(4.5, 42.0)
        self.assertEqual(feed.mode, 'normal')
        st = system_buttons.npu_feed_state()['auto']
        self.assertIn('NPU 1', st['why']); self.assertEqual(st['period']['state'], 'over')   # one core is all there is now
        self.assertEqual(sum(1 for l in self.lines() if l.startswith('FEED period: cannot be kept')), 1)

    def test_a_conflict_between_asking_and_switching_leaves_everything_as_it_was(self):
        feed = self.feed
        with patch.object(feed, '_apply_mode'): self.frames(5.0, 42.0)
        self.assertEqual((feed.pending, feed.auto_to), ('auto', 'arrival'))
        self.assertFalse(system_buttons.request_npu_feed('normal')[0])   # one change at a time
        APP.MODEL['busy'] = True
        with self.live.cond: feed._apply_mode()
        self.assertEqual((feed.mode, feed.pending, feed.auto_to, feed.reason), ('normal', None, None, None))
        self.assertIn('검출 모델', feed.mon.state(True)['why'])
        self.said.assert_not_called()
        APP.MODEL['busy'] = False
        self.frames(2.0, 42.0); self.assertEqual(feed.mode, 'normal')
        self.frames(2.0, 42.0); self.assertEqual(feed.mode, 'arrival')

    def test_a_model_change_right_after_the_rule_asked_drops_the_switch(self):
        feed = self.feed
        with patch.object(feed, '_apply_mode'): self.frames(5.0, 42.0)
        self.assertEqual((feed.pending, feed.auto_to), ('auto', 'arrival'))
        feed.cur = None; feed.redo = deque(); feed.st.update(swaps=0, redo=0); feed.eng = [SimpleNamespace(b=Mock()), SimpleNamespace(b=Mock())]
        with patch.dict(vars(APP), a=feed):
            feed.pause()                              # the frame loop took a model change before the switch could be applied
        self.assertEqual((feed.pending, feed.auto_to, feed.paused, feed.mode), (None, None, True, 'normal'))
        self.said.assert_not_called()

    def test_choosing_auto_on_the_page_and_at_the_start(self):
        feed = self.feed; APP.FEED_MODE = 'arrival'; feed.mode = 'arrival'
        handler = object.__new__(page_handler.make_lv_Handler())
        handler.path = '/system/npu-feed?mode=auto'
        handler.headers = {'Host': 'localhost:8090', 'Origin': 'http://localhost:8090', 'X-Gimbal': '1'}
        handler.wfile = io.BytesIO(); handler.send_response = Mock(); handler.send_header = Mock(); handler.end_headers = Mock(); handler.send_error = Mock()
        with patch('builtins.print'): handler.do_POST()
        handler.send_response.assert_called_once_with(202)
        self.assertEqual((feed.pending, feed.auto_to), ('auto', None))
        with self.live.cond: feed._apply_mode()
        self.saved.assert_called_once_with('auto')
        self.assertEqual((APP.FEED_MODE, feed.mode, feed.pending), ('auto', 'normal', None))   # auto starts on core 0 only
        st = system_buttons.npu_feed_state()
        self.assertEqual((st['requested'], st['mode'], st['active']), ('auto', 'normal', False))
        feed.eng.pop()
        self.assertFalse(feed.request_mode('auto')[0])    # no second engine: as for arrival
        self.assertTrue(feed.request_mode('normal')[0])

    def test_start_keeps_both_engines_for_auto_and_refuses_with_a_line(self):
        args = SimpleNamespace(live_port=8090, pipeline=True, no_vit=True, npu_depth=1, live_wait=0, zoom=False)
        made = []
        class FakeFeed:
            def __init__(self, second, *, hot_switch=False):
                made.append((second, hot_switch)); self.mode = 'normal'; self.mon = monitor(); self.reason = None
        second = object()
        with patch.dict(vars(APP), args=args, FEED_HOT_SWITCH=False, FEED_INCOMPATIBLE='', det_names=['m']), patch.object(frame_feed, 'Feed', FakeFeed), \
                patch.object(frame_feed, '_switch_interval', return_value=0), patch.object(frame_feed, 'make_second', return_value=second):
            self.assertIsNotNone(frame_feed.start())
            self.assertEqual(made, [(second, True)])          # both engines without LWIR_NPU_HOT_SWITCH=1
            self.assertTrue(self.lines()[-1].startswith('FEED auto on: NPU core 0 only; cores 0 and 1 when detection p90 > 0.90 x camera period for 3 s'), self.lines())
            APP.FEED_MODE = 'normal'
            self.assertIsNone(frame_feed.start())             # as before: no feed without the variable
            APP.FEED_MODE = 'auto'; APP.FEED_INCOMPATIBLE = 'not with --zoom'
            self.assertIsNone(frame_feed.start())
            self.assertEqual(self.lines()[-1], 'FEED auto refused: not with --zoom. The normal path is used.')
            APP.FEED_INCOMPATIBLE = ''; args.live_wait = 5
            self.assertIsNone(frame_feed.start())
            self.assertIn('--live-wait 0', self.lines()[-1])
        self.assertEqual(len(made), 1)

    def test_an_error_in_the_rule_stops_the_rule_not_the_frames(self):
        feed = self.feed
        with patch.object(feed.mon, 'step', side_effect=ZeroDivisionError), patch('builtins.print') as said, self.live.cond:
            feed._auto()
        self.assertIsNone(feed.mon); self.assertEqual(said.call_count, 1)
        with self.live.cond: feed._auto()
        self.assertEqual((feed.mode, feed.pending), ('normal', None))
        self.assertNotIn('auto', frame_feed.status())


class SettingTests(unittest.TestCase):
    def test_auto_is_read_from_the_environment_and_the_saved_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'npu_feed.json'
            with patch.object(frame_feed, 'MODE_FILE', path), patch.dict(os.environ, LWIR_NPU_FEED='auto'):
                self.assertEqual(frame_feed.load_mode(), 'auto')
                frame_feed.save_mode('normal'); self.assertEqual(frame_feed.load_mode(), 'normal')
                frame_feed.save_mode('auto'); self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'mode': 'auto'})
                self.assertEqual(frame_feed.load_mode(), 'auto')
                path.unlink()
                with patch.dict(os.environ, LWIR_NPU_FEED='fastest'), patch('builtins.print'):
                    self.assertEqual(frame_feed.load_mode(), 'normal')


class Engine:
    """An imitation NPU engine: one job at a time, the result is ready service[0] seconds after the submit."""
    def __init__(self, core, service, log):
        self.core = core; self.name = 'core%d' % core; self.b = self; self.packed = np.empty(1, np.float32); self.service = service
        self.log = log; self.t = None; self.jobs = 0; self.over = 0

    def run_async(self, image, *, user_arg, output_buffer):
        if self.t is not None: self.over += 1     # a second job on a core that is still busy
        self.t = time.perf_counter(); self.jobs += 1; self.log.append(self.core)
        return (user_arg['f'], self.t + self.service[0])

    def wait(self):
        f, ready = APP.pending.pop(self.name)
        left = ready - time.perf_counter()
        if left > 0: time.sleep(left)
        self.t = None
        return [np.array([f], np.float32)]


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.service = [0.020]; self.cores = []; self.stop = threading.Event()
        self.pixels = np.zeros((480, 640), dtype=np.uint8)
        self.live = SimpleNamespace(cond=threading.Condition(), q=deque(maxlen=3), q_drop=0, stopped=False, cam_lost=False, prev_t=None, last=None,
                                    stabq=deque(), arr=deque(), rawq=deque(), restart_pending=False, restart_stop=False, cam_hold=False, replay=None, gap_n=0)
        self.e0 = Engine(0, self.service, self.cores); self.e1 = Engine(1, self.service, self.cores)

        def submit_tiles(image, seq):
            APP.a.submit(image, seq); return time.perf_counter_ns()
        # short times for the test; as with the real values the hold is longer than the window (one stall of this PC must not switch the cores)
        env = {'LWIR_NPU_FEED_AUTO_' + k: v for k, v in dict(WINDOW_S='1', UP_HOLD_S='1.2', DOWN_HOLD_S='1.5', DWELL_S='0.3').items()}
        self.lines = []
        for p_ in (patch.dict(os.environ, env),
                   patch.dict(vars(APP), dict(LIVE=self.live, a=self.e0, FEED=None, FEED_MODE='auto', FEED_READY=True, GB=None, pending={}, FRAME_DY=[0.0],
                                              MODEL=dict(req=None, busy=False, lock=threading.Lock()), lv_to8=lambda raw: raw,
                                              args=SimpleNamespace(pipeline=True, gmc_thread=False, live_wait=0)), clear=True),
                   patch.dict(frame_feed.STAB, on=False), patch.object(frame_feed, 'stab_off', return_value=None),
                   patch.object(frame_feed.camera_extra.replay_eval, 'EV', None),
                   patch.object(frame_feed.camera_extra.replay_eval, 'reset_first_input', return_value=False),
                   patch.object(frame_feed.detect_frame, 'submit_tiles', side_effect=submit_tiles),
                   patch.object(e2e_live, 'skip_detector', return_value=False),
                   patch.object(resident_det, 'RES', resident_det._State()),
                   patch.object(frame_feed, 'save_mode'),
                   patch.object(frame_feed.camera_extra, 'say', side_effect=self.lines.append)):
            p_.start(); self.addCleanup(p_.stop)
        resident_det.RES.want = False
        self.addCleanup(sys.setswitchinterval, sys.getswitchinterval()); sys.setswitchinterval(0.001)   # as frame_feed.start() does on the server (1 ms)

    def camera(self):   # as camera_link's reader: a frame every period; when the line is full the oldest is lost and counted
        nxt = time.perf_counter()
        while not self.stop.is_set():
            nxt += P; time.sleep(max(0.0, nxt - time.perf_counter()))
            with self.live.cond:
                if len(self.live.q) == self.live.q.maxlen: self.live.q_drop += 1
                self.live.q.append((time.perf_counter(), self.pixels)); self.live.cond.notify_all()

    def loop(self, feed, taken, errors):   # the frame loop of app.py: read, wait for the detector, 4 ms of work
        try:
            k = 0
            while not self.stop.is_set():
                g, _ = feed.read_frame(k)
                if g is None or self.stop.is_set(): break
                feed.submit_time(g, k); t_arr = self.live.arr[-1]; out = feed.wait()
                time.sleep(0.004)
                taken.append((int(out[0][0]), t_arr, time.perf_counter(), feed.mode)); k += 1
        except Exception as exc:
            errors.append(exc)

    def until(self, what, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            if what(): return True
            time.sleep(0.02)
        return what()

    def test_slow_then_fast_detection_on_the_running_feed(self):
        taken = []; errors = []
        feed = frame_feed.Feed(self.e1, hot_switch=True); APP.FEED = feed
        threads = [threading.Thread(target=self.camera, daemon=True), threading.Thread(target=self.loop, args=(feed, taken, errors), daemon=True)]
        try:
            for th in threads: th.start()
            time.sleep(0.6)                                       # the first frames of the new threads are not the steady state
            with self.live.cond: feed.mon.reset()
            time.sleep(2.0)                                       # 20 ms: core 0 only
            first = frame_feed.status()['auto']
            if (first['service_ms']['p90'] or 99.0) > 25.0 or first['misses']['dropped']:   # the sleeps of a busy PC (the whole suite) are too late to stand for 20 ms
                self.skipTest('this PC is too busy for the real-time part: 20 ms measured as p90 %s ms, %d dropped' % (first['service_ms']['p90'], first['misses']['dropped']))
            self.assertEqual((feed.mode, self.e1.jobs), ('normal', 0), self.lines)
            n_fast = len(taken); late_fast = feed.mon.n_late
            self.assertGreater(n_fast, 50)                        # about 30 a second on one core
            self.assertEqual(first['period']['state'], 'ok')
            t_slow = time.perf_counter(); self.service[0] = 0.042   # the NPU slows down
            self.assertTrue(self.until(lambda: feed.mode == 'arrival', 4.0), frame_feed.status())
            took_up = time.perf_counter() - t_slow
            self.assertGreaterEqual(took_up, 1.2)                 # not before the hold
            self.assertLess(took_up, 1.2 + 1.0 + 0.6)             # the hold, at most one window, and slack for this PC
            time.sleep(1.5)
            self.assertEqual(feed.mode, 'arrival')                # still slow: it stays
            n_a = len(taken); t_a = time.perf_counter(); time.sleep(1.5)
            fps_two = (len(taken) - n_a) / (time.perf_counter() - t_a)
            self.assertGreater(fps_two, 24.5)                     # 42 ms a frame on two cores keeps the camera rate (one core: 1 / 0.042 s = 23.8 at most)
            self.assertEqual(frame_feed.status()['auto']['period']['state'], 'late')   # every frame processed, each later than one period
            t_fast = time.perf_counter(); self.service[0] = 0.020
            self.assertTrue(self.until(lambda: feed.mode == 'normal', 8.0), frame_feed.status())
            took_down = time.perf_counter() - t_fast
            self.assertGreaterEqual(took_down, 1.5)               # not before the longer hold
            with self.live.cond:
                status = system_buttons.npu_feed_state(); st = dict(feed.st); q_drop = self.live.q_drop
        finally:
            self.stop.set(); self.live.stopped = True
            with self.live.cond: self.live.cond.notify_all()
            for th in threads: th.join(3)
            feed._stop_thread()
        self.assertFalse(errors); self.assertIsNone(feed.err)
        auto = status['auto']
        self.assertEqual((status['requested'], status['mode'], APP.FEED_MODE), ('auto', 'normal', 'auto'))
        self.assertEqual((auto['switches']['up'], auto['switches']['down']), (1, 1))
        self.assertEqual([l[:36] for l in self.lines[:2]], ['FEED auto: core 0 only -> cores 0 an', 'FEED auto: cores 0 and 1 -> core 0 o'])
        frame_feed.save_mode.assert_not_called()                  # the setting stays auto, nothing is written
        self.assertAlmostEqual(auto['period_ms'], 33.3, delta=1.5)
        self.assertAlmostEqual(auto['service_ms']['p50'], 20.0, delta=4.0)
        # real-time properties
        seqs = [x[0] for x in taken]
        self.assertEqual(seqs, sorted(set(seqs)))                 # in arrival order, each once
        self.assertEqual(st['order_bad'], 0)
        self.assertLessEqual(st['max_ahead'], frame_feed.AHEAD)   # at most one frame in the detector ahead of the loop
        self.assertEqual((self.e0.over, self.e1.over), (0, 0))    # one frame on a core at a time
        self.assertGreater(self.e1.jobs, 30)
        # counters: every frame the feeding thread dropped is in all three counts; late = result later than arrival + one period
        self.assertEqual(auto['misses']['dropped'], st['drop_wait'])
        self.assertLessEqual(st['drop_wait'], q_drop)
        self.assertGreater(auto['misses']['dropped_busy'], 0)     # the slow time on one core dropped frames behind the busy core
        self.assertAlmostEqual(auto['misses']['frames'], st['taken'], delta=1)
        late = sum(1 for _, t_arr, t_done, _ in taken if t_done - t_arr > P)
        self.assertAlmostEqual(auto['misses']['late'], late, delta=max(10, late // 5))
        self.assertLess(late_fast, n_fast // 4)                   # 20 ms: in time
        self.assertGreater(late, 30)                              # 42 ms is later than one period whatever the cores

    def test_auto_feeds_the_newest_waiting_frame_and_the_other_settings_feed_the_oldest_as_before(self):
        for setting, first in (('auto', 3.0), ('arrival', 1.0), ('normal', 1.0)):
            APP.FEED_MODE = setting; APP.a = self.e0; APP.pending.clear()
            self.live.q.clear(); self.live.q.extend((float(k), self.pixels) for k in (1, 2, 3)); self.live.q_drop = 0   # three frames waited (e.g. during a switch)
            self.live.arr.clear(); self.live.prev_t = None
            feed = frame_feed.Feed(self.e1, hot_switch=True)
            try:
                g, _ = feed.read_frame(0); feed.wait()
                self.assertEqual(self.live.arr[0], first, setting)
                if setting == 'auto':
                    self.assertEqual((self.live.q_drop, feed.st['drop_wait'], feed.mon.n_drop, feed.mon.n_drop_busy), (2, 2, 2, 0))
            finally:
                feed._stop_thread()


if __name__ == '__main__':
    unittest.main()
