"""11-22: the camera reset by itself after a frozen camera (cam_autoreset.py). No camera, no USB, no sudo: mocks, a fake clock, and (Linux with g++)
the real cam_pipe.cpp built against the stand-in SDK of test_cam_pipe_frozen.py for the exit codes and log lines the cause is read from.
Run from tracking_server/:  python3 -m unittest src.camera.test_cam_autoreset
"""
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from ..common.state import APP
from . import cam_autoreset as ar
from . import camera_link
from . import test_cam_frozen as frozen_base

OPEN_FAILED = "DIAG open: op=4(send) result=-7(libusb_timeout)\nOPEN_FAILED\n"   # the 11-22 10:16-10:33 lines, one try
NO_CAMERA = "OPEN_FAILED: camera not found on USB\n"
FROZEN = frozen_base.FROZEN_LINE


class Cause(unittest.TestCase):
    def test_usb_cause(self):
        self.assertEqual(ar.usb_cause(2, OPEN_FAILED), 'libusb_timeout')       # first open failed, send timed out
        self.assertEqual(ar.usb_cause(3, 'DIAG open: op=5(receive) result=-7(libusb_timeout)\n'), 'libusb_timeout')   # reopen after a reset failed
        self.assertEqual(ar.usb_cause(6, 'DIAG open: op=4(send) result=-7(libusb_timeout)\n'), 'libusb_timeout')
        # an earlier stall of the same run timed out, but the reopen then failed because the device is gone (real lines of the stand-in SDK run): not a frozen chip
        self.assertEqual(ar.usb_cause(3, 'DIAG stall: op=5(receive) result=-7(libusb_timeout)\nDIAG open: op=2(open) result=7(device_not_connected)\nOPEN_FAILED\n'), '')
        self.assertEqual(ar.usb_cause(6, 'x libusb_timeout x'), '')   # a timeout that is not the open's
        self.assertEqual(ar.usb_cause(8, ''), 'CAMERA_FROZEN')                 # exit 8 is the frozen exit on its own
        self.assertEqual(ar.usb_cause(2, FROZEN), 'CAMERA_FROZEN')
        for rc_, text_ in ((2, NO_CAMERA), (3, 'libusb_no_device'), (0, OPEN_FAILED), (7, OPEN_FAILED), (1, OPEN_FAILED), (2, ''), (None, None)):
            self.assertEqual(ar.usb_cause(rc_, text_), '', (rc_, text_))      # cable out, normal end, another cam_pipe still closing, nothing seen

    def test_switch_values(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LWIR_CAM_AUTORESET', None); ar.set_mode(None)
            self.assertEqual(ar.mode(), '1')                                      # default on (user approval 11-22)
            for value_, want_ in (('0', '0'), ('off', '0'), ('dry', 'dry'), ('1', '1'), ('on', '1'), ('', '1'), ('garbage', '1')):
                os.environ['LWIR_CAM_AUTORESET'] = value_; self.assertEqual(ar.mode(), want_, value_)
            ar.set_mode('dry'); self.assertEqual(ar.mode(), 'dry')               # the page / rules choice wins over the start value
            ar.set_mode('x'); self.assertEqual(ar.mode(), '1')
        ar.set_mode(None)

    def test_rules_key(self):
        from ..web import id_rule_switches
        ar.set_mode(None); self.addCleanup(ar.set_mode, None)
        app_ = NS()
        ok_, _ = id_rule_switches.request(app_, {'cam_autoreset': ['dry']}, lambda q_: (True, 'rest'))
        self.assertTrue(ok_); self.assertEqual(ar.mode(), 'dry')
        ok_, text_ = id_rule_switches.request(app_, {'cam_autoreset': ['maybe']}, lambda q_: (True, 'rest'))
        self.assertFalse(ok_); self.assertEqual(ar.mode(), 'dry')                # refused: unchanged
        id_rule_switches.request(app_, {'cam_autoreset': ['off']}, lambda q_: (True, 'rest')); self.assertEqual(ar.mode(), '0')
        self.assertIn('cam_autoreset', ar.switch_state())


class ExitCounting(frozen_base.LiveCase):
    """lv_cam_wait with the hook: the failed opens in a row that have the USB cause."""
    def spawn(self):
        super().spawn(); ar.on_spawn(self.live)

    def fail(self, rc_, text_=''):   # a cam_pipe try began (log offset), wrote text_ and ended with rc_
        ar.on_spawn(self.live); self.log.write(text_); self.log.flush()
        self.live.cam_next_spawn = None; self.ended(rc_); camera_link.lv_cam_wait()

    def test_five_failed_opens_with_libusb_timeout(self):
        for i_ in range(4): self.fail(2, OPEN_FAILED)
        self.assertEqual(ar.state(self.live)['usb_n'], 4); self.assertEqual(self.live.cam_fail_n, 4)
        self.fail(2, OPEN_FAILED)
        self.assertEqual(ar.state(self.live)['usb_n'], 5); self.assertEqual(ar.state(self.live)['cause'], 'libusb_timeout')

    def test_a_frame_in_between_starts_a_new_chain(self):
        for i_ in range(3): self.fail(2, OPEN_FAILED)
        self.live.cam_fail_n = 0   # the reader got a frame (it resets cam_fail_n)
        self.fail(2, OPEN_FAILED)
        self.assertEqual(ar.state(self.live)['usb_n'], 1)

    def test_another_cause_breaks_the_chain_and_exit_7_does_not(self):
        for i_ in range(3): self.fail(2, OPEN_FAILED)
        self.fail(7, OPEN_FAILED)   # the old cam_pipe was still closing: neutral
        self.assertEqual(ar.state(self.live)['usb_n'], 3)
        self.fail(2, NO_CAMERA)     # cable out: not a frozen chip
        self.assertEqual(ar.state(self.live)['usb_n'], 0)

    def test_exit_8_counts_and_keeps_the_frozen_wait(self):
        self.cam('1-1.3', 1, 8)
        self.fail(8, FROZEN)
        self.assertEqual(ar.state(self.live)['usb_n'], 1); self.assertEqual(ar.state(self.live)['cause'], 'CAMERA_FROZEN')
        self.assertIsNotNone(self.live.cam_frozen)   # the existing frozen handling is untouched
        self.assertAlmostEqual(self.live.cam_next_spawn - __import__('time').monotonic(), 60, delta=2)

    def test_old_log_lines_do_not_count(self):
        self.log.write(FROZEN + OPEN_FAILED); self.log.flush()   # lines of earlier tries
        self.fail(2, NO_CAMERA)
        self.assertEqual(ar.state(self.live)['usb_n'], 0)


class FakeHooks:
    def __init__(self, live_):
        self.live = live_; self.replay = False; self.meas = False; self.manual = ''; self.place = 'hub'; self.ok = (True, '카메라 초기화 시작')
        self.text = 'USB 전원 껐다 켬, 카메라 다시 보임. cam_pipe 다시 시작함. 영상은 10~60초 뒤'
        self.g = dict(present=True, alive=True, ready=True, error=False, dev_ok=True); self.calls = []; self.said = []

    def say(self, t_): self.said.append(t_)
    def reset_request(self):
        self.calls.append('reset')
        if self.ok[0]: self.live.cam_hold = True
        return self.ok
    def replay_active(self): return self.replay
    def measuring(self): return self.meas
    def manual_active(self, now_): return self.manual
    def camera_place(self): return self.place
    def reset_text(self): return self.text
    def gimbal(self): return dict(self.g)
    def gimbal_restart(self, kind_): self.calls.append('gimbal-' + kind_)


class WatcherCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='autoreset_'); self.addCleanup(shutil.rmtree, self.dir, True)
        self.now = [1000.0]; self.up = [5000.0]; self.boot = ['boot-1']
        self.live = NS(stopped=False, cam_fail_n=5, cam_hold=False, last_frame_t=-1e9)
        self.hooks = FakeHooks(self.live)
        self.store = ar.Store(os.path.join(self.dir, 'state.json'), uptime_=lambda: self.up[0], boot_=lambda: self.boot[0])
        self.day = ['2026-11-22']
        self.w = self.make()
        ar.state(self.live)['usb_n'] = 5; ar.state(self.live)['cause'] = 'libusb_timeout'
        ar.set_mode(None); self.addCleanup(ar.set_mode, None)
        patcher = patch.dict(os.environ, {}); patcher.start(); self.addCleanup(patcher.stop); os.environ.pop('LWIR_CAM_AUTORESET', None); os.environ.pop('LWIR_CAM_AUTORESET_N', None)

    def make(self):
        return ar.Watcher(self.live, self.hooks, self.store, clock_=lambda: self.now[0], day_=lambda: self.day[0])

    def tick(self, secs_=2.0):
        self.now[0] += secs_; self.up[0] += secs_; return self.w.step()

    def finish_ok(self):   # the reset thread ends, the camera delivers, the gimbal is up
        self.live.cam_hold = False; self.live.cam_fail_n = 0; self.live.last_frame_t = self.now[0] + 1; self.tick(); return self.tick()


class Trigger(WatcherCase):
    def test_below_n_nothing_at_n_one_reset(self):
        ar.state(self.live)['usb_n'] = 4
        self.assertEqual(self.tick(), 'idle'); self.assertEqual(self.hooks.calls, [])
        ar.state(self.live)['usb_n'] = 5
        self.assertEqual(self.tick(), 'started'); self.assertEqual(self.hooks.calls, ['reset'])
        self.assertIn('CAMAUTORESET starting the camera reset', self.hooks.said[-1]); self.assertIn('5 failed opens in a row, cause libusb_timeout', self.hooks.said[-1])
        self.assertEqual(self.tick(), 'resetting'); self.assertEqual(self.hooks.calls, ['reset'])   # no second request while it runs
        self.assertEqual(self.finish_ok(), 'done')
        self.assertIn('CAMAUTORESET result', self.hooks.said[-1]); self.assertIn('camera frames again, gimbal ok', self.hooks.said[-1])
        h_ = ar.health(self.live)
        self.assertEqual((h_['mode'], h_['count_today'], h_['phase'], h_['usb_fail_n']), ('1', 1, 'idle', 0)); self.assertTrue(h_['last'])

    def test_n_from_the_environment(self):
        os.environ['LWIR_CAM_AUTORESET_N'] = '3'; ar.state(self.live)['usb_n'] = 3
        self.assertEqual(self.tick(), 'started')

    def test_off_and_dry(self):
        ar.set_mode('0'); self.assertEqual(self.tick(), 'off'); self.assertEqual(self.hooks.calls, [])
        ar.set_mode('dry'); self.assertEqual(self.tick(), 'dry'); self.assertEqual(self.hooks.calls, [])
        self.assertIn('dry run', self.hooks.said[-1]); self.assertEqual(ar.health(self.live)['dry_today'], 1); self.assertEqual(ar.health(self.live)['count_today'], 0)
        self.assertEqual(self.tick(), 'skip')   # the 30 minute rule holds for the decision too: no log every 2 s
        self.assertEqual(len([s_ for s_ in self.hooks.said if 'dry run' in s_]), 1)

    def test_a_frame_or_a_reset_ends_the_chain(self):
        self.live.cam_fail_n = 0
        self.assertEqual(self.tick(), 'idle'); self.assertEqual(ar.state(self.live)['usb_n'], 0)

    def test_stopped_server(self):
        self.live.stopped = True; self.assertEqual(self.tick(), 'stopped'); self.assertEqual(self.hooks.calls, [])


class Guards(WatcherCase):
    def test_manual_control_waits_and_goes_when_idle(self):
        self.hooks.manual = 'a move command 3 s ago'
        for _ in range(5): self.assertEqual(self.tick(), 'skip')
        self.assertEqual(self.hooks.calls, []); self.assertIn('drives the gimbal by hand', ar.state(self.live)['skip'])
        self.assertEqual(len([s_ for s_ in self.hooks.said if 'waiting' in s_]), 1)   # said once a minute, not every poll
        self.hooks.manual = ''
        self.assertEqual(self.tick(), 'started'); self.assertEqual(self.hooks.calls, ['reset']); self.assertEqual(ar.state(self.live)['skip'], '')

    def test_manual_hooks_on_the_real_state(self):
        gb_ = NS(mover=None, last_manual_t=-1e9, state={'moving': 0}); h_ = ar.Hooks(NS(GB=gb_))
        self.assertEqual(h_.manual_active(100.0), '')
        gb_.last_manual_t = 95.0; self.assertIn('move command', h_.manual_active(100.0))
        self.assertEqual(h_.manual_active(106.0), '')                      # 11 s after: idle
        gb_.mover = '10.42.0.2'; self.assertIn('page holds', h_.manual_active(100.0)); gb_.mover = None
        gb_.state['moving'] = 1; self.assertIn('bridge reports', h_.manual_active(200.0))
        self.assertEqual(ar.Hooks(NS(GB=None)).manual_active(1.0), '')

    def test_replay_measurement_running_reset(self):
        self.hooks.replay = True; self.assertEqual(self.tick(), 'skip'); self.assertIn('replay', ar.state(self.live)['skip'])
        self.hooks.replay = False; self.hooks.meas = True; self.assertEqual(self.tick(), 'skip'); self.assertIn('measurement', ar.state(self.live)['skip'])
        self.hooks.meas = False; self.live.cam_hold = True; self.assertEqual(self.tick(), 'skip')
        self.live.cam_hold = False; self.assertEqual(self.hooks.calls, [])
        self.assertEqual(self.tick(), 'started')

    def test_replay_hook_on_the_real_namespace(self):
        live_ = NS(replay=None, meas=None)
        with patch('src.camera.replay_link.blocking', return_value=False):
            self.assertFalse(ar.Hooks(NS(LIVE=live_)).replay_active())
            live_.replay = {'name': 'x'}; self.assertTrue(ar.Hooks(NS(LIVE=live_)).replay_active())   # a replay does not use the camera
            live_.replay = None
        with patch('src.camera.replay_link.blocking', return_value=True): self.assertTrue(ar.Hooks(NS(LIVE=live_)).replay_active())

    def test_camera_not_behind_the_hub_uses_no_budget(self):
        self.hooks.place = 'root'; self.assertEqual(self.tick(), 'skip'); self.assertIn('not behind the switchable hub', ar.state(self.live)['skip'])
        self.assertEqual(self.hooks.calls, [])
        self.hooks.place = None; self.assertEqual(self.tick(), 'skip')
        self.hooks.place = 'hub'; self.assertEqual(self.tick(), 'started')

    def test_a_refused_request_uses_no_budget(self):
        self.hooks.ok = (False, '측정 중에는 초기화하지 않음')
        self.assertEqual(self.tick(), 'skip'); self.assertIn('refused', ar.state(self.live)['skip']); self.assertEqual(self.store.load('2026-11-22')['up_s'], None)
        self.hooks.ok = (True, 'x'); self.assertEqual(self.tick(), 'started')


class Limit(WatcherCase):
    def test_one_reset_per_30_minutes(self):
        self.assertEqual(self.tick(), 'started'); self.finish_ok()
        ar.state(self.live)['usb_n'] = 5; self.live.cam_fail_n = 5
        for _ in range(3): self.assertEqual(self.tick(60.0), 'skip')
        self.assertEqual(self.hooks.calls, ['reset']); self.assertIn('at most one reset per 30 min', ar.state(self.live)['skip'])
        self.assertGreater(ar.health(self.live)['next_in_s'], 1000)
        self.up[0] += 1500; self.assertEqual(self.tick(), 'skip')              # about 28 minutes after the first
        self.up[0] += 200; self.assertEqual(self.tick(), 'started')            # more than 30
        self.assertEqual(self.hooks.calls, ['reset', 'reset']); self.assertEqual(ar.health(self.live)['count_today'], 2)

    def test_the_limit_survives_a_server_restart_in_the_same_boot_but_not_a_reboot(self):
        self.assertEqual(self.tick(), 'started'); self.finish_ok()
        self.live = NS(stopped=False, cam_fail_n=5, cam_hold=False, last_frame_t=-1e9); self.hooks.live = self.live   # the server restarted: new objects, same file
        self.up[0] += 60; self.w = self.make(); ar.state(self.live)['usb_n'] = 5; ar.state(self.live)['cause'] = 'CAMERA_FROZEN'
        self.assertEqual(self.tick(), 'skip'); self.assertEqual(self.hooks.calls, ['reset'])
        self.boot[0] = 'boot-2'; self.w = self.make(); ar.state(self.live)['usb_n'] = 5
        self.assertEqual(self.tick(), 'started')                               # a reboot: free (the board has no RTC, the uptime restarts)

    def test_count_today_resets_with_the_date(self):
        self.assertEqual(self.tick(), 'started'); self.finish_ok()
        self.assertEqual(ar.health(self.live)['count_today'], 1)
        self.day[0] = '2026-11-23'; self.up[0] += 2000; ar.state(self.live)['usb_n'] = 5; self.live.cam_fail_n = 5
        self.assertEqual(self.tick(), 'started'); self.assertEqual(ar.health(self.live)['count_today'], 1)
        self.assertEqual(json.load(open(self.store.path))['day'], '2026-11-23')


class AfterReset(WatcherCase):
    def start_reset(self):
        self.assertEqual(self.tick(), 'started'); self.live.cam_hold = False; self.assertEqual(self.tick(), 'checking')

    def test_no_power_step_is_retried_soon_and_given_up_after_three(self):
        self.hooks.text = 'USB 전원 껐다 켜기 실패(보드에 sudo 허용 설정 필요). 케이블을 뽑았다 꽂으세요. '
        self.assertEqual(self.tick(), 'started'); self.live.cam_hold = False
        self.assertEqual(self.tick(), 'no-power-step'); self.assertIn('did not cut the power', self.hooks.said[-1]); self.assertEqual(ar.health(self.live)['count_today'], 0)
        ar.state(self.live)['usb_n'] = 5; self.live.cam_fail_n = 5
        self.assertEqual(self.tick(60.0), 'skip'); self.assertIn('retry in', ar.state(self.live)['skip'])
        self.assertEqual(self.tick(300.0), 'started'); self.live.cam_hold = False; self.assertEqual(self.tick(), 'no-power-step')
        ar.state(self.live)['usb_n'] = 5; self.live.cam_fail_n = 5
        self.assertEqual(self.tick(400.0), 'started'); self.live.cam_hold = False; self.assertEqual(self.tick(), 'no-power-step')   # third: now the 30 minutes hold
        ar.state(self.live)['usb_n'] = 5; self.live.cam_fail_n = 5
        self.assertEqual(self.tick(400.0), 'skip'); self.assertIn('at most one reset', ar.state(self.live)['skip'])

    def test_a_cut_that_did_not_bring_the_camera_back_still_counts_as_the_one_reset(self):
        self.hooks.text = 'USB 전원을 껐다 켰지만 카메라가 보이지 않음 (케이블을 뽑았다 꽂으세요). '
        self.assertEqual(self.tick(), 'started'); self.live.cam_hold = False
        self.assertEqual(self.tick(), 'checking')                                   # not 'no-power-step': the hub was cycled, no second cycle within 30 minutes
        ar.state(self.live)['usb_n'] = 5; self.live.cam_fail_n = 5; ar.state(self.live)['phase'] = 'idle'
        self.assertEqual(self.tick(600.0), 'skip'); self.assertIn('at most one reset', ar.state(self.live)['skip'])

    def test_gimbal_bridge_dead_is_started_then_ready(self):
        self.start_reset()
        self.hooks.g = dict(present=True, alive=False, ready=False, error=False, dev_ok=True)
        self.assertEqual(self.tick(20.0), 'checking'); self.assertNotIn('gimbal-spawn', self.hooks.calls)     # gb_watch gets its time first
        self.assertEqual(self.tick(20.0), 'checking'); self.assertEqual(self.hooks.calls.count('gimbal-spawn'), 1)
        self.assertEqual(self.tick(10.0), 'checking'); self.assertEqual(self.hooks.calls.count('gimbal-spawn'), 1)   # not again within 20 s
        self.assertEqual(self.tick(15.0), 'checking'); self.assertEqual(self.hooks.calls.count('gimbal-spawn'), 2)   # retried
        self.hooks.g = dict(present=True, alive=True, ready=True, error=False, dev_ok=True); self.live.cam_fail_n = 0; self.live.last_frame_t = self.now[0]
        self.assertEqual(self.tick(), 'done'); self.assertIn('gimbal ok', self.hooks.said[-1])

    def test_gimbal_alive_but_not_ready_is_ended_twice_at_most(self):
        self.start_reset()
        self.hooks.g = dict(present=True, alive=True, ready=False, error=False, dev_ok=True)
        self.tick(40.0); self.assertNotIn('gimbal-end', self.hooks.calls)
        self.tick(10.0); self.assertEqual(self.hooks.calls.count('gimbal-end'), 1)
        self.tick(25.0); self.assertEqual(self.hooks.calls.count('gimbal-end'), 2)
        for _ in range(3): self.tick(20.0)
        self.assertEqual(self.hooks.calls.count('gimbal-end'), 2)

    def test_result_when_nothing_came_back(self):
        self.start_reset()
        self.hooks.g = dict(present=True, alive=False, ready=False, error=False, dev_ok=False)   # the adapter is not even on USB
        r_ = None
        for _ in range(8):
            r_ = self.tick(20.0)
            if r_ == 'done': break
        self.assertEqual(r_, 'done'); self.assertIn('camera still no frame, gimbal not ready', self.hooks.said[-1]); self.assertEqual(self.hooks.calls.count('gimbal-spawn'), 0)   # no device: nothing to start
        self.assertEqual(ar.state(self.live)['phase'], 'idle'); self.assertEqual(self.tick(), 'idle')   # chain restarts from 0

    def test_without_a_gimbal(self):
        self.hooks.g = dict(present=False); self.start_reset(); self.live.cam_fail_n = 0; self.live.last_frame_t = self.now[0]
        self.assertEqual(self.tick(), 'done'); self.assertIn('gimbal ok', self.hooks.said[-1])


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('g++'), 'the stand-in SDK build needs Linux with g++')
class RealCamPipe(unittest.TestCase):
    """The real cam_pipe.cpp against the stand-in SDK: the exit code and the log text of the three situations the cause is decided from."""
    def test_causes_from_the_real_cam_pipe(self):
        from . import test_cam_pipe_frozen as fake
        root_ = tempfile.mkdtemp(prefix='ar_pipe_'); self.addCleanup(shutil.rmtree, root_, True)
        exe_ = os.environ.get('CAM_PIPE_FAKE_BIN') or fake.build(root_)[0]
        shutil.copy(os.path.join(fake.HERE, 'cam_shutter.txt'), os.path.dirname(exe_))   # the shutter file next to the binary, as on the board
        names_ = ('first_open_fails', 'frozen_control', 'reopen_not_connected')
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(names_)) as pool:
            out_ = dict(zip(names_, pool.map(lambda n_: fake.run_case(exe_, root_, n_), names_)))
        text_ = lambda r_: r_['err']
        self.assertEqual((out_['first_open_fails']['rc'], ar.usb_cause(out_['first_open_fails']['rc'], text_(out_['first_open_fails']))), (2, 'libusb_timeout'))
        self.assertEqual((out_['frozen_control']['rc'], ar.usb_cause(out_['frozen_control']['rc'], text_(out_['frozen_control']))), (8, 'CAMERA_FROZEN'))
        self.assertEqual(ar.usb_cause(out_['reopen_not_connected']['rc'], text_(out_['reopen_not_connected'])), '')   # the cable is out: not a frozen chip


if __name__ == '__main__':
    unittest.main()
