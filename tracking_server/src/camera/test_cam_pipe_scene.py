"""11-15: cam_pipe v2.13 against the stand-in SDK (fake_i3/), without a camera: calibration reuse, open count, the no-scene check, no close during a calibration.

On 11-15 the picture became a flat field (the middle of the picture one value, frames still at 30 fps) after many camera opens in a short time, each of which
moved the shutter. v2.13 (see the top of cam_pipe.cpp):
  (a) the first open of a process skips ShutterCalibrationOn when the state file says the last calibration of ours is recent and on the same camera power-on;
  (b) an OPENS line counts the opens of this camera power-on;
  (c) a picture whose middle is one value for LWIR_CAM_BLANK_S seconds gets one engine reset per camera power-on, then only lines (SCENE_LOST / SCENE_BACK);
  (d) a stop signal right after a calibration waits before CloseTE.
The camera's USB entry is a fake sysfs folder (LWIR_CAM_SYSFS); a replug is a new devnum in it. Each case has its own HOME (lock and state file).
Linux with g++ only (skipped elsewhere), or CAM_PIPE_FAKE_BIN=<cam_pipe built with fake_i3/>. The cases run at the same time; the whole file takes about 50 s.
Run from the tracking_server folder: python3 -m unittest src.camera.test_cam_pipe_scene
"""
import concurrent.futures
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from . import test_cam_pipe_frozen as base

HERE = os.path.dirname(os.path.abspath(__file__))


def fake_sysfs(root_, devnum_):   # one camera entry 3-1 (0547:0080) on bus 3 with this device number
    d_ = os.path.join(root_, 'sys', '3-1'); os.makedirs(d_, exist_ok=True)
    for k_, v_ in (('idVendor', '0547'), ('idProduct', '0080'), ('busnum', '3'), ('devnum', str(devnum_))):
        with open(os.path.join(d_, k_), 'w') as f_: f_.write(v_ + '\n')
    return os.path.join(root_, 'sys')


def run(exe_, home_, plan_, secs_, extra_=None, term_at_=None):   # one cam_pipe run in home_: dict(rc, err, log, secs, term_to_exit)
    os.makedirs(home_, exist_ok=True)
    logp_ = os.path.join(home_, 'sdk.log')
    try: os.unlink(logp_)
    except OSError: pass
    env_ = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home_, FAKE_I3_PLAN=plan_, FAKE_I3_LOG=logp_, FAKE_I3_LOG_CAL='1', **(extra_ or {}))
    t0_ = time.monotonic(); term_to_exit_ = None
    p_ = subprocess.Popen([exe_, '-1', str(secs_)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env_)
    if term_at_ is not None:
        time.sleep(term_at_); t_term_ = time.monotonic(); p_.send_signal(signal.SIGTERM)
        err_ = p_.communicate(timeout=60)[1]; term_to_exit_ = time.monotonic() - t_term_
    else: err_ = p_.communicate(timeout=secs_ + 90)[1]
    try:
        with open(logp_) as f_: log_ = f_.read().split('\n')[:-1]
    except OSError: log_ = []
    return dict(rc=p_.returncode, err=err_.decode('utf-8', 'replace'), log=log_, secs=time.monotonic() - t0_, term_to_exit=term_to_exit_)


def seq_reuse(exe_, root_):   # (a)+(b): calibrate, reuse, replug -> calibrate, LWIR_CAM_CAL_REUSE_S=0 -> calibrate, short run -> no commit
    home_ = os.path.join(root_, 'reuse'); sysfs_ = fake_sysfs(home_, 5); env_ = dict(LWIR_CAM_SYSFS=sysfs_)
    out_ = dict(first=run(exe_, home_, 'ok,-1,0,1,1', 8, env_))
    out_['second'] = run(exe_, home_, 'ok,-1,0,1,1', 3, env_)
    fake_sysfs(home_, 6)   # replug: new device number
    out_['replugged'] = run(exe_, home_, 'ok,-1,0,1,1', 3, env_)
    out_['reuse_off'] = run(exe_, home_, 'ok,-1,0,1,1', 3, dict(env_, LWIR_CAM_CAL_REUSE_S='0'))
    home2_ = os.path.join(root_, 'short'); env2_ = dict(LWIR_CAM_SYSFS=fake_sysfs(home2_, 5))
    out_['short'] = run(exe_, home2_, 'ok,-1,0,1,1', 2, env2_)                     # ends 2 s after the calibration: not counted
    out_['after_short'] = run(exe_, home2_, 'ok,-1,0,1,1', 3, env2_)
    return out_


def seq_scene(exe_, root_):   # (c): blank picture -> one engine reset, the scene comes back; the next process on the same power-on gets no second reset
    home_ = os.path.join(root_, 'scene'); env_ = dict(LWIR_CAM_SYSFS=fake_sysfs(home_, 5), LWIR_CAM_BLANK_S='2')
    out_ = dict(reset=run(exe_, home_, 'ok,-1,0,1,1,30,-1;ok,-1,0,1,1', 40, env_))
    out_['again'] = run(exe_, home_, 'ok,-1,0,1,1,30,-1', 12, env_)
    return out_


def seq_term(exe_, root_):   # (d): SIGTERM 0.5 s after the open (its calibration) waits before CloseTE
    home_ = os.path.join(root_, 'term')
    return dict(term=run(exe_, home_, 'ok,-1,0,1,1', 30, dict(LWIR_CAM_SYSFS=fake_sysfs(home_, 5)), term_at_=0.5))


def one(name_, plan_, secs_, extra_):
    def go(exe_, root_):
        home_ = os.path.join(root_, name_)
        return {name_: run(exe_, home_, plan_, secs_, dict(extra_, LWIR_CAM_SYSFS=fake_sysfs(home_, 5)))}
    return go


JOBS = [seq_reuse, seq_scene, seq_term,
        one('reset_off', 'ok,-1,0,1,1,30,180', 16, dict(LWIR_CAM_BLANK_S='2', LWIR_CAM_BLANK_RESET='0')),   # blank 5 s, lines only, then the scene comes back
        one('check_off', 'ok,-1,0,1,1,30,-1', 8, dict(LWIR_CAM_BLANK_S='0')),
        one('normal', 'ok,-1,0,1,1', 10, dict(LWIR_CAM_BLANK_S='2')),
        one('no_recover', 'ok,-1,0,1,1,30,-1', 8, dict(LWIR_CAM_BLANK_S='2', CAM_NO_RECOVER='1'))]


class CamPipeScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = None
        exe_ = os.environ.get('CAM_PIPE_FAKE_BIN', '')
        if not exe_:
            if not sys.platform.startswith('linux') or shutil.which('g++') is None: raise unittest.SkipTest('needs Linux with g++ (or CAM_PIPE_FAKE_BIN)')
            cls.tmp = tempfile.mkdtemp(prefix='cam_pipe_scene_')
            exe_, cls.build_out = base.build(cls.tmp)
            shutil.copy(os.path.join(HERE, 'cam_shutter.txt'), cls.tmp)
        else: cls.build_out = ''
        cls.runs_root = tempfile.mkdtemp(prefix='cam_pipe_scene_runs_')
        cls.res = {}
        with concurrent.futures.ThreadPoolExecutor(len(JOBS)) as ex_:
            for f_ in [ex_.submit(j_, exe_, cls.runs_root) for j_ in JOBS]: cls.res.update(f_.result())

    @classmethod
    def tearDownClass(cls):
        for d_ in (cls.tmp, getattr(cls, 'runs_root', None)):
            if d_: shutil.rmtree(d_, ignore_errors=True)

    def lines(self, r_, head_):
        return [ln_ for ln_ in r_['err'].splitlines() if ln_.startswith(head_)]

    def cals(self, r_):
        return [x_ for x_ in r_['log'] if x_.startswith('cal ')]

    def test_build_has_no_warning_from_cam_pipe(self):
        self.assertNotIn('cam_pipe.cpp', self.build_out, self.build_out)

    def test_first_open_calibrates_and_the_next_process_reuses_it(self):
        a_, b_ = self.res['first'], self.res['second']
        self.assertEqual(a_['rc'], 0, a_['err']); self.assertEqual(b_['rc'], 0, b_['err'])
        self.assertEqual(self.cals(a_), ['cal 1'], a_['log'])
        self.assertIn('CALIBRATION at this open: first calibration of ours since the board started', a_['err'])
        self.assertEqual(self.cals(b_), [], b_['log'])
        self.assertTrue(any(x_.startswith('CALIBRATION REUSED (no ShutterCalibrationOn at this open): the last calibration of ours was ') and 'on this camera power-on (USB 3-1:3:5)' in x_
                            for x_ in b_['err'].splitlines()), b_['err'])
        self.assertIn(' calibration_at_open=-2 ', b_['err'])
        self.assertIn('set_mode=1 set_value=1', b_['err'])   # the shutter mode and time are still set

    def test_replug_and_reuse_off_calibrate_again(self):
        r_ = self.res['replugged']
        self.assertEqual(self.cals(r_), ['cal 1'], r_['log'])
        self.assertIn('the camera was plugged in or powered again since the last calibration (USB 3-1:3:5 -> 3-1:3:6)', r_['err'])
        o_ = self.res['reuse_off']
        self.assertEqual(self.cals(o_), ['cal 1'], o_['log'])
        self.assertIn('LWIR_CAM_CAL_REUSE_S=0: calibration at every open', o_['err'])

    def test_a_calibration_counts_only_after_frames_ran_after_it(self):
        self.assertEqual(self.cals(self.res['short']), ['cal 1'])
        self.assertEqual(self.cals(self.res['after_short']), ['cal 1'], self.res['after_short']['err'])
        self.assertIn('CLOSE_WAIT the shutter calibration of this open began ', self.res['short']['err'])

    def test_opens_are_counted_per_camera_power_on(self):
        n_ = [re.search(r'^OPENS this camera power-on \(USB ([^)]*)\): (\d+) in the last 10 min', self.res[k_]['err'], re.M) for k_ in ('first', 'second', 'replugged', 'reuse_off')]
        self.assertTrue(all(n_), [self.res[k_]['err'] for k_ in ('first', 'second')])
        self.assertEqual([(m_.group(1), int(m_.group(2))) for m_ in n_], [('3-1:3:5', 1), ('3-1:3:5', 2), ('3-1:3:6', 1), ('3-1:3:6', 2)])
        self.assertIn(', started by ', self.res['first']['err'])

    def test_no_scene_gets_one_engine_reset_and_the_scene_comes_back(self):
        r_ = self.res['reset']
        self.assertEqual(r_['rc'], 0, r_['err'])
        lost_ = self.lines(r_, 'SCENE_LOST ')
        self.assertEqual(len(lost_), 1, r_['err'])
        self.assertIn('one engine reset now (ResetMainBoard, close, 20 s, reopen with a calibration), the only one on this camera power-on (USB 3-1:3:5)', lost_[0])
        self.assertEqual(len(self.lines(r_, 'CAMERA_RESET_FOR_SCENE ')), 1, r_['err'])
        self.assertEqual(r_['log'][:6], ['open 1 ok', 'cal 1', 'reset 1 sent=1', 'close 1', 'open 2 ok', 'cal 2'], r_['log'])
        self.assertEqual(len(self.lines(r_, 'SCENE_BACK ')), 1, r_['err'])
        self.assertNotIn('CAMERA_STALL ', r_['err'])

    def test_no_second_engine_reset_on_the_same_camera_power_on(self):
        r_ = self.res['again']
        self.assertEqual(r_['rc'], 0, r_['err'])
        lost_ = self.lines(r_, 'SCENE_LOST ')
        self.assertEqual(len(lost_), 1, r_['err'])
        self.assertIn('no automatic repair left (the one engine reset of this camera power-on was at ', lost_[0])
        self.assertIn('unplug the camera USB cable for 10 s and plug it back', lost_[0])
        self.assertFalse(any(x_.startswith('reset') for x_ in r_['log']), r_['log'])

    def test_reset_switch_off_gives_lines_only_and_scene_back(self):
        r_ = self.res['reset_off']
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertIn('no automatic repair (LWIR_CAM_BLANK_RESET=0)', r_['err'])
        self.assertEqual(len(self.lines(r_, 'SCENE_BACK ')), 1, r_['err'])
        self.assertFalse(any(x_.startswith('reset') for x_ in r_['log']), r_['log'])

    def test_check_off_and_normal_pictures_and_no_recover(self):
        self.assertIn('SCENE_WATCH off (LWIR_CAM_BLANK_S=0)', self.res['check_off']['err'])
        self.assertEqual(self.lines(self.res['check_off'], 'SCENE_LOST'), [])
        self.assertEqual(self.lines(self.res['normal'], 'SCENE_'), ['SCENE_WATCH on: no scene = centre_same >= 0.5 for 2 s, then one engine reset per camera power-on'])
        nr_ = self.res['no_recover']
        self.assertIn('no automatic repair (CAM_NO_RECOVER)', nr_['err'])
        self.assertFalse(any(x_.startswith('reset') for x_ in nr_['log']), nr_['log'])

    def test_stop_right_after_the_calibration_waits_for_it(self):
        r_ = self.res['term']
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertIn('CLOSE_WAIT the shutter calibration of this open began ', r_['err'])
        self.assertGreater(r_['term_to_exit'], 1.5)
        self.assertLess(r_['term_to_exit'], 5.0)
        self.assertEqual(r_['log'], ['open 1 ok', 'cal 1', 'close 1'])


if __name__ == '__main__':
    unittest.main()
