"""11-15: cam_pipe v2.12 "frozen camera" exit (code 8) and the repair paths it must leave alone, run against a stand-in SDK (fake_i3/) without a camera.

The camera's own USB chip sometimes stops answering while it stays on the bus (DIAG control/send libusb_timeout, ResetMainBoard refused); only a power cut clears
it. cam_pipe now exits with code 8 at once in that case (and when the reopen after a stall times out the same way after frames had streamed), instead of the
20 s reopen loop. Everything else must behave as before: libusb_no_device stalls and stalls with an accepted ResetMainBoard are repaired by the reopen, a first
open that fails ends with 2, other reopen failures with 3, CAM_NO_RECOVER with 7.

Linux with g++ only (skipped elsewhere). cam_pipe.cpp is built with fake_i3/ into a temporary folder, or CAM_PIPE_FAKE_BIN=<binary built that way> is used.
All cases run at the same time, each with its own HOME (cam_pipe's lock file) and plan (FAKE_I3_PLAN, see fake_i3/fake_i3.cpp). cam_pipe keeps its real times
(6 s stall, 20 s wait before the reopen, 10 scans of 1 s), so the whole file takes about 40 s.
Run from the tracking_server folder: python3 -m unittest src.camera.test_cam_pipe_frozen
"""
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_TAIL = ": the camera's USB chip does not answer; only a power cut (unplug ~10 s, or the 5 V step on a switchable port) clears it, at "

# name: (FAKE_I3_PLAN, extra environment, seconds argument). Plan steps: ok,<frames>,<op>,<result>,<reset> | fail,<op>,<result> | gone
CASES = dict(
    frozen_control=('ok,30,3,-7,0', {}, 60),               # 11-13 20:33 on the board: DIAG stall op=3 libusb_timeout, ResetMainBoard sent=0
    frozen_send=('ok,30,4,-7,0', {}, 60),
    no_device=('ok,30,3,-4,0;ok,-1,0,1,1', {}, 32),       # 11-13 16:55 / 17:02: libusb_no_device, back by itself 20 s later
    reset_accepted=('ok,30,3,-7,1;ok,-1,0,1,1', {}, 32),  # the engine took the reset: the old repair
    receive_timeout=('ok,30,5,-7,0;ok,-1,0,1,1', {}, 32), # the old soft stall (receive): not a frozen chip
    reopen_frozen=('ok,30,5,-7,0;fail,4,-7', {}, 60),      # the reopen after a stall times out in send after frames had streamed
    reopen_not_connected=('ok,30,5,-7,0;fail,2,7', {}, 60),
    reopen_gone=('ok,30,5,-7,0;gone', {}, 60),
    first_open_fails=('fail,4,-7', {}, 60),               # nothing streamed in this process: exit 2 as before
    no_recover=('ok,30,3,-7,0', {'CAM_NO_RECOVER': '1'}, 60),
    test_hook_frozen=('ok,-1,0,1,1', {'CAM_PIPE_TEST_STALL_AT': '1', 'CAM_PIPE_TEST_FROZEN': '1'}, 60),
    test_hook_plain=('ok,-1,0,1,1;ok,-1,0,1,1', {'CAM_PIPE_TEST_STALL_AT': '1'}, 32),
)


def build(dest_):   # (binary path, compiler output)
    exe_ = os.path.join(dest_, 'cam_pipe_fake')
    cmd_ = ['g++', '-O2', '-Wall', '-Wextra', '-D_LINUX', '-I' + os.path.join(HERE, 'fake_i3'), os.path.join(HERE, 'cam_pipe.cpp'),
            os.path.join(HERE, 'fake_i3', 'fake_i3.cpp'), '-o', exe_, '-lpthread']
    r_ = subprocess.run(cmd_, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    if r_.returncode != 0: raise RuntimeError('build failed:\n' + r_.stdout.decode('utf-8', 'replace'))
    return exe_, r_.stdout.decode('utf-8', 'replace')


def run_case(exe_, root_, name_):   # dict(rc, err, log, secs)
    plan_, extra_, secs_ = CASES[name_]
    home_ = os.path.join(root_, name_); os.makedirs(home_)
    env_ = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home_, FAKE_I3_PLAN=plan_, FAKE_I3_LOG=os.path.join(home_, 'sdk.log'), **extra_)
    t0_ = time.monotonic()
    p_ = subprocess.run([exe_, '-1', str(secs_)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env_, timeout=secs_ + 60)
    secs_used_ = time.monotonic() - t0_
    try:
        with open(env_['FAKE_I3_LOG']) as f_: log_ = f_.read().split('\n')[:-1]
    except OSError: log_ = []
    return dict(rc=p_.returncode, err=p_.stderr.decode('utf-8', 'replace'), log=log_, secs=secs_used_)


class CamPipeFrozen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = None
        exe_ = os.environ.get('CAM_PIPE_FAKE_BIN', '')
        if not exe_:
            if not sys.platform.startswith('linux') or shutil.which('g++') is None: raise unittest.SkipTest('needs Linux with g++ (or CAM_PIPE_FAKE_BIN)')
            cls.tmp = tempfile.mkdtemp(prefix='cam_pipe_frozen_')
            exe_, cls.build_out = build(cls.tmp)
            shutil.copy(os.path.join(HERE, 'cam_shutter.txt'), cls.tmp)   # 11-15: the shutter file of the repository next to the binary, as on the board
        else: cls.build_out = ''
        root_ = tempfile.mkdtemp(prefix='cam_pipe_frozen_runs_')
        cls.runs_root = root_
        with concurrent.futures.ThreadPoolExecutor(len(CASES)) as ex_:
            futs_ = {n_: ex_.submit(run_case, exe_, root_, n_) for n_ in CASES}
            cls.res = {n_: f_.result() for n_, f_ in futs_.items()}

    @classmethod
    def tearDownClass(cls):
        for d_ in (cls.tmp, getattr(cls, 'runs_root', None)):
            if d_: shutil.rmtree(d_, ignore_errors=True)

    def r(self, name_):
        return self.res[name_]

    def frozen_lines(self, err_):
        return [ln_ for ln_ in err_.splitlines() if ln_.startswith('CAMERA_FROZEN ')]

    def frames(self, err_):
        m_ = re.search(r'^FRAMES (\d+) in', err_, re.M)
        return int(m_.group(1)) if m_ else -1

    def test_build_has_no_warning_from_cam_pipe(self):
        self.assertNotIn('cam_pipe.cpp', self.build_out, self.build_out)

    def test_repository_shutter_file_gives_t120(self):   # 11-15: src/camera/cam_shutter.txt (setting on the first line, comments after it) is read as T120
        if self.tmp is None: self.skipTest('CAM_PIPE_FAKE_BIN given: the shutter file is not copied next to it')
        self.assertIn('SHUTTER setting=T120 (file) mode=TIME value=120 ', self.r('frozen_control')['err'])

    def test_stall_with_control_timeout_and_refused_reset_exits_8_at_once(self):
        r_ = self.r('frozen_control')
        self.assertEqual(r_['rc'], 8, r_['err'])
        fl_ = self.frozen_lines(r_['err'])
        self.assertEqual(len(fl_), 1, r_['err'])
        self.assertTrue(fl_[0].startswith('CAMERA_FROZEN at a stall after 30 frames (control libusb_timeout, ResetMainBoard refused)' + FROZEN_TAIL), fl_[0])
        self.assertEqual(r_['log'], ['open 1 ok', 'stop 1', 'reset 1 sent=0', 'close 1'])   # one ResetMainBoard, handle closed, no second open
        self.assertNotIn('REOPEN_FAILED', r_['err'])
        self.assertLess(r_['secs'], 15.0)                                                   # 1 s of frames + the 6 s stall, no 20 s wait
        self.assertEqual(self.frames(r_['err']), 30)                                        # the end-of-run lines are still written

    def test_stall_with_send_timeout_and_refused_reset_exits_8(self):
        r_ = self.r('frozen_send')
        self.assertEqual(r_['rc'], 8, r_['err'])
        self.assertIn('CAMERA_FROZEN at a stall after 30 frames (send libusb_timeout, ResetMainBoard refused)' + FROZEN_TAIL, r_['err'])
        self.assertEqual(r_['log'], ['open 1 ok', 'stop 1', 'reset 1 sent=0', 'close 1'])

    def assert_repaired(self, name_):   # the old repair: ResetMainBoard, close, 20 s, reopen, frames again, normal end at the seconds limit
        r_ = self.r(name_)
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertEqual(self.frozen_lines(r_['err']), [])
        self.assertEqual(r_['log'][:5], ['open 1 ok', 'stop 1', r_['log'][2], 'close 1', 'open 2 ok'], r_['log'])
        self.assertTrue(r_['log'][2].startswith('reset 1 sent='), r_['log'])
        self.assertGreater(self.frames(r_['err']), 60, r_['err'])
        return r_

    def test_no_device_stall_is_still_repaired_by_the_reopen(self):
        r_ = self.assert_repaired('no_device')
        self.assertIn('DIAG stall: op=3(control) result=-4(libusb_no_device)', r_['err'])

    def test_stall_with_accepted_reset_is_still_repaired(self):
        r_ = self.assert_repaired('reset_accepted')
        self.assertIn('ResetMainBoard sent=1', r_['err'])

    def test_receive_timeout_stall_is_not_taken_as_frozen(self):
        r_ = self.assert_repaired('receive_timeout')
        self.assertIn('ResetMainBoard sent=0', r_['err'])

    def test_reopen_that_times_out_after_streaming_exits_8(self):
        r_ = self.r('reopen_frozen')
        self.assertEqual(r_['rc'], 8, r_['err'])
        self.assertIn('DIAG open: op=4(send) result=-7(libusb_timeout)', r_['err'])
        self.assertIn('CAMERA_FROZEN at the reopen after a stall, after 30 frames (send libusb_timeout)' + FROZEN_TAIL, r_['err'])
        self.assertNotIn('REOPEN_FAILED', r_['err'])
        self.assertEqual(r_['log'], ['open 1 ok', 'stop 1', 'reset 1 sent=0', 'close 1', 'open 2 fail'])
        self.assertGreater(r_['secs'], 26.0)   # the stall went through the old 20 s wait first (its DIAG was receive)

    def test_other_reopen_failures_keep_exit_3(self):
        for name_, line_ in (('reopen_not_connected', 'DIAG open: op=2(open) result=7(device_not_connected)'), ('reopen_gone', 'NO_CAMERA_FOUND')):
            with self.subTest(name_):
                r_ = self.r(name_)
                self.assertEqual(r_['rc'], 3, r_['err'])
                self.assertIn(line_, r_['err'])
                self.assertIn('REOPEN_FAILED', r_['err'])
                self.assertEqual(self.frozen_lines(r_['err']), [])

    def test_first_open_that_times_out_keeps_exit_2(self):
        r_ = self.r('first_open_fails')
        self.assertEqual(r_['rc'], 2, r_['err'])
        self.assertIn('OPEN_FAILED', r_['err'])
        self.assertEqual(self.frozen_lines(r_['err']), [])
        self.assertEqual(r_['log'], ['open 1 fail'])

    def test_cam_no_recover_keeps_exit_7_without_reset(self):
        r_ = self.r('no_recover')
        self.assertEqual(r_['rc'], 7, r_['err'])
        self.assertIn('CAMERA_STALL_NO_RECOVER', r_['err'])
        self.assertEqual(self.frozen_lines(r_['err']), [])
        self.assertFalse(any(x_.startswith('reset') for x_ in r_['log']), r_['log'])

    def test_test_hook_takes_the_frozen_path_on_a_healthy_camera(self):
        r_ = self.r('test_hook_frozen')
        self.assertEqual(r_['rc'], 8, r_['err'])
        self.assertIn('TEST: the simulated stall is taken as frozen', r_['err'])
        self.assertEqual(len(self.frozen_lines(r_['err'])), 1)
        self.assertEqual(r_['log'], ['open 1 ok', 'close 1'])   # ResetMainBoard not called

    def test_test_hook_without_frozen_keeps_the_old_repair(self):
        r_ = self.r('test_hook_plain')
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertIn('TEST: simulated stall', r_['err'])
        self.assertEqual(r_['log'][:4], ['open 1 ok', 'reset 1 sent=1', 'close 1', 'open 2 ok'])
        self.assertEqual(self.frozen_lines(r_['err']), [])


if __name__ == '__main__':
    unittest.main()
