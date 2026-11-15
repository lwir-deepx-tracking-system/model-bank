"""11-15: cam_pipe v2.14 read pacing against the stand-in SDK (fake_i3/) with a camera frame clock, without a camera.

Measured on the board on 11-15 (every read logged): the camera makes a picture every 33.333 ms. cam_pipe v2.7 to v2.13 started each read 33.333 ms
after the ACTUAL start of the read before, so every wake-up delay (64 us on the board) added up: reads every 33.400 ms. The reads slid later until,
7.2 ms after the place where the stream was picked up, the camera gave the same picture 4 to 5 times and then no frame (0.23 s without a new
picture, every 3.7 s). v2.14 (see the top of cam_pipe.cpp): the reads sit on a grid counted from the PLANNED start, and the period follows the
camera's own frame period (learned from how long a grid lasted before reads came too early or too late; kept in the state file).

The stand-in camera (FAKE_I3_FRAME_US) behaves as measured: more than 7.2 ms late = 5 repeats then an empty read, more than 11 ms early = an empty
read, the first read after an empty one picks the stream up. Each case has its own HOME (lock and state file). Linux with g++ only (skipped
elsewhere), or CAM_PIPE_FAKE_BIN=<cam_pipe built with fake_i3/>. The cases run at the same time; the whole file takes about 95 s.
Run from the tracking_server folder: python3 -m unittest src.camera.test_cam_pipe_pacing
"""
import concurrent.futures
import os
import re
import shutil
import sys
import tempfile
import unittest

from . import test_cam_pipe_frozen as base
from . import test_cam_pipe_scene as scene

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = 'ok,-1,0,1,1'
# A lesson needs reads that started on time: within 250 us on the board. A PC (WSL) wakes up 200-400 us late on average, so the tests allow 2 ms;
# the stand-in camera's band of repeats at the late edge is 2.5 ms wide, so a read that late still cannot jump over it.
ON_TIME = dict(LWIR_CAM_PACE_ON_TIME_US='2000')


def one(name_, secs_, extra_):
    def go(exe_, root_):
        home_ = os.path.join(root_, name_)
        return {name_: scene.run(exe_, home_, PLAN, secs_, dict(ON_TIME, **extra_, LWIR_CAM_SYSFS=scene.fake_sysfs(home_, 5)))}
    return go


def seq_slow_camera(exe_, root_):   # the camera's frame is 10 us longer than our period: the reads come too early, the period is learned, the next process starts with it
    home_ = os.path.join(root_, 'slow'); env_ = dict(ON_TIME, FAKE_I3_FRAME_US='33343', LWIR_CAM_SYSFS=scene.fake_sysfs(home_, 5))
    out_ = dict(slow=scene.run(exe_, home_, PLAN, 80, env_))
    try:
        with open(os.path.join(home_, '.cam_pipe_state')) as f_: out_['slow_state'] = f_.read()
    except OSError: out_['slow_state'] = ''
    out_['slow_again'] = scene.run(exe_, home_, PLAN, 12, env_)
    return out_


def seq_read_log(exe_, root_):   # every read in a file, settings from a file while running
    home_ = os.path.join(root_, 'hooks'); sysfs_ = scene.fake_sysfs(home_, 5)
    logp_ = os.path.join(home_, 'reads.txt'); ctl_ = os.path.join(home_, 'pace.txt')
    with open(ctl_, 'w') as f_: f_.write('period_us=20000 mode=grid\n')   # read at the first look, 1 s after the start
    r_ = scene.run(exe_, home_, PLAN, 6, dict(LWIR_CAM_READ_LOG=logp_, LWIR_CAM_PACE_FILE=ctl_, LWIR_CAM_SYSFS=sysfs_))
    with open(logp_) as f_: r_['reads'] = [x_.split() for x_ in f_.read().split('\n')[:-1]]
    return dict(hooks=r_)


JOBS = [one('old', 20, dict(FAKE_I3_FRAME_US='33333', LWIR_CAM_PACE_MODE='after')),      # v2.7 pacing against a camera with exactly our period
        one('grid', 20, dict(FAKE_I3_FRAME_US='33333')),                                 # the same camera, v2.14 default
        seq_slow_camera,
        one('fast', 80, dict(FAKE_I3_FRAME_US='33323')),                                 # the camera's frame is 10 us shorter: the reads come too late
        one('no_trim', 30, dict(FAKE_I3_FRAME_US='33383', LWIR_CAM_PACE_TRIM='0')),      # 50 us longer and learning off
        one('filter_off', 20, dict(FAKE_I3_FRAME_US='33283', LWIR_CAM_DROP_DUP='0', LWIR_CAM_PACE_TRIM='0')),   # 50 us shorter, duplicate filter off: late must still be told from early
        seq_read_log]


def stats(r_):   # the last READ_STATS line as numbers
    ln_ = [x_ for x_ in r_['err'].splitlines() if x_.startswith('READ_STATS ')][-1]
    m_ = re.search(r'reads (\d+), new pictures (\d+) \(([\d.]+) a second\), repeats (\d+), empty reads (\d+), longest time without a new picture (\d+) ms, reads too early (\d+), too late (\d+), '
                   r'grid places skipped after a late start (\d+), period in use ([\d.]+) us \(mode (\w+)\)', ln_)
    assert m_, ln_
    k_ = ('reads', 'new', 'new_s', 'repeats', 'empty', 'gap_ms', 'early', 'late', 'skipped', 'period', 'mode')
    return {a_: (b_ if a_ == 'mode' else float(b_)) for a_, b_ in zip(k_, m_.groups())}


class CamPipePacing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = None
        exe_ = os.environ.get('CAM_PIPE_FAKE_BIN', '')
        if not exe_:
            if not sys.platform.startswith('linux') or shutil.which('g++') is None: raise unittest.SkipTest('needs Linux with g++ (or CAM_PIPE_FAKE_BIN)')
            cls.tmp = tempfile.mkdtemp(prefix='cam_pipe_pacing_')
            exe_, cls.build_out = base.build(cls.tmp)
            shutil.copy(os.path.join(HERE, 'cam_shutter.txt'), cls.tmp)
        else: cls.build_out = ''
        cls.runs_root = tempfile.mkdtemp(prefix='cam_pipe_pacing_runs_')
        cls.res = {}
        with concurrent.futures.ThreadPoolExecutor(len(JOBS)) as ex_:
            for f_ in [ex_.submit(j_, exe_, cls.runs_root) for j_ in JOBS]: cls.res.update(f_.result())

    @classmethod
    def tearDownClass(cls):
        for d_ in (cls.tmp, getattr(cls, 'runs_root', None)):
            if d_: shutil.rmtree(d_, ignore_errors=True)

    def lines(self, r_, head_):
        return [ln_ for ln_ in r_['err'].splitlines() if ln_.startswith(head_)]

    def test_build_has_no_warning_from_cam_pipe(self):
        self.assertNotIn('cam_pipe.cpp', self.build_out, self.build_out)

    def test_old_pacing_slides_late_and_stalls_on_a_camera_with_exactly_our_period(self):
        r_ = self.res['old']; s_ = stats(r_)
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertIn('READ_PACING period=33333 us (0 = off) mode=after', r_['err'])
        self.assertEqual(s_['mode'], 'after')
        self.assertGreaterEqual(s_['late'], 1, r_['err'])                 # wake-up delays added up until the reads were 7.2 ms late
        self.assertEqual(s_['early'], 0, r_['err'])
        self.assertGreaterEqual(len(self.lines(r_, 'REPEAT_RUN 5 identical frames in a row')), 1, r_['err'])
        self.assertGreaterEqual(s_['gap_ms'], 200, r_['err'])            # the stall the gimbal saw
        self.assertEqual(self.lines(r_, 'PACE_RESYNC'), [])               # the old mode prints nothing new
        self.assertIn('late', r_['log'])

    def test_grid_keeps_every_picture_on_the_same_camera(self):
        r_ = self.res['grid']; s_ = stats(r_)
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertIn('READ_PACING period=33333 us (0 = off) mode=grid, period in use 33333.000 us (follows the camera: yes), rests -4000 us from the pick-up place', r_['err'])
        self.assertEqual((s_['repeats'], s_['empty'], s_['early'], s_['late']), (0, 0, 0, 0), r_['err'])
        self.assertGreaterEqual(s_['new_s'], 29.8, r_['err'])
        self.assertLess(s_['gap_ms'], 60, r_['err'])
        self.assertEqual(self.lines(r_, 'REPEAT_RUN'), [])
        self.assertEqual(self.lines(r_, 'PACE_RESYNC'), [])
        self.assertEqual(r_['log'], ['open 1 ok', 'cal 1', 'close 1'], r_['log'])   # the camera saw no read too early or too late

    def test_slow_camera_reads_come_early_and_the_period_is_learned(self):
        r_ = self.res['slow']; s_ = stats(r_); ev_ = self.lines(r_, 'PACE_RESYNC ')
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertGreaterEqual(len(ev_), 2, r_['err'])
        self.assertTrue(all(x_.startswith('PACE_RESYNC reads came too early (an empty read) after ') for x_ in ev_), ev_)
        self.assertIn('nothing learned', ev_[0])                          # the first grid began at the open: its place is not known
        self.assertIn('read period 33333.000 -> 33338.000 us', ev_[1])    # 7 ms in about 700 reads = 10 us a read; one lesson moves at most 5 us
        self.assertEqual((s_['late'], s_['repeats']), (0, 0), r_['err'])
        self.assertLess(s_['gap_ms'], 100, r_['err'])                     # a read that came too early costs one frame, not a stall
        self.assertEqual(self.lines(r_, 'REPEAT_RUN'), [])
        self.assertGreaterEqual(s_['period'], 33337.9)
        self.assertLessEqual(s_['period'], 33343.0)
        m_ = re.search(r'^pace_trim_ns ([\d.]+)$', self.res['slow_state'], re.M)
        self.assertTrue(m_, self.res['slow_state'])
        self.assertGreaterEqual(float(m_.group(1)), 4999.0)
        self.assertIn('pace_period_us 33333', self.res['slow_state'])

    def test_the_next_process_starts_with_the_learned_period(self):
        r_ = self.res['slow_again']; s_ = stats(r_)
        self.assertEqual(r_['rc'], 0, r_['err'])
        ln_ = self.lines(r_, 'READ_PACING ')[0]
        self.assertIn('period from the state file', ln_)
        self.assertRegex(ln_, r'period in use 3333[89]\.\d\d\d us|period in use 3334[0-3]\.\d\d\d us')
        self.assertEqual((s_['early'], s_['late'], s_['repeats'], s_['empty']), (0, 0, 0, 0), r_['err'])

    def test_fast_camera_reads_come_late_and_the_period_is_shortened(self):
        r_ = self.res['fast']; s_ = stats(r_); ev_ = self.lines(r_, 'PACE_RESYNC ')
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertGreaterEqual(len(ev_), 2, r_['err'])
        self.assertTrue(all(x_.startswith('PACE_RESYNC reads came too late (repeated pictures, then an empty read) after ') for x_ in ev_), ev_)
        self.assertIn('nothing learned', ev_[0])
        self.assertIn('read period 33333.000 -> 33328.000 us', ev_[1])    # 11.2 ms in about 1120 reads = 10 us a read; one lesson moves at most 5 us
        self.assertEqual(s_['early'], 0, r_['err'])
        self.assertLessEqual(s_['period'], 33328.1)
        self.assertGreaterEqual(s_['period'], 33323.0)
        n_ = [int(re.search(r'after (\d+) reads', x_).group(1)) for x_ in ev_[:2]]
        self.assertGreater(n_[1], n_[0] * 1.3, ev_)                       # resting 4 ms earlier leaves more room on the late side: 11.2 ms instead of 7.2

    def test_learning_off_keeps_the_period(self):
        r_ = self.res['no_trim']; s_ = stats(r_); ev_ = self.lines(r_, 'PACE_RESYNC ')
        self.assertIn('(follows the camera: no)', r_['err'])
        self.assertGreaterEqual(len(ev_), 3, r_['err'])
        self.assertTrue(all('read period 33333.000 -> 33333.000 us (not following the camera)' in x_ for x_ in ev_), ev_)
        self.assertEqual(s_['period'], 33333.0)
        self.assertEqual((s_['late'], s_['repeats']), (0, 0), r_['err'])

    def test_late_is_told_from_early_with_the_duplicate_filter_off(self):
        r_ = self.res['filter_off']; s_ = stats(r_)
        self.assertIn('DUPLICATE_FILTER holds back up to 0 repeated frames in a row (0 = off)', r_['err'])
        self.assertGreaterEqual(s_['late'], 2, r_['err'])
        self.assertEqual(s_['early'], 0, r_['err'])
        self.assertGreaterEqual(s_['repeats'], 10, r_['err'])

    def test_read_log_and_settings_file(self):
        r_ = self.res['hooks']
        self.assertEqual(r_['rc'], 0, r_['err'])
        self.assertIn('READ_LOG ', r_['err']); self.assertIn('settings file on (LWIR_CAM_PACE_FILE)', r_['err'])
        ln_ = self.lines(r_, 'READ_PACING ')
        self.assertEqual(len(ln_), 2, ln_)
        self.assertIn('READ_PACING period=20000 us (0 = off) mode=grid, period in use 20000.000 us', ln_[1])
        rd_ = r_['reads']
        self.assertTrue(all(len(x_) == 7 for x_ in rd_), rd_[:3])
        self.assertEqual(len(rd_), int(stats(r_)['reads']))
        fast_ = [float(b_[0]) - float(a_[0]) for a_, b_ in zip(rd_, rd_[1:]) if b_[5] == '20000.000' and a_[5] == '20000.000']
        self.assertGreater(len(fast_), 150, len(fast_))
        self.assertAlmostEqual(sum(fast_) / len(fast_), 20.0, delta=0.02)   # the grid: late wake-ups do not add up
        slow_ = [float(b_[0]) - float(a_[0]) for a_, b_ in zip(rd_, rd_[1:]) if b_[5] == '33333.000' and a_[5] == '33333.000']
        self.assertAlmostEqual(sum(slow_) / len(slow_), 33.333, delta=0.03)


if __name__ == '__main__':
    unittest.main()
