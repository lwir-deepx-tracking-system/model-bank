"""Back from a recording to the camera: the camera program is started once and stays (2026-11-15).

On the board, 11-15 22:44: 'stop the recording' started cam_pipe, cam_pipe opened the camera, got SIGTERM at once and
closed it again (exit 0); the server noticed 20 s later and opened the camera a second time. 25 s without a picture and
two shutter calibrations. cam_pipe asks for SIGTERM when its parent ends (PR_SET_PDEATHSIG), and Linux sends that when
the THREAD that started it ends: the replay switch thread ends right after the start. The camera reset button's thread
had the same fault. spawn_keeper.popen starts the program from a thread that lives as long as the server.

Here:
  - the rule itself (any system),
  - Linux: a child that asks for the same signal dies when started straight from a short-lived thread (the fault) and
    lives when started through spawn_keeper,
  - Linux with g++ (or CAM_PIPE_FAKE_BIN=<cam_pipe built with fake_i3/>): the real cam_pipe with the stand-in SDK, both
    ways, with the camera opens counted in the SDK's log,
  - the page state 'returning' (replay_link) and the server's two start paths.

Run from the tracking_server folder: python3 -m unittest src.camera.test_spawn_keeper
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..common.state import APP
from . import camera_extra, camera_link, replay_link, spawn_keeper

LINUX = sys.platform.startswith('linux')
CHILD = ("import ctypes, signal, sys, time\n"
         "ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)\n"   # 1 = PR_SET_PDEATHSIG, as cam_pipe.cpp main()
         "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
         "time.sleep(30)\n")


def from_short_thread(work, linger=0.0):
    """work() in a thread that ends linger seconds after it, like replay_link._switch after lv_spawn (on the board that
    thread still writes its message, a few ms: longer than cam_pipe needs to ask for the signal). -> what work returned."""
    box = []

    def run():
        box.append(work())
        time.sleep(linger)

    thread = threading.Thread(target=run)
    thread.start(); thread.join(20)
    assert not thread.is_alive() and box
    return box[0]


class KeeperRule(unittest.TestCase):
    def test_main_thread_starts_it_itself(self):
        seen = []
        self.assertEqual(spawn_keeper.popen(lambda *a, **k: seen.append((threading.current_thread(), a, k)) or 'p', 1, x=2), 'p')
        self.assertEqual(seen, [(threading.main_thread(), (1,), dict(x=2))])

    def test_another_thread_hands_it_to_one_thread_that_stays(self):
        seen = []
        make = lambda *a, **k: seen.append(threading.current_thread()) or len(seen)
        self.assertEqual(from_short_thread(lambda: spawn_keeper.popen(make, 'cam_pipe')), 1)
        self.assertEqual(from_short_thread(lambda: spawn_keeper.popen(make, 'cam_pipe')), 2)
        self.assertIs(seen[0], seen[1])
        self.assertIsNot(seen[0], threading.main_thread())
        self.assertTrue(seen[0].is_alive() and seen[0].daemon)
        self.assertEqual(seen[0].name, 'lwir-camera-start')

    def test_an_error_goes_to_the_caller_and_the_keeper_stays(self):
        def fails(*a, **k):
            raise FileNotFoundError('no cam_pipe')

        def call():
            try: spawn_keeper.popen(fails)
            except FileNotFoundError as exc: return str(exc)

        self.assertEqual(from_short_thread(call), 'no cam_pipe')
        self.assertEqual(from_short_thread(lambda: spawn_keeper.popen(lambda: 'after')), 'after')

    def test_a_keeper_that_is_gone_is_replaced(self):
        from_short_thread(lambda: spawn_keeper.popen(lambda: None))
        with patch.dict(spawn_keeper._KEEPER, thread=NS(is_alive=lambda: False)):
            self.assertEqual(from_short_thread(lambda: spawn_keeper.popen(threading.current_thread)).name, 'lwir-camera-start')


@unittest.skipUnless(LINUX, 'the parent-death signal is a Linux rule')
class ParentDeathSignal(unittest.TestCase):
    def start(self, through_keeper):
        def work():
            make = lambda: subprocess.Popen([sys.executable, '-c', CHILD], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            child = spawn_keeper.popen(make) if through_keeper else make()
            self.assertEqual(child.stdout.readline(), b'ready\n')   # the request is in place before the starting thread ends
            return child

        child = from_short_thread(work)
        self.addCleanup(child.stdout.close)
        self.addCleanup(lambda: (child.kill(), child.wait(5)) if child.poll() is None else None)
        return child

    def test_started_from_a_short_thread_it_is_stopped_when_that_thread_ends(self):
        child = self.start(False)
        self.assertEqual(child.wait(5), -signal.SIGTERM)

    def test_started_through_the_keeper_it_lives(self):
        child = self.start(True)
        with self.assertRaises(subprocess.TimeoutExpired):
            child.wait(1.5)


def fake_cam_pipe(dest):
    """cam_pipe built with the stand-in SDK (as test_cam_pipe_frozen), or None when there is no compiler."""
    exe = os.environ.get('CAM_PIPE_FAKE_BIN')
    if exe:
        return exe
    if not shutil.which('g++'):
        return None
    from .test_cam_pipe_frozen import build
    return build(dest)[0]


@unittest.skipUnless(LINUX, 'cam_pipe is built for Linux')
class CamPipeAfterReplay(unittest.TestCase):
    """The real cam_pipe (stand-in SDK) started the way the replay switch thread starts it. No calibration of ours is on
    record (a recording played longer than cam_pipe's reuse window, or the first hand-back after a power-on)."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix='spawn_keeper_')
        cls.exe = fake_cam_pipe(cls.root)
        if cls.exe is None:
            shutil.rmtree(cls.root, ignore_errors=True)
            raise unittest.SkipTest('needs g++ or CAM_PIPE_FAKE_BIN')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def start(self, home, through_keeper, seconds):
        """One cam_pipe in home, started from a thread that ends 0.3 s later; watched for seconds."""
        from .test_cam_pipe_scene import fake_sysfs
        os.makedirs(home, exist_ok=True)
        log = os.path.join(home, 'sdk.log')
        if os.path.exists(log): os.unlink(log)
        env = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home, FAKE_I3_PLAN='ok,-1,0,1,1', FAKE_I3_LOG=log,
                   FAKE_I3_LOG_CAL='1', LWIR_CAM_SYSFS=fake_sysfs(home, 5))
        with open(os.path.join(home, 'err.txt'), 'wb') as err:
            make = lambda: subprocess.Popen([self.exe, '-1', '60'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, bufsize=0, env=env)
            child = from_short_thread(lambda: spawn_keeper.popen(make) if through_keeper else make(), linger=0.3)
            got = [0]

            def drain():
                while True:
                    data = child.stdout.read(65536)
                    if not data: return
                    got[0] += len(data)

            reader = threading.Thread(target=drain, daemon=True); reader.start()
            try: code = child.wait(seconds); alive = False
            except subprocess.TimeoutExpired:
                alive = True; child.terminate(); code = child.wait(20)
            reader.join(5); child.stdout.close()
        with open(os.path.join(home, 'err.txt'), encoding='utf-8', errors='replace') as f: lines = f.read()
        with open(log, encoding='utf-8', errors='replace') as f: sdk = f.read().splitlines()
        size = [int(v) for v in lines.split('CAMERA ', 1)[1].split()[:2]] if 'CAMERA ' in lines else [640, 480]
        return NS(alive=alive, code=code, frames=got[0] // (size[0] * size[1] * 2), err=lines,
                  opens=sum(x.startswith('open ') for x in sdk), cals=sum(x.startswith('cal ') for x in sdk))

    def test_through_the_keeper_one_open_one_calibration_and_the_picture_stays(self):
        r = self.start(os.path.join(self.root, 'kept'), True, 5.0)
        self.assertTrue(r.alive, r.err)                      # still streaming 5 s after the thread that asked for it ended
        self.assertGreater(r.frames, 100, r.err)
        self.assertEqual((r.opens, r.cals), (1, 1), r.err)
        self.assertNotIn('CLOSE_WAIT', r.err.split('FRAMES')[0])   # no stop before the frames (the one at the end is this test's terminate)

    def test_without_it_two_opens_and_two_calibrations(self):
        home = os.path.join(self.root, 'short_thread')
        first = self.start(home, False, 8.0)
        self.assertFalse(first.alive, first.err)             # the fault of 11-15 22:44: stopped when the starting thread ended
        self.assertEqual(first.code, 0, first.err)
        self.assertIn('CLOSE_WAIT', first.err)
        self.assertLess(first.frames, 30, first.err)
        again = self.start(home, True, 5.0)                  # the server's next start (20 s later on the board)
        self.assertTrue(again.alive, again.err)
        self.assertEqual((first.opens + again.opens, first.cals + again.cals), (2, 2), first.err + again.err)


class ServerStartPaths(unittest.TestCase):
    """camera_link.lv_spawn and camera_extra.spawn start the camera program through spawn_keeper; the replay program not."""

    def setUp(self):
        self.starts = []

        def popen(argv, **kwargs):
            self.starts.append((argv, threading.current_thread()))
            return NS(poll=lambda: None, stdout=None)

        self.live = NS(replay=None, log=None, cond=threading.Condition(), jcond=threading.Condition(), stopped=False,
                       cam_stamp=dict(on=False, fallback=''), cam_stamp_recent=[])
        patches = [patch.dict(vars(APP), LIVE=self.live, args=NS(live_cam='cam_pipe', live_wait=0),
                              lv_sp=NS(Popen=popen, PIPE=-1), lv_th=NS(Thread=Mock()), lv_reader=Mock()),
                   patch.object(camera_link, 'invalidate_preview'), patch.object(camera_link.cam_scene, 'on_spawn'),
                   patch.object(camera_link.cam_frozen, 'on_spawn'), patch.object(camera_extra, 'log_start'),
                   patch.object(camera_extra, 'install_producer', side_effect=lambda process: setattr(self.live, 'proc', process)),
                   patch.object(camera_extra, '_ring_once'),
                   patch.object(camera_extra, 'note')]
        for item in patches:
            item.start(); self.addCleanup(item.stop)

    def keeper(self):
        return spawn_keeper._KEEPER['thread']

    def test_plain_start_from_a_short_thread(self):
        with patch.object(camera_extra, 'wanted', return_value=False):
            from_short_thread(lambda: camera_link.lv_spawn() or True)
        self.assertEqual([argv for argv, _ in self.starts], [['cam_pipe']])
        self.assertIs(self.starts[0][1], self.keeper())

    def test_stamped_start_from_a_short_thread(self):
        with patch.object(camera_extra, 'wanted', return_value=True), patch.dict(camera_extra._S, stamp_off='', shm_off=''):
            from_short_thread(lambda: camera_link.lv_spawn() or True)
        self.assertEqual([argv for argv, _ in self.starts], [['cam_pipe']])
        self.assertIs(self.starts[0][1], self.keeper())

    def test_start_from_the_frame_loop_is_unchanged(self):
        with patch.object(camera_extra, 'wanted', return_value=True), patch.dict(camera_extra._S, stamp_off='', shm_off=''):
            camera_link.lv_spawn()
        self.assertIs(self.starts[0][1], threading.main_thread())


class ReturningState(unittest.TestCase):
    """GET /replay state.returning: from 'stop the recording' to the first frame of the camera program started for it."""

    def setUp(self):
        now = time.monotonic()
        self.live = NS(replay=dict(name='fixture', fps=25.0, loop=False, bits=8, frames=10, fin0=0), stopped=False,
                       cam_hold=False, replay_busy=False, sys_lock=threading.Lock(), cam_lock=threading.Lock(),
                       proc_t0=now - 30, last_frame_t=now - 0.1, frames_in=5, replay_msg=None, preview_worker=None)
        self.thread = Mock()
        patches = [patch.dict(vars(APP), LIVE=self.live, GB=None, args=NS(live_wait=0, live_cam='camera'),
                              lv_th=NS(Thread=self.thread)),
                   patch.object(replay_link, 'set_preview_producer'), patch.object(replay_link.replay_eval, 'EV', None)]
        for item in patches:
            item.start(); self.addCleanup(item.stop)

    def test_not_returning_while_a_recording_plays_or_the_camera_runs(self):
        self.assertFalse(replay_link.state()['returning'])
        self.live.replay = None
        self.assertFalse(replay_link.state()['returning'])

    def test_stop_until_the_first_camera_frame(self):
        code, text = replay_link.stop_request()
        self.assertEqual(code, 200)
        self.assertIn('카메라로 돌아가는 중', text)
        self.assertNotIn('10~60', text)
        self.assertTrue(replay_link.state()['returning'])            # the switch thread has not run yet
        self.live.last_frame_t = time.monotonic()                     # a late hold packet of the replay program
        self.assertTrue(replay_link.state()['returning'])
        self.live.replay = None; self.live.replay_busy = False        # the switch started the camera program ...
        self.live.proc_t0 = time.monotonic()
        self.assertTrue(replay_link.state()['returning'])            # ... which is opening the camera
        self.live.last_frame_t = time.monotonic() + 0.001             # its first frame
        self.assertFalse(replay_link.state()['returning'])
        self.assertIsNone(self.live.replay_return_t)

    def test_gives_up_after_a_minute(self):
        replay_link.stop_request()
        self.live.replay = None; self.live.replay_busy = False
        self.live.replay_return_t -= replay_link.RETURN_MAX_S + 1     # the camera did not come back: the page's camera notices take over
        self.assertFalse(replay_link.state()['returning'])

    def test_a_new_recording_ends_it(self):
        replay_link.stop_request()
        self.live.replay = None; self.live.replay_busy = False; self.live.cam_hold = False
        with patch.object(replay_link, 'find', return_value=(dict(name='test', bits=16, frames=2), 'images')), \
             patch.object(replay_link, '_gimbal_stop'):
            self.assertEqual(replay_link.start_request({'seq': ['test']})[0], 200)
        self.assertIsNone(self.live.replay_return_t)

    def test_replay_program_that_ends_by_itself(self):
        live = self.live
        live.cam_lost = True; live.proc = NS(poll=lambda: 0); live.q = []; live.cond = threading.Condition()
        with patch.object(replay_link.replay_eval, 'activate'), patch.dict(vars(APP), lv_spawn=Mock()), \
             patch.object(replay_link, '_msg') as said:
            self.assertTrue(replay_link.on_cam_lost(time.monotonic()))
        self.assertIsNotNone(live.replay_return_t)
        self.assertNotIn('10~60', said.call_args.args[0])
        live.proc_t0 = time.monotonic() - 1; live.last_frame_t = live.proc_t0 - 1
        self.assertTrue(replay_link.state()['returning'])


if __name__ == '__main__':
    unittest.main()
