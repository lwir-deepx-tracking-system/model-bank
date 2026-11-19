"""No-device GUI/control lane tests; optional real bridge uses a PTY only.

LWIR_TEST_BRIDGE=/path/to/compiled/v27 python -m unittest ...test_picture_transport
"""
from collections import deque
import os
import time
from threading import Condition, Event, Thread
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ..common.state import APP
from . import gimbal_link


class PictureTransportTests(unittest.TestCase):
    def setUp(self):
        self.saved = APP.__dict__.copy()
        APP.CTL = dict(lv=None, lead_px=0)
        self.lane = NS(fd=123, condition=Condition(), frame=None, closed=False)
        self.proc = NS(_picture_lane=self.lane)
        APP.GB = NS(wcond=Condition(), wopen=True, wq=deque(), wframe=None,
                    skipped=0, proc=self.proc, picture_lane_supported=True, state={})
        self.header = 'G 42 123456 640 480 1 17 100 50 40 80 .9 OBS 1 LEAD 17 1 2 9'
        self.picture = b'\x00\n' * (640 * 480)

    def tearDown(self):
        APP.__dict__.clear()
        APP.__dict__.update(self.saved)

    def test_negotiated_control_has_dimensions_tracks_and_suffixes_without_pixels(self):
        gimbal_link.gb_send_frame(self.header, self.picture)
        self.assertEqual(APP.GB.wframe, ('O' + self.header[1:] + '\n').encode())
        self.assertEqual(self.lane.frame, ((self.header + '\n').encode(), self.picture))
        self.assertIs(self.lane.frame[1], self.picture)
        self.assertLess(len(APP.GB.wframe), 200)

    def test_old_bridge_and_control_only_F_remain_compatible(self):
        for supported in (False, True):
            APP.GB.picture_lane_supported = supported
            gimbal_link.gb_send_frame('F 1 2 0 OBS 1', b'')
            self.assertEqual(APP.GB.wframe, b'F 1 2 0 OBS 1\n')
        APP.GB.picture_lane_supported = False
        gimbal_link.gb_send_frame(self.header, self.picture)
        self.assertEqual(APP.GB.wframe, (self.header + '\n').encode() + self.picture)
        self.assertIsNone(self.lane.frame)

    def test_missing_lane_uses_legacy_even_if_flag_was_set(self):
        del self.proc._picture_lane
        gimbal_link.gb_send_frame(self.header, self.picture)
        self.assertEqual(APP.GB.wframe, (self.header + '\n').encode() + self.picture)

    def test_failed_picture_lane_never_falls_back_to_blocking_control_with_pixels(self):
        self.lane.closed = True
        gimbal_link.gb_send_frame(self.header, self.picture)
        self.assertEqual(APP.GB.wframe, ('O' + self.header[1:] + '\n').encode())
        self.assertIsNone(self.lane.frame)

    def test_capability_requires_current_process_and_an_inherited_lane(self):
        for current, has_lane in ((True, True), (True, False), (False, True)):
            APP.GB.picture_lane_supported = False
            self.proc._picture_lane = self.lane if has_lane else None
            APP.GB.proc = self.proc if current else object()
            # Stop after the first message, before bridge-exit device cleanup.
            self.proc.stdout = NS(readline=unittest.mock.Mock(side_effect=[b'PICTURE_LANE 1\n', RuntimeError('end')]))
            with self.assertRaisesRegex(RuntimeError, 'end'):
                gimbal_link.gb_reader(self.proc)
            self.assertEqual(APP.GB.picture_lane_supported, current and has_lane)

    def test_blocked_picture_does_not_block_observation_or_stop_barrier(self):
        blocked = Event()
        release = Event()
        written = bytearray()
        def write(fd, data):
            self.assertEqual(fd, self.lane.fd)
            if not release.is_set():
                blocked.set()
                raise BlockingIOError()
            count = min(len(data), 12345)
            written.extend(data[:count])
            return count
        class Poll:
            def register(self, *args): pass
            def poll(self, timeout): release.wait(timeout / 1000)
        gimbal_link.gb_send_frame(self.header, self.picture)
        with patch.object(gimbal_link.os, 'write', side_effect=write), \
             patch.object(gimbal_link.os, 'set_blocking', create=True), \
             patch.object(gimbal_link.os, 'close'), \
             patch.object(gimbal_link.select, 'POLLOUT', 4, create=True), \
             patch.object(gimbal_link.select, 'poll', Poll, create=True):
            worker = Thread(target=gimbal_link._picture_writer, args=(self.lane,))
            worker.start()
            try:
                self.assertTrue(blocked.wait(1))
                # A partial image is held by its own writer. Newer images replace
                # just the pending slot; the control writer remains runnable.
                for frame in range(43, 53):
                    gimbal_link.gb_send_frame(self.header.replace('42 ', f'{frame} ', 1), self.picture)
                self.assertTrue(self.lane.frame[0].startswith(b'G 52 '))
                sent = []
                self.proc.stdin = NS(write=sent.append, flush=lambda: setattr(APP.GB, 'wopen', False))
                APP.GB.observation_barrier = b'F 40 123455 0 OBS 0\n'
                APP.GB.wq.append(b'H 19\n')
                gimbal_link.gb_writer(self.proc)
                self.assertEqual(sent[:2], [b'F 40 123455 0 OBS 0\n', b'H 19\n'])
                self.assertTrue(sent[2].startswith(b'O 52 '))
                self.assertNotIn(self.picture, b''.join(sent))
            finally:
                gimbal_link._close_picture_lane(self.proc)
                release.set()
                worker.join(1)
            self.assertFalse(worker.is_alive())

    def test_partial_writes_preserve_complete_header_pixel_pair(self):
        packet = (self.header + '\n').encode() + self.picture
        done = Event()
        written = bytearray()
        def write(fd, data):
            size = min(997, len(data))
            written.extend(data[:size])
            if len(written) == len(packet): done.set()
            return size
        gimbal_link.gb_send_frame(self.header, self.picture)
        with patch.object(gimbal_link.os, 'write', side_effect=write), \
             patch.object(gimbal_link.os, 'set_blocking', create=True), patch.object(gimbal_link.os, 'close'):
            worker = Thread(target=gimbal_link._picture_writer, args=(self.lane,))
            worker.start()
            try:
                self.assertTrue(done.wait(2))
            finally:
                gimbal_link._close_picture_lane(self.proc)
                worker.join(1)
        self.assertEqual(written, packet)
        self.assertFalse(worker.is_alive())


@unittest.skipUnless(os.name == 'posix' and os.environ.get('LWIR_TEST_BRIDGE'),
                     'requires a compiled Linux bridge; uses no physical device')
class BridgePictureIsolationTests(unittest.TestCase):
    def test_partial_picture_does_not_defer_stop_or_next_observation(self):
        from . import test_gimbal_bridge as fixtures
        fixtures.BRIDGE = os.environ['LWIR_TEST_BRIDGE']
        read_fd, write_fd = os.pipe()
        original = fixtures.subprocess.Popen
        def spawn(*args, **kwargs):
            kwargs['pass_fds'] = (read_fd,)
            kwargs['env'] = dict(os.environ, LWIR_GUI_PICTURE_FD=str(read_fd))
            return original(*args, **kwargs)
        rig = None
        try:
            with patch.object(fixtures.subprocess, 'Popen', side_effect=spawn):
                rig = fixtures.Rig('--gui-host', '127.0.0.1', '--gui-command-port', '0', '--gui-codec', 'jpeg')
            self.assertEqual(rig.wait_line('PICTURE_LANE'), 'PICTURE_LANE 1')
            os.write(write_fd, b'G 999 1 640 480 0\n\x00')
            rig.send('P')
            rig.frame((17, 500, 240))
            rig.send('A 17')
            self.assertEqual(rig.wait_line('R A'), 'R A OK')
            rig.frame((17, 500, 240))
            rig.wait_state(dict(auto=1, id=17))
            rig.send('H 771')
            self.assertEqual(rig.wait_line('HALTED', timeout=.5), 'HALTED 771')
            self.assertTrue(rig.until_stop())
            # The same stalled video packet cannot hide newer observations.
            rig.send(f'O 1001 {time.monotonic_ns()} 640 480 1 23 400 100 40 80 .9 OBS 1')
            rig.send('A 23')
            self.assertEqual(rig.wait_line('R A'), 'R A OK')
        finally:
            if rig is not None:
                rig.close()
                rig.proc.stdout.close()
            os.close(read_fd)
            os.close(write_fd)


if __name__ == '__main__':
    unittest.main()
