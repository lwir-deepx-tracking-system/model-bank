"""Verify single-play EOF behavior through the real replay subprocess and wire headers."""
import os
from pathlib import Path
import queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from . import replay_cam


class ReplayEndTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.raw = Path(tmp.name) / "three.raw"
        self.raw.write_bytes(b"".join(bytes([i]) * replay_cam.FRAME_BYTES for i in (1, 2, 3)))
        self.env = {**os.environ, "LWIR_CAM_STAMP": "1"}

    def start(self, *args):
        cmd = [sys.executable, "-B", str(Path(replay_cam.__file__)), "--dir", str(self.raw),
               "--fps", "25", "--gap", ".05", "--rep-fps", "10", "--tag", *args]
        p = subprocess.Popen(cmd, env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        self.addCleanup(self.close, p)
        frames = queue.Queue()

        def read():
            while buf := p.stdout.read(replay_cam.HDR.size):
                head = replay_cam.HDR.unpack(buf)
                body = p.stdout.read(head[7])
                frames.put((head[6], bool(head[2] & replay_cam.FLAG_REPEAT), head[8] >> 20, body[0]))
            frames.put(None)

        reader = Thread(target=read, daemon=True)
        reader.start()
        self.addCleanup(reader.join, 2)  # close the process before waiting for its reader
        self.addCleanup(self.stop, p)
        return p, lambda: frames.get(timeout=4)

    @staticmethod
    def stop(p):
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=3)

    @classmethod
    def close(cls, p):
        cls.stop(p)
        p.stdout.close()
        p.stdin.close()

    @staticmethod
    def command(p, command):
        p.stdin.write(command + b"\n")
        p.stdin.flush()

    def fresh(self, frame):
        for _ in range(12):
            value = frame()
            if value is None or not value[1]:
                return value
        self.fail("no fresh frame after control command")

    def test_web_holds_final_and_restarts_on_play(self):
        p, frame = self.start("--control", "--loops", "1", "--no-skip")
        self.assertEqual([frame() for _ in range(3)], [(0, False, 0, 1), (1, False, 0, 2), (2, False, 0, 3)])
        self.assertEqual([frame() for _ in range(6)], [(2, True, 0, 3)] * 6)
        self.assertIsNone(p.poll())
        self.command(p, b"PLAY")
        self.assertEqual(self.fresh(frame), (0, False, 1, 1))
        self.assertEqual([frame(), frame(), frame()], [(1, False, 1, 2), (2, False, 1, 3), (2, True, 1, 3)])
        self.command(p, b"SEEK 1")
        self.assertEqual(self.fresh(frame), (1, False, 2, 2))
        self.assertEqual([frame(), frame()], [(2, False, 2, 3), (2, True, 2, 3)])

    def test_existing_command_line_and_loop_behavior(self):
        p, frame = self.start("--loops", "1")
        self.assertEqual([frame() for _ in range(4)], [(0, False, 0, 1), (1, False, 0, 2), (2, False, 0, 3), None])
        self.assertEqual(p.wait(timeout=3), 0)
        _, frame = self.start("--control", "--loops", "0", "--no-skip")
        self.assertEqual([frame() for _ in range(4)], [(0, False, 0, 1), (1, False, 0, 2), (2, False, 0, 3), (0, False, 1, 1)])


if __name__ == "__main__":
    unittest.main()
