"""11-14: the server's reader of cam_pipe's temperature ring (src/camera/temp_ring.py), on ring files written byte by byte from the v2.11
layout note at the top of cam_pipe.cpp (the formats are written out here again on purpose, so a change on one side fails these tests).
With CAM_TEMP_RING_FILE=<file> it also reads a ring written by cam_temp_ring_test.cpp (cam_pipe's own writer functions)."""
import contextlib
import io
import os
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.camera import temp_ring as tr

HERE = os.path.dirname(os.path.abspath(__file__))
RING_FMT = '<4s7IQ2If2I4x'        # magic, version, ring_id, nslots, stride, payload, width, height, pid, every, state, emissivity, max_ms, maps
SLOT_FMT = '<Q4sHHQIHHfIIIHH12x'   # frame, magic, version, flags, t_mono_ns, calc_us, width, height, emissivity, map_seq, payload, ring_id, t_offset, t_div
HEAD = 64
MAPS_AT = 56


class RingFile:
    """a ring file in a temp folder, written the way cam_pipe v2.11 writes it (frame 0, slot header + map, frame, then maps in the file header)"""
    def __init__(self, w=64, h=48, nslots=4, ring_id=77, every=3, emissivity=0.98, state=1):
        self.dir = tempfile.mkdtemp(); self.path = os.path.join(self.dir, 'lwir_temp_ring_test')
        self.w, self.h, self.n, self.id, self.emis = w, h, nslots, ring_id, emissivity
        self.payload = w * h * 2; self.sd = (HEAD + self.payload + 63) & ~63; self.maps = 0
        b = bytearray(HEAD + nslots * self.sd)
        struct.pack_into(RING_FMT, b, 0, b'LWT1', 1, ring_id, nslots, self.sd, self.payload, w, h, 4242, every, state, emissivity, 50, 0)
        with open(self.path, 'wb') as f:
            f.write(bytes(b))

    def _at(self, off, data):
        with open(self.path, 'r+b') as f:
            f.seek(off); f.write(data)

    def put(self, frame, t_ns, values, calc_us=1500, flags=1, map_seq=None):
        k = self.maps + 1; off = HEAD + (self.maps % self.n) * self.sd
        self._at(off, struct.pack('<Q', 0))
        head = bytearray(HEAD)
        struct.pack_into(SLOT_FMT, head, 0, 0, b'LWTS', 1, flags, t_ns, calc_us, self.w, self.h, self.emis, k if map_seq is None else map_seq, self.payload, self.id, 5000, 100)
        self._at(off + 8, bytes(head[8:]) + np.ascontiguousarray(values, '<u2').tobytes())
        self._at(off, struct.pack('<Q', frame))
        self.maps = k; self._at(MAPS_AT, struct.pack('<I', k))

    def set_frame(self, k, frame):   # the frame number of the slot that holds map k
        self._at(HEAD + ((k - 1) % self.n) * self.sd, struct.pack('<Q', frame))

    def set_maps(self, k):
        self._at(MAPS_AT, struct.pack('<I', k))

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def tmap(w, h, c):
    return tr.from_celsius(np.full((h, w), c, np.float32))


class LayoutTests(unittest.TestCase):
    def test_constants_match_the_cam_pipe_note(self):
        fmt = lambda s: s.format if isinstance(s.format, str) else s.format.decode()
        self.assertEqual((fmt(tr.RING_HDR), fmt(tr.SLOT_HDR)), (RING_FMT, SLOT_FMT))
        with open(os.path.join(HERE, 'cam_pipe.cpp'), encoding='utf-8') as f:
            src = f.read()
        for s in ("['%s']" % RING_FMT, "['%s']" % SLOT_FMT, '"LWT1"', '"LWTS"', 'lwir_temp_ring_', 'TEMP_SLOTS = 4'):
            self.assertIn(s, src)
        self.assertEqual((tr.HDR_N, tr.MAPS_OFF, tr.RING_MAGIC, tr.SLOT_MAGIC, tr.VERSION), (HEAD, MAPS_AT, b'LWT1', b'LWTS', 1))
        self.assertEqual(struct.calcsize('<4s7IQ2If1I'), MAPS_AT)
        self.assertEqual(tr.stride(640, 480), 614464)
        self.assertTrue(tr.ring_path().endswith('lwir_temp_ring_%d' % (os.getuid() if hasattr(os, 'getuid') else 0)))

    def test_celsius_scale(self):
        v = tr.from_celsius([22.0, 34.2, 45.0, -10.0])
        self.assertEqual(v.tolist(), [7200, 8420, 9500, 4000])
        np.testing.assert_allclose(tr.to_celsius(v), [22.0, 34.2, 45.0, -10.0], atol=1e-4)
        np.testing.assert_allclose(tr.to_celsius([7300], 5000, 50), [46.0], atol=1e-4)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.f = RingFile()
        self.addCleanup(self.f.close)
        self.r = tr.TempRing(self.f.path)

    def test_missing_empty_and_no_map_yet(self):
        self.assertIsNone(tr.TempRing(os.path.join(self.f.dir, 'nothing')).newest())
        z = os.path.join(self.f.dir, 'zeros')
        with open(z, 'wb') as fh: fh.write(b'\0' * 4096)
        self.assertIsNone(tr.TempRing(z).newest())
        self.assertIsNone(self.r.newest())   # a ring with no map yet
        self.assertEqual(self.r.stats['empty'], 1); self.assertEqual(self.r.info['state'], 1)

    def test_newest_map_after_wrap(self):
        w, h = self.f.w, self.f.h
        for k in range(1, 7):   # 6 maps in 4 slots
            self.f.put(3 * k, 10 ** 9 * k + 7, tmap(w, h, 20.0 + k), calc_us=1000 + k)
        m = self.r.newest()
        self.assertEqual((m.frame, m.map_seq, m.ring_id, m.t_ns, m.calc_us, m.flags, m.t_offset, m.t_div, m.state, m.every), (18, 6, 77, 6 * 10 ** 9 + 7, 1006, 1, 5000, 100, 1, 3))
        self.assertAlmostEqual(m.emissivity, 0.98, places=5)
        self.assertEqual((m.values.shape, m.values.dtype), ((h, w), np.dtype('<u2')))
        np.testing.assert_allclose(tr.to_celsius(m.values, m.t_offset, m.t_div), 26.0, atol=1e-4)
        self.assertIs(self.r.newest(), m)   # no new map: no second copy
        self.assertEqual((self.r.stats['maps'], self.r.stats['same']), (1, 1))
        self.f.put(21, 7 * 10 ** 9, tmap(w, h, 30.0))
        m = self.r.newest()
        self.assertEqual((m.frame, m.map_seq), (21, 7)); np.testing.assert_allclose(tr.to_celsius(m.values), 30.0, atol=1e-4)

    def test_newest_being_written_gives_the_map_before(self):
        w, h = self.f.w, self.f.h
        self.f.put(3, 1000, tmap(w, h, 30.0)); self.f.put(6, 2000, tmap(w, h, 34.0))
        self.f.set_frame(2, 0)   # cam_pipe is in the middle of a copy into map 2's slot
        self.assertEqual(self.r.newest().frame, 3)
        self.assertEqual(self.r.stats['writing'], 1)

    def test_torn_copy_is_dropped(self):
        w, h = self.f.w, self.f.h
        self.f.put(3, 1000, tmap(w, h, 30.0)); self.f.put(6, 2000, tmap(w, h, 34.0))
        calls = []

        def overwrite():   # cam_pipe starts writing map 2's slot again while we copy it
            if not calls: self.f.set_frame(2, 0)
            calls.append(1)
        self.r._between = overwrite
        m = self.r.newest()
        self.assertEqual(m.map_seq, 1); self.assertEqual(self.r.stats['torn'], 1)
        np.testing.assert_allclose(tr.to_celsius(m.values), 30.0, atol=1e-4)
        self.f.put(9, 3000, tmap(w, h, 34.0))
        self.r._between = lambda: (self.f.set_frame(3, 0), self.f.set_frame(2, 0))   # map 3 torn, map 2 then being written: no map
        self.assertIsNone(self.r.newest())
        self.assertEqual((self.r.stats['torn'], self.r.stats['writing']), (2, 1))

    def test_overwritten_before_the_first_look(self):
        w, h = self.f.w, self.f.h
        for k in range(1, 7):
            self.f.put(3 * k, 1000 * k, tmap(w, h, 20.0 + k))
        self.f.set_maps(7)   # the counter says 7, but map 7's slot still holds map 3
        m = self.r.newest()
        self.assertEqual(m.map_seq, 6); self.assertEqual(self.r.stats['gone'], 1)

    def test_new_cam_pipe_same_map_number_is_read_again(self):
        w, h = self.f.w, self.f.h
        self.f.put(3, 1000, tmap(w, h, 30.0))
        self.assertEqual(self.r.newest().ring_id, 77)
        g = RingFile(ring_id=78); self.addCleanup(g.close)
        g.put(3, 1000, tmap(w, h, 33.0))
        shutil.copyfile(g.path, self.f.path)   # cam_pipe started again: a new ring in the same place
        m = self.r.newest()
        self.assertEqual((m.ring_id, m.map_seq), (78, 1)); np.testing.assert_allclose(tr.to_celsius(m.values), 33.0, atol=1e-4)

    def test_foreign_or_short_file(self):
        w, h = self.f.w, self.f.h
        self.f.put(3, 1000, tmap(w, h, 30.0))
        self.f._at(0, b'LWR1')   # the picture ring is not a temperature ring
        self.assertIsNone(self.r.newest()); self.assertEqual(self.r.stats['bad'], 1)
        self.f._at(0, b'LWT1'); self.f._at(4, struct.pack('<I', 2))   # another version
        self.assertIsNone(self.r.newest())
        self.f._at(4, struct.pack('<I', 1))
        self.assertEqual(self.r.newest().map_seq, 1)
        with open(self.f.path, 'r+b') as fh: fh.truncate(HEAD + self.f.sd)   # shorter than its header says
        self.assertIsNone(self.r.newest())

    def test_stopped_ring_is_still_read(self):
        w, h = self.f.w, self.f.h
        self.f.put(3, 1000, tmap(w, h, 30.0))
        self.f._at(44, struct.pack('<I', 2))   # state 2: TEMP_OFF (the last map stays; the server sees it get old)
        m = self.r.newest()
        self.assertEqual((m.map_seq, m.state), (1, 2))

    def test_full_size_ring(self):
        f = RingFile(640, 480); self.addCleanup(f.close)
        v = tmap(640, 480, 22.0); v[100:120, 300:316] = tr.from_celsius(34.5)
        f.put(10, 99, v)
        m = tr.TempRing(f.path).newest()
        self.assertEqual((m.frame, m.width, m.height), (10, 640, 480))
        self.assertAlmostEqual(float(tr.to_celsius(m.values[110, 310])), 34.5, places=3)


@unittest.skipUnless(os.environ.get('CAM_TEMP_RING_FILE'), 'CAM_TEMP_RING_FILE not set (a ring written by cam_temp_ring_test)')
class WrittenByCamPipe(unittest.TestCase):
    def test_ring_from_cam_temp_ring_test(self):   # 6 maps of 64 x 48, every 3: map k = frame 3k, t = k * 1e9 + 7, calc_us 1000 + k, pixel i = 5000 + 100 * (20 + k) + i % 50
        r = tr.TempRing(os.environ['CAM_TEMP_RING_FILE'])
        m = r.newest()
        self.assertEqual((m.frame, m.map_seq, m.t_ns, m.calc_us, m.flags, m.width, m.height, m.every, m.t_offset, m.t_div), (18, 6, 6 * 10 ** 9 + 7, 1006, 1, 64, 48, 3, 5000, 100))
        self.assertIn(m.state, (1, 2)); self.assertAlmostEqual(m.emissivity, 0.98, places=6)
        want = (5000 + 100 * 26 + np.arange(64 * 48) % 50).astype(np.uint16).reshape(48, 64)
        self.assertTrue(np.array_equal(m.values, want))
        self.assertAlmostEqual(float(tr.to_celsius(m.values, m.t_offset, m.t_div)[0, 1]), 26.01, places=4)


class StartOptionTests(unittest.TestCase):
    def test_cam_temp_option_and_environment(self):
        with patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()):
            os.environ.pop(tr.ENV, None)
            self.assertEqual(tr.apply_cli(0), '')   # default: nothing passed to cam_pipe
            self.assertNotIn(tr.ENV, os.environ)
            self.assertEqual(tr.apply_cli(None), '')
            self.assertNotIn(tr.ENV, os.environ)
            os.environ[tr.ENV] = '3'   # given in the environment (live.env): passed on unchanged
            self.assertEqual(tr.apply_cli(0), '3')
            self.assertEqual(tr.apply_cli(10), '10')   # the option wins
            self.assertEqual(os.environ[tr.ENV], '10')
        self.assertNotEqual(os.environ.get(tr.ENV), '10')


if __name__ == '__main__':
    unittest.main()
