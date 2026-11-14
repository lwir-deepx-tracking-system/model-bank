"""11-14: writer side of cam_pipe's temperature ring (cam_pipe.cpp v2.11, LWIR_CAM_TEMP=N) against the server's reader (temp_ring.py). Runs without a camera.

The ring /dev/shm/lwir_temp_ring_<uid>: a 64-byte file header, then 4 slots of a 64-byte slot header + width*height uint16 (T_C = (v - 5000) / 100).
The layout is stated here again on purpose (RING_HDR, SLOT_HDR below), so that a change on either side fails:
  1. the formats are the ones in the v2.11 note of cam_pipe.cpp and in temp_ring.py, and the field offsets are those the C++ static_asserts hold;
  2. a ring built here byte by byte from the note is read by temp_ring.TempRing;
  3. with CAM_TEMP_RING_FILE=<file>: a ring written by cam_temp_ring_test.cpp (cam_pipe's own temp_ring_open / temp_ring_put / temp_off) is checked byte by byte
     and read by temp_ring.TempRing.
"""
import os
import shutil
import struct
import tempfile
import unittest

import numpy as np

from . import temp_ring as tr

HERE = os.path.dirname(os.path.abspath(__file__))
RING_HDR = struct.Struct('<4s7IQ2If2I4x')        # magic LWT1, version, ring_id, nslots, stride, payload, width, height, pid, every, state, emissivity, max_ms, maps
SLOT_HDR = struct.Struct('<Q4sHHQIHHfIIIHH12x')   # frame, magic LWTS, version, flags, t_mono_ns, calc_us, width, height, emissivity, map_seq, payload, ring_id, t_offset, t_div
HDR_N = 64
SLOTS = 4                                        # cam_pipe TEMP_SLOTS
STATE_ON, STATE_STOPPED = 1, 2
FLAG_AGC_READ = 1
assert RING_HDR.size == HDR_N and SLOT_HDR.size == HDR_N


def fmt(s_):
    return s_.format if isinstance(s_.format, str) else s_.format.decode()


def stride(w_, h_):   # cam_pipe: (sizeof(TempSlotHead) + payload + 63) & ~63
    return (HDR_N + w_ * h_ * 2 + 63) & ~63


def build_ring(maps_, w_=8, h_=6, every_=2, emis_=0.0, state_=STATE_ON, ring_id_=77):   # as cam_pipe leaves it: map k in slot (k - 1) % 4, maps = number written
    sd_ = stride(w_, h_); pay_ = w_ * h_ * 2
    b_ = bytearray(HDR_N + SLOTS * sd_)
    RING_HDR.pack_into(b_, 0, b'LWT1', 1, ring_id_, SLOTS, sd_, pay_, w_, h_, 4242, every_, state_, emis_, 50, len(maps_))
    for k_, (frame_, m_) in enumerate(maps_, 1):
        off_ = HDR_N + ((k_ - 1) % SLOTS) * sd_
        SLOT_HDR.pack_into(b_, off_, frame_, b'LWTS', 1, FLAG_AGC_READ, 10 ** 9 * k_, 900 + k_, w_, h_, emis_, k_, pay_, ring_id_, 5000, 100)
        b_[off_ + HDR_N:off_ + HDR_N + pay_] = np.ascontiguousarray(m_, '<u2').tobytes()
    return b_


class Layout(unittest.TestCase):
    def test_formats_in_cam_pipe_and_temp_ring(self):
        with open(os.path.join(HERE, 'cam_pipe.cpp'), encoding='utf-8') as f_: src_ = f_.read()
        for s_ in (RING_HDR, SLOT_HDR): self.assertIn("['%s']" % fmt(s_), src_)
        self.assertEqual((fmt(tr.RING_HDR), fmt(tr.SLOT_HDR)), (fmt(RING_HDR), fmt(SLOT_HDR)))
        self.assertEqual((tr.RING_MAGIC, tr.SLOT_MAGIC, tr.VERSION, tr.HDR_N, tr.MAPS_OFF), (b'LWT1', b'LWTS', 1, HDR_N, 56))
        for s_ in ('/dev/shm/lwir_temp_ring_%u', 'sizeof(TempRingHead) == 64', 'sizeof(TempSlotHead) == 64', 'TEMP_SLOTS = 4', '"LWT1"', '"LWTS"'): self.assertIn(s_, src_)

    def test_field_offsets(self):   # the offsets the C++ static_asserts hold
        self.assertEqual([struct.calcsize(p_) for p_ in ('<4s7I', '<4s7IQ', '<4s7IQ2I', '<4s7IQ2If2I')], [32, 40, 48, 60])         # pid, every, emissivity, after maps
        self.assertEqual([struct.calcsize(p_) for p_ in ('<Q4sHH', '<Q4sHHQ', '<Q4sHHQIHH', '<Q4sHHQIHHfII', '<Q4sHHQIHHfIII')], [16, 24, 32, 44, 48])   # t_mono_ns, calc_us, emissivity, ring_id, t_offset
        for w_, h_ in ((640, 480), (384, 288), (64, 48), (7, 3)): self.assertEqual(tr.stride(w_, h_), stride(w_, h_))
        self.assertEqual(stride(640, 480), 614464)

    def test_ring_built_from_the_note_is_read_by_temp_ring(self):
        d_ = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d_, True); p_ = os.path.join(d_, 'ring')
        with open(p_, 'wb') as f_: f_.write(bytes(build_ring([(2 * k, tr.from_celsius(np.full((6, 8), 30.0 + k))) for k in range(1, 7)], emis_=0.98)))
        m_ = tr.TempRing(p_).newest()   # 6 maps in 4 slots: map 6 in slot 1
        self.assertEqual((m_.frame, m_.map_seq, m_.ring_id, m_.t_ns, m_.calc_us, m_.flags, m_.state, m_.every, m_.values.shape), (12, 6, 77, 6 * 10 ** 9, 906, FLAG_AGC_READ, STATE_ON, 2, (6, 8)))
        np.testing.assert_allclose(tr.to_celsius(m_.values, m_.t_offset, m_.t_div), 36.0, atol=1e-4)


@unittest.skipUnless(os.environ.get('CAM_TEMP_RING_FILE'), 'CAM_TEMP_RING_FILE not set (a ring written by cam_temp_ring_test)')
class WrittenByCamPipe(unittest.TestCase):
    def test_ring_from_cam_temp_ring_test(self):
        path_ = os.environ['CAM_TEMP_RING_FILE']
        with open(path_, 'rb') as f_: b_ = f_.read()
        w_, h_ = 64, 48; sd_ = stride(w_, h_); pay_ = w_ * h_ * 2
        magic_, ver_, rid_, ns_, sd2_, pay2_, w2_, h2_, pid_, every_, state_, emis_, max_ms_, maps_ = RING_HDR.unpack_from(b_, 0)
        self.assertEqual((magic_, ver_, ns_, sd2_, pay2_, w2_, h2_, every_, max_ms_, maps_), (b'LWT1', 1, SLOTS, sd_, pay_, w_, h_, 3, 50, 6))
        self.assertIn(state_, (STATE_ON, STATE_STOPPED)); self.assertGreater(pid_, 1); self.assertNotEqual(rid_, 0)
        self.assertAlmostEqual(emis_, 0.98, places=6)
        self.assertEqual(len(b_), HDR_N + SLOTS * sd_)
        want_ = lambda k_: (5000 + 100 * (20 + k_) + np.arange(w_ * h_) % 50).astype(np.uint16).reshape(h_, w_)
        for slot_, k_ in ((0, 5), (1, 6), (2, 3), (3, 4)):   # maps 1 and 2 were overwritten by 5 and 6
            s_ = SLOT_HDR.unpack_from(b_, HDR_N + slot_ * sd_)
            self.assertEqual(s_[:6] + s_[6:8] + s_[9:], (3 * k_, b'LWTS', 1, FLAG_AGC_READ, 10 ** 9 * k_ + 7, 1000 + k_, w_, h_, k_, pay_, rid_, 5000, 100))
            self.assertAlmostEqual(s_[8], 0.98, places=6)
            self.assertTrue(np.array_equal(np.frombuffer(b_, '<u2', w_ * h_, HDR_N + slot_ * sd_ + HDR_N).reshape(h_, w_), want_(k_)))
        r_ = tr.TempRing(path_); m_ = r_.newest()   # the server's reader
        self.assertEqual((m_.frame, m_.map_seq, m_.ring_id, m_.t_ns, m_.calc_us, m_.flags, m_.state, m_.every, m_.width, m_.height), (18, 6, rid_, 6 * 10 ** 9 + 7, 1006, 1, state_, 3, w_, h_))
        self.assertTrue(np.array_equal(m_.values, want_(6)))
        self.assertAlmostEqual(float(tr.to_celsius(m_.values, m_.t_offset, m_.t_div)[0, 1]), 26.01, places=4)


if __name__ == '__main__':
    unittest.main()
