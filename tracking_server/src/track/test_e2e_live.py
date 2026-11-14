"""11-14 checks of the live end-to-end FastTrackTr path with fake NPU engines: outputs decoded to canvas rows, the decoder state (traj) handed on,
IoU linking keeps the number, switching on / off happens in apply(). Run from tracking_server/: python -m unittest src.track.test_e2e_live"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from . import e2e_live as E


class FakeEngine:
    def __init__(self, path, opt):
        self.kind = os.path.basename(path).split('.')[0]; self.calls = []
    def get_output_tensors_info(self):
        return {'ae': [dict(name=n) for n in ('kv4', 'tgt4', 'ref4')], 'b0': [dict(name='tgt_out'), dict(name='ref_out')],
                'b1': [dict(name=n) for n in ('logits', 'boxes', 'hs', 'id_words', 'query_pos')]}[self.kind]
    def get_input_tensors_info(self):
        return [dict(name='tgt', shape=[1, 300, 256], dtype=np.float32), dict(name='ref_unact', shape=[1, 300, 4], dtype=np.float32),
                dict(name='kv', shape=[1, 300, 256], dtype=np.float32), dict(name='traj', shape=[1, 300, 256], dtype=np.float32)]
    SCRIPT = None   # optional: frame number (1, 2, ...) -> list of (cx, cy, w, h, id vector) people, all normalised 0..1

    def run(self, xs):
        self.calls.append(xs)
        if self.kind == 'ae': return [np.zeros((1, 15, 20, 256)), np.zeros((1, 15, 20, 256)), np.zeros((1, 15, 20, 4))]
        if self.kind == 'b0': return [np.zeros((1, 300, 256)), np.zeros((1, 300, 4))]
        lg = np.full((1, 300, 1), -10.0); bx = np.zeros((1, 300, 4)); idw = np.zeros((1, 300, 60))
        people = FakeEngine.SCRIPT(len(self.calls)) if FakeEngine.SCRIPT else [(0.5, 0.5, 0.1, 0.4, np.eye(60)[0])]
        for q, (cx, cy, w, h, v) in enumerate(people): lg[0, q, 0] = 5.0; bx[0, q] = [cx, cy, w, h]; idw[0, q] = v
        return [lg, bx, np.full((1, 300, 256), 0.5), idw, np.zeros((1, 300, 256))]


class FakeDX:
    InferenceEngine = FakeEngine
    class InferenceOption:
        BOUND_OPTION = SimpleNamespace(NPU_0=0, NPU_1=1, NPU_2=2)
        def __init__(self): self.devices = []; self.bound_option = None; self.use_ort = False


class E2ELiveTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); E._DIR[0] = self.d
        for k in ('ae', 'b0', 'b1'): open(os.path.join(self.d, k + '.dxnn'), 'wb').close()
        json.dump(dict(label='FastTrackTr test', files=dict(ae='ae.dxnn', b0='b0.dxnn', b1='b1.dxnn')), open(os.path.join(self.d, 'INFO.json'), 'w'))
        E.E2E.update(want='off', on='off', eng=None, err=None, ms=[], link='auto', auto_link='iou'); FakeEngine.SCRIPT = None

    def test_rows_traj_and_ids(self):
        E.E2E['want'] = 'ftt'; E.apply(FakeDX); self.assertEqual(E.E2E['on'], 'ftt')
        g = np.zeros((512, 640), np.uint8)
        r1 = E.step(g); r2 = E.step(g)
        self.assertEqual(len(r1), 1); x1, y1, x2, y2, tid, sc = r1[0][:6]
        np.testing.assert_allclose([x1, y1, x2, y2], [288, 144 + 16, 352, 336 + 16]); self.assertGreater(sc, 0.99)
        self.assertEqual(r2[0][4], tid)                                  # same place -> same number
        b1 = E.E2E['eng'].ebs[0]; self.assertTrue(np.all(b1.calls[0][3] == 0)); self.assertTrue(np.all(b1.calls[1][3] == 0.5))   # traj: zeros, then the last hs
        st = E.state(); self.assertTrue(st['available']); self.assertEqual(st['on'], 'ftt')
        self.assertTrue(E.uses_own())                                    # frame loop skips the detector and KeepID
        E.E2E['want'] = 'off'; E.apply(FakeDX); self.assertIsNone(E.step(g)); self.assertFalse(E.uses_own())

    def test_skip_detector_for_stored_replay(self):   # stored end-to-end results on a recording: the server detector / tracker are not used
        from ..camera import replay_eval as R
        old = R.EV
        try:
            R.EV = SimpleNamespace(own_boxes=True); self.assertTrue(E.skip_detector())
            R.EV = SimpleNamespace(own_boxes=False); self.assertFalse(E.skip_detector())
            R.EV = None; self.assertFalse(E.skip_detector())
        finally:
            R.EV = old

    def _run(self, link, script, frames):
        FakeEngine.SCRIPT = script; E.E2E.update(link=link, want='ftt'); E.apply(FakeDX); g = np.zeros((512, 640), np.uint8)
        out = [E.step(g) for _ in range(frames)]; E.E2E['want'] = 'off'; E.apply(FakeDX); return out

    def test_id_link_keeps_numbers_through_a_crossing(self):
        a, b = np.eye(60)[3], np.eye(60)[7]   # two people with different number vectors walk past each other
        def script(n): x = 0.2 + 0.6 * min(n, 20) / 20; return [(x, 0.5, 0.12, 0.5, a), (1.0 - x, 0.5, 0.12, 0.5, b)]
        def ids_of_a(rows): return [min(r, key=lambda z: abs((z[0] + z[2]) / 2 - (0.2 + 0.6 * min(k + 1, 20) / 20) * 640))[4] for k, r in enumerate(rows)]
        self.assertEqual(len(set(ids_of_a(self._run('id', script, 20)))), 1)            # id: person a keeps one number
        self.assertGreater(len(set(ids_of_a(self._run('iou', script, 20)))), 1)          # overlap only: it gets swapped at the crossing

    def test_id_link_brings_back_a_person_after_a_gap(self):
        a = np.eye(60)[5]
        def script(n):
            if n <= 5: return [(0.30, 0.5, 0.1, 0.4, a)]
            if n <= 25: return []                                                         # hidden for 20 frames
            return [(0.42, 0.5, 0.1, 0.4, a)]                                            # back, moved (no overlap with the last box)
        r = self._run('id', script, 27); self.assertEqual(r[0][0][4], r[-1][0][4])
        r = self._run('iou', script, 27); self.assertNotEqual(r[0][0][4], r[-1][0][4])
        self.assertEqual(E.state()['link'], 'iou')

    def test_auto_follows_info(self):
        json.dump(dict(label='x', id_link=True, files=dict(ae='ae.dxnn', b0='b0.dxnn', b1='b1.dxnn')), open(os.path.join(self.d, 'INFO.json'), 'w'))
        E.E2E['want'] = 'ftt'; E.apply(FakeDX); self.assertEqual(E.link_mode(), 'id'); self.assertEqual(E.state()['link_used'], 'id')
        E.E2E['link'] = 'iou'; self.assertEqual(E.link_mode(), 'iou')

    def test_missing_files(self):
        os.remove(os.path.join(self.d, 'b1.dxnn')); E.E2E['want'] = 'ftt'; E.apply(FakeDX)
        self.assertEqual(E.E2E['on'], 'off'); self.assertIn('b1', E.E2E['err'])


if __name__ == '__main__':
    unittest.main()
