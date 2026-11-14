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
    def run(self, xs):
        self.calls.append(xs)
        if self.kind == 'ae': return [np.zeros((1, 15, 20, 256)), np.zeros((1, 15, 20, 256)), np.zeros((1, 15, 20, 4))]
        if self.kind == 'b0': return [np.zeros((1, 300, 256)), np.zeros((1, 300, 4))]
        lg = np.full((1, 300, 1), -10.0); lg[0, 0, 0] = 5.0; bx = np.zeros((1, 300, 4)); bx[0, 0] = [0.5, 0.5, 0.1, 0.4]
        return [lg, bx, np.full((1, 300, 256), 0.5), np.zeros((1, 300, 60)), np.zeros((1, 300, 256))]


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
        E.E2E.update(want='off', on='off', eng=None, err=None, ms=[])

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

    def test_missing_files(self):
        os.remove(os.path.join(self.d, 'b1.dxnn')); E.E2E['want'] = 'ftt'; E.apply(FakeDX)
        self.assertEqual(E.E2E['on'], 'off'); self.assertIn('b1', E.E2E['err'])


if __name__ == '__main__':
    unittest.main()
