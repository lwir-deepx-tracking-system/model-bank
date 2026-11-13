"""검출 모델 출력 종류 시험: 파일 읽기(bank_probe), 목록 올리기(bank_add_), 엔진 출력 칸(Engine), 상자 만들기(_detect).

NPU 없이 돈다. parse_model 의 글과 NPU 출력은 지어낸 값이다 (실제 장치의 값이 아니다).
'v8' = 출력 하나 [1,20,N], 'yolox' = 출력 하나 [1,N,21] (model_scan.RAW_DETS 의 설명)."""
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..common.state import APP
from ..common.shared_values import HERE, CORE, DET, anchors
from . import detect_frame, model_scan, npu_engine

F = np.float32
HEADS = ['box_s8', 'box_s16', 'box_s32', 'cls_s8', 'cls_s16', 'cls_s32']
_MISSING = object()


def six_outputs(h, w):
    return [(pre + str(st), (1, c, h // st, w // st), 'float32') for c, pre in ((4, 'box_s'), (16, 'cls_s')) for st in (8, 16, 32)]


def parse_model_text(inputs, outputs, inner=(), colour=False):
    """Text in the form bank_probe reads: tensor lines '+- name {shape: [..], dtype: ..' and the 'Model Output Tensors:' name block. Invented layout, not a device capture."""
    def line(t): return '      +- %s {shape: [%s], dtype: %s, layout: N/A}' % (t[0], ', '.join(str(v) for v in t[1]), t[2])
    rows = ['DXNN Model Ver. : 8', 'Model Input Tensors:'] + ['  - ' + t[0] for t in inputs]
    rows += ['Model Output Tensors:'] + ['  - ' + t[0] for t in outputs] + ['', 'Tasks:', '  Task[0] npu_0, NPU', '    inputs'] + [line(t) for t in inputs]
    rows += ['    outputs'] + [line(t) for t in (list(inner) or outputs)]
    if inner: rows += ['  Task[1] cpu_0, CPU', '    inputs'] + [line(t) for t in inner] + ['    outputs'] + [line(t) for t in outputs]
    text = '\n'.join(rows) + '\n'
    if colour: text = text.replace('+- ', '\x1b[1;32m+- \x1b[0m').replace('Model Output Tensors:', '\x1b[36mModel Output Tensors:\x1b[0m')
    return text


def probe(inputs, outputs, **kw):
    done = SimpleNamespace(stdout=parse_model_text(inputs, outputs, **kw).encode('utf-8'))
    with patch.object(model_scan.subprocess, 'run', return_value=done): return model_scan.bank_probe('x.dxnn')


def image(h, w, dtype='uint8'): return [('images', (1, h, w, 3), dtype)]


def ref_nms(y):
    """numpy copy of post.cpp nms_decoded: score > 0.001 for every (anchor, class), stable sort by score, at most 30,000, class-wise greedy NMS (IoU > 0.7 dropped), best 300."""
    assert y.dtype == np.float32 and y.flags['C_CONTIGUOUS'] and y.ndim == 3 and y.shape[:2] == (1, 20)
    a = y[0]; half = F(.5); cand = []
    ii, cc = np.nonzero(a[4:].T > F(.001))
    for i, c in zip(ii, cc):
        x, yc, w, h = a[0, i], a[1, i], a[2, i], a[3, i]
        cand.append((x - w * half, yc - h * half, x + w * half, yc + h * half, a[4 + c, i], int(c)))
    cand = [cand[k] for k in sorted(range(len(cand)), key=lambda k: -float(cand[k][4]))][:30000]
    dead = [False] * len(cand)
    for c in range(16):
        ids = [k for k, v in enumerate(cand) if v[5] == c]
        for p, i in enumerate(ids):
            if dead[i]: continue
            u = cand[i]; area = (u[2] - u[0]) * (u[3] - u[1])
            for j in ids[p + 1:]:
                if dead[j]: continue
                z = cand[j]
                iw = max(F(0), min(u[2], z[2]) - max(u[0], z[0])); ih = max(F(0), min(u[3], z[3]) - max(u[1], z[1]))
                inter = iw * ih; other = (z[2] - z[0]) * (z[3] - z[1])
                if inter / (area + other - inter) > F(.7): dead[j] = True
    keep = [v for k, v in enumerate(cand) if not dead[k]][:300]
    return np.array(keep, np.float32).reshape(-1, 6)


def grid(h, w):
    """anchor x index, y index, stride for the N anchors: strides 8, 16, 32, row-major inside a level (the order of the model files and of post.cpp)."""
    gx, gy, st = [], [], []
    for s in (8, 16, 32):
        hh, ww = h // s, w // s; j = np.arange(hh * ww); gx.append(j % ww); gy.append(j // ww); st.append(np.full(hh * ww, s))
    return [np.concatenate(v).astype(np.float32) for v in (gx, gy, st)]


def anchor_of(h, w, stride, cx, cy):
    """index of the anchor of this stride whose cell holds the point (cx, cy) in model pixels"""
    off = 0
    for s in (8, 16, 32):
        if s == stride: return off + int(cy // s) * (w // s) + int(cx // s)
        off += (h // s) * (w // s)


def v8_file_tail(dfl, logits, h, w):
    """What the CPU part inside the YOLOv8 files computes (read from the embedded graph): dfl [4,N] = left, top, right, bottom distances in grid units,
    logits [16,N] -> output0 [1,20,N] = cx, cy, w, h in model pixels + sigmoid(logits)."""
    gx, gy, st = grid(h, w); ax, ay = gx + F(.5), gy + F(.5); out = np.empty((1, 20, dfl.shape[1]), np.float32)
    out[0, 0] = (ax + (dfl[2] - dfl[0]) * F(.5)) * st; out[0, 1] = (ay + (dfl[3] - dfl[1]) * F(.5)) * st
    out[0, 2] = (dfl[0] + dfl[2]) * st; out[0, 3] = (dfl[1] + dfl[3]) * st
    out[0, 4:] = 1 / (1 + np.exp(-logits)); return out


def yolox_file_tail(reg, obj, cls, h, w):
    """What the CPU part inside the YOLOX files computes (read from the embedded graph): reg [N,4] raw, obj [N] and cls [N,16] already 0..1
    -> output0 [1,N,21] = (reg xy + cell index) * stride, exp(reg wh) * stride, obj, cls."""
    gx, gy, st = grid(h, w); out = np.empty((1, reg.shape[0], 21), np.float32)
    out[0, :, 0] = (reg[:, 0] + gx) * st; out[0, :, 1] = (reg[:, 1] + gy) * st
    out[0, :, 2] = np.exp(reg[:, 2]) * st; out[0, :, 3] = np.exp(reg[:, 3]) * st
    out[0, :, 4] = obj; out[0, :, 5:] = cls; return out


class AppCase(unittest.TestCase):
    """Sets values on the shared APP object and on the module tables, and puts every one of them back afterwards."""
    def app(self, **kw):
        for k, v in kw.items():
            old = APP.__dict__.get(k, _MISSING)
            self.addCleanup((lambda k_=k: APP.__dict__.pop(k_, None)) if old is _MISSING else (lambda k_=k, o_=old: setattr(APP, k_, o_)))
            setattr(APP, k, v)

    def keep_tables(self):
        for t in (model_scan.BANK, model_scan.BANK_SEEN, model_scan.BANK_REJ, model_scan.RAW_DETS, model_scan.RAW_KEYS, DET, CORE, detect_frame.RAW_BUF):
            self.addCleanup(lambda t_=t, c_=dict(t): (t_.clear(), t_.update(c_)))
        for t in (model_scan.W480_DETS, model_scan.S6_DETS, model_scan.E2E_DETS, model_scan.E2E_KEYS):
            self.addCleanup(lambda t_=t, c_=set(t): (t_.clear(), t_.update(c_)))


class ProbeTests(unittest.TestCase):
    def test_six_heads_and_end_to_end_answers_are_as_before(self):
        for h in (480, 640):
            self.assertEqual(probe(image(h, 640), six_outputs(h, 640)), ((h, 640, 'six'), None))
            self.assertEqual(probe(image(h, 640), [('output0', (1, 300, 6), 'float32')]), ((h, 640, 'e2e'), None))
        self.assertEqual(probe(image(640, 640), six_outputs(640, 640), colour=True), ((640, 640, 'six'), None))
        bad = six_outputs(640, 640); bad[1] = ('box_s16', (1, 4, 40, 41), 'float32')
        self.assertEqual(probe(image(640, 640), bad), (None, "head box_s16 has shape ((1, 4, 40, 41), 'float32')"))
        swapped = six_outputs(640, 640)[::-1]
        got, why = probe(image(640, 640), swapped)
        self.assertIsNone(got); self.assertIn('neither the six detector heads', why)

    def test_one_decoded_output_gives_the_two_new_kinds(self):
        inner8 = [('/model.22/dfl/conv/Conv_output_0', (1, 1, 4, 8400), 'float32'), ('/model.22/cv3.0/cv3.0.2/Conv_output_0', (1, 16, 80, 80), 'float32')]
        self.assertEqual(probe(image(640, 640), [('output0', (1, 20, 8400), 'float32')], inner=inner8), ((640, 640, 'v8'), None))
        self.assertEqual(probe(image(640, 640), [('output0', (1, 8400, 21), 'float32')], colour=True), ((640, 640, 'yolox'), None))
        self.assertEqual(anchors(480, 640), 6300)
        self.assertEqual(probe(image(480, 640), [('out', (1, 20, 6300), 'float32')]), ((480, 640, 'v8'), None))
        self.assertEqual(probe(image(480, 640), [('out', (1, 6300, 21), 'float32')]), ((480, 640, 'yolox'), None))

    def test_other_single_outputs_are_rejected_with_the_shape_in_the_reason(self):
        cases = [((640, 640), (1, 84, 8400), 'float32'),     # 80 classes
                 ((640, 640), (1, 21, 8400), 'float32'),     # YOLOX columns on the wrong axis
                 ((640, 640), (1, 8400, 20), 'float32'),     # YOLOv8 rows on the wrong axis
                 ((640, 640), (1, 8400, 85), 'float32'),     # YOLOX with 80 classes
                 ((640, 640), (1, 20, 6300), 'float32'),     # anchors of the other input size
                 ((480, 640), (1, 20, 8400), 'float32'),
                 ((480, 640), (1, 8400, 21), 'float32'),
                 ((640, 640), (20, 8400), 'float32'),        # no batch axis
                 ((640, 640), (1, 20, 8400), 'uint8'),
                 ((640, 640), (1, 8400, 21), 'float16'),
                 ((640, 640), (1, 300, 7), 'float32'),
                 ((640, 640), (1, 128), 'float32')]
        for (h, w), shape, dtype in cases:
            got, why = probe(image(h, w), [('output0', shape, dtype)])
            self.assertIsNone(got, (h, w, shape, dtype))
            self.assertIn('the one output output0 is %s %s' % (list(shape), dtype), why)
            n = anchors(h, w)
            self.assertIn('[1,300,6]', why); self.assertIn('[1,20,%d]' % n, why); self.assertIn('[1,%d,21]' % n, why)

    def test_input_and_output_count_rules_did_not_loosen(self):
        v8 = [('output0', (1, 20, 8400), 'float32')]
        self.assertEqual(probe(image(640, 640, 'float32'), v8)[0], None)
        self.assertIn('is not 640x480 / 640x640 uint8', probe(image(640, 640, 'float32'), v8)[1])
        self.assertIn('is not 640x480 / 640x640 uint8', probe(image(320, 320), [('output0', (1, 20, 2100), 'float32')])[1])
        self.assertIn('is not 640x480 / 640x640 uint8', probe([('images', (1, 3, 640, 640), 'uint8')], v8)[1])
        self.assertEqual(probe([('input', (1, 640, 640, 3), 'uint8')], v8), (None, 'not a readable model file'))
        two = [('output0', (1, 20, 8400), 'float32'), ('output1', (1, 8400, 21), 'float32')]
        got, why = probe(image(640, 640), two)
        self.assertIsNone(got); self.assertIn("['output0', 'output1'] are neither the six detector heads", why)
        done = SimpleNamespace(stdout=b'Model Output Tensors:\n  - output0\n      +- images {shape: [1, 640, 640, 3], dtype: uint8}\n')
        with patch.object(model_scan.subprocess, 'run', return_value=done):
            self.assertEqual(model_scan.bank_probe('x.dxnn')[0], None)
            self.assertIn('the one output output0 is unreadable', model_scan.bank_probe('x.dxnn')[1])
        with patch.object(model_scan.subprocess, 'run', side_effect=OSError()):
            self.assertEqual(model_scan.bank_probe('x.dxnn'), (None, 'parse_model missing'))

    def test_raw_shape(self):
        self.assertEqual(model_scan.raw_shape('v8', 8400), (1, 20, 8400))
        self.assertEqual(model_scan.raw_shape('yolox', 8400), (1, 8400, 21))


class ListingTests(AppCase):
    OUT = {'yolov8n_nms_640x640': (640, [('output0', (1, 20, 8400), 'float32')]), 'yolox_s_nms_640x640': (640, [('output0', (1, 8400, 21), 'float32')]),
           'wide_v8': (480, [('output0', (1, 20, 6300), 'float32')]), 'endtoend': (640, [('output0', (1, 300, 6), 'float32')]),
           'heads': (640, six_outputs(640, 640)), 'classes80': (640, [('output0', (1, 84, 8400), 'float32')])}

    def test_scan_lists_each_kind_on_its_own_path_and_keeps_the_old_kinds(self):
        self.keep_tables()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for i, stem in enumerate(self.OUT):
                sub = folder / ('yolov8' if 'v8' in stem else 'other'); sub.mkdir(exist_ok=True); (sub / (stem + '.dxnn')).write_bytes(b'model %d' % i)
            (folder / 'yolov8' / 'yolov8n_nms_640x640_compile_metadata.json').write_text(json.dumps({'header': {}}), encoding='utf-8')
            (folder / 'yolov8' / 'yolov8n_nms_640x640.json').write_text(json.dumps({'display_name': 'YOLOv8n_(FLIR) 640x640'}), encoding='utf-8')
            self.app(MODEL=dict(lock=threading.Lock(), catalog=[dict(id=1, det='w480', key='NEWMRW480', name='first')]), man={}, det_shape={}, BANK_DIR=folder)

            def run(cmd, **kw):
                h, outs = self.OUT[Path(cmd[2]).stem]
                return SimpleNamespace(stdout=parse_model_text(image(h, 640), outs).encode('utf-8'))
            now = time.time() + 100.0
            with patch.object(model_scan.subprocess, 'run', side_effect=run), patch('builtins.print') as said:
                model_scan.bank_add_(now)
                self.assertEqual(len(APP.MODEL['catalog']), 1)   # first scan only remembers size and time
                model_scan.bank_add_(now)
            cat = {c['key']: c for c in APP.MODEL['catalog'][1:]}
            self.assertEqual(sorted(cat), ['endtoend', 'heads', 'wide_v8', 'yolov8n_nms_640x640', 'yolox_s_nms_640x640'])
            self.assertEqual([c['id'] for c in APP.MODEL['catalog']], [1, 2, 3, 4, 5, 6])
            self.assertEqual(cat['yolov8n_nms_640x640']['name'], 'YOLOv8n_(FLIR) 640x640 (자동 등록)')
            self.assertEqual(cat['yolox_s_nms_640x640']['name'], 'yolox_s_nms_640x640 (자동 등록)')
            self.assertEqual(cat['wide_v8']['name'], 'wide_v8 640x480 (자동 등록)')
            self.assertEqual(cat['endtoend']['name'], 'endtoend 640x640 [NMS-free] (자동 등록)')
            self.assertEqual(cat['heads']['name'], 'heads 640x640 (자동 등록)')
            self.assertEqual({k: v for k, v in model_scan.RAW_DETS.items() if k.startswith('bank:')},
                             {'bank:yolov8n_nms_640x640': 'v8', 'bank:yolox_s_nms_640x640': 'yolox', 'bank:wide_v8': 'v8'})
            self.assertEqual(model_scan.RAW_KEYS['yolox_s_nms_640x640'], 'yolox'); self.assertEqual(model_scan.RAW_KEYS['wide_v8'], 'v8')
            self.assertNotIn('heads', model_scan.RAW_KEYS); self.assertNotIn('endtoend', model_scan.RAW_KEYS)
            self.assertIn('bank:endtoend', model_scan.E2E_DETS); self.assertIn('endtoend', model_scan.E2E_KEYS)
            for det in ('bank:yolov8n_nms_640x640', 'bank:yolox_s_nms_640x640', 'bank:wide_v8', 'bank:heads'):
                self.assertNotIn(det, model_scan.E2E_DETS)
            for det in ('bank:yolov8n_nms_640x640', 'bank:yolox_s_nms_640x640', 'bank:heads', 'bank:endtoend'):   # input canvas of 640x640 models
                self.assertIn(det, model_scan.S6_DETS); self.assertEqual(DET[det]['yoff'], -64.0)
            self.assertIn('bank:wide_v8', model_scan.W480_DETS); self.assertEqual(DET['bank:wide_v8']['yoff'], 16.0)
            self.assertEqual(APP.det_shape['yolox_s_nms_640x640'], (640, 640)); self.assertEqual(APP.det_shape['wide_v8'], (480, 640))
            lines = [' '.join(str(v) for v in c.args) for c in said.call_args_list]
            self.assertEqual(len([v for v in lines if v.startswith('BANK file not listed: classes80.dxnn (the one output output0 is [1, 84, 8400] float32;')]), 1, lines)

    def test_the_four_repository_files_have_a_name_tag_that_matches_the_manifest(self):
        files = json.loads((HERE / 'models' / 'MANIFEST.json').read_text(encoding='utf-8'))['_files']
        want = {'yolov8/yolov8n_nms_640x640': 'YOLOv8n_(FLIR) 640x640', 'yolov8/yolov8s_nms_640x640': 'YOLOv8s_(FLIR) 640x640',
                'yolox/yolox_nano_nms_640x640': 'YOLOX-nano_(FLIR) 640x640', 'yolox/yolox_s_nms_640x640': 'YOLOX-s_(FLIR) 640x640'}
        for rel, name in want.items():
            tag = json.loads((HERE / 'models' / 'detect' / (rel + '.json')).read_text(encoding='utf-8'))
            self.assertEqual(tag['display_name'], name)
            self.assertEqual(tag['sha256'], files['detect/' + rel + '.dxnn'])
            meta = json.loads((HERE / 'models' / 'detect' / (rel + '_compile_metadata.json')).read_text(encoding='utf-8'))
            self.assertNotIn('display_name', meta)   # bank_scan reads this file first; a name there would hide the name tag

    def test_models_readme_names_every_detection_file_and_gives_the_calibration_folder_without_a_personal_path(self):
        files = json.loads((HERE / 'models' / 'MANIFEST.json').read_text(encoding='utf-8'))['_files']
        text = (HERE / 'models' / 'README.md').read_text(encoding='utf-8')
        detect = sorted(k[len('detect/'):] for k in files if k.startswith('detect/') and k.endswith('.dxnn'))
        self.assertEqual(len(detect), 13)   # 4 linked by MANIFEST _links + 9 found by the scan; the top-level README states this count
        for rel in detect: self.assertIn('`%s`' % rel, text)
        # the side files hide the calibration folder ('<YOUR_PATH>'); the .dxnn files name it, so the README must not call it unknown
        for word in ('<YOUR_PATH>', 'calibration_train_only', 'yolox_nano_canonical_canvas_dxnn_20261020_v2', 'yolox_s_canonical_canvas_dxnn_20261020_v2'):
            self.assertIn(word, text)
        tags = [(HERE / 'models' / 'detect' / (rel[:-5] + '.json')).read_text(encoding='utf-8') for rel in detect if rel.startswith(('yolov8/', 'yolox/'))]
        self.assertEqual(len(tags), 4)
        for written in [text] + tags:   # public repository: no personal folder in the text files of this change
            for lead in ('/Users/', 'Users' + chr(92), '/home/', '/mnt/c/'):
                self.assertNotIn(lead, written)


class DecodeTests(AppCase):
    """_detect for the two new kinds, with the numpy copy of the host NMS (NativeDecodeTests repeats it with post.cpp itself)."""
    nms = staticmethod(ref_nms)

    def use(self, det, yoff, kind=None, hw=(640, 640)):
        self.keep_tables()
        self.app(args=SimpleNamespace(det=det, min_person_h=0.0), det_yoff=yoff, fast_nms=self.nms, _detect=detect_frame._detect)
        if kind: model_scan.RAW_DETS[det] = kind
        (model_scan.W480_DETS if hw[0] == 480 else model_scan.S6_DETS).add(det)

    def check_rows(self, got, want):
        self.assertEqual(got.dtype, np.float32); self.assertEqual(got.shape, (len(want), 6), got)
        np.testing.assert_allclose(got, np.array(want, np.float32), rtol=0, atol=2e-4)

    def test_v8_person_box_reaches_the_frame_canvas(self):
        # a person on camera rows 200..320, columns 300..340 = model rows 280..400 (the picture sits in rows 80:560 of the 640x640 input) = canvas rows 216..336
        self.use('bank:yolov8n_nms_640x640', -64.0, 'v8')
        n = 8400; dfl = np.zeros((4, n), np.float32); logits = np.full((16, n), -12.0, np.float32)
        a = anchor_of(640, 640, 16, 320, 340); gx, gy, st = grid(640, 640)
        self.assertEqual((gx[a], gy[a], st[a]), (20, 21, 16))   # anchor centre (328, 344)
        dfl[:, a] = [(328 - 300) / 16, (344 - 280) / 16, (340 - 328) / 16, (400 - 344) / 16]; logits[0, a] = 2.0
        b = anchor_of(640, 640, 8, 322, 342)                    # the same person seen by a stride-8 anchor, a little weaker: removed by NMS
        dfl[:, b] = [(gx[b] * 8 + 4 - 301) / 8, (gy[b] * 8 + 4 - 282) / 8, (341 - gx[b] * 8 - 4) / 8, (399 - gy[b] * 8 - 4) / 8]; logits[0, b] = 1.0
        c = anchor_of(640, 640, 32, 100, 500)                   # another class (2) elsewhere, and a second class (6) on the same anchor (multi-label)
        dfl[:, c] = [1, 1, 1, 1]; logits[2, c] = 0.0; logits[6, c] = -1.0
        out0 = v8_file_tail(dfl, logits, 640, 640); self.assertEqual(out0.shape, (1, 20, 8400))
        kept = out0.copy(); got = detect_frame.detect([out0], None)
        s = lambda v: 1 / (1 + np.exp(-v))
        self.check_rows(got, [[300, 216, 340, 336, s(2.0), 0], [80, 464 - 64, 144, 528 - 64, 0.5, 2], [80, 464 - 64, 144, 528 - 64, s(-1.0), 6]])
        np.testing.assert_array_equal(out0, kept)               # the engine's buffer is not written to
        self.assertTrue((got[:, 4] > 0.001).all())              # sigmoid(-12) = 6e-6 backgrounds are under the 0.001 threshold

    def test_yolox_person_box_reaches_the_frame_canvas(self):
        self.use('bank:yolox_s_nms_640x640', -64.0, 'yolox')
        n = 8400; reg = np.zeros((n, 4), np.float32); obj = np.full(n, 1e-4, np.float32); cls = np.full((n, 16), 0.9, np.float32)   # high class score alone is not a box
        gx, gy, st = grid(640, 640)
        a = anchor_of(640, 640, 16, 320, 340); self.assertEqual((gx[a], gy[a]), (20, 21))   # cell corner (320, 336)
        reg[a] = [(320 - 320) / 16, (340 - 336) / 16, np.log(40 / 16), np.log(120 / 16)]; obj[a] = 0.9; cls[a] = 0.001; cls[a, 0] = 0.8
        b = anchor_of(640, 640, 8, 322, 342)
        reg[b] = [(321 - gx[b] * 8) / 8, (340.5 - gy[b] * 8) / 8, np.log(40 / 8), np.log(117 / 8)]; obj[b] = 0.7; cls[b] = 0.001; cls[b, 0] = 0.8
        c = anchor_of(640, 640, 32, 100, 500); self.assertEqual((gx[c] * 32, gy[c] * 32), (96, 480))
        reg[c] = [0.5, 0.5, 0.0, np.log(2.0)]; obj[c] = 0.5; cls[c] = 0.0; cls[c, 6] = 0.6; cls[c, 2] = 0.4
        d = anchor_of(640, 640, 8, 600, 100); obj[d] = 0.99; cls[d] = 0.001                # sure object, no class over the threshold: 0.99 * 0.001 < 0.001
        out0 = yolox_file_tail(reg, obj, cls, 640, 640); self.assertEqual(out0.shape, (1, 8400, 21))
        kept = out0.copy(); got = detect_frame.detect([out0], None)
        self.check_rows(got, [[300, 280 - 64, 340, 400 - 64, 0.9 * 0.8, 0], [96, 464 - 64, 128, 528 - 64, 0.5 * 0.6, 6], [96, 464 - 64, 128, 528 - 64, 0.5 * 0.4, 2]])
        np.testing.assert_array_equal(out0, kept)

    def test_640x480_input_uses_its_own_offset(self):
        n = 6300; y = np.zeros((1, 20, n), np.float32); y[0, :4, 77] = [100, 200, 20, 60]; y[0, 4 + 0, 77] = 0.5
        self.use('bank:wide_v8', 16.0, 'v8', hw=(480, 640))
        self.check_rows(detect_frame._detect([y], None), [[90, 170 + 16, 110, 230 + 16, 0.5, 0]])
        o = np.zeros((1, n, 21), np.float32); o[0, 77, :6] = [100, 200, 20, 60, 0.5, 0.5]
        self.use('bank:wide_yolox', 16.0, 'yolox', hw=(480, 640))
        self.check_rows(detect_frame._detect([o], None), [[90, 170 + 16, 110, 230 + 16, 0.25, 0]])

    def test_same_boxes_give_the_same_rows_on_the_six_head_path_and_on_both_new_paths(self):
        rng = np.random.default_rng(31); n = 8400
        boxes = rng.uniform(0.2, 6.0, (1, 4, n)).astype(np.float32); logits = np.full((1, 16, n), -9.0, np.float32)
        hot = rng.choice(n, 220, replace=False); logits[0, rng.integers(0, 16, 220), hot] = rng.uniform(-4, 4, 220).astype(np.float32)
        logits[0, 0, hot[:60]] = rng.uniform(-2, 3, 60).astype(np.float32)

        def decode_raw(box, scores, out, h, w):   # numpy copy of post.cpp decode_raw
            gx, gy, st = grid(h, w); ax, ay = gx + F(.5), gy + F(.5); b = box.reshape(4, -1)
            l, t, r, bt = ax - b[0], ay - b[1], ax + b[2], ay + b[3]
            out[0, 0] = (l + r) * F(.5) * st; out[0, 1] = (t + bt) * F(.5) * st; out[0, 2] = (r - l) * st; out[0, 3] = (bt - t) * st
        self.use('s6', -64.0)
        self.app(native_post=SimpleNamespace(lib=SimpleNamespace(decode_raw=decode_raw)), cv2=SimpleNamespace(exp=np.exp))
        self.app(dec_s=npu_engine.Decoder(640, 640))
        six = detect_frame._detect([boxes, logits.copy()], None)
        self.assertGreater(len(six), 100); self.assertLess(len(six), 300)
        y = APP.dec_s.out.copy()                                # the candidate form of the six-head path: [1,20,N]
        APP.args.det = 'bank:v'; model_scan.RAW_DETS['bank:v'] = 'v8'
        np.testing.assert_array_equal(detect_frame._detect([y.copy()], None), six)
        # YOLOX form of the same candidates: objectness 1 keeps the scores; objectness 0.5 with doubled class scores gives them back as well
        o = np.empty((1, n, 21), np.float32); o[0, :, :4] = y[0, :4].T; o[0, :, 4] = 1.0; o[0, :, 5:] = y[0, 4:].T
        APP.args.det = 'bank:x'; model_scan.RAW_DETS['bank:x'] = 'yolox'
        np.testing.assert_array_equal(detect_frame._detect([o], None), six)
        o[0, :, 4] = 0.5; o[0, :, 5:] *= 2
        np.testing.assert_array_equal(detect_frame._detect([o], None), six)

    def test_raw_decoded_form(self):
        self.keep_tables()
        rng = np.random.default_rng(5); o = rng.uniform(0, 1, (1, 8400, 21)).astype(np.float32)
        y = detect_frame.raw_decoded('yolox', o)
        self.assertEqual((y.shape, y.dtype, y.flags['C_CONTIGUOUS']), ((1, 20, 8400), np.dtype('float32'), True))
        for i in (0, 6399, 6400, 8399):
            np.testing.assert_array_equal(y[0, :4, i], o[0, i, :4]); np.testing.assert_array_equal(y[0, 4:, i], o[0, i, 5:] * o[0, i, 4])
        self.assertIs(detect_frame.raw_decoded('yolox', o), y)                    # one work buffer per anchor count
        self.assertEqual(detect_frame.raw_decoded('yolox', o[:, :6300]).shape, (1, 20, 6300))
        v = rng.uniform(0, 1, (1, 20, 8400)).astype(np.float32); z = detect_frame.raw_decoded('v8', v)
        self.assertTrue(np.shares_memory(z, v)); self.assertTrue(z.flags['C_CONTIGUOUS'])   # no copy when the engine output is already in that form
        z = detect_frame.raw_decoded('v8', v.ravel()); self.assertEqual(z.shape, (1, 20, 8400))

    def test_end_to_end_path_is_still_taken_first_and_unchanged(self):
        self.use('bank:e', -64.0); model_scan.E2E_DETS.add('bank:e')
        self.app(fast_nms=lambda y: self.fail('the end-to-end path must not call the host NMS'))
        p = np.zeros((1, 300, 6), np.float32); p[0, 0] = [10, 100, 30, 160, 0.9, 0]; p[0, 1] = [50, 50, 40, 90, 0.8, 0]; p[0, 2] = [1, 2, 3, 4, 0.001, 0]
        self.check_rows(detect_frame._detect([p], None), [[10, 36, 30, 96, 0.9, 0]])


@unittest.skipUnless(shutil.which('g++'), 'g++ is needed to build post.cpp')
class NativeDecodeTests(DecodeTests):
    """The same decode tests through the real host NMS (post.cpp, built into a temporary folder as scripts/setup.sh builds post.so)."""
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(); so = Path(cls.folder.name) / 'post.so'
        subprocess.run(['g++', '-O3', '-shared', '-fPIC', '-o', str(so), str(Path(__file__).with_name('post.cpp'))], check=True)
        lib = ctypes.CDLL(str(so)); arr = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        lib.nms_decoded.argtypes = [arr, arr, ctypes.c_int]; lib.nms_decoded.restype = ctypes.c_int; out = np.empty((300, 6), np.float32)

        def fast_nms(y):   # = native_post.fast_nms
            n = lib.nms_decoded(y, out, y.shape[2]); assert 0 <= n <= 300
            return out[:n].copy()
        cls.nms = staticmethod(fast_nms)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_numpy_copy_of_the_nms_matches_post_cpp(self):
        rng = np.random.default_rng(7); n = 8400; y = np.zeros((1, 20, n), np.float32)
        y[0, :2] = rng.uniform(100, 540, (2, n)); y[0, 2:4] = rng.uniform(20, 200, (2, n))
        hot = rng.choice(n, 600, replace=False); y[0, 4 + rng.integers(0, 3, 600), hot] = rng.uniform(0, 1, 600).astype(np.float32)
        a, b = self.nms(y), ref_nms(y)
        self.assertEqual(a.shape, b.shape); self.assertGreater(len(a), 50); np.testing.assert_allclose(a, b, rtol=0, atol=1e-4)


class FakeOption:
    BOUND_OPTION = SimpleNamespace(NPU_0=0, NPU_1=1, NPU_2=2)


class FakeNpu:
    """Stand-in for dx_engine.InferenceEngine: reports the given tensors and gives back values written by the test."""
    spec = {}

    def __init__(self, model, option):
        self.input, self.outs = self.spec[model]; self.fill = None
    def get_input_tensors_info(self): return [dict(name='images', shape=list(self.input), dtype=np.uint8)]
    def get_output_tensors_info(self): return [dict(name=n, shape=list(s), dtype=np.float32) for n, s in self.outs]
    def get_output_tensor_sizes(self): return [int(np.prod(s)) * 4 for n, s in self.outs]
    def run_async(self, x, user_arg, output_buffer): self.buf = output_buffer; return 1
    def wait(self, job):
        vals, k = [], 0
        for (n, s), v in zip(self.outs, self.fill):
            size = int(np.prod(s)); self.buf[k:k + size] = np.asarray(v, np.float32).ravel(); vals.append(self.buf[k:k + size].reshape(s)); k += size
        return vals


class EngineTests(AppCase):
    def start(self, spec, det_shape, raw=(), e2e=()):
        self.keep_tables(); FakeNpu.spec = spec
        self.app(dx=SimpleNamespace(InferenceOption=FakeOption, InferenceEngine=FakeNpu), args=SimpleNamespace(mock=True, split6=True, pipeline=False, det='', min_person_h=0.0),
                 det_shape=dict(det_shape), pending={})
        model_scan.RAW_KEYS.update(raw); model_scan.E2E_KEYS.update(e2e)

    def test_new_kinds_get_one_output_slot_of_the_file_shape(self):
        self.start({'v': ((1, 640, 640, 3), [('output0', (1, 20, 8400))]), 'x': ((1, 640, 640, 3), [('output0', (1, 8400, 21))]),
                    'w': ((1, 480, 640, 3), [('output0', (1, 6300, 21))])},
                   {'v': (640, 640), 'x': (640, 640), 'w': (480, 640)}, raw={'v': 'v8', 'x': 'yolox', 'w': 'yolox'})
        for model, shape, nbytes in (('v', (1, 20, 8400), 672000), ('x', (1, 8400, 21), 705600), ('w', (1, 6300, 21), 529200)):
            e = npu_engine.Engine(model, 0)
            self.assertEqual([s.shape for s in e.slots], [shape]); self.assertEqual(e.packed.nbytes, nbytes); self.assertFalse(e.split6)
            second = npu_engine.Engine(model + '#core1', 1, 1, model=model)       # the second engine of the alternate-core feed
            self.assertEqual([s.shape for s in second.slots], [shape])

    def test_new_kinds_refuse_a_file_whose_tensors_differ_from_the_probe(self):
        self.start({'v': ((1, 640, 640, 3), [('output0', (1, 8400, 21))]), 'x': ((1, 640, 640, 3), [('output0', (1, 20, 8400))]),
                    'i': ((1, 480, 640, 3), [('output0', (1, 20, 8400))]), 't': ((1, 640, 640, 3), [('output0', (1, 20, 8400)), ('extra', (1, 4))])},
                   {'v': (640, 640), 'x': (640, 640), 'i': (640, 640), 't': (640, 640)}, raw={'v': 'v8', 'x': 'yolox', 'i': 'v8', 't': 'v8'})
        for model in ('v', 'x', 'i', 't'):
            with self.assertRaises(AssertionError): npu_engine.Engine(model, 0)

    def test_engine_output_goes_through_detect(self):
        self.start({'v': ((1, 640, 640, 3), [('output0', (1, 20, 8400))]), 'x': ((1, 640, 640, 3), [('output0', (1, 8400, 21))])},
                   {'v': (640, 640), 'x': (640, 640)}, raw={'v': 'v8', 'x': 'yolox'})
        model_scan.RAW_DETS.update({'bank:v': 'v8', 'bank:x': 'yolox'}); self.app(det_yoff=-64.0, fast_nms=ref_nms)
        y = np.zeros((1, 20, 8400), np.float32); y[0, :4, 5000] = [320, 340, 40, 120]; y[0, 4, 5000] = 0.75
        o = np.zeros((1, 8400, 21), np.float32); o[0, 5000, :6] = [320, 340, 40, 120, 0.75, 1.0]
        for model, val in (('v', y), ('x', o)):
            e = npu_engine.Engine(model, 0); e.b.fill = [val]; e.submit(np.zeros(e.in_shape, np.uint8), 0); APP.args.det = 'bank:' + model
            ra = e.wait(); self.assertEqual([v.shape for v in ra], [val.shape])
            np.testing.assert_allclose(detect_frame._detect(ra, None), [[300, 216, 340, 336, 0.75, 0]], rtol=0, atol=1e-5)
        e.b.fill = [np.where(o == 120, np.inf, o)]; e.submit(np.zeros(e.in_shape, np.uint8), 1)
        with self.assertRaises(AssertionError): e.wait()   # as for every model: a value that is not finite stops the run

    def test_six_heads_and_end_to_end_engines_are_as_before(self):
        heads = [(n, s) for n, s, d in six_outputs(640, 640)]
        self.start({'s': ((1, 640, 640, 3), heads), 'e': ((1, 640, 640, 3), [('output0', (1, 300, 6))])}, {'s': (640, 640), 'e': (640, 640)}, e2e={'e'})
        e = npu_engine.Engine('s', 0); self.assertTrue(e.split6); self.assertEqual([s.shape for s in e.slots], [s for n, s in heads])
        rng = np.random.default_rng(3); vals = [rng.normal(size=s).astype(np.float32) for n, s in heads]
        e.b.fill = vals; e.submit(np.zeros(e.in_shape, np.uint8), 0); ra = e.wait()
        self.assertEqual([v.shape for v in ra], [(1, 4, 8400), (1, 16, 8400)])
        np.testing.assert_array_equal(ra[0], np.concatenate([v.reshape(1, 4, -1) for v in vals[:3]], axis=2))
        np.testing.assert_array_equal(ra[1], np.concatenate([v.reshape(1, 16, -1) for v in vals[3:]], axis=2))
        e = npu_engine.Engine('e', 0); self.assertFalse(e.split6); self.assertEqual([s.shape for s in e.slots], [(1, 300, 6)])


if __name__ == '__main__':
    unittest.main()
