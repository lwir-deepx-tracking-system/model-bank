"""Check of Occlude v4 (pose_partial.py, office occluder bank + heavy-share ramp) through the full training augmentation. No training.

Usage: python test_occlude_bank.py <otp2_dataset dir> <bank.npz> <template_points.npy> <work dir (outside OneDrive)> <out dir> [n=200]
                                   [--v3-file <copy of pose_partial.py before v4>]

1. Tiny OTP2-style set in <work dir>: n OTP2 train pictures (evenly spaced) with their labels; in every second picture the eye/ear slots
   (1-4) get flag 3 (IGNORE) with their coordinates kept, so flag-3 points also lie under pasted patches.
2. Training pipeline as in training (mosaic off, affine, flips, PartialCrop 0.5, body boxes, ThermalLook), Occlude p=1.0 with the bank,
   bank_share 0.5, heavy_ramp (0.5, 0.8, 9): first half of the samples at epoch 0, second half at epoch 9.
   A spy around Occlude compares what goes in and what comes out of every call:
   - boxes of the kept people unchanged, keypoint coordinates unchanged, flags only 2 -> 1, flag 3 unchanged both ways;
   - no visible (2) point where the pasted opacity >= Occlude.COVER (office) or inside the pasted rectangle (v3 shapes);
   - every 2 -> 1 change is under a patch;
   - border blending of office patches: per feathered side, the largest 1-px step of the side-averaged profile from 2 px outside to the
     feather middle ("soft"), the same in the picture before the paste ("natural"), and the step a hard paste would make there ("hard").
     Pass: soft <= natural + 0.5 x hard + 2 grey levels on >= 98 % of sides, and median soft/hard <= 0.4 where hard >= 10.
     The v3 rectangles (hard pastes) are measured the same way for comparison only.
   After the whole pipeline: in the flag-3 pictures slots 1-4 are 3 for every person, nowhere else is a 3.
3. Heavy share seen at epoch 0 and 9 (expected 0.5 and 0.8), and the shared epoch value read inside 2 DataLoader worker processes.
4. --v3-file: Occlude without bank and ramp must give the same pictures and labels as the old class for the same inputs and seeds.
5. Contact sheet <out dir>/occlude_bank_sheet.jpg: final training pictures (left, after ThermalLook and flips) and the same with marks
   (right): yellow = pasted office patch, cyan = visible point, red = hidden point, grey = flag 3 point, green = body box.
Result: <out dir>/occlude_bank_test.json, printed as well.
"""
import argparse
import copy
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_partial  # noqa: E402

FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
LINES = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


class ShareSet(torch.utils.data.Dataset):
    """For the worker check: every item is the heavy share an Occlude object sees inside the worker process."""

    def __init__(self, occ): self.occ = occ

    def __len__(self): return 8

    def __getitem__(self, i): return torch.tensor(self.occ.heavy_share())


def make_tiny_set(otp2, work, n):
    src_i, src_l = Path(otp2) / 'train' / 'images', Path(otp2) / 'train' / 'labels'
    imgs = sorted(src_i.glob('*.png'), key=lambda p: p.name)
    imgs = [p for p in imgs if (src_l / (p.stem + '.txt')).exists()]
    pick = imgs[:: max(1, len(imgs) // n)][:n]
    di, dl = Path(work) / 'data' / 'train' / 'images', Path(work) / 'data' / 'train' / 'labels'
    if (Path(work) / 'data').exists(): shutil.rmtree(Path(work) / 'data')
    di.mkdir(parents=True); dl.mkdir(parents=True)
    ign = set()
    for j, p in enumerate(pick):
        shutil.copy2(p, di / p.name)
        rows = []
        for line in (src_l / (p.stem + '.txt')).read_text().split('\n'):
            v = line.split()
            if not v: continue
            if j % 2 == 0:                  # eyes / ears -> flag 3, coordinates kept
                for s in range(1, 5): v[5 + 3 * s + 2] = '3'
            rows.append(' '.join(v))
        (dl / (p.stem + '.txt')).write_text('\n'.join(rows) + '\n')
        if j % 2 == 0: ign.add(p.name)
    y = Path(work) / 'data' / 'tiny.yaml'
    y.write_text('path: %s\ntrain: train/images\nval: train/images\nkpt_shape: [17, 3]\nflip_idx: %s\nnames:\n  0: person\n'
                 % (str(Path(work) / 'data').replace('\\', '/'), FLIP))
    return y, ign


def side_metrics(before, after, reg, w, h):
    """Per feathered side of one office region: (soft, natural, hard, feather, w, h). before / after = grey picture right before and right
    after THIS region was pasted (a monitor and the desk below it are measured one at a time)."""
    rx1, ry1, rx2, ry2 = reg['rect']; X1, Y1, X2, Y2 = reg['full']; a = reg['alpha']; P = reg['patch']; fe = reg['feather']
    b0, a0 = before.astype(np.float32), after.astype(np.float32)
    half = max(1, int(np.ceil(fe / 2)))
    out = []
    rows_mid = np.flatnonzero(a.max(1) >= 0.99) + ry1          # rows away from the top/bottom feather
    cols_mid = np.flatnonzero(a.max(0) >= 0.99) + rx1
    for side in ('l', 'r', 't', 'b'):
        if side == 'l' and (X1 <= 0 or rx1 < 2): continue
        if side == 'r' and (X2 >= w or rx2 > w - 2): continue
        if side == 't' and (Y1 <= 0 or ry1 < 2): continue
        if side == 'b' and (Y2 >= h or ry2 > h - 2): continue
        if side in 'lr':
            if len(rows_mid) < 3: continue
            cols = np.arange(rx1 - 2, rx1 + half + 1) if side == 'l' else np.arange(rx2 + 1, rx2 - half - 2, -1)
            cols = cols[(cols >= 0) & (cols < w)]
            pa, pb = a0[rows_mid][:, cols].mean(0), b0[rows_mid][:, cols].mean(0)
            pin = P[rows_mid - ry1, 0 if side == 'l' else -1].mean(); pout = b0[rows_mid, rx1 - 1 if side == 'l' else rx2].mean()
        else:
            if len(cols_mid) < 3: continue
            rows = np.arange(ry1 - 2, ry1 + half + 1) if side == 't' else np.arange(ry2 + 1, ry2 - half - 2, -1)
            rows = rows[(rows >= 0) & (rows < h)]
            pa, pb = a0[rows][:, cols_mid].mean(1), b0[rows][:, cols_mid].mean(1)
            pin = P[0 if side == 't' else -1, cols_mid - rx1].mean(); pout = b0[ry1 - 1 if side == 't' else ry2, cols_mid].mean()
        out.append((float(np.abs(np.diff(pa)).max()), float(np.abs(np.diff(pb)).max()), float(abs(pin - pout)), float(fe), rx2 - rx1, ry2 - ry1))
    return out


def rect_side_metrics(before, after, rect, w, h):
    """v3 rectangles (hard paste): the same profile step, for comparison."""
    rx1, ry1, rx2, ry2 = rect; b0, a0 = before[..., 0].astype(np.float32), after[..., 0].astype(np.float32); out = []
    for side in ('l', 'r', 't', 'b'):
        if side == 'l' and rx1 >= 2 and ry2 - ry1 > 6: pa, pb = a0[ry1:ry2, rx1 - 2:rx1 + 3].mean(0), b0[ry1:ry2, rx1 - 2:rx1 + 3].mean(0)
        elif side == 'r' and rx2 <= w - 2 and ry2 - ry1 > 6: pa, pb = a0[ry1:ry2, rx2 - 3:rx2 + 2].mean(0), b0[ry1:ry2, rx2 - 3:rx2 + 2].mean(0)
        elif side == 't' and ry1 >= 2 and rx2 - rx1 > 6: pa, pb = a0[ry1 - 2:ry1 + 3, rx1:rx2].mean(1), b0[ry1 - 2:ry1 + 3, rx1:rx2].mean(1)
        elif side == 'b' and ry2 <= h - 2 and rx2 - rx1 > 6: pa, pb = a0[ry2 - 3:ry2 + 2, rx1:rx2].mean(1), b0[ry2 - 3:ry2 + 2, rx1:rx2].mean(1)
        else: continue
        out.append((float(np.abs(np.diff(pa)).max()), float(np.abs(np.diff(pb)).max())))
    return out


class Spy:
    """Wraps the Occlude transform: checks every call and keeps what the contact sheet needs."""

    def __init__(self, occ):
        self.occ = occ; occ.debug = True
        self.st = dict(calls=0, office=0, box=0, none=0, kinds={}, heavy=[[], []], people_in=0, people_dropped=0, points_2_to_1=0,
                       boxes_changed=0, coords_changed=0, bad_flag_change=0, flag3_changed=0, visible_under_patch=0, hidden_not_under=0,
                       office_sides=[], box_sides=[], level_diff=[])
        self.phase = 0; self.records = []; self.inputs = []

    def __call__(self, labels):
        inst = labels['instances']
        cin = copy.deepcopy(inst); cin.convert_bbox(format='xyxy')
        kin = None if cin.keypoints is None else cin.keypoints.copy(); bin_ = cin.bboxes.copy(); before = labels['img'].copy()
        if len(self.inputs) < 120: self.inputs.append(copy.deepcopy(labels))
        out = self.occ(labels)
        st = self.st; st['calls'] += 1; last = self.occ.last
        self.records.append(None)
        if last is None: st['none'] += 1; return out
        st[last['src']] += 1; st['heavy'][self.phase].append(bool(last['heavy']))
        o = out['instances']; o.convert_bbox(format='xyxy'); keep = last['keep']
        kout, bout = o.keypoints, o.bboxes; img = out['img']; h, w = img.shape[:2]
        st['people_in'] += len(keep); st['people_dropped'] += int((~keep).sum())
        st['boxes_changed'] += int(not np.allclose(bin_[keep], bout, atol=1e-4))
        ki = kin[keep]
        st['coords_changed'] += int(not np.allclose(ki[..., :2], kout[..., :2], atol=1e-4))
        fi, fo = ki[..., 2], kout[..., 2]
        st['bad_flag_change'] += int(((fi != fo) & ~((fi == 2) & (fo == 1))).sum())
        st['flag3_changed'] += int(((fi == 3) != (fo == 3)).sum())
        st['points_2_to_1'] += int(((fi == 2) & (fo == 1)).sum())
        ins = ~((kout[..., 0] < 0) | (kout[..., 1] < 0) | (kout[..., 0] > w) | (kout[..., 1] > h))
        if last['src'] == 'office':
            for r in last['regions']: st['kinds'][r['kind']] = st['kinds'].get(r['kind'], 0) + 1
            cov = last['cover']; xi = np.clip(kout[..., 0], 0, w - 1).astype(int); yi = np.clip(kout[..., 1], 0, h - 1).astype(int)
            under = ins & (cov[yi, xi] >= self.occ.COVER)
            st['visible_under_patch'] += int(((fo == 2) & under).sum())
            st['hidden_not_under'] += int(((fi == 2) & (fo == 1) & ~under).sum())
            regs = last['regions']
            for q, r in enumerate(regs):
                aft = regs[q + 1]['bg'] if q + 1 < len(regs) else img[..., 0].astype(np.float32)
                st['office_sides'] += side_metrics(r['bg'], aft, r, w, h)
                # level: mean of the pasted patch vs mean of the ring of picture around it (before the paste)
                x1, y1, x2, y2 = r['rect']; b = max(3, int(0.15 * min(x2 - x1, y2 - y1)))
                e = r['bg'][max(0, y1 - b):y2 + b, max(0, x1 - b):x2 + b]; m = np.ones(e.shape, bool)
                m[y1 - max(0, y1 - b):y2 - max(0, y1 - b), x1 - max(0, x1 - b):x2 - max(0, x1 - b)] = False
                pad = np.abs(e - 114) < 0.5
                if (m & ~pad).sum() >= 0.3 * max(1, m.sum()): m &= ~pad          # grey padding is not background (same rule as Occlude)
                if m.sum() > 20: st['level_diff'].append(float(abs(r['patch'].mean() - e[m].mean())))
            self.records.append(dict(regions=[r['rect'] for r in last['regions']], kinds=[r['kind'] for r in last['regions']], heavy=last['heavy'], boxes=bout.copy(), w=w))
            self.records.pop(-2)
        else:
            x1, y1, x2, y2 = last['regions'][0]['rect']
            inside = ins & (kout[..., 0] >= x1) & (kout[..., 0] <= x2) & (kout[..., 1] >= y1) & (kout[..., 1] <= y2)
            st['visible_under_patch'] += int(((fo == 2) & inside).sum())
            st['hidden_not_under'] += int(((fi == 2) & (fo == 1) & ~inside).sum())
            st['box_sides'] += rect_side_metrics(before, img, (x1, y1, x2, y2), w, h)
        return out


def v3_equivalence(v3_file, inputs, bank_free_new):
    spec = importlib.util.spec_from_file_location('pose_partial_v3', v3_file); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    a, b = old.Occlude(p=1.0), bank_free_new(p=1.0)
    same = 0
    for j, lab in enumerate(inputs):
        la, lb = copy.deepcopy(lab), copy.deepcopy(lab)
        random.seed(1000 + j); ra = a(la); random.seed(1000 + j); rb = b(lb)
        ia, ib = ra['instances'], rb['instances']
        ok = (np.array_equal(ra['img'], rb['img']) and np.array_equal(ra['cls'], rb['cls']) and len(ia) == len(ib)
              and (len(ia) == 0 or (np.array_equal(ia.bboxes, ib.bboxes) and np.array_equal(ia.keypoints, ib.keypoints))))
        same += int(ok)
    return dict(samples=len(inputs), identical=same)


def draw(img, rec, boxes, kps):
    a = np.ascontiguousarray(img.copy())
    if rec:
        for (x1, y1, x2, y2) in rec['regions']: cv2.rectangle(a, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), (0, 255, 255), 1)
    for b, k in zip(boxes, kps):
        cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 1)
        f = k[:, 2].round().astype(int)
        for i, j in LINES:
            if f[i] in (1, 2) and f[j] in (1, 2): cv2.line(a, (int(k[i, 0]), int(k[i, 1])), (int(k[j, 0]), int(k[j, 1])), (150, 150, 150), 1)
        for (x, y, _), v in zip(k, f):
            if v > 0: cv2.circle(a, (int(x), int(y)), 4, {2: (255, 255, 0), 1: (0, 0, 255), 3: (128, 128, 128)}[v], -1)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('otp2'); ap.add_argument('bank'); ap.add_argument('template'); ap.add_argument('work'); ap.add_argument('out')
    ap.add_argument('n', nargs='?', type=int, default=200); ap.add_argument('--v3-file', default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    yaml_p, ign = make_tiny_set(a.otp2, a.work, a.n)
    pose_partial.install(); pose_partial.set_body_box(a.template)
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG
    np.random.seed(0); torch.manual_seed(0); random.seed(0)
    cfg = get_cfg(DEFAULT_CFG, overrides=dict(task='pose', mode='train', imgsz=640, mosaic=0.0, close_mosaic=0, scale=0.5, translate=0.2, degrees=5.0,
                                               fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, mixup=0.0, copy_paste=0.0, data=str(yaml_p)))
    data = check_det_dataset(str(yaml_p))
    ds = build_yolo_dataset(cfg, data['train'], batch=4, data=data, mode='train', rect=False, stride=32)
    pose_partial.add_partial_crop(ds, p=0.5, seed=0, occlude_p=1.0, thermal_look=True, office_bank=a.bank, bank_share=0.5, heavy_ramp=(0.5, 0.8, 9))
    t = ds.transforms.transforms
    k = next(j for j, x in enumerate(t) if isinstance(x, pose_partial.Occlude)); occ = t[k]; spy = Spy(occ); t[k] = spy
    n = len(ds)
    fin = dict(samples=n, people=0, ign_pictures=0, ign_slots_not3=0, flag3_elsewhere=0)
    tiles = []
    for i in range(n):
        spy.phase = 0 if i < n // 2 else 1; occ.set_epoch(0 if i < n // 2 else 9)
        s = ds[i]; img = s['img'].permute(1, 2, 0).numpy()[..., ::-1]; hh, ww = img.shape[:2]
        kp = s['keypoints'].numpy().copy(); kp[..., 0] *= ww; kp[..., 1] *= hh
        bx = s['bboxes'].numpy(); bxy = np.stack([(bx[:, 0] - bx[:, 2] / 2) * ww, (bx[:, 1] - bx[:, 3] / 2) * hh, (bx[:, 0] + bx[:, 2] / 2) * ww, (bx[:, 1] + bx[:, 3] / 2) * hh], 1) if len(bx) else np.zeros((0, 4))
        f = kp[..., 2].round().astype(int); fin['people'] += len(kp)
        if Path(s['im_file']).name in ign:
            fin['ign_pictures'] += 1; fin['ign_slots_not3'] += int((f[:, 1:5] != 3).sum()); fin['flag3_elsewhere'] += int((f[:, [0] + list(range(5, 17))] == 3).sum())
        else:
            fin['flag3_elsewhere'] += int((f == 3).sum())
        rec = spy.records[-1]
        if rec is not None and len(bxy) == len(rec['boxes']) and len(bxy):
            # the left-right flip comes after Occlude: mirror the pasted rectangles when the final boxes are the mirrored Occlude boxes
            rb = rec['boxes']; W = rec['w']
            if np.allclose(bxy[:, 0], W - rb[:, 2], atol=1.0) and not np.allclose(bxy[:, 0], rb[:, 0], atol=1.0):
                rec = dict(rec, regions=[(W - x2, y1, W - x1, y2) for (x1, y1, x2, y2) in rec['regions']])
        if rec is not None and len(tiles) < 12 and len(kp) and (len(tiles) < 6 or rec['heavy']):
            c = np.ascontiguousarray(img)
            tiles.append(np.hstack([cv2.resize(c, (320, 320)), cv2.resize(draw(c, rec, bxy, kp), (320, 320))]))
            cv2.putText(tiles[-1], '%s %s' % ('+'.join(rec['kinds']), 'heavy' if rec['heavy'] else 'light'), (326, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    st = spy.st
    so = np.array(st.pop('office_sides') or [(0, 0, 0, 0, 0, 0)]); sb = np.array(st.pop('box_sides') or [(0, 0)])
    ok_side = so[:, 0] <= so[:, 1] + 0.5 * so[:, 2] + 2
    excess = so[:, 0] - (so[:, 1] + 0.5 * so[:, 2] + 2)
    big = so[:, 2] >= 10
    ratio = so[big, 0] / so[big, 2] if big.any() else np.array([0.0])
    hv = st.pop('heavy'); ld = np.array(st.pop('level_diff') or [0.0])
    res = dict(occlude_calls=st, final=fin,
               blending=dict(office_sides=len(so), office_soft_median=round(float(np.median(so[:, 0])), 2), office_soft_p95=round(float(np.percentile(so[:, 0], 95)), 2),
                             office_natural_median=round(float(np.median(so[:, 1])), 2), office_hard_median=round(float(np.median(so[:, 2])), 2),
                             office_hard_p95=round(float(np.percentile(so[:, 2], 95)), 2), office_sides_ok_share=round(float(ok_side.mean()), 4),
                             office_soft_over_hard_median_where_hard_ge10=round(float(np.median(ratio)), 3), sides_with_hard_ge10=int(big.sum()),
                             v3_rect_sides=len(sb), v3_step_median=round(float(np.median(sb[:, 0])), 2), v3_step_p95=round(float(np.percentile(sb[:, 0], 95)), 2),
                             v3_natural_median=round(float(np.median(sb[:, 1])), 2),
                             office_worst_sides=[dict(soft=round(v[0], 1), natural=round(v[1], 1), hard=round(v[2], 1), feather=round(v[3], 1), region_wh=[int(v[4]), int(v[5])])
                                                 for v in so[np.argsort(-excess)[:5]].tolist()]),
               level=dict(patch_minus_ring_abs_median=round(float(np.median(ld)), 2), p90=round(float(np.percentile(ld, 90)), 2), regions=len(ld)),
               heavy_share=dict(epoch0=round(float(np.mean(hv[0])), 3) if hv[0] else None, epoch9=round(float(np.mean(hv[1])), 3) if hv[1] else None,
                                n0=len(hv[0]), n9=len(hv[1]), expected=[0.5, 0.8]))
    # shared epoch inside DataLoader workers (spawned processes on Windows)
    occ2 = pose_partial.Occlude(p=1.0, office_bank=a.bank, heavy_ramp=(0.5, 0.8, 9))
    dl = torch.utils.data.DataLoader(ShareSet(occ2), batch_size=4, num_workers=2, persistent_workers=True)
    seen = {}
    for e in (0, 9, 4):
        occ2.set_epoch(e); seen[e] = sorted(set(round(float(v), 4) for b in dl for v in b))
    res['worker_epoch'] = dict(seen={str(k): v for k, v in seen.items()}, expected={'0': [0.5], '9': [0.8], '4': [round(0.5 + 0.3 * 4 / 9, 4)]})
    if a.v3_file:
        res['v3_equivalence'] = v3_equivalence(a.v3_file, spy.inputs, pose_partial.Occlude)
    checks = dict(office_used=st['office'] > 0 and st['box'] > 0,
                  boxes_unchanged=st['boxes_changed'] == 0, coords_unchanged=st['coords_changed'] == 0, flags_only_2_to_1=st['bad_flag_change'] == 0,
                  flag3_unchanged_in_occlude=st['flag3_changed'] == 0, no_visible_point_under_patch=st['visible_under_patch'] == 0,
                  hidden_only_under_patch=st['hidden_not_under'] == 0, points_hidden=st['points_2_to_1'] > 0,
                  flag3_after_pipeline=fin['ign_pictures'] > 0 and fin['ign_slots_not3'] == 0 and fin['flag3_elsewhere'] == 0,
                  edges_blended=bool(ok_side.mean() >= 0.98 and float(np.median(ratio)) <= 0.4),
                  heavy_ramp=bool(hv[0] and hv[1] and abs(np.mean(hv[0]) - 0.5) <= 0.15 and abs(np.mean(hv[1]) - 0.8) <= 0.15),
                  worker_epoch=seen[0] == [0.5] and seen[9] == [0.8] and seen[4] == [round(0.5 + 0.3 * 4 / 9, 4)])
    if a.v3_file: checks['same_as_v3_without_bank'] = res['v3_equivalence']['identical'] == res['v3_equivalence']['samples'] > 0
    res['checks'] = checks; res['ALL_OK'] = all(checks.values())
    if tiles:
        while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / 'occlude_bank_sheet.jpg'), np.vstack([np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / 'occlude_bank_test.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
