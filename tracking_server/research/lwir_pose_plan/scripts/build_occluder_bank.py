"""Occluder bank from our people-free office recordings (PLAN_training_settings_20261115.md, stage 2 + new occlusion synthesis).

Cuts desk-, monitor- and chair-like patches out of empty-office thermal frames and stores them for pose_partial.Occlude(office_bank=...).

Input: 16-bit raw recordings (frames.raw.gz + frames.json, 640x480 '<u2'), e.g. LIG-Data/close_range/raw16/empty/a and b
(= device LWIR_labels/empty_office_350 and empty_office2_449; the device folders only hold contact sheets of the same frames).
8-bit look: per-frame NORM_MINMAX, the same mapping that made our recording PNGs (checked on rec_inside_20261110_165240: mean error
0.1 grey level against raw16/MEASURE_20261110_165240). Level and contrast are matched again when a patch is pasted.

Patch kinds (w/h aspect, size, where in the frame):
  desk    wide-low   aspect 1.6-4.0, width 25-75 % of the frame, centre in the lower part (y 55-95 %)
  monitor square-ish aspect 0.8-1.6, width 15-45 %,             centre in the middle (y 30-80 %)
  chair   tall       aspect 0.3-0.7, height 30-85 %,            centre middle/lower (y 40-85 %)
Score = mean gradient magnitude (Sobel after a Gaussian blur of sigma 2, clipped at 40 so one hot-lamp edge does not dominate). Per frame and kind the best of
60 random windows is a candidate; a candidate is dropped if it is flat (std < 8) or more than 35 % saturated (<= 5 or >= 250). Per kind the
candidates are taken best first, above the kind's 40th score percentile, skipping near-duplicates (thumbnail correlation > 0.92 with a patch
already taken), up to --max-per-kind. Long side capped at 320 px.

Output .npz (loads with allow_pickle=False): pix (all patches, uint8, flattened), off (start of each patch in pix), hw (h, w), kind (0/1/2),
kinds (names), src (recording index, frame, x, y, w, h), score, recs (recording names). Plus <out>.json (summary) and <out>_sheet.jpg.

Usage: python build_occluder_bank.py <out.npz> <raw recording dir> [<raw recording dir> ...] [--max-per-kind 400] [--seed 0] [--sheet <jpg>]
"""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

KINDS = ('desk', 'monitor', 'chair')
SPEC = dict(desk=dict(asp=(1.6, 4.0), wfrac=(0.25, 0.75), yc=(0.55, 0.95)),
            monitor=dict(asp=(0.8, 1.6), wfrac=(0.15, 0.45), yc=(0.30, 0.80)),
            chair=dict(asp=(0.3, 0.7), hfrac=(0.30, 0.85), yc=(0.40, 0.85)))
TRIES, MIN_STD, MAX_SAT, PCT_KEEP, DUP_CORR, MAX_SIDE, GRAD_CAP, BLUR = 60, 8.0, 0.35, 40, 0.92, 320, 40.0, 2.0


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load_raw(d):
    meta = json.loads((d / 'frames.json').read_text())
    w, h = int(meta['width']), int(meta['height'])
    raw = np.frombuffer(gzip.open(d / 'frames.raw.gz').read(), dtype='<u2').reshape(-1, h, w)
    return raw, meta


def to8(frame16):
    # same mapping as our recording PNGs (per-frame min-max)
    return cv2.normalize(frame16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def integral(a):
    return cv2.integral(a.astype(np.float64))


def box_sum(ii, x, y, w, h):
    return ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]


def window(rng, kind, W, H):
    s = SPEC[kind]; asp = rng.uniform(*s['asp'])
    if kind == 'chair':
        h = int(rng.uniform(*s['hfrac']) * H); w = int(h * asp)
    else:
        w = int(rng.uniform(*s['wfrac']) * W); h = int(w / asp)
    w, h = max(24, min(w, W - 1)), max(24, min(h, H - 1))
    yc = rng.uniform(*s['yc']) * H
    y = int(np.clip(yc - h / 2, 0, H - h)); x = int(rng.integers(0, W - w + 1))
    return x, y, w, h


def thumb(p):
    t = cv2.resize(p, (24, 12), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean(); n = np.linalg.norm(t)
    return t / n if n > 1e-6 else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('recs', nargs='+')
    ap.add_argument('--max-per-kind', type=int, default=400); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sheet', default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); sheet = Path(a.sheet) if a.sheet else out.with_name(out.stem + '_sheet.jpg')
    cand = {k: [] for k in KINDS}; rec_info = []
    for ri, rd in enumerate(map(Path, a.recs)):
        raw, meta = load_raw(rd)
        n_flat = 0
        for fi in range(len(raw)):
            g8 = to8(raw[fi]); H, W = g8.shape
            if float(g8.std()) < 6: n_flat += 1; continue          # shutter / flat frame
            b = cv2.GaussianBlur(g8.astype(np.float32), (0, 0), BLUR)          # blur first: sensor noise of stretched flat frames is not structure
            gm = np.minimum(np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)), GRAD_CAP)
            ig, i1, i2 = integral(gm), integral(g8), integral(g8.astype(np.float64) ** 2)
            isat = integral(((g8 <= 5) | (g8 >= 250)).astype(np.uint8))
            for kind in KINDS:
                best = None
                for _ in range(TRIES):
                    x, y, w, h = window(rng, kind, W, H); n = float(w * h)
                    mu = box_sum(i1, x, y, w, h) / n; sd = max(0.0, box_sum(i2, x, y, w, h) / n - mu * mu) ** 0.5
                    if sd < MIN_STD or box_sum(isat, x, y, w, h) / n > MAX_SAT: continue
                    s = box_sum(ig, x, y, w, h) / n
                    if best is None or s > best[0]: best = (s, x, y, w, h)
                if best is not None:
                    s, x, y, w, h = best
                    cand[kind].append((float(s), ri, fi, x, y, w, h, g8[y:y + h, x:x + w].copy()))
        rec_info.append(dict(dir=str(rd).replace('\\', '/'), frames=len(raw), flat_frames_skipped=n_flat, source=meta.get('source'),
                             sha256_raw=sha(rd / 'frames.raw.gz')))
        print('read', rd, len(raw), 'frames', flush=True)

    pix, off, hw, kd, src, sc, per_kind = [], [], [], [], [], [], {}
    pos = 0
    for k_i, kind in enumerate(KINDS):
        c = sorted(cand[kind], key=lambda t: -t[0])
        if not c: per_kind[kind] = dict(candidates=0, kept=0); continue
        thr = float(np.percentile([t[0] for t in c], PCT_KEEP))
        taken, thumbs, dup = [], [], 0
        for t in c:
            if t[0] < thr or len(taken) >= a.max_per_kind: break
            p = t[7]
            if max(p.shape) > MAX_SIDE:
                f = MAX_SIDE / max(p.shape); p = cv2.resize(p, (max(8, round(p.shape[1] * f)), max(8, round(p.shape[0] * f))), interpolation=cv2.INTER_AREA)
            th = thumb(p)
            if thumbs and float(np.max(np.stack(thumbs) @ th)) > DUP_CORR: dup += 1; continue
            thumbs.append(th); taken.append((t, p))
        for t, p in taken:
            pix.append(p.ravel()); off.append(pos); pos += p.size; hw.append(p.shape); kd.append(k_i)
            src.append(t[1:7]); sc.append(t[0])
        sizes = np.array([p.shape for _, p in taken]) if taken else np.zeros((0, 2))
        per_kind[kind] = dict(candidates=len(c), score_threshold=round(thr, 2), duplicates_skipped=dup, kept=len(taken),
                              score_median=round(float(np.median([t[0] for t, _ in taken])), 2) if taken else None,
                              h_median=int(np.median(sizes[:, 0])) if taken else None, w_median=int(np.median(sizes[:, 1])) if taken else None,
                              from_recording=[int(sum(1 for t, _ in taken if t[1] == r)) for r in range(len(rec_info))])
        print(kind, per_kind[kind], flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pix=np.concatenate(pix).astype(np.uint8), off=np.array(off, np.int64), hw=np.array(hw, np.int32),
                        kind=np.array(kd, np.int8), kinds=np.array(KINDS), src=np.array(src, np.int32), score=np.array(sc, np.float32),
                        recs=np.array([r['dir'] for r in rec_info]))

    # contact sheet: one row band per kind, best patches first
    rows = []
    for k_i, kind in enumerate(KINDS):
        idx = [i for i in range(len(kd)) if kd[i] == k_i]
        pick = idx[:: max(1, len(idx) // 24)][:24]
        tiles = []
        for i in pick:
            p = pix[i].reshape(hw[i]); s = 110 / p.shape[0]
            tiles.append(cv2.resize(p, (max(8, min(330, round(p.shape[1] * s))), 110), interpolation=cv2.INTER_AREA))
        line, lines, wsum = [], [], 0
        for t in tiles:
            if wsum + t.shape[1] + 4 > 1600 and line: lines.append(line); line, wsum = [], 0
            line.append(t); wsum += t.shape[1] + 4
        if line: lines.append(line)
        for j, ln in enumerate(lines):
            canvas = np.full((114, 1600), 30, np.uint8); x = 2
            for t in ln: canvas[2:112, x:x + t.shape[1]] = t; x += t.shape[1] + 4
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if j == 0: cv2.putText(canvas, kind, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            rows.append(canvas)
    if rows: cv2.imwrite(str(sheet), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

    summary = dict(made=time.strftime('%Y-%m-%d %H:%M:%S'), script='close_range_20261113/tools/build_occluder_bank.py', seed=a.seed,
                   eight_bit='per-frame cv2.NORM_MINMAX (same as our recording PNGs)', recordings=rec_info, patches=len(kd), kinds=list(KINDS),
                   per_kind=per_kind, settings=dict(spec=SPEC, tries=TRIES, min_std=MIN_STD, max_saturated=MAX_SAT, keep_above_percentile=PCT_KEEP,
                                                    duplicate_corr=DUP_CORR, max_side=MAX_SIDE, grad_cap=GRAD_CAP, blur_sigma=BLUR, max_per_kind=a.max_per_kind),
                   npz=str(out).replace('\\', '/'), npz_bytes=out.stat().st_size, npz_sha256=sha(out), sheet=str(sheet).replace('\\', '/'))
    out.with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
