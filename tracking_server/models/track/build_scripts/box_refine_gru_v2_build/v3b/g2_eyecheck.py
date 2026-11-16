"""gru_v2/scripts/g2_eyecheck.py - by-eye check of the drawn boxes on the device-rate recordings (REPORT.md doubt 3).
Per recording and detector stream (near, far): the server ByteTrack rows (g2_device.streams), the GRU v2 engine + size filter, and picture sheets:
white = tracker row, green = GRU v2 output (drawn / gimbal box), thin orange = the matched detection, dots = shoulder (cyan) / hip (magenta) keypoints >= 0.5.
Also numbers: where the hip keypoints sit inside each box ((hip y - box top) / box height) per detector and detection height bin.
Usage: python g2_eyecheck.py <out dir> <npz weights> <raw16 dir> <rec> [...]
"""
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import g2_lib as L
import g2_device as GD

W_, H_ = 640, 480
BINS = ((0, 60), (60, 100), (100, 160), (160, 240), (240, 999))


class To8:
    def __init__(self): self.lo = None; self.hi = None

    def __call__(self, raw):
        hist = np.bincount((raw >> 4).ravel(), minlength=4096).cumsum(); tot = hist[-1]
        lo = float(np.searchsorted(hist, tot * 0.01)) * 16; hi = float(np.searchsorted(hist, tot * 0.99)) * 16 + 16
        if hi - lo < 200: hi = lo + 200
        if self.lo is None: self.lo, self.hi = lo, hi
        else: self.lo += 0.1 * (lo - self.lo); self.hi += 0.1 * (hi - self.hi)
        return np.clip((raw.astype(np.float32) - self.lo) * (255.0 / (self.hi - self.lo)), 0, 255).astype(np.uint8)


def frames8(rec_dir, want):
    meta = json.loads((rec_dir / 'frames.json').read_text(encoding='utf-8')); n = int(meta['frames']); nb = W_ * H_ * 2; to8 = To8(); out = {}
    with gzip.open(rec_dir / 'frames.raw.gz', 'rb') as f:
        for i in range(n):
            b = f.read(nb); g = to8(np.frombuffer(b, '<u2').reshape(H_, W_))
            if i in want: out[i] = g.copy()
            if i >= max(want): break
    return out


def rect(img, b, col, th, s):
    """b cxcywh on the 640x512 canvas -> picture (rows - 16), scaled by s."""
    x1, y1, x2, y2 = b[0] - b[2] / 2, b[1] - b[3] / 2 - L.P.Y0, b[0] + b[2] / 2, b[1] + b[3] / 2 - L.P.Y0
    cv2.rectangle(img, (int(round(x1 * s)), int(round(y1 * s))), (int(round(x2 * s)), int(round(y2 * s))), col, th, cv2.LINE_AA)


def hip_frac(box, kp):
    """(mean hip y - top) / h and (mean shoulder y - top) / h, nan when not both visible."""
    top = box[:, 1] - box[:, 3] / 2; o = []
    for a, b in ((11, 12), (5, 6)):
        ok = (kp[:, a, 2] >= 0.5) & (kp[:, b, 2] >= 0.5); y = (kp[:, a, 1] + kp[:, b, 1]) / 2
        o.append(np.where(ok, (y - top) / box[:, 3], np.nan))
    return o


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); net = L.load_net(sys.argv[2]); raw = Path(sys.argv[3]); recs = sys.argv[4:]
    z = np.load(sys.argv[2]); size = tuple(float(v) for v in z['size_euro']) if 'size_euro' in z.files else (5.0, 0.0, 1.0)
    if 'lut_tab' in z.files:   # the exported table (so a re-exported conversion is the one drawn)
        tab, lhc, arc = z['lut_tab'].astype(np.float64), z['lut_lh_cen'].astype(np.float64), z['lut_ar_cen'].astype(np.float64)
        L.conv_delta.__defaults__ = (tab, lhc, arc)
    S = 2; panels = []; stats = {k: {'%d-%d' % b: dict(n=0, r=[], hip_det=[], hip_out=[], hip_trk=[], sh_out=[], sh_det=[]) for b in BINS} for k in ('near', 'far')}; picks = []
    for ri, rec in enumerate(recs):
        St, info = GD.streams(rec, ri); want = {}; sel = {}
        for st in ('near', 'far'):
            inp = St[st]; eo = L.run_rows(inp['ids'], inp['W'], inp['fps'], inp['n'], net, size, inp['times']); rows = []
            for tid, e in inp['ids'].items():
                o = eo[tid]['out_f']; hd, sd = hip_frac(e['z'], e['kp']); ho, so = hip_frac(o, e['kp']); ht, _ = hip_frac(e['trk'], e['kp'])
                for j in range(len(e['f'])):
                    for b in BINS:
                        if b[0] <= e['z'][j, 3] < b[1]:
                            d = stats[st]['%d-%d' % b]; d['n'] += 1; d['r'].append(o[j, 3] / e['trk'][j, 3]); d['hip_det'].append(hd[j]); d['hip_out'].append(ho[j]); d['hip_trk'].append(ht[j]); d['sh_out'].append(so[j]); d['sh_det'].append(sd[j])
                    if eo[tid]['k'][j] >= 5: rows.append((e['trk'][j, 3], int(e['f'][j]), tid))
            rows.sort(); n_pick = 6 if ri == 0 else 3; fs = []
            for q in np.linspace(0.03, 0.97, n_pick):   # spread over the person sizes of this recording
                c = rows[int(q * (len(rows) - 1))]; k = 0
                while c[1] in fs and k < 50: k += 1; c = rows[min(int(q * (len(rows) - 1)) + k, len(rows) - 1)]
                fs.append(c[1])
            sel[st] = (fs, inp, eo)
            for f in fs: want[f] = 1
        g8 = frames8(raw / rec, want)
        for st in ('near', 'far'):
            fs, inp, eo = sel[st]
            for f in fs:
                img = cv2.cvtColor(cv2.resize(g8[f], (W_ * S, H_ * S), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR); txt = []
                for tid, e in inp['ids'].items():
                    j = np.flatnonzero(e['f'] == f)
                    if not len(j): continue
                    j = int(j[0]); o = eo[tid]['out_f'][j]; rect(img, e['z'][j], (0, 140, 255), 1, S); rect(img, e['trk'][j], (255, 255, 255), 2, S); rect(img, o, (0, 255, 0), 2, S)
                    for a, col in ((5, (255, 255, 0)), (6, (255, 255, 0)), (11, (255, 0, 255)), (12, (255, 0, 255)), (0, (0, 0, 255))):
                        if e['kp'][j, a, 2] >= 0.5: cv2.circle(img, (int(e['kp'][j, a, 0] * S), int((e['kp'][j, a, 1] - L.P.Y0) * S)), 5, col, -1, cv2.LINE_AA)
                    txt.append('id%d trk h %.0f det h %.0f v3 h %.0f (%.2f)' % (tid, e['trk'][j, 3], e['z'][j, 3], o[3], o[3] / e['trk'][j, 3]))
                    picks.append(dict(rec=rec, stream=st, frame=f, id=int(tid), trk=[round(float(v), 1) for v in e['trk'][j]], det=[round(float(v), 1) for v in e['z'][j]], v2=[round(float(v), 1) for v in o]))
                cv2.putText(img, '%s  %s detector  frame %d' % (rec[8:], st, f), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                for q, s_ in enumerate(txt[:6]): cv2.putText(img, s_, (8, 58 + 26 * q), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                panels.append(img)
    for k in range(0, len(panels), 4):
        p = panels[k:k + 4]
        while len(p) < 4: p.append(np.zeros_like(panels[0]))
        sheet = np.vstack([np.hstack(p[:2]), np.hstack(p[2:])]); cv2.imwrite(str(out / ('sheet_%02d.jpg' % (k // 4 + 1))), cv2.resize(sheet, (W_ * 2, H_ * 2), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / ('sheet_%02d_full.jpg' % (k // 4 + 1))), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    med = lambda v: round(float(np.nanmedian(v)), 3) if len(v) and np.isfinite(v).any() else None
    res = {k: {b: dict(n=d['n'], h_over_tracker=med(d['r']), hip_in_det=med(d['hip_det']), hip_in_tracker=med(d['hip_trk']), hip_in_v2=med(d['hip_out']), shoulder_in_det=med(d['sh_det']), shoulder_in_v2=med(d['sh_out']))
                   for b, d in v.items() if d['n']} for k, v in stats.items()}
    (out / 'eyecheck.json').write_text(json.dumps(dict(weights=Path(sys.argv[2]).name, stats=res, picks=picks), indent=1))
    for k, v in res.items():
        for b, d in v.items(): print('STAT %-4s det h %-8s n %5d  v2 h / tracker h %s  hip in det %s  hip in tracker %s  hip in v2 %s  shoulder in det %s in v2 %s' % (k, b, d['n'], d['h_over_tracker'], d['hip_in_det'], d['hip_in_tracker'], d['hip_in_v2'], d['shoulder_in_det'], d['shoulder_in_v2']))
    print('EYECHECK_DONE %d panels' % len(panels), flush=True)


if __name__ == '__main__':
    main()
