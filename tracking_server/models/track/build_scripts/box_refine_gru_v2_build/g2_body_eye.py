"""gru_v2/scripts/g2_body_eye.py - by-eye sheets of the server mode 'body': tracker row (white), body mode output (green), detection (thin orange),
shoulder (cyan) / hip (magenta) keypoints >= 0.5 where the detector gives them.
  close: the three device-rate recordings with stored detections (v2/data/det), near and far detector streams;
  mid:   frame ranges of another device recording: our two pose detectors on the PC CPU for those frames only (dev_det's code), server ByteTrack;
  far:   public thermal val-B frames (small whole-person boxes), tracker rows from rows_valB.pkl, cropped and enlarged x4.
Usage: python g2_body_eye.py <tracking_server dir> <npz> <out dir> <raw16 dir> <rows_valB.pkl> <near.pt> <far.pt> <mid rec> <f0-f1,f0-f1> <close rec> [...]
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import g2_lib as L
import g2_device as GD
import g2_eyecheck as EY
import g2_body as GB
from g2_body import BR
from g2_lib import P

W_, H_ = 640, 480


def draw(g8, people, title, crop=None, S=2):
    """people: list of (trk, body, det, kp or None, label)."""
    if crop is not None:
        x0, y0, cw, ch = crop; S = W_ * 2 // cw
    else: x0 = y0 = 0; cw, ch = W_, H_
    img = cv2.cvtColor(cv2.resize(g8[y0:y0 + ch, x0:x0 + cw], (cw * S, ch * S), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)

    def rect(b, col, th):
        x1, y1, x2, y2 = b[0] - b[2] / 2 - x0, b[1] - b[3] / 2 - P.Y0 - y0, b[0] + b[2] / 2 - x0, b[1] + b[3] / 2 - P.Y0 - y0
        cv2.rectangle(img, (int(round(x1 * S)), int(round(y1 * S))), (int(round(x2 * S)), int(round(y2 * S))), col, th, cv2.LINE_AA)
    txt = []
    for trk, body, det, kp, lab in people:
        if det is not None: rect(det, (0, 140, 255), 1)
        rect(trk, (255, 255, 255), 2); rect(body, (0, 255, 0), 2)
        if kp is not None:
            for a, col in ((5, (255, 255, 0)), (6, (255, 255, 0)), (11, (255, 0, 255)), (12, (255, 0, 255))):
                if kp[a, 2] >= 0.5: cv2.circle(img, (int((kp[a, 0] - x0) * S), int((kp[a, 1] - P.Y0 - y0) * S)), 5, col, -1, cv2.LINE_AA)
        txt.append('%s trk %.0fx%.0f body %.0fx%.0f (h %.2f)' % (lab, trk[2], trk[3], body[2], body[3], body[3] / trk[3]))
    cv2.putText(img, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    for q, s_ in enumerate(txt[:7]): cv2.putText(img, s_, (8, 58 + 26 * q), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(img, (W_ * 2, H_ * 2))


def sheets(out, tag, panels):
    for k in range(0, len(panels), 4):
        p = panels[k:k + 4]
        while len(p) < 4: p.append(np.zeros_like(panels[0]))
        cv2.imwrite(str(out / ('%s_%02d.jpg' % (tag, k // 4 + 1))), cv2.resize(np.vstack([np.hstack(p[:2]), np.hstack(p[2:])]), (W_ * 2, H_ * 2), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])


def spread(rows, n):
    """rows [(height, frame)] -> n frames spread over the heights."""
    rows = sorted(rows); fs = []
    for q in np.linspace(0.04, 0.96, n):
        i = int(q * (len(rows) - 1)); k = 0
        while rows[min(i + k, len(rows) - 1)][1] in fs and k < 200: k += 1
        fs.append(rows[min(i + k, len(rows) - 1)][1])
    return fs


def close(model, raw, recs, stats):
    panels = []
    for ri, rec in enumerate(recs):
        St, _ = GD.streams(rec, ri); want = {}; sel = {}
        for st in ('near', 'far'):
            inp = St[st]; o, kk, _ = GB.server(model, 'body', inp['ids'], inp['W'], inp['n'], inp['times'], inp['fps'])
            rows = [(e['trk'][j, 3], int(e['f'][j])) for tid, e in inp['ids'].items() for j in range(len(e['f'])) if kk[tid][j] >= 5]
            fs = spread(rows, 2 if ri == 0 else 1); sel[st] = (fs, inp, o)
            for f in fs: want[f] = 1
            for tid, e in inp['ids'].items(): stats['close'].append((e['trk'], o[tid], e['kp']))
        g8 = EY.frames8(raw / rec, want)
        for st in ('near', 'far'):
            fs, inp, o = sel[st]
            for f in fs:
                ppl = [(e['trk'][j], o[tid][j], e['z'][j], e['kp'][j], 'id%d' % tid) for tid, e in inp['ids'].items() for j in np.flatnonzero(e['f'] == f)]
                panels.append(draw(g8[f], ppl, 'close  %s  %s detector  frame %d' % (rec[8:], st, f)))
    return panels


def mid(model, raw, rec, ranges, wn, wf, stats):
    import torch
    sys.path.insert(0, str(L.ROOT / 'scripts')); sys.path.insert(0, str(L.ROOT / 'v2' / 'scripts'))
    import det_run as D
    import gmc_run as G
    import track_run as TR
    from ultralytics import YOLO
    torch.set_num_threads(4); nets = {k: D.Heads(YOLO(w).model.float().eval()).eval() for k, w in (('near', wn), ('far', wf))}
    meta = json.loads((raw / rec / 'frames.json').read_text(encoding='utf-8')); ts = np.asarray(meta['times_s'], np.float64); want = {f: 1 for a, b in ranges for f in range(a, b)}
    g8 = EY.frames8(raw / rec, want); panels = []
    for a, b in ranges:
        n = b - a; gmc = G.GMC(2); gmc.fp = dict(gmc.fp, maxCorners=300, minDistance=5); Wp = np.zeros((n, 2, 3)); dets = {k: [] for k in nets}
        for i in range(n): Wp[i], _ = gmc.estimate(G.to_canvas(g8[a + i]))
        for i0 in range(0, n, 4):
            x = torch.from_numpy(np.stack([g8[a + i] for i in range(i0, min(i0 + 4, n))])).float()[:, None].expand(-1, 3, -1, -1).contiguous()
            for k, wrap in nets.items():
                with torch.inference_mode(): bo, cl, kq = wrap(x)
                bo, cl, kq = bo.float().numpy(), cl.float().numpy(), kq.float().numpy()
                for j in range(len(x)):
                    bx, sc, kp, _ = D.post(bo[j], cl[j], kq[j]); bx = bx.astype(np.float64).copy(); kp = kp.astype(np.float64).copy(); bx[:, [1, 3]] += P.Y0; kp[:, :, 1] += P.Y0; dets[k].append((bx, sc.astype(np.float64), kp))
        for k in nets:
            fd = [(np.c_[dets[k][i][0], dets[k][i][1], np.zeros(len(dets[k][i][1]))], np.arange(len(dets[k][i][1])), np.zeros(len(dets[k][i][1]), np.int64)) for i in range(n)]
            rows = np.array(TR.run(n, Wp, fd), np.float64).reshape(-1, 8); ids = defaultdict(list)
            for r in rows: ids[int(r[1])].append(r)
            E = {}
            for i, rr in ids.items():
                rr = np.array(rr); rr = rr[np.argsort(rr[:, 0])]; f = rr[:, 0].astype(np.int64); di = rr[:, 7].astype(np.int64)
                E[i] = dict(f=f, z=P.xyxy2cxcywh(np.array([dets[k][ff][0][d] for ff, d in zip(f, di)])), kp=np.array([dets[k][ff][2][d] for ff, d in zip(f, di)]), trk=P.xyxy2cxcywh(rr[:, 2:6]), src=np.full(len(f), L.DETS.index(k)))
            if not E: continue
            o, kk, _ = GB.server(model, 'body', E, Wp, n, ts[a:b], (n - 1) / (ts[b - 1] - ts[a]))
            rows = [(e['trk'][j, 3], int(e['f'][j])) for tid, e in E.items() for j in range(len(e['f'])) if kk[tid][j] >= 4]
            for tid, e in E.items(): stats['mid'].append((e['trk'], o[tid], e['kp']))
            for f in spread(rows, 2):
                ppl = [(e['trk'][j], o[tid][j], e['z'][j], e['kp'][j], 'id%d' % tid) for tid, e in E.items() for j in np.flatnonzero(e['f'] == f)]
                panels.append(draw(g8[a + f], ppl, 'mid  %s  %s detector  frame %d' % (rec[8:], k, a + f)))
        print('MID range %d-%d done' % (a, b), flush=True)
    return panels


def far(model, rows_path, stats):
    d = pickle.loads(open(rows_path, 'rb').read()); recs, Wd, nd = d['recs'], d['W'], d['n']; groups = defaultdict(list); panels = []
    for r in recs:
        if r['ds'] == 'TMOT' and r['stream'] in ('near', 'far'): groups[(r['name'], r['stream'])].append(r)
    cand = []
    for key, rr in sorted(groups.items()):
        ids = {r['id']: dict(f=r['e']['f'], z=r['e']['z'], kp=r['e']['kp'], trk=r['o']['tracker'], src=r['e']['src']) for r in rr}
        o, kk, _ = GB.server(model, 'body', ids, Wd[key[0]], nd[key[0]], None, rr[0]['fps']); cnt = defaultdict(list)
        for tid, e in ids.items():
            stats['far'].append((e['trk'], o[tid], e['kp']))
            for j in range(len(e['f'])):
                if kk[tid][j] >= 4 and e['trk'][j, 3] < 110: cnt[int(e['f'][j])].append((tid, j))
        if cnt:
            f = max(cnt, key=lambda q: (len(cnt[q]) >= 2, -abs(len(cnt[q]) - 3))); cand.append((key, f, ids, o, cnt[f]))
    for key, f, ids, o, its in cand[:8]:
        sd = Path(P.PATHS['tmot_val']) / key[0]; dd = sd / 'img1' if (sd / 'img1').is_dir() else sd / 'images'; files = sorted(p for p in dd.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
        g = cv2.imread(str(files[f]), cv2.IMREAD_GRAYSCALE)
        if g.shape != (H_, W_): g = cv2.resize(g, (W_, H_))
        cx = float(np.mean([ids[t]['trk'][j, 0] for t, j in its])); cy = float(np.mean([ids[t]['trk'][j, 1] for t, j in its])) - P.Y0
        x0 = int(np.clip(cx - 160, 0, W_ - 320)); y0 = int(np.clip(cy - 120, 0, H_ - 240))
        ppl = [(e['trk'][j], o[tid][j], e['z'][j], e['kp'][j], 'id%d' % tid) for tid, e in ids.items() for j in np.flatnonzero(e['f'] == f)
               if x0 <= e['trk'][j, 0] <= x0 + 320 and y0 <= e['trk'][j, 1] - P.Y0 <= y0 + 240]
        panels.append(draw(g, ppl, 'far  public thermal %s  %s detector  frame %d (x4)' % (key[0], key[1], f), crop=(x0, y0, 320, 240)))
    return panels


def main():
    npz, out, raw, rows_path, wn, wf, mrec, mr = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]; crecs = sys.argv[10:]
    out.mkdir(parents=True, exist_ok=True); z = np.load(npz, allow_pickle=False); model = BR.G2Model({k: z[k] for k in z.files}); stats = defaultdict(list)
    sheets(out, 'close', close(model, raw, crecs, stats)); print('CLOSE done', flush=True)
    sheets(out, 'far', far(model, rows_path, stats)); print('FAR done', flush=True)
    sheets(out, 'mid', mid(model, raw, mrec, [tuple(int(v) for v in s.split('-')) for s in mr.split(',')], wn, wf, stats)); res = {}
    for k, v in stats.items():   # where the detector's own hip / shoulder keypoints sit inside the tracker row and inside the body box, by tracker-row height
        trk = np.concatenate([a for a, _, _ in v]); bod = np.concatenate([b for _, b, _ in v]); kp = np.concatenate([c for _, _, c in v]); res[k] = {}
        (ht, st), (hb, sb) = EY.hip_frac(trk, kp), EY.hip_frac(bod, kp)
        for a, b in GB.HB:
            m = (trk[:, 3] >= a) & (trk[:, 3] < b)
            if m.sum() < 20: continue
            med = lambda x: round(float(np.nanmedian(x[m])), 3) if np.isfinite(x[m]).any() else None
            res[k]['%d-%d' % (a, b)] = dict(n=int(m.sum()), n_hip=int(np.isfinite(ht[m]).sum()), h_over_tracker=round(float(np.median(bod[m, 3] / trk[m, 3])), 3), w_over_tracker=round(float(np.median(bod[m, 2] / trk[m, 2])), 3),
                                            hip_in_tracker=med(ht), hip_in_body=med(hb), shoulder_in_tracker=med(st), shoulder_in_body=med(sb))
            print('STAT %-5s trk h %-9s n %5d (hips %5d)  body h / trk h %.2f  w %.2f  hip in tracker %s  in body %s  shoulder in tracker %s  in body %s' % (k, '%d-%d' % (a, b), m.sum(), res[k]['%d-%d' % (a, b)]['n_hip'],
                  res[k]['%d-%d' % (a, b)]['h_over_tracker'], res[k]['%d-%d' % (a, b)]['w_over_tracker'], med(ht), med(hb), med(st), med(sb)), flush=True)
    (out / 'body_eye.json').write_text(json.dumps(dict(weights=Path(npz).name, stats=res), indent=1)); print('BODY_EYE_DONE', flush=True)


if __name__ == '__main__':
    main()
