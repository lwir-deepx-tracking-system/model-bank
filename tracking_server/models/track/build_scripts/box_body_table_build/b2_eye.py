"""body_v2/scripts/b2_eye.py - by-eye sheets of body mode v2: tracker row (white), body v2 (green, server Refiner, committed table file), the matched
detection (thin orange), shoulders (cyan) / hips (magenta) where the pose detector gives them; every green box is numbered.
  close : the three device-rate recordings (our two pose detectors, server ByteTrack), frames spread over the tracker-row heights (rows >= 5 of a track)
  mid   : frames of device recording MEASURE_20261113_192922_1 (both pose detectors re-run on the PC CPU for those frames, server GMC + ByteTrack)
  pub   : public val-B: TMOT thermal far people (crops x4) and DanceTrack mid / close people
  ofar  : our own camera's outdoor far frames (stage3a far_frames, model F = the 'general' pose detector's predictions, score >= 0.25): single frames,
          every detection converted with the pose table WITHOUT the time filter (crops x2-x4)
  plain : the device recordings with the plain YOLO26x FLIR detector (b2_yolo.py rec), body v2 with the whole-person constant
Usage: python b2_eye.py <out dir> <table npz> <raw16 dir> <near.pt> <far.pt> <stage3a dir> <parts comma list>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import b2_lib as B
import b2_eval as E
from b2_lib import BR, L, P

W_, H_ = 640, 480
REC_CLOSE = ('MEASURE_20261109_140236_2', 'MEASURE_20261110_165240', 'MEASURE_20261110_170437')
REC_MID, MID_RANGES = 'MEASURE_20261113_192922_1', ((640, 668), (1040, 1068))
LOG = []


def table(p):
    z = np.load(p, allow_pickle=False); return BR.BodyTable({k: z[k] for k in z.files})


def draw(g8, people, title, crop=None, y0c=P.Y0):
    """people: list of (trk or None, body, det or None, kp or None, label); boxes cxcywh in canvas coords (picture y + y0c)."""
    if crop is not None: x0, y0, cw, ch = crop; S = W_ * 2 // cw
    else: x0 = y0 = 0; cw, ch = W_, H_; S = 2
    img = cv2.cvtColor(cv2.resize(g8[y0:y0 + ch, x0:x0 + cw], (cw * S, ch * S), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)

    def rect(b, col, th):
        x1, y1, x2, y2 = b[0] - b[2] / 2 - x0, b[1] - b[3] / 2 - y0c - y0, b[0] + b[2] / 2 - x0, b[1] + b[3] / 2 - y0c - y0
        cv2.rectangle(img, (int(round(x1 * S)), int(round(y1 * S))), (int(round(x2 * S)), int(round(y2 * S))), col, th, cv2.LINE_AA)
        return int(round(x1 * S)), int(round(y1 * S))
    txt = []
    for n_, (trk, body, det, kp, lab) in enumerate(people, 1):
        if det is not None: rect(det, (0, 140, 255), 1)
        if trk is not None: rect(trk, (255, 255, 255), 2)
        px, py = rect(body, (0, 255, 0), 2); cv2.putText(img, str(n_), (px + 3, py + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if kp is not None:
            for a, col in ((5, (255, 255, 0)), (6, (255, 255, 0)), (11, (255, 0, 255)), (12, (255, 0, 255))):
                if kp[a, 2] >= 0.5: cv2.circle(img, (int((kp[a, 0] - x0) * S), int((kp[a, 1] - y0c - y0) * S)), 4, col, -1, cv2.LINE_AA)
        ref = trk if trk is not None else det
        txt.append('%d %s ref %.0fx%.0f body %.0fx%.0f (h %.2f)' % (n_, lab, ref[2], ref[3], body[2], body[3], body[3] / ref[3]))
    cv2.putText(img, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    for q, s_ in enumerate(txt[:8]): cv2.putText(img, s_, (8, 52 + 22 * q), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    LOG.append(dict(title=title, people=txt))
    return cv2.resize(img, (W_ * 2, H_ * 2))


def sheets(out, tag, panels):
    for k in range(0, len(panels), 4):
        p = panels[k:k + 4]
        while len(p) < 4: p.append(np.zeros_like(panels[0]))
        cv2.imwrite(str(out / ('%s_%02d.jpg' % (tag, k // 4 + 1))), cv2.resize(np.vstack([np.hstack(p[:2]), np.hstack(p[2:])]), (W_ * 2, H_ * 2), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])


def spread(rows, n):
    rows = sorted(rows); fs = []
    for q in np.linspace(0.06, 0.94, n):
        i = int(q * (len(rows) - 1)); k = 0
        while rows[min(i + k, len(rows) - 1)][1] in fs and k < 300: k += 1
        fs.append(rows[min(i + k, len(rows) - 1)][1])
    return fs


def run_body(T, ids, W, n, times, fps, keys, family='pose'):
    gp_ids = {tid: dict(f=e['f'], z=e['z'], trk=e['trk'], src=e['src']) for tid, e in ids.items()}
    o, kk, _, _ = E.run(gp_ids, E.items_of(gp_ids), W, n, keys, times, fps, 'body', T, None, 0.5, family)
    return o, kk


def frames8(rec_dir, want):
    import gzip
    import dev_det as DD
    meta = json.loads((rec_dir / 'frames.json').read_text(encoding='utf-8')); n = int(meta['frames']); nb = W_ * H_ * 2; to8 = DD.To8(); out = {}
    with gzip.open(rec_dir / 'frames.raw.gz', 'rb') as f:
        for i in range(n):
            b = f.read(nb); g = to8(np.frombuffer(b, '<u2').reshape(H_, W_))
            if i in want: out[i] = g.copy()
            if i >= max(want): break
    return out


def close(T, raw):
    import g2_device as GD
    panels = []
    for ri, rec in enumerate(REC_CLOSE):
        S, _ = GD.streams(rec, ri); want = {}; sel = {}
        for st in ('near', 'far'):
            inp = S[st]; keys = [st] * inp['n']; o, kk = run_body(T, inp['ids'], inp['W'], inp['n'], inp['times'], inp['fps'], keys)
            rows = [(e['trk'][j, 3], int(e['f'][j])) for tid, e in inp['ids'].items() for j in range(len(e['f'])) if kk[tid][j] >= 5]
            fs = spread(rows, 3 if ri == 0 else 2); sel[st] = (fs, inp, o)
            for f in fs: want[f] = 1
        g8 = frames8(raw / rec, want)
        for st in ('near', 'far'):
            fs, inp, o = sel[st]
            for f in fs:
                ppl = [(e['trk'][j], o[tid][j], e['z'][j], e['kp'][j], 'id%d' % tid) for tid, e in inp['ids'].items() for j in np.flatnonzero(e['f'] == f)]
                panels.append(draw(g8[f], ppl, 'close %s %s det frame %d' % (rec[8:], st, f)))
    return panels


def plain(T, raw):
    panels = []
    for ri, rec in enumerate(REC_CLOSE):
        inp = E.plain_stream(rec)
        if inp is None: continue
        keys = ['yolo26x_flir'] * inp['n']; o, kk = run_body(T, inp['ids'], inp['W'], inp['n'], inp['times'], inp['fps'], keys, 'person')
        rows = [(e['trk'][j, 3], int(e['f'][j])) for tid, e in inp['ids'].items() for j in range(len(e['f'])) if kk[tid][j] >= 5]
        if not rows: continue
        fs = spread(rows, 2); g8 = frames8(raw / rec, {f: 1 for f in fs})
        for f in fs:
            ppl = [(e['trk'][j], o[tid][j], e['z'][j], None, 'id%d' % tid) for tid, e in inp['ids'].items() for j in np.flatnonzero(e['f'] == f)]
            panels.append(draw(g8[f], ppl, 'plain YOLO26x %s frame %d (person constant)' % (rec[8:], f)))
    return panels


def mid(T, raw, wn, wf):
    import torch
    sys.path.insert(0, str(B.ROOT / 'scripts')); sys.path.insert(0, str(B.ROOT / 'v2' / 'scripts'))
    import det_run as D
    import gmc_run as G
    import track_run as TR
    from ultralytics import YOLO
    torch.set_num_threads(4); nets = {k: D.Heads(YOLO(w).model.float().eval()).eval() for k, w in (('near', wn), ('far', wf))}
    meta = json.loads((raw / REC_MID / 'frames.json').read_text(encoding='utf-8')); ts = np.asarray(meta['times_s'], np.float64); want = {f: 1 for a, b in MID_RANGES for f in range(a, b)}
    g8 = frames8(raw / REC_MID, want); panels = []
    for a, b in MID_RANGES:
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
            Ed = {}
            for i, rr in ids.items():
                rr = np.array(rr); rr = rr[np.argsort(rr[:, 0])]; f = rr[:, 0].astype(np.int64); di = rr[:, 7].astype(np.int64)
                Ed[i] = dict(f=f, z=P.xyxy2cxcywh(np.array([dets[k][ff][0][d] for ff, d in zip(f, di)])), kp=np.array([dets[k][ff][2][d] for ff, d in zip(f, di)]), trk=P.xyxy2cxcywh(rr[:, 2:6]), src=np.full(len(f), 0))
            if not Ed: continue
            o, kk = run_body(T, Ed, Wp, n, ts[a:b], (n - 1) / (ts[b - 1] - ts[a]), [k] * n)
            rows = [(e['trk'][j, 3], int(e['f'][j])) for tid, e in Ed.items() for j in range(len(e['f'])) if kk[tid][j] >= 4]
            if not rows: continue
            for f in spread(rows, 2):
                ppl = [(e['trk'][j], o[tid][j], e['z'][j], e['kp'][j], 'id%d' % tid) for tid, e in Ed.items() for j in np.flatnonzero(e['f'] == f)]
                panels.append(draw(g8[a + f], ppl, 'mid %s %s det frame %d' % (REC_MID[8:], k, a + f)))
    return panels


def pub(T):
    """public val-B (rows of gru_v2/v3b): TMOT far people (crops x4), DanceTrack mid / close people"""
    panels_far = []; panels_dt = []; gr = E.groups('valB')
    for gp in gr:
        if gp['stream'] not in ('near', 'far'): continue
        o, kk = run_body(T, gp['ids'], gp['W'], gp['n'], None, gp['fps'], gp['keys'])
        cnt = defaultdict(list)
        for tid, e in gp['ids'].items():
            for j in range(len(e['f'])):
                if kk[tid][j] >= 4: cnt[int(e['f'][j])].append((tid, j))
        if not cnt: continue
        if gp['ds'] == 'TMOT':
            small = {f: v for f, v in cnt.items() if sum(gp['ids'][t]['trk'][j, 3] < 110 for t, j in v) >= 2}
            if not small or len(panels_far) >= 8: continue
            f = max(small, key=lambda q: -abs(len(small[q]) - 3)); its = small[f]
            sd = Path(P.PATHS['tmot_val']) / gp['name']; dd = sd / 'img1' if (sd / 'img1').is_dir() else sd / 'images'; files = sorted(p for p in dd.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
            g = cv2.imread(str(files[f]), cv2.IMREAD_GRAYSCALE)
            if g.shape != (H_, W_): g = cv2.resize(g, (W_, H_))
            cx = float(np.mean([gp['ids'][t]['trk'][j, 0] for t, j in its])); cy = float(np.mean([gp['ids'][t]['trk'][j, 1] for t, j in its])) - P.Y0
            x0 = int(np.clip(cx - 160, 0, W_ - 320)); y0 = int(np.clip(cy - 120, 0, H_ - 240))
            ppl = [(e['trk'][j], o[tid][j], e['z'][j], None, 'id%d' % tid) for tid, e in gp['ids'].items() for j in np.flatnonzero(e['f'] == f) if x0 <= e['trk'][j, 0] <= x0 + 320 and y0 <= e['trk'][j, 1] - P.Y0 <= y0 + 240]
            panels_far.append(draw(g, ppl, 'pub TMOT %s %s det frame %d (x4)' % (gp['name'], gp['stream'], f), crop=(x0, y0, 320, 240)))
        elif len(panels_dt) < 4 and gp['stream'] == ('far' if len(panels_dt) % 2 else 'near'):
            f = max(cnt, key=lambda q: len(cnt[q])); sub = 'val' if gp['name'] in ('dancetrack0087', 'dancetrack0096', 'dancetrack0098', 'dancetrack0099') else 'train'
            files = sorted((Path(P.PATHS['dance']) / sub / gp['name'] / 'img1').iterdir()); g = cv2.imread(str(files[f]), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, (640, 360)); gg = np.full((H_, W_), 114, np.uint8); gg[60:420] = g   # the letterboxed grey copy the detectors saw
            ppl = [(e['trk'][j], o[tid][j], e['z'][j], None, 'id%d' % tid) for tid, e in gp['ids'].items() for j in np.flatnonzero(e['f'] == f)]
            panels_dt.append(draw(gg, ppl[:8], 'pub DanceTrack %s %s det frame %d' % (gp['name'], gp['stream'], f)))
    return panels_far, panels_dt


def ofar(T, s3a):
    """our camera's outdoor frames (stage3a eval8/far, model F predictions): each detection converted by the pose table, no time filter"""
    z = np.load(Path(s3a) / 'preds' / 'eval_F.npz', allow_pickle=True); fn = np.array([str(f).replace('\\', '/') for f in z['fname']]); sc = z['score']; bx = z['box']
    files = sorted({f for f in fn[(sc >= 0.25)] if f.startswith('far/')}); rng = np.random.default_rng(0); panels = []; stats = []
    # frames with >= 2 people, spread over recordings and person sizes
    byf = defaultdict(list)
    for i in np.flatnonzero(sc >= 0.25):
        if fn[i].startswith('far/'): byf[fn[i]].append(i)
    cand = [(f, v) for f, v in byf.items() if len(v) >= 2]; seqs = sorted({f.split('/')[1] for f, _ in cand}); pick = []
    for s in seqs:
        cs = [(f, v) for f, v in cand if f.split('/')[1] == s]
        if cs: pick.append(cs[int(rng.integers(len(cs)))])
    for f, v in pick[:12]:
        g = cv2.imread(str(Path(s3a) / 'eval8' / f), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        Z = P.xyxy2cxcywh(bx[v].astype(np.float64)); d = T.conv_delta(Z, 'pose')
        Cb = np.stack([Z[:, 0] + d[:, 0] * Z[:, 3], Z[:, 1] + d[:, 1] * Z[:, 3], Z[:, 2] * np.exp(d[:, 2]), Z[:, 3] * np.exp(d[:, 3])], 1)
        for q in range(len(v)): stats.append(dict(frame=f, h=float(Z[q, 3]), ar=float(Z[q, 2] / Z[q, 3]), ratio=float(Cb[q, 3] / Z[q, 3]), hip=float(B.hip_pos(Z[q:q + 1], z['kpt'][v[q]][None].astype(np.float64))[0])))
        cx, cy = float(np.median(Z[:, 0])), float(np.median(Z[:, 1])); hm = float(np.median(Z[:, 3])); cw = 160 if hm < 40 else 320; ch = cw * 3 // 4
        x0 = int(np.clip(cx - cw / 2, 0, W_ - cw)); y0 = int(np.clip(cy - ch / 2, 0, H_ - ch))
        ppl = [(None, Cb[q], Z[q], z['kpt'][v[q]].astype(np.float64), 'det %.2f' % sc[v[q]]) for q in range(len(v)) if x0 <= Z[q, 0] <= x0 + cw and y0 <= Z[q, 1] <= y0 + ch]
        if ppl: panels.append(draw(g, ppl, 'our far %s (F, no filter, x%d)' % (f[4:], W_ * 2 // cw // 2), crop=(x0, y0, cw, ch), y0c=0.0))
    return panels, stats


def main():
    out, tp, raw, wn, wf, s3a, parts = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7].split(',')
    out.mkdir(parents=True, exist_ok=True); T = table(tp); res = {}
    sys.path.insert(0, str(B.ROOT / 'v2' / 'scripts'))
    if 'close' in parts: sheets(out, 'close', close(T, raw)); print('CLOSE done', flush=True)
    if 'mid' in parts: sheets(out, 'mid', mid(T, raw, wn, wf)); print('MID done', flush=True)
    if 'pub' in parts:
        pf, pd = pub(T); sheets(out, 'pub_tmot_far', pf); sheets(out, 'pub_dancetrack', pd); print('PUB done', flush=True)
    if 'ofar' in parts:
        po, st = ofar(T, s3a); sheets(out, 'our_far', po); res['our_far_stats'] = st; print('OFAR done (%d detections drawn)' % len(st), flush=True)
    if 'plain' in parts:
        pp = plain(T, raw)
        if pp: sheets(out, 'plain', pp)
        print('PLAIN done (%d panels)' % len(pp), flush=True)
    (out / ('eye_%s.json' % '_'.join(parts))).write_text(B.jdump(dict(table=Path(tp).name, panels=LOG, **res)), encoding='utf-8'); print('EYE_DONE', flush=True)


if __name__ == '__main__':
    main()
