"""Independent check of indoor_score_v2.py (2026-11-15 review).

dump:  python indoor_score_v2_check.py dump <weights> <boxes.json> <recording dir> ...
       Every 3rd image (name order), imgsz 640, conf 0.05, 8-bit images only; saves xyxy, conf, cls per image.
check: python indoor_score_v2_check.py check <boxes.json> <indoor_score_v2 result.json> <recordings root>
       Re-scores from the dump with separate code (augmenting-path matching, no scipy) and compares per frame
       (match count, false boxes, old-rule count) with the result file. Then explains old-found / new-missed people
       and shows how much the 50%-of-visible-area condition costs per size bin (alternative: also allow a pair when
       >= 70% of the predicted box lies inside the visible box).
"""
import sys, json, collections
from pathlib import Path
import numpy as np

SIZE = [('<80', 0, 80), ('80-160', 80, 160), ('160-320', 160, 320), ('>=320', 320, 1e9)]


def dump(w, out, recs):
    import cv2
    from ultralytics import YOLO
    m = YOLO(w); res = {}
    for rd in map(Path, recs):
        for f in sorted((rd / 'images').glob('*.png'))[::3]:
            im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            assert im.dtype == np.uint8, (f, im.dtype)
            if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            b = m.predict(im, imgsz=640, conf=0.05, verbose=False)[0].boxes
            res[f'{rd.name}/{f.stem}'] = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()].tolist() if b is not None else []
        print(rd.name, len(res), flush=True)
    Path(out).write_text(json.dumps(res))


def max_match(adj):
    """Maximum bipartite matching size (Kuhn); adj[i] = allowed box indices of person i. Returns {person: box}."""
    mt = {}
    def tr(i, seen):
        for j in adj[i]:
            if j in seen: continue
            seen.add(j)
            if j not in mt or tr(mt[j], seen): mt[j] = i; return True
        return False
    for i in range(len(adj)): tr(i, set())
    return {i: j for j, i in mt.items()}


def pair_stats(vb, b):
    """vb visible xyxy, b predicted xyxy -> centre inside, share of visible area inside, share of predicted area inside visible, iou."""
    x1, y1, x2, y2 = vb; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; a = (x2 - x1) * (y2 - y1)
    bx1, by1, bx2, by2 = b; ab = (bx2 - bx1) * (by2 - by1)
    ins = bx1 <= cx <= bx2 and by1 <= cy <= by2
    it = max(0, min(x2, bx2) - max(x1, bx1)) * max(0, min(y2, by2) - max(y1, by1))
    return ins, it / a, it / max(ab, 1e-9), it / (a + ab - it)


def check(boxes, result, root):
    B = json.loads(Path(boxes).read_text())
    J = json.loads(Path(result).read_text(encoding='utf-8'))
    jp = collections.defaultdict(list)
    for p in J['people']: jp[(p['rec'], p['frame'])].append(p)
    jf = {(r, f['frame']): f for r, fs in J['frames'].items() for f in fs}
    tot = collections.Counter(); mism = []; why = collections.Counter(); bins = collections.defaultdict(lambda: [0, 0, 0])
    for rec in J['recordings']:
        d = Path(root) / rec
        gt = np.loadtxt(d / 'tracks_edited_mot.txt', delimiter=',', ndmin=2)
        for key in sorted(k for k in B if k.split('/')[0] == rec):
            fn = int(key.split('/')[1]); g = gt[(gt[:, 0] == fn) & (gt[:, 7] != 1)]
            b = [x[:4] for x in B[key] if x[4] >= 0.25 and x[5] == 0]
            V = [(x, y, x + w, y + h) for x, y, w, h in g[:, 2:6]]
            st = [[pair_stats(v, bb) for bb in b] for v in V]
            adj = [[j for j, s in enumerate(r) if s[0] and s[1] >= 0.5] for r in st]
            adj_alt = [[j for j, s in enumerate(r) if s[0] and (s[1] >= 0.5 or s[2] >= 0.7)] for r in st]
            old = [any(s[0] and s[3] >= 0.2 for s in r) for r in st]
            m, m_alt = max_match(adj), max_match(adj_alt)
            ps = jp[(rec, fn)]
            tot['frames'] += 1; tot['people'] += len(V); tot['found'] += len(m); tot['false'] += len(b) - len(m); tot['old'] += sum(old)
            if (len(ps) != len(V) or sum(p['found'] for p in ps) != len(m) or jf[(rec, fn)]['false'] != len(b) - len(m)
                    or sum(p['old_found'] for p in ps) != sum(old)):
                mism.append(key)
            # result people are written in label-row order, so index i is the same person
            for i, p in enumerate(ps):
                if p['old_found'] and not p['found']:
                    why['a pair >= 50% exists, lost to one-to-one' if adj[i] else 'centre inside but < 50% of visible area inside'] += 1
            for i, (x1, y1, x2, y2) in enumerate(V):
                s = next(n for n, lo, hi in SIZE if lo <= y2 - y1 < hi)
                for k in (s, 'all'): c = bins[k]; c[0] += 1; c[1] += i in m; c[2] += i in m_alt
    lines = [f'independent re-score: {dict(tot)}, found share {round(tot["found"] / tot["people"], 3)}; result file pooled: '
             f'{J["pooled"]["found"]}/{J["pooled"]["people"]} found, {J["pooled"]["false_boxes"]} false',
             f'frames that differ from the result file (match count, false boxes, old-rule count): {len(mism)} {mism[:10]}',
             f'old rule found but new rule missed: {dict(why)}',
             'found by size bin, rule as specified (50% of visible area) vs alternative (or >= 70% of predicted box inside visible box); '
             'bin counts can differ by 1-2 from the result file because ties between people are broken differently:']
    for k in [n for n, _, _ in SIZE] + ['all']:
        n, a, c = bins[k]
        lines.append(f'  {k}: n {n}  specified {a} ({a / n:.3f})  alternative {c} ({c / n:.3f})' if n else f'  {k}: n 0')
    print('\n'.join(lines))


if __name__ == '__main__':
    if sys.argv[1] == 'dump': dump(sys.argv[2], sys.argv[3], sys.argv[4:])
    else: check(*sys.argv[2:5])
