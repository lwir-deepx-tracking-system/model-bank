"""11-20 (NOT RECOMMENDED: measured worse, HANDOFF 11-20) number guards after the tracker (any engine: Python / C++ BoT-SORT): page switches 'pre_capture' and 'merge_split', default off.

User: two people overlap almost completely, the detector keeps one box, and the hidden person's number slides onto the front person.
This layer works on the tracker's output rows (like feat_guard works after the tracker) and only renames rows: it keeps a one-to-one map
tracker id -> shown number (a label). Labels are tracker ids (identity until the layer acts) or fresh numbers from FRESH up. Off = not one
line of it runs (rows unchanged).

merge_split: one detection covers two recently shown numbers a, b (each one's predicted box lies >= MG_CONTAIN inside it, or their
  predicted boxes overlap IoU > MG_IOU) and no other row is near them -> a MERGED GROUP. The group's box is shown with ONE number (the one
  whose memory the box matches by >= MARGIN; else the one whose predicted box overlaps it most = continuity); the other is held: not lost,
  not given to any other box (a row with a group number away from the group gets a fresh number), the judge may not hand it to a birth and
  stores no feature for either number (OnlineJudge.ms_hold). When two rows are near the group again and overlap IoU < SPLIT_APART (split),
  the two numbers go to the two boxes by memory (cosine, the better pairing by >= MARGIN), else by motion = the tracker's own pairing (its
  tracks carry the group numbers; centre distances only when neither row shows one), and that is counted.
  A group older than HOLD_S seconds, or without a near row for GONE_N frames, is dissolved (the held number takes the normal lost path).
pre_capture: two shown numbers approach (IoU >= AP_IOU or centre distance < AP_DIST x the wider box): their crops are embedded now (the
  BoT-SORT batch feature when it has one, else an extra crop, at most PC_MAX extra per frame) and stored while no other box covers more than
  PC_COVER of them; at pair IoU >= FREEZE_IOU (or a group) the memory is frozen until the pair is apart again. When a pair that reached
  FREEZE_IOU is apart again (IoU < AP_IOU and centre distance >= AP_DIST), the two numbers are swapped if their current crops fit the frozen
  memories crosswise better by >= MARGIN (once per episode). Without pre_capture, merge_split uses only free batch features of boxes covered
  < CLEAN_COVER by any other detection.
Constants fixed before the measurement (id_steal_20261120/merge/RULE_BEFORE_RUN.txt); MARGIN was the one varied value.
"""
import time

import numpy as np

from ..common.shared_values import PERSON_CLS
from ..reid.cos_map import cmap as _cmap

SW = dict(pre_capture=False, merge_split=False)
AP_IOU, AP_DIST, PC_COVER, CLEAN_COVER, FREEZE_IOU, MEM_N, PC_MAX = 0.10, 1.0, 0.35, 0.15, 0.30, 5, 2
MG_CONTAIN, MG_IOU, NEAR_IOU, SEEN_GAP, HOLD_S, GONE_N, DUP_IOU, SPLIT_APART = 0.6, 0.5, 0.3, 3, 3.0, 10, 0.6, 0.3
MARGIN = 0.05   # picked by the rule among 0.05 / 0.10 / 0.15 (fewest lasting steals with both switches)
THIRD = 'fresh'   # a group number on a box away from its group: 'fresh' = a fresh number (as asked); 'release' = the group is dissolved (exploratory)
MIN_H = 32.0
FRESH = 10 ** 7


def _iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    i = w * h; return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def _inside(a, b):
    """share of box a's area inside box b"""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def _ctr(b): return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def _unit(v):
    v = np.asarray(v, np.float64); return v / max(float(np.linalg.norm(v)), 1e-12)


def _shift(b, v, n):
    return (b[0] + v[0] * n, b[1] + v[1] * n, b[2] + v[0] * n, b[3] + v[1] * n)


def active():
    return bool(SW['pre_capture'] or SW['merge_split'])


class MergeSplit:
    def __init__(self):
        self.stats = dict(frames=0, groups=0, split_app=0, split_motion=0, split_nomem=0, hold_expired=0, gone=0, third_box=0, group_relabel=0,
                          front_app=0, front_iou=0, pc_swaps=0, pc_checks=0, pc_nomem=0, extra_crops=0, max_extra=0, ms_total=0.0, ms_max=0.0)
        self.extra_log = []   # extra crops per frame (harness)
        self.ms_log = []      # layer CPU ms per frame (harness)
        self.log = None       # harness: a list -> (frame, event, details) of every action
        self.reset()

    def reset(self):
        self.M, self.inv = {}, {}          # tracker id -> label, label -> tracker id
        self.L = {}                        # label -> dict(box, f, v, mem [feats], frozen (unit mean) or None)
        self.groups = []                   # dict(a shown, b held, f0, box, v, f, gone, pb held box, vb, fb)
        self.pairs = {}                    # (a, b) a < b -> dict(peak) of approaching pairs (pre_capture)
        self.fresh = FRESH; self.f = -1

    def hold_labels(self):
        return {x for g in self.groups for x in (g['a'], g['b'])}

    def groups_now(self):
        return [dict(shown=int(g['a']), held=int(g['b']), frames=int(self.f - g['f0'])) for g in self.groups]

    # ------------------------------------------------------------ label map (a permutation of tracker ids plus fresh labels)
    def _label(self, raw):
        if raw not in self.M: self.M[raw] = raw; self.inv[raw] = raw
        return self.M[raw]

    def _relabel(self, raw, x):
        y = self.M[raw]
        if y == x: return
        r2 = self.inv.get(x)
        self.M[raw] = x; self.inv[x] = raw
        if r2 is not None and r2 != raw: self.M[r2] = y; self.inv[y] = r2
        else: self.inv.pop(y, None)

    def _fresh(self, raw):
        y = self.M[raw]; self.fresh += 1; x = self.fresh
        self.M[raw] = x; self.inv[x] = raw; self.inv.pop(y, None)
        return x

    def _mem(self, lab):
        s = self.L.get(lab)
        if s is None: return None
        if s.get('frozen') is not None: return s['frozen']
        return _unit(np.mean(s['mem'], 0)) if s['mem'] else None

    def _pred(self, lab):
        s = self.L[lab]; return _shift(s['box'], s['v'], self.f - s['f'])

    # ------------------------------------------------------------ one frame
    def step(self, rows, dets, feats, embed, fps=30.0, gate=None):
        """rows: tracker rows [x1, y1, x2, y2, id, conf, cls, k, ...]; dets: [n, >=5] this frame's detections (k indexes them);
        feats: {k: L2 feature} already embedded this frame (BoT-SORT batch); embed(list of k) -> {k: feature} (synchronous extra crops) or None.
        Returns the rows (same order, same boxes) with person ids renamed."""
        t0 = time.perf_counter(); self.f += 1; f = self.f; st = self.stats; st['frames'] += 1
        feats = dict(feats or {}); extra = [0]
        P = [i for i, r in enumerate(rows) if int(r[6]) == PERSON_CLS]
        raw = {i: int(rows[i][4]) for i in P}
        for i in P: self._label(raw[i])
        nd = len(dets) if dets is not None else 0
        dbox = {}
        for i in P:
            k = int(rows[i][7]); dbox[i] = tuple(float(v) for v in (dets[k, :4] if 0 <= k < nd else rows[i][:4]))
        pdet = [tuple(float(v) for v in dets[j, :4]) for j in range(nd) if (dets.shape[1] < 6 or int(dets[j, 5]) == PERSON_CLS) and float(dets[j, 4]) >= 0.1]

        def cover(i):   # largest share of row i's detection box covered by another person detection
            b = dbox[i]; return max((_inside(b, q) for q in pdet if q != b), default=0.0)

        def get_feats(want):   # want: rows -> {row: feature}; extra crops within PC_MAX per frame
            out, need = {}, []
            for i in want:
                k = int(rows[i][7])
                if k in feats: out[i] = _unit(feats[k])
                elif embed is not None and 0 <= k < nd and dbox[i][3] - dbox[i][1] >= MIN_H and k not in need: need.append(k)
            room = PC_MAX - extra[0]
            if need and room > 0:
                need = need[:room]; got = embed(need) or {}; extra[0] += len(need)
                for k, v in got.items(): feats[k] = _unit(v)
                for i in want:
                    k = int(rows[i][7])
                    if k in feats: out[i] = _unit(feats[k])
            return out

        def lab(i): return self.M[raw[i]]
        held = self.hold_labels()
        # ---- merged groups: continue, split, expire
        if SW['merge_split']:
            hold_n = int(round(HOLD_S * fps))
            for g in list(self.groups):
                gp = _shift(g['box'], g['v'], f - g['f'])
                ok = lambda i, g_=g: lab(i) in (g_['a'], g_['b']) or self.L.get(lab(i)) is None or self.L[lab(i)]['f0'] >= g_['f0']   # not a third person passing
                near = sorted([i for i in P if ok(i) and (_iou(dbox[i], gp) >= 0.2 or (gp[0] <= _ctr(dbox[i])[0] <= gp[2] and gp[1] <= _ctr(dbox[i])[1] <= gp[3]))],
                              key=lambda i: -_iou(dbox[i], gp))
                if f - g['f0'] > hold_n:
                    self.groups.remove(g); st['hold_expired'] += 1; self._ev('hold_expired', g['a'], g['b']); continue
                if not near:
                    g['gone'] += 1
                    if g['gone'] > GONE_N: self.groups.remove(g); st['gone'] += 1; self._ev('gone', g['a'], g['b'])
                    continue
                g['gone'] = 0
                r1 = near[0]; r2 = next((i for i in near[1:] if _iou(dbox[i], dbox[r1]) < DUP_IOU), None)
                if r2 is None:   # still one box: it shows the group's number
                    if lab(r1) != g['a']: self._relabel(raw[r1], g['a']); st['group_relabel'] += 1
                    self._move(g, dbox[r1]); continue
                if _iou(dbox[r1], dbox[r2]) >= SPLIT_APART:   # two boxes, still overlapping: their crops are mixed, decide when apart
                    u1, u2 = dbox[r1], dbox[r2]; self._move(g, (min(u1[0], u2[0]), min(u1[1], u2[1]), max(u1[2], u2[2]), max(u1[3], u2[3]))); continue   # the group box spans both
                # split: two boxes near the group again
                a, b = g['a'], g['b']; ma, mb = self._mem(a), self._mem(b)
                fe = get_feats([r1, r2]) if (ma is not None and mb is not None) else {}
                swap = None; mode = 'app'
                if ma is not None and mb is not None and r1 in fe and r2 in fe:
                    keep = _cmap(float(fe[r1] @ ma)) + _cmap(float(fe[r2] @ mb)); crs = _cmap(float(fe[r1] @ mb)) + _cmap(float(fe[r2] @ ma))   # 11-20: per-model cosine map (reid/cos_map.py)
                    if abs(keep - crs) / 2.0 >= MARGIN: swap = crs > keep; st['split_app'] += 1
                if swap is None:
                    mode = 'split_motion' if (ma is not None and mb is not None) else 'split_nomem'; st[mode] += 1
                    # motion = the tracker's own pairing (its position memory carried the hidden track); only when neither row
                    # shows a group number, the centre distances to the group's / the held number's predicted places
                    l1, l2 = lab(r1), lab(r2)
                    if l1 in (a, b) or l2 in (a, b): swap = (l1 == b) or (l1 not in (a, b) and l2 == a)
                    else:
                        pa = gp; pb = _shift(g['pb'], g['vb'], f - g['fb'])
                        c1, c2 = _ctr(dbox[r1]), _ctr(dbox[r2]); qa, qb = _ctr(pa), _ctr(pb)
                        dk = np.hypot(c1[0] - qa[0], c1[1] - qa[1]) + np.hypot(c2[0] - qb[0], c2[1] - qb[1])
                        dx = np.hypot(c1[0] - qb[0], c1[1] - qb[1]) + np.hypot(c2[0] - qa[0], c2[1] - qa[1])
                        swap = dx < dk
                la, lb = (b, a) if swap else (a, b); self._ev('split', mode, a, b, la, lb, raw[r1], raw[r2], round(dbox[r1][0]), round(dbox[r2][0]))
                self._relabel(raw[r1], la); self._relabel(raw[r2], lb)
                self.groups.remove(g)
            held = self.hold_labels()
            # a group number on a box away from its group: a fresh number (neither number may be given to a third box)
            gnear = set()
            for g in self.groups:
                gp = g['box']
                gnear |= {i for i in P if _iou(dbox[i], gp) >= 0.2}
            for i in P:
                if lab(i) in held and i not in gnear:
                    self._ev('third', lab(i), raw[i]); st['third_box'] += 1
                    if THIRD == 'fresh': self._fresh(raw[i])
                    else:   # 'release' (exploratory, after the measurement): the tracker found the held person elsewhere -> the group was wrong
                        x = lab(i); self.groups = [g for g in self.groups if x not in (g['a'], g['b'])]
            held = self.hold_labels()
            # ---- new groups: one row covers two recently shown numbers and no other row is near them
            shown = [x for x, s in self.L.items() if f - s['f'] <= SEEN_GAP and x not in held]
            if len(shown) >= 2:
                pr = {x: self._pred(x) for x in shown}
                used = set()
                for xi in range(len(shown)):
                    for yi in range(xi + 1, len(shown)):
                        a, b = shown[xi], shown[yi]
                        if a in used or b in used: continue
                        pa, pb = pr[a], pr[b]
                        nr = [i for i in P if _iou(dbox[i], pa) >= NEAR_IOU or _iou(dbox[i], pb) >= NEAR_IOU]
                        if len(nr) != 1: continue
                        i = nr[0]; D = dbox[i]
                        if lab(i) not in (a, b): continue
                        if not ((_inside(pa, D) >= MG_CONTAIN and _inside(pb, D) >= MG_CONTAIN) or _iou(pa, pb) > MG_IOU): continue
                        # which number the one box shows: its memory by >= MARGIN, else the predicted box overlapping it most
                        ma, mb = self._mem(a), self._mem(b); front = None
                        if ma is not None and mb is not None:
                            fe = get_feats([i])
                            if i in fe:
                                d = (_cmap(float(fe[i] @ ma)) - _cmap(float(fe[i] @ mb)))
                                if abs(d) >= MARGIN: front = a if d > 0 else b; st['front_app'] += 1
                        if front is None:
                            ia, ib = _iou(D, pa), _iou(D, pb)
                            front = a if ia > ib else (b if ib > ia else lab(i)); st['front_iou'] += 1
                        back = b if front == a else a
                        if lab(i) != front: self._relabel(raw[i], front); st['group_relabel'] += 1
                        sb = self.L[back]
                        self.groups.append(dict(a=front, b=back, f0=f, box=D, v=self.L[front]['v'], f=f, gone=0, pb=sb['box'], vb=sb['v'], fb=sb['f']))
                        st['groups'] += 1; used |= {a, b}; self._ev('group', front, back, raw[i], ma is not None, mb is not None)
                        for x in (a, b):   # the memory as it was before the merge
                            s = self.L[x]
                            if s.get('frozen') is None: s['frozen'] = _unit(np.mean(s['mem'], 0)) if s['mem'] else None
            held = self.hold_labels()
        # ---- pre_capture: approaching pairs, crops now, freeze, swap check when apart again
        if SW['pre_capture']:
            cur = {lab(i): i for i in P if lab(i) not in held}
            ls = sorted(cur)
            close = set()
            for xi in range(len(ls)):
                for yi in range(xi + 1, len(ls)):
                    a, b = ls[xi], ls[yi]; A, B = dbox[cur[a]], dbox[cur[b]]
                    u = _iou(A, B); ca, cb = _ctr(A), _ctr(B); w = max(A[2] - A[0], B[2] - B[0])
                    if u >= AP_IOU or np.hypot(ca[0] - cb[0], ca[1] - cb[1]) < AP_DIST * w:
                        close.add((a, b)); p = self.pairs.setdefault((a, b), dict(peak=0.0)); p['peak'] = max(p['peak'], u)
            # crops of approaching numbers whose memory is not frozen
            want = sorted({cur[x] for pq in close for x in pq if self.L.get(x, {}).get('frozen') is None and cover(cur[x]) < PC_COVER},
                          key=lambda i: -(dbox[i][3] - dbox[i][1]))
            if want:
                fe = get_feats(want)
                for i, v in fe.items(): self._store(lab(i), v)
            for (a, b) in close:
                if self.pairs[(a, b)]['peak'] >= FREEZE_IOU:
                    for x in (a, b):
                        s = self.L.get(x)
                        if s is not None and s.get('frozen') is None: s['frozen'] = _unit(np.mean(s['mem'], 0)) if s['mem'] else None
            # pairs apart again
            for (a, b) in [pq for pq in self.pairs if pq not in close]:
                p = self.pairs.pop((a, b))
                if p['peak'] < FREEZE_IOU or a not in cur or b not in cur: continue
                ma, mb = self._mem(a), self._mem(b); st['pc_checks'] += 1
                if ma is None or mb is None: st['pc_nomem'] += 1; continue
                fe = get_feats([cur[a], cur[b]])
                if cur[a] not in fe or cur[b] not in fe: st['pc_nomem'] += 1; continue
                keep = _cmap(float(fe[cur[a]] @ ma)) + _cmap(float(fe[cur[b]] @ mb)); crs = _cmap(float(fe[cur[a]] @ mb)) + _cmap(float(fe[cur[b]] @ ma))   # 11-20: per-model cosine map (reid/cos_map.py)
                if (crs - keep) / 2.0 >= MARGIN:
                    ra, rb = raw[cur[a]], raw[cur[b]]; self._relabel(ra, b); self._relabel(rb, a); st['pc_swaps'] += 1
                    cur[a], cur[b] = cur[b], cur[a]; self._ev('pc_swap', a, b)
        # ---- label states: boxes, motion, memories (no writes for group numbers)
        held = self.hold_labels(); in_pair = {x for pq in self.pairs for x in pq}
        free = [i for i in P if lab(i) not in held and int(rows[i][7]) in feats and cover(i) < CLEAN_COVER]
        for i in P:
            x = lab(i); b = dbox[i]; s = self.L.get(x)
            if s is None: s = self.L[x] = dict(box=b, f=f, f0=f, v=(0.0, 0.0), mem=[], frozen=None)
            else:
                n = max(f - s['f'], 1); c0, c1 = _ctr(s['box']), _ctr(b)
                s['v'] = (0.5 * s['v'][0] + 0.5 * (c1[0] - c0[0]) / n, 0.5 * s['v'][1] + 0.5 * (c1[1] - c0[1]) / n); s['box'] = b; s['f'] = f
            if x not in held and x not in in_pair and s.get('frozen') is not None: s['frozen'] = None   # apart again: memory learns again
        for i in free:
            if self.L[lab(i)].get('frozen') is None: self._store(lab(i), feats[int(rows[i][7])])
        if f % 300 == 0:   # forget labels not shown for a long time
            for x in [x for x, s in self.L.items() if f - s['f'] > 900 and x not in held]: del self.L[x]
        out = []
        for i, r in enumerate(rows):
            rr = list(r)
            if i in raw: rr[4] = self.M[raw[i]]
            out.append(rr)
        ms = (time.perf_counter() - t0) * 1000.0
        st['extra_crops'] += extra[0]; st['max_extra'] = max(st['max_extra'], extra[0]); st['ms_total'] += ms; st['ms_max'] = max(st['ms_max'], ms)
        self.extra_log.append(extra[0]); self.ms_log.append(ms)
        return out

    def _ev(self, *a):
        if self.log is not None: self.log.append((self.f,) + a)

    def _store(self, x, v):
        s = self.L.get(x)
        if s is None or s.get('frozen') is not None: return
        s['mem'].append(_unit(v))
        if len(s['mem']) > MEM_N: del s['mem'][0]

    def _move(self, g, box):
        c0, c1 = _ctr(g['box']), _ctr(box); n = max(self.f - g['f'], 1)
        g['v'] = (0.5 * g['v'][0] + 0.5 * (c1[0] - c0[0]) / n, 0.5 * g['v'][1] + 0.5 * (c1[1] - c0[1]) / n); g['box'] = box; g['f'] = self.f
