"""끊긴 추적 조각을 외형 특징으로 다시 잇는 부분 (Stitcher)."""
import numpy as np


class Stitcher:
    """--stitch-delay D (default off): delayed tracklet stitching on the OUTPUT stream; it never feeds back into the tracker. Every
    frame's track rows are held for D frames. Per output id: gallery = EMA (0.9/0.1, as Track.set_feat) of the ViT embeddings its
    tracks received (crop h >= min_h) and the last frame it was emitted. A NEW tracker id (person, first row h >= min_h) that has
    received >= m embeddings by D frames after its first row compares its mean embedding with every output id that ended at most W
    frames before that first row (so the two never coexist); best cos >= S and margin >= .05 to the second best -> ALL its rows carry
    the old output id (its first row is still in the delay buffer). An old tracker id that reappears in a frame where its output id is
    already carried by another track gets a fresh id (conflicts), so a frame never holds one id twice. push() returns the frames now
    due (payload, rows with output ids); flush() at a video end / reset returns the rest."""
    FRESH = 100000

    def __init__(self, delay, sim, m=1, window=60, min_h=32.0, margin=0.05, cls=0):
        self.D, self.S, self.m, self.W, self.min_h, self.margin, self.cls = int(delay), float(sim), int(m), int(window), float(min_h), float(margin), int(cls)
        self.stitches = 0; self.conflicts = 0; self.log = []; self._clear()

    def _clear(self):
        self.n = -1; self.buf = []; self.raw = {}; self.outs = {}; self.fresh = self.FRESH

    def _ema(self, o, f):
        f = np.asarray(f, np.float64); f = f / max(np.linalg.norm(f), 1e-12)
        g = f if o['gal'] is None else 0.9 * o['gal'] + 0.1 * f
        o['gal'] = g / max(np.linalg.norm(g), 1e-12); o['nemb'] += 1

    def _out(self, oid, last):
        return self.outs.setdefault(oid, dict(gal=None, nemb=0, last=last, carrier=None, cls=None))

    def _new_raw(self, rid):
        return self.raw.setdefault(int(rid), dict(first=None, last=None, out=None, embs=[], elig=False, cls=None))

    def _decide(self, rid, r):
        oid = rid
        if r['elig'] and len(r['embs']) >= self.m:
            q = np.mean(np.asarray(r['embs'], np.float64), 0); q /= max(np.linalg.norm(q), 1e-12); sc = []
            for x, o in self.outs.items():
                if o['gal'] is None or o['last'] >= r['first'] or r['first'] - o['last'] > self.W or o['cls'] != r['cls']: continue
                sc.append((float(o['gal'] @ q), x))
            sc.sort(reverse=True)
            if sc and sc[0][0] >= self.S and (len(sc) < 2 or sc[0][0] - sc[1][0] >= self.margin):
                oid = sc[0][1]; self.stitches += 1
                self.log.append(dict(frame=self.n, new_id=int(rid), old_id=int(oid), sim=round(sc[0][0], 4), second=(round(sc[1][0], 4) if len(sc) > 1 else None),
                                     gap=int(r['first'] - self.outs[oid]['last']), n_emb=len(r['embs'])))
        r['out'] = oid; o = self._out(oid, r['last']); o['cls'] = r['cls']; o['carrier'] = rid; o['last'] = max(o['last'], r['last'])
        for f in r['embs']: self._ema(o, f)
        for e in self.buf:
            for k, row in enumerate(e[1]):
                if int(row[4]) == rid and e[2][k] is None: e[2][k] = oid

    def push(self, payload, rows, embs=None, reset=False):
        """rows: tracker rows [x1,y1,x2,y2,id,conf,cls,det_idx] of this frame; embs: {tracker id: [L2 features received this frame]}."""
        done = self.flush() if reset else []
        self.n += 1; n = self.n
        for rid, fs in (embs or {}).items():
            r = self._new_raw(rid)
            for f in fs:
                if r['out'] is None: r['embs'].append(np.asarray(f, np.float64))
                else: self._ema(self.outs[r['out']], f)
        oids = []
        for row in rows:
            rid = int(row[4]); r = self._new_raw(rid)
            if r['first'] is None:
                r['first'] = n; r['cls'] = int(row[6]); r['elig'] = int(row[6]) == self.cls and (row[3] - row[1]) >= self.min_h
            r['last'] = n
            if r['out'] is None and not r['elig']:
                r['out'] = rid; o = self._out(rid, n); o['cls'] = r['cls']; o['carrier'] = rid
                for f in r['embs']: self._ema(o, f)
            oids.append(r['out'])
        seen = {}
        for k, row in enumerate(rows):          # conflicts: two decided tracker ids on one output id in this frame -> the non-carrier gets a fresh id
            oid = oids[k]
            if oid is None: continue
            if oid in seen:
                a = seen[oid]; keep, mv = (a, k) if self.outs[oid]['carrier'] == int(rows[a][4]) else (k, a)
                self.fresh += 1; r = self.raw[int(rows[mv][4])]; r['out'] = self.fresh; o = self._out(self.fresh, n); o['cls'] = r['cls']; o['carrier'] = int(rows[mv][4])
                oids[mv] = self.fresh; self.conflicts += 1; seen[oid] = keep
            else: seen[oid] = k
        for oid in oids:
            if oid is not None: self.outs[oid]['last'] = n
        self.buf.append([payload, rows, oids])
        for rid in sorted((x for x, r in self.raw.items() if r['out'] is None and r['first'] is not None and n - r['first'] >= self.D), key=lambda x: self.raw[x]['first']):
            self._decide(rid, self.raw[rid])
        while len(self.buf) > self.D: done.append(self._emit(self.buf.pop(0)))
        return done

    def _emit(self, e):
        return e[0], [list(row[:4]) + [int(o if o is not None else row[4])] + list(row[5:]) for row, o in zip(e[1], e[2])]

    def flush(self):
        for rid in sorted((x for x, r in self.raw.items() if r['out'] is None and r['first'] is not None), key=lambda x: self.raw[x]['first']):
            self._decide(rid, self.raw[rid])
        done = [self._emit(e) for e in self.buf]; self._clear(); return done


def resolve_embs(tracker, delivered, sync_got):
    """--stitch-*: embeddings received this frame -> {tracker id: [features]}. delivered: [(track id or None, crop box, feature)] from the
    asynchronous queue (track id after attach_embedding); sync_got: {det_idx: feature} embedded synchronously in this frame's update()."""
    out = {}
    for tid, box, f in delivered:
        if tid is not None and (box[3] - box[1]) >= 32.0: out.setdefault(int(tid), []).append(f)
    if sync_got:
        by = {t.last_det[1]: t for t in tracker.tracked if t.last_det is not None and t.last_det[0] == tracker.frame_id}
        for i, f in sync_got.items():
            t = by.get(int(i))
            if t is not None: out.setdefault(int(t.track_id), []).append(f)
    return out
