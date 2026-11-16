"""Online learned relink judge for the M1 tracker script (Claude 2026-11-02, feattrk; replaces KeepID's hand rule when m1_track_eval_v10.py
runs with --judge <RELINK_JUDGE_FROZEN.json>). numpy only (M1 host + PC); no sklearn / torch.

The feature code is a verbatim copy of failure_analysis_20261102/learned_relink/build/relink_data.py (make_obs, cv_velocity, gallery,
pair_features + the rank / min-other / max-other columns of build_sequence) and the model inference / decision rule of
learned_relink/train/relink_judge.py (prep_apply, predict_json, relabel_online), rewritten frame by frame:
  - every output person row of every frame -> an observation (box, confidence, overlap / neighbour counts against the frame's person
    detections P, crop height, RepVGG feature or None); rows are rounded like TRACKS_BASE.jsonl (box 0.01, conf 1e-5) so the online
    features equal an offline rebuild from the saved rows;
  - birth = a tracker person ID output for the first time in the sequence (frames since the tracker reset); decided AT ITS BIRTH FRAME
    (K = max_wait_frames = 0 of the frozen model; offsets_scored [0]); candidates = tracker person IDs output earlier, not output now,
    lost <= 90 frames (the candidate set includes retired / handed-over tracker IDs, because n_cand, rank_d, rank_cv, *_min_other,
    m1_cos_mean5_max_other are defined over that set; they are filtered at the decision only, as relabel_online does);
  - decision = relabel_online: pairs with p >= threshold, candidate still holds an ID, not output now, its output ID not output now;
    greedy by (-p, birth id, candidate id); one candidate per birth, one birth per candidate; the birth takes the candidate's output ID for
    the rest of its life (forward only); the candidate tracker ID is retired (a fresh ID if it is output again); boxes / confidences /
    classes never change.
Appearance (feature set B) and the crop budget (--judge-crops N per frame, default 2) on NPU core 2:
  - eligible crop = output person row whose detection k is in P (person detections conf >= 0.1) with detection height >= 32 px, ANY
    overlap (KeepID embedded clean crops only);
  - priority: (1) the crops of this frame's births that have >= 1 in-window candidate (smallest ID first) - the judge needs the new box's
    own feature at the decision frame, (2) other births, (3) refresh of eligible tracks: fewest features among the last 5 observations
    first, then the oldest last feature, then the smallest ID. (1)+(2) are embedded synchronously before the decision; (3) may be
    asynchronous (embed_async, --rl-async): collected at the start of the next frame and attached to that observation, which gives the
    same features as the synchronous path for every later decision (a candidate's history is only read at later frames).
  - missing features = the availability flags of the training tables: m1_new_avail 0 and NaN cosines when the new box has no feature,
    m1_cand_n5 0 / m1_cand_nclean 0 and NaN when the candidate has none; the lr model imputes the training median of slog(x) per column and
    adds the learned missing flag (prep 'miss' columns) - exactly as in training, where features existed only for crops >= 32 px.
    With the budget, candidate galleries are sparser than in the saved-feature training tables (every crop >= 32 px existed there);
    the PC replay (verify_judge_pc.py) quantifies the difference.
"""
import json, math, time, hashlib
from pathlib import Path
import numpy as np

VIEW = (0.0, 16.0, 640.0, 496.0)      # RT480 detector view in frame coords (rows 16..496 of the 640x512 frame)
WINDOW = 90                            # candidate window (frames), = KeepID LOST_MAX
NEAR_H = 2.0                           # "near" = centre distance <= 2 heights
CLEAN_H = 32.0                         # KeepID crop height rule
CV_N = 5                               # constant-velocity fit on the last 5 output boxes
GAL_N = 5                              # mean of the last 5 available features
MIN_H = 32.0                           # crops below this height get no feature (crop_from_pred.py rule)
NAN = float('nan')


# ---------------------------------------------------------------- relink_data.py (verbatim; fm1 / ffp replaced by the feature value)
def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def unit(v):
    v = np.asarray(v, np.float64); return v / max(float(np.linalg.norm(v)), 1e-12)


def make_obs(t, row, P, prow, m1=None):
    """one output person row at frame t -> observation dict (everything here is from frame t only); m1 = its RepVGG feature or None"""
    x1, y1, x2, y2 = [float(v) for v in row[:4]]; tid, k = int(row[4]), int(row[7])
    w, h = x2 - x1, y2 - y1; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    db = P.get(k)                                     # the detection box (crop source; KeepID's overlap rule uses it)
    ref = db if db is not None else [x1, y1, x2, y2]
    others = [b for j, b in P.items() if j != k]
    ious = [iou(ref, b) for b in others]
    rc = ((ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2); rh = ref[3] - ref[1]
    n_near_det = sum(1 for b in others if math.hypot(rc[0] - (b[0] + b[2]) / 2, rc[1] - (b[1] + b[3]) / 2) <= NEAR_H * rh)
    n_near_trk = sum(1 for r in prow if int(r[4]) != tid and math.hypot(cx - (r[0] + r[2]) / 2, cy - (r[1] + r[3]) / 2) <= NEAR_H * h)
    return dict(t=t, tid=tid, box=(x1, y1, x2, y2), conf=float(row[5]), k=k, w=w, h=h, cx=cx, cy=cy,
                crop_h=(db[3] - db[1]) if db is not None else NAN, in_P=db is not None,
                miou=max(ious, default=0.0), n_ovl=sum(1 for v in ious if v > 0), n_near_det=n_near_det, n_near_trk=n_near_trk,
                ex=min(x1 - VIEW[0], VIEW[2] - x2), ey=min(y1 - VIEW[1], VIEW[3] - y2),
                m1=(m1 if k >= 0 else None), fp=None)


def cv_velocity(obs):
    """least-squares centre velocity (px / frame) over the last CV_N observations; (vx, vy, n used)"""
    o = obs[-CV_N:]
    if len(o) < 2: return 0.0, 0.0, len(o)
    t = np.asarray([x['t'] for x in o], np.float64)
    if t.max() - t.min() <= 0: return 0.0, 0.0, len(o)
    tc = t - t.mean(); den = float(tc @ tc)
    vx = float(tc @ (np.asarray([x['cx'] for x in o]) - np.mean([x['cx'] for x in o]))) / den
    vy = float(tc @ (np.asarray([x['cy'] for x in o]) - np.mean([x['cy'] for x in o]))) / den
    return vx, vy, len(o)


def gallery(obs, key, clean=False):
    """the last GAL_N available features of an observation list (any overlap / height unless clean) -> (list of obs, most recent last)"""
    out = []
    for o in reversed(obs):
        if o[key] is None: continue
        if clean and not (o['crop_h'] >= CLEAN_H and o['miou'] == 0.0): continue
        out.append(o)
        if len(out) >= GAL_N: break
    return out[::-1]


def cos(a, b):
    return NAN if a is None or b is None else float(np.dot(a, b))


def pair_features(nobs, cobs, t0, t_now, rows_now, coexist):
    """nobs = the new ID's observations t0..t_now (time order), cobs = the candidate's observations before t0; rows_now = output person
    rows of frame t_now. Returns a dict of causal features (no GT)."""
    n, c = nobs[-1], cobs[-1]
    gap = n['t'] - c['t']; hl, wl = max(c['h'], 1e-6), max(c['w'], 1e-6)
    dx, dy = n['cx'] - c['cx'], n['cy'] - c['cy']; d = math.hypot(dx, dy)
    vx, vy, nv = cv_velocity(cobs)
    px, py = c['cx'] + vx * gap, c['cy'] + vy * gap
    pbox = (px - c['w'] / 2, py - c['h'] / 2, px + c['w'] / 2, py + c['h'] / 2)
    sp = math.hypot(vx, vy)
    dcos = (vx * dx + vy * dy) / (sp * d) if sp > 1e-6 and d > 1e-6 else NAN
    keepid_gate = 3.5 * hl + 1.0 * hl * gap / 10.0
    f = dict(gap=gap, gap_birth=t0 - c['t'], delay=t_now - t0, new_out_now=int(n['t'] == t_now), new_nobs=len(nobs), cand_coexist=int(coexist),
             d_last_px=d, d_last_h=d / hl, dx_last_h=dx / hl, dy_last_h=dy / hl, d_over_keepid_gate=d / keepid_gate,
             cv_n=nv, cv_vx_h=vx / hl, cv_vy_h=vy / hl, speed_h=sp / hl, d_cv_h=math.hypot(n['cx'] - px, n['cy'] - py) / hl,
             iou_cv=iou(n['box'], pbox), iou_last=iou(n['box'], c['box']), dir_cos=dcos, dir_valid=int(dcos == dcos),
             cv_out_view=int(not (VIEW[0] <= px <= VIEW[2] and VIEW[1] <= py <= VIEW[3])),
             hr=n['h'] / hl, wr=n['w'] / wl, log_hr=math.log(max(n['h'], 1e-6) / hl), log_wr=math.log(max(n['w'], 1e-6) / wl),
             h_new=n['h'], w_new=n['w'], h_last=c['h'], w_last=c['w'], aspect_new=n['w'] / max(n['h'], 1e-6), aspect_last=c['w'] / hl,
             conf_new=n['conf'], conf_new_mean=float(np.mean([o['conf'] for o in nobs])), conf_last=c['conf'],
             conf_last5=float(np.mean([o['conf'] for o in cobs[-5:]])),
             edge_x_new=n['ex'], edge_y_new=n['ey'], edge_new_h=min(n['ex'], n['ey']) / max(n['h'], 1e-6),
             edge_x_last=c['ex'], edge_y_last=c['ey'], edge_last_h=min(c['ex'], c['ey']) / hl,
             miou_new=n['miou'], n_ovl_new=n['n_ovl'], n_near_det_new=n['n_near_det'], n_near_trk_new=n['n_near_trk'],
             overlap_level_new=0 if n['miou'] == 0 else (1 if n['miou'] <= 0.2 else (2 if n['miou'] <= 0.4 else 3)),
             miou_last=c['miou'], n_ovl_last=c['n_ovl'], n_near_det_last=c['n_near_det'], n_near_trk_last=c['n_near_trk'],
             n_trk_near_lastpos_now=sum(1 for r in rows_now if int(r[4]) != n['tid'] and
                                         math.hypot(c['cx'] - (r[0] + r[2]) / 2, c['cy'] - (r[1] + r[3]) / 2) <= NEAR_H * hl),
             cand_len=len(cobs), cand_age=c['t'] - cobs[0]['t'], cand_born_first_frame=int(cobs[0]['t'] == 0),
             new_crop_h=n['crop_h'], new_in_P=int(n['in_P']))
    for s in ('m1', 'fp'):
        g = gallery(cobs, s); nf = [o for o in nobs if o[s] is not None]
        fl = n[s]; fm = unit(np.mean([o[s] for o in nf], 0)) if nf else None
        g1 = g[-1][s] if g else None; g5 = unit(np.mean([o[s] for o in g], 0)) if g else None
        f.update({s + '_new_avail': int(fl is not None), s + '_new_n': len(nf), s + '_cand_n5': len(g),
                  s + '_cos_last1': cos(fl, g1), s + '_cos_mean5': cos(fl, g5), s + '_cos_nmean_last1': cos(fm, g1), s + '_cos_nmean_mean5': cos(fm, g5),
                  s + '_cand_feat_age': (n['t'] - g[-1]['t']) if g else NAN, s + '_cand_last1_h': g[-1]['crop_h'] if g else NAN,
                  s + '_cand_last1_miou': g[-1]['miou'] if g else NAN, s + '_cand_mean5_miou_max': max(o['miou'] for o in g) if g else NAN,
                  s + '_new_feat_h': n['crop_h'] if fl is not None else NAN, s + '_new_feat_miou': n['miou'] if fl is not None else NAN})
        if s == 'm1':
            gc = gallery(cobs, s, clean=True); gc5 = unit(np.mean([o[s] for o in gc], 0)) if gc else None
            f.update(m1_cand_nclean=len(gc), m1_cos_clean5=cos(fl, gc5), m1_new_clean=int(fl is not None and n['crop_h'] >= CLEAN_H and n['miou'] == 0.0))
    return f


def group_features(fs):
    """the per-birth columns of build_sequence: rank_d, rank_cv, d_cv_h_min_other, d_last_h_min_other, {m1,fp}_cos_mean5_max_other (in place)"""
    for key, rk in (('d_last_h', 'rank_d'), ('d_cv_h', 'rank_cv')):
        order = sorted(range(len(fs)), key=lambda i: (fs[i][key], fs[i]['cand_tid']))
        for r_, i in enumerate(order): fs[i][rk] = r_ + 1
    for i, f in enumerate(fs):
        oth = [g for j, g in enumerate(fs) if j != i]
        f['d_cv_h_min_other'] = min((g['d_cv_h'] for g in oth), default=NAN)
        f['d_last_h_min_other'] = min((g['d_last_h'] for g in oth), default=NAN)
        for s in ('m1', 'fp'):
            v = [g[s + '_cos_mean5'] for g in oth if g[s + '_cos_mean5'] == g[s + '_cos_mean5']]
            f[s + '_cos_mean5_max_other'] = max(v) if v else NAN


# ---------------------------------------------------------------- relink_judge.py (verbatim): exported-model inference in numpy
def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def prep_apply(P, X):
    med, mu, sd = np.asarray(P['med']), np.asarray(P['mu']), np.asarray(P['sd'])
    Xs = slog(X); nan = np.isnan(Xs); Xi = np.where(nan, med, Xs); Z = np.clip((Xi - mu) / sd, -5.0, 5.0)
    if P['miss']: Z = np.hstack([Z, nan[:, P['miss']].astype(np.float64)])
    return Z


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))


def predict_json(M, X):
    """pure numpy inference of an exported model dict -> p(same person)"""
    if M['kind'] == 'lr':
        return sigmoid(prep_apply(M['prep'], X) @ np.asarray(M['coef']) + M['intercept'])
    if M['kind'] == 'mlp':
        h = prep_apply(M['prep'], X)
        for i, (W, b) in enumerate(zip(M['W'], M['b'])):
            h = h @ np.asarray(W) + np.asarray(b)
            if i < len(M['W']) - 1: h = np.maximum(h, 0.0)
        return sigmoid(h[:, 0])
    raw = np.full(len(X), M['baseline'])
    for T in M['trees']:
        f, thr, nl, L, Rr, leaf, val = (np.asarray(T[k]) for k in ('feature', 'thr', 'nan_left', 'left', 'right', 'leaf', 'value'))
        node = np.zeros(len(X), int)
        for _ in range(64):
            act = leaf[node] == 0
            if not act.any(): break
            idx = np.flatnonzero(act); n = node[idx]; x = X[idx, f[n]]
            go_left = np.where(np.isnan(x), nl[n] == 1, x <= thr[n])
            node[idx] = np.where(go_left, L[n], Rr[n])
        raw += val[node]
    return sigmoid(raw)


def load_spec(path):
    """RELINK_JUDGE_FROZEN.json -> (spec dict, sha256 of the file text)"""
    txt = Path(path).read_text(encoding='utf-8'); spec = json.loads(txt)
    assert spec['decision']['max_wait_frames'] == 0 and spec['decision']['offsets_scored'] == [0], 'only K = 0 (decide at the birth frame) is implemented online'
    assert spec['decision']['candidate_window_frames'] == WINDOW, spec['decision']['candidate_window_frames']
    assert spec['model']['kind'] in ('lr', 'mlp', 'hgb'), spec['model']['kind']
    return spec, hashlib.sha256(txt.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- the online judge
class OnlineJudge:
    """Drop-in for relabel_online.OnlineRelabel (same step / reset / stats / events / per_frame interface) with the learned judge.
    crops = per-frame crop budget on NPU core 2 (0 = unlimited: every eligible crop every frame; the PC replay uses it)."""
    def __init__(self, spec_path, crops=2):
        self.spec, self.spec_sha = load_spec(spec_path)
        self.M, self.feats, self.thr = self.spec['model'], list(self.spec['features']), float(self.spec['decision']['threshold'])
        self.crops = int(crops)
        self.stats = dict(new_ids=0, births=0, births_with_candidates=0, pairs_scored=0, relabelled=0, retired_reappearances=0, frames=0,
                          crops_new=0, crops_refresh=0, crops_embedded=0, crops_new_births_with_candidates=0, births_with_candidates_no_feature=0,
                          decided_with_feature=0)
        self.events, self.per_frame = [], []
        self.feat_ms, self.model_ms, self.cands_per_eval = [], [], []
        self.keep_rows = False; self.rows_log = []
        self.pred_gate = True   # 11-14 (user): only candidates whose growing search range (relabel_online.search_range) holds the birth (default on after the replay check); False = the old candidate set                      # PC verification: (frame, t, birth id, [cand ids], X, p) per evaluation
        self.reset()

    def reset(self):
        """call at the first frame of every video (drains an asynchronous refresh of the previous video; its features are dropped)"""
        if getattr(self, 'deferred', None) is not None: self.deferred[1]()
        self.deferred = None
        self.t = -1; self.fresh = 10 ** 6
        self.hist = {}            # tracker id -> [obs] (time order), every output person row of the sequence
        self.id_map = {}          # tracker id -> output id (popped when retired)
        self.retired = set()      # tracker ids handed over; a fresh output id if output again
        self.last_emb = {}        # tracker id -> t of its last feature (refresh priority)

    def _elig(self, prow, P):
        """tracker id -> detection k of the rows that can be cropped (k in P, detection height >= 32 px, any overlap)"""
        e = {}
        for r in prow:
            k = int(r[7])
            if k >= 0 and k in P and P[k][3] - P[k][1] >= MIN_H: e[int(r[4])] = k
        return e

    def step(self, rows, P, embed, frame=None, embed_async=None):
        """rows: this frame's tracker rows [x1, y1, x2, y2, id, conf, cls, k, ...]; P: {k: [x1, y1, x2, y2]} person detections (conf >= 0.1)
        keyed like the rows' k; embed(list of k) -> {k: L2-normalised feature}; embed_async(list of k) -> collect() -> {k: feature}.
        Returns the rows (same order, same boxes) with relabelled person IDs."""
        self.t += 1; t = self.t; st = self.stats; st['frames'] += 1
        if self.deferred is not None:                                     # last frame's asynchronous refresh joins its observations now
            recs_d, collect_d = self.deferred; self.deferred = None; got_d = collect_d() or {}
            for o_, k_ in recs_d:
                if k_ in got_d: o_['m1'] = got_d[k_]; self.last_emb[o_['tid']] = o_['t']; st['crops_embedded'] += 1
        prow = [r for r in rows if int(r[6]) == 0]
        # rows rounded like TRACKS_BASE.jsonl, so the features equal an offline rebuild from the saved rows
        prow_r = [[round(float(v), 2) for v in r[:4]] + [int(r[4]), round(float(r[5]), 5), int(r[6]), int(r[7])] for r in prow]
        now = [int(r[4]) for r in prow_r]; now_set = set(now)
        births = [b for b in now if b not in self.hist]
        # candidates of every birth (frozen at t0): tracker ids output before, not output now, lost <= WINDOW frames
        cands = {}
        if births:
            cl = [(c, co) for c, co in self.hist.items() if c not in now_set and t - co[-1]['t'] <= WINDOW]
            for b in births: cands[b] = cl
            if self.pred_gate:   # 11-14: candidate kept only if the birth centre lies in its growing search range around the predicted place
                from .relabel_online import SR_BASE, SR_GROW, SR_MAX, SR_VMAX
                rb = {int(r[4]): r for r in prow_r}
                for b in births:
                    bx, by = (rb[b][0] + rb[b][2]) / 2, (rb[b][1] + rb[b][3]) / 2; keep = []
                    for c, co in cl:
                        last = co[-1]; hl = max(last['h'], 1e-6); gap = t - last['t']; vx, vy, _ = cv_velocity(co); sp = math.hypot(vx, vy)
                        if sp > SR_VMAX * hl: vx, vy = vx * SR_VMAX * hl / sp, vy * SR_VMAX * hl / sp
                        if math.hypot(bx - (last['cx'] + vx * gap), by - (last['cy'] + vy * gap)) <= hl * min(SR_BASE + SR_GROW * gap, SR_MAX): keep.append((c, co))
                    cands[b] = keep
        # ---- crop budget: births with candidates, other births, refresh (fewest features of the last 5 obs, oldest last feature, smallest id)
        elig = self._elig(prow_r, P)
        bc = sorted(b for b in births if cands.get(b) and b in elig); bo = sorted(b for b in births if not cands.get(b) and b in elig)
        req = [(b, elig[b]) for b in bc + bo]
        if self.crops > 0: req = req[:self.crops]
        n_new = len(req); taken = {b for b, _ in req}
        rest = self.crops - n_new if self.crops > 0 else len(elig)
        if rest > 0:
            ref = sorted((len(gallery(self.hist.get(bid, []), 'm1')), self.last_emb.get(bid, -1), bid, k) for bid, k in elig.items() if bid not in taken)
            req += [(bid, k) for _, _, bid, k in ref[:rest]]
        recs = None
        if embed_async is not None and len(req) > n_new:                # refresh asynchronously, the births' crops now
            got = (embed([k for _, k in req[:n_new]]) or {}) if n_new else {}
            recs = req[n_new:]; collect = embed_async([k for _, k in recs])
        else:
            got = (embed([k for _, k in req]) or {}) if req else {}
        feat = {bid: got[k] for bid, k in req if k in got}
        st['crops_new'] += n_new; st['crops_refresh'] += len(req) - n_new; st['crops_embedded'] += len(feat); self.per_frame.append((len(req), len(feat)))
        st['crops_new_births_with_candidates'] += sum(1 for b in bc[:n_new] if b in feat)
        # ---- observations of this frame (not yet in hist: candidate histories are the frames before t0)
        obs = {int(r[4]): make_obs(t, r, P, prow_r, feat.get(int(r[4]))) for r in prow_r}
        # ---- id bookkeeping (relabel_online): retired ids reappearing get a fresh id; births get their own id
        for b in now:
            if b in self.retired:
                self.id_map[b] = self.fresh; self.fresh += 1; self.retired.discard(b); st['retired_reappearances'] += 1
            if b not in self.id_map:
                self.id_map[b] = b; st['new_ids'] += 1
        out_ids = {self.id_map[x] for x in now}
        # ---- judge every birth with candidates (K = 0: features at t_now = t0, nobs = [the birth observation])
        st['births'] += len(births); pairs = []
        for b in births:
            cl = cands.get(b)
            if not cl: continue
            st['births_with_candidates'] += 1
            if obs[b]['m1'] is None: st['births_with_candidates_no_feature'] += 1
            else: st['decided_with_feature'] += 1
            t1 = time.perf_counter_ns(); fs = []
            for c, cb in cl:
                f = pair_features([obs[b]], cb, t, t, prow_r, False)
                f.update(cand_tid=c, n_cand=len(cl), n_births_frame=len(births), n_out_frame=len(now_set)); fs.append(f)
            group_features(fs)
            X = np.asarray([[float(f[c_]) for c_ in self.feats] for f in fs], np.float64)
            t2 = time.perf_counter_ns(); p = predict_json(self.M, X); t3 = time.perf_counter_ns()
            self.feat_ms.append((t2 - t1) / 1e6); self.model_ms.append((t3 - t2) / 1e6); self.cands_per_eval.append(len(cl)); st['pairs_scored'] += len(cl)
            if self.keep_rows: self.rows_log.append((frame, t, b, [c for c, _ in cl], X, p))
            for (c, _), pv in zip(cl, p):
                pv = float(pv)
                if pv >= self.thr and c != b and c in self.id_map and c not in now_set and self.id_map[c] not in out_ids:
                    pairs.append((pv, b, c))
        ub, uc = set(), set()
        for pv, b, c in sorted(pairs, key=lambda x: (-x[0], x[1], x[2])):
            if b in ub or c in uc or c not in self.id_map or b not in self.id_map: continue
            ub.add(b); uc.add(c); oid = self.id_map[c]; old_new = self.id_map[b]
            self.id_map[b] = oid; self.retired.add(c); self.id_map.pop(c, None); st['relabelled'] += 1
            self.events.append(dict(frame=frame, t=t, tracker_id=b, new_id=old_new, old_id=oid, cand_tid=c, p=round(pv, 4), n_cand=len(cands[b])))
        # ---- histories + deferred refresh
        for b, o in obs.items():
            self.hist.setdefault(b, []).append(o)
            if o['m1'] is not None: self.last_emb[b] = t
        if recs is not None: self.deferred = ([(obs[bid], k) for bid, k in recs if bid in obs], collect)
        out = []
        for r in rows:
            rr = list(r)
            if int(r[6]) == 0: rr[4] = self.id_map[int(r[4])]
            out.append(rr)
        return out

    def timing(self):
        def dist(v):
            v = np.asarray(v, np.float64)
            return dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), max=float(v.max())) if len(v) else None
        tot = [a + b for a, b in zip(self.feat_ms, self.model_ms)]
        return dict(ms_per_decision=dist(tot), feature_ms=dist(self.feat_ms), model_ms=dist(self.model_ms), candidates_per_decision=dist(self.cands_per_eval))


def replay(rows, P_of, feat_of, spec_path, crops=0):
    """PC replay over saved project rows (frame dicts with tracker_reset) with saved features: P_of(frame dict) -> {k: box},
    feat_of(image_id, k) -> unit feature or None. crops = 0 reproduces the training tables (every crop with a feature is 'embedded').
    Returns (relabelled rows, judge)."""
    J = OnlineJudge(spec_path, crops=crops); out = []
    for fr in rows:
        if fr['tracker_reset']: J.reset()
        P = P_of(fr); img = fr['image_id']
        embed = lambda ks, img=img: {k: f for k in ks for f in [feat_of(img, k)] if f is not None}
        new = J.step(fr['tracks_xyxy_id_conf_class_detectionindex'], P, embed, frame=img)
        out.append(dict(fr, tracks_xyxy_id_conf_class_detectionindex=new))
    return out, J
