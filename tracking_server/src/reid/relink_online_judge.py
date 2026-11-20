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
from ..common import frame_math

VIEW = (0.0, 16.0, 640.0, 496.0)      # RT480 detector view in frame coords (rows 16..496 of the 640x512 frame)
WINDOW = 90                            # candidate window (frames), = KeepID LOST_MAX
NEAR_H = 2.0                           # "near" = centre distance <= 2 heights
CLEAN_H = 32.0                         # KeepID crop height rule
CV_N = 5                               # constant-velocity fit on the last 5 output boxes
GAL_N = 5                              # mean of the last 5 available features
FOLLOW_HISTORY_N = 12                  # gimbal appearance looks at the last 12 observations
MIN_H = 32.0                           # crops below this height get no feature (crop_from_pred.py rule)
NAN = float('nan')


# ---------------------------------------------------------------- relink_data.py (verbatim; fm1 / ffp replaced by the feature value)
def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def unit(v):
    v = np.asarray(v, np.float64); return v / max(float(np.linalg.norm(v)), 1e-12)


def make_obs(t, row, P, prow, m1=None, *, geometry=None):
    """one output person row at frame t -> observation dict (everything here is from frame t only); m1 = its RepVGG feature or None"""
    x1, y1, x2, y2 = [float(v) for v in row[:4]]; tid, k = int(row[4]), int(row[7])
    w, h = x2 - x1, y2 - y1; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    db = P.get(k)                                     # the detection box (crop source; KeepID's overlap rule uses it)
    if geometry is None:
        ref = db if db is not None else [x1, y1, x2, y2]
        others = [b for j, b in P.items() if j != k]
        ious = [iou(ref, b) for b in others]
        rc = ((ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2); rh = ref[3] - ref[1]
        n_near_det = sum(1 for b in others if math.hypot(rc[0] - (b[0] + b[2]) / 2, rc[1] - (b[1] + b[3]) / 2) <= NEAR_H * rh)
        n_near_trk = sum(1 for r in prow if int(r[4]) != tid and math.hypot(cx - (r[0] + r[2]) / 2, cy - (r[1] + r[3]) / 2) <= NEAR_H * h)
        miou, n_ovl = max(ious, default=0.0), sum(1 for v in ious if v > 0)
    else:
        miou, n_ovl, n_near_det, n_near_trk = geometry
    return dict(t=t, tid=tid, box=(x1, y1, x2, y2), conf=float(row[5]), k=k, w=w, h=h, cx=cx, cy=cy,
                crop_h=(db[3] - db[1]) if db is not None else NAN, in_P=db is not None,
                miou=miou, n_ovl=n_ovl, n_near_det=n_near_det, n_near_trk=n_near_trk,
                ex=min(x1 - VIEW[0], VIEW[2] - x2), ey=min(y1 - VIEW[1], VIEW[3] - y2),
                m1=(m1 if k >= 0 else None), fp=None)


def make_observations(t, rows, detections, features):
    geometry = frame_math.keepid_geometry(rows, detections, NEAR_H)
    return {row[4]: make_obs(t, row, detections, rows, features.get(row[4]),
                             geometry=None if geometry is None else geometry[i])
            for i, row in enumerate(rows)}


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


class _LiveObservationHistory(list):
    """Bound the geometry window while preserving identity features and sparse galleries."""

    __slots__ = ('first_t', 'total', 'features')

    def __init__(self):
        super().__init__()
        self.first_t = None
        self.total = 0
        self.features = {}

    def append(self, observation):
        if not self:
            self.first_t = observation['t']
        self.total += 1
        super().append(observation)
        if len(self) > FOLLOW_HISTORY_N:
            del self[0]
        for key in ('m1', 'fp'):
            self.add_feature(observation, key)

    def add_feature(self, observation, key):
        if observation[key] is None:
            return
        self._keep((key, False), observation)
        if observation['crop_h'] >= CLEAN_H and observation['miou'] == 0.0:
            self._keep((key, True), observation)

    def _keep(self, key, observation):
        observations = self.features.setdefault(key, [])
        observations.append(observation)
        if len(observations) > GAL_N:
            del observations[0]


def gallery(obs, key, clean=False):
    """the last GAL_N available features of an observation list (any overlap / height unless clean) -> (list of obs, most recent last)"""
    if isinstance(obs, _LiveObservationHistory):
        return list(obs.features.get((key, clean), ()))
    out = []
    for o in reversed(obs):
        if o[key] is None: continue
        if clean and not (o['crop_h'] >= CLEAN_H and o['miou'] == 0.0): continue
        out.append(o)
        if len(out) >= GAL_N: break
    return out[::-1]


def cos(a, b):
    return NAN if a is None or b is None else float(np.dot(a, b))


def _gallery_summary(observations, key, *, clean=False):
    entries = gallery(observations, key, clean=clean)
    if not entries:
        return 0, None, None, NAN
    return (len(entries), entries[-1], unit(np.mean([o[key] for o in entries], 0)),
            max(o['miou'] for o in entries))


def _candidate_summary(observations):
    if isinstance(observations, _LiveObservationHistory):
        count, first = observations.total, observations.first_t
    else:
        count, first = len(observations), observations[0]['t']
    return (count, first, float(np.mean([o['conf'] for o in observations[-5:]])),
            {key: _gallery_summary(observations, key) for key in ('m1', 'fp')},
            _gallery_summary(observations, 'm1', clean=True))


def _birth_summary(observations):
    appearance = {}
    for key in ('m1', 'fp'):
        values = [o[key] for o in observations if o[key] is not None]
        appearance[key] = len(values), unit(np.mean(values, 0)) if values else None
    return float(np.mean([o['conf'] for o in observations])), appearance


def pair_features(nobs, cobs, t0, t_now, rows_now, coexist, *,
                  candidate=None, birth=None, velocity=None):
    """nobs = the new ID's observations t0..t_now (time order), cobs = the candidate's observations before t0; rows_now = output person
    rows of frame t_now. Returns a dict of causal features (no GT)."""
    n, c = nobs[-1], cobs[-1]
    candidate_count, candidate_first_t, conf_last5, galleries, clean = (
        _candidate_summary(cobs) if candidate is None else candidate)
    conf_new, appearance = _birth_summary(nobs) if birth is None else birth
    gap = n['t'] - c['t']; hl, wl = max(c['h'], 1e-6), max(c['w'], 1e-6)
    dx, dy = n['cx'] - c['cx'], n['cy'] - c['cy']; d = math.hypot(dx, dy)
    vx, vy, nv = cv_velocity(cobs) if velocity is None else velocity
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
             conf_new=n['conf'], conf_new_mean=conf_new, conf_last=c['conf'], conf_last5=conf_last5,
             edge_x_new=n['ex'], edge_y_new=n['ey'], edge_new_h=min(n['ex'], n['ey']) / max(n['h'], 1e-6),
             edge_x_last=c['ex'], edge_y_last=c['ey'], edge_last_h=min(c['ex'], c['ey']) / hl,
             miou_new=n['miou'], n_ovl_new=n['n_ovl'], n_near_det_new=n['n_near_det'], n_near_trk_new=n['n_near_trk'],
             overlap_level_new=0 if n['miou'] == 0 else (1 if n['miou'] <= 0.2 else (2 if n['miou'] <= 0.4 else 3)),
             miou_last=c['miou'], n_ovl_last=c['n_ovl'], n_near_det_last=c['n_near_det'], n_near_trk_last=c['n_near_trk'],
             n_trk_near_lastpos_now=sum(1 for r in rows_now if int(r[4]) != n['tid'] and
                                         math.hypot(c['cx'] - (r[0] + r[2]) / 2, c['cy'] - (r[1] + r[3]) / 2) <= NEAR_H * hl),
             cand_len=candidate_count, cand_age=c['t'] - candidate_first_t, cand_born_first_frame=int(candidate_first_t == 0),
             new_crop_h=n['crop_h'], new_in_P=int(n['in_P']))
    for s in ('m1', 'fp'):
        count, last, g5, miou_max = galleries[s]
        n_features, fm = appearance[s]
        fl = n[s]; g1 = last[s] if last is not None else None
        f.update({s + '_new_avail': int(fl is not None), s + '_new_n': n_features, s + '_cand_n5': count,
                  s + '_cos_last1': cos(fl, g1), s + '_cos_mean5': cos(fl, g5), s + '_cos_nmean_last1': cos(fm, g1), s + '_cos_nmean_mean5': cos(fm, g5),
                  s + '_cand_feat_age': (n['t'] - last['t']) if last is not None else NAN,
                  s + '_cand_last1_h': last['crop_h'] if last is not None else NAN,
                  s + '_cand_last1_miou': last['miou'] if last is not None else NAN, s + '_cand_mean5_miou_max': miou_max,
                  s + '_new_feat_h': n['crop_h'] if fl is not None else NAN, s + '_new_feat_miou': n['miou'] if fl is not None else NAN})
        if s == 'm1':
            clean_count, _, clean_mean, _ = clean
            f.update(m1_cand_nclean=clean_count, m1_cos_clean5=cos(fl, clean_mean),
                     m1_new_clean=int(fl is not None and n['crop_h'] >= CLEAN_H and n['miou'] == 0.0))
    return f


def _other_extreme(values, choose):
    indices = [i for i, value in enumerate(values) if value == value]
    if not indices:
        return [NAN] * len(values)
    best = choose(indices, key=values.__getitem__)
    second = choose((i for i in indices if i != best), key=values.__getitem__, default=None)
    return [values[best] if i != best else values[second] if second is not None else NAN
            for i in range(len(values))]


def group_features(fs):
    """the per-birth columns of build_sequence: rank_d, rank_cv, d_cv_h_min_other, d_last_h_min_other, {m1,fp}_cos_mean5_max_other (in place)"""
    for key, rk in (('d_last_h', 'rank_d'), ('d_cv_h', 'rank_cv')):
        order = sorted(range(len(fs)), key=lambda i: (fs[i][key], fs[i]['cand_tid']))
        for r_, i in enumerate(order): fs[i][rk] = r_ + 1
    for key, choose in (('d_cv_h', min), ('d_last_h', min), ('m1_cos_mean5', max), ('fp_cos_mean5', max)):
        values = [f[key] for f in fs]
        if choose is min and any(value != value for value in values):
            # Python min retains NaN if it is the first remaining value.
            result = [min((v for j, v in enumerate(values) if j != i), default=NAN)
                      for i in range(len(values))]
        else:
            result = _other_extreme(values, choose)
        for f, value in zip(fs, result):
            f[key + ('_min_other' if choose is min else '_max_other')] = value


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


def mutual_best_pairs(ranked_pairs):
    """Keep only unique first choices on both sides of the eligible candidate list."""
    best_birth, best_owner = {}, {}
    tied_birth, tied_owner = set(), set()
    for pair in ranked_pairs:
        score, birth, owner = pair
        if birth not in best_birth:
            best_birth[birth] = pair
        elif score == best_birth[birth][0]:
            tied_birth.add(birth)
        if owner not in best_owner:
            best_owner[owner] = pair
        elif score == best_owner[owner][0]:
            tied_owner.add(owner)
    return [pair for pair in ranked_pairs
            if pair[1] not in tied_birth and pair[2] not in tied_owner
            and best_birth[pair[1]] == pair and best_owner[pair[2]] == pair]


# ---------------------------------------------------------------- the online judge
class OnlineJudge:
    """Drop-in for relabel_online.OnlineRelabel (same step / reset / stats / events / per_frame interface) with the learned judge.
    crops = per-frame crop budget on NPU core 2 (0 = unlimited: every eligible crop every frame; the PC replay uses it)."""
    def __init__(self, spec_path, crops=2, *, joint_candidates=False, compact_ids=False):
        self.spec, self.spec_sha = load_spec(spec_path)
        self.M, self.feats, self.thr = self.spec['model'], list(self.spec['features']), float(self.spec['decision']['threshold'])
        self.crops = int(crops)
        self.joint_candidates = joint_candidates
        self.compact_ids = bool(compact_ids)
        self._history_type = _LiveObservationHistory if compact_ids else list
        self.stats = dict(new_ids=0, births=0, births_with_candidates=0, pairs_scored=0, relabelled=0, retired_reappearances=0, frames=0,
                          crops_new=0, crops_refresh=0, crops_embedded=0, crops_new_births_with_candidates=0, births_with_candidates_no_feature=0,
                          decided_with_feature=0)
        if joint_candidates:
            self.stats['joint_withheld_births'] = 0
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
        self.next_public_id = 1
        self.public_identity_token = object() if self.compact_ids else None
        self.hist = {}            # tracker id -> observation history, preserving raw IDs after disappearance
        self.current_raw_ids = ()
        self.follow_transfers = []
        self.id_map = {}          # tracker id -> output id (popped when retired)
        self.retired = set()      # tracker ids handed over; a fresh output id if output again
        self.last_emb = {}        # tracker id -> t of its last feature (refresh priority)

    def _allocate_compact_public_id(self):
        """Allocate a fresh public ID once its first publication is decided."""
        public = self.next_public_id
        self.next_public_id += 1
        return public

    def confirm_follow_transfer(self, old_public, new_public, active_raw, *, namespace, expected_t):
        """Transfer only a uniquely owned, absent public ID in this exact judge frame."""
        active = tuple(int(raw) for raw in active_raw)
        if (namespace is not self.id_map or expected_t != self.t or self.t < 0
                or old_public == new_public or old_public <= 0 or new_public <= 0
                or len(active) != len(set(active)) or set(active) != set(self.current_raw_ids)):
            return None
        public_now = [self.id_map.get(raw) for raw in active]
        if (None in public_now or len(public_now) != len(set(public_now))
                or old_public in public_now or public_now.count(new_public) != 1):
            return None
        old_owners = [raw for raw, public in self.id_map.items() if public == old_public]
        new_owners = [raw for raw, public in self.id_map.items() if public == new_public]
        if len(old_owners) != 1 or len(new_owners) != 1:
            return None
        old_raw, new_raw = old_owners[0], new_owners[0]
        if (old_raw in active or new_raw not in active or old_raw in self.retired
                or new_raw in self.retired or not self.hist.get(old_raw) or not self.hist.get(new_raw)
                or self.hist[old_raw][-1]['t'] >= self.t or self.hist[new_raw][-1]['t'] != self.t):
            return None
        self.id_map[new_raw] = old_public
        self.id_map.pop(old_raw)
        self.retired.add(old_raw)
        self.follow_transfers.append(dict(t=self.t, old_public=old_public, new_public=new_public,
                                         old_raw=old_raw, new_raw=new_raw))
        return new_raw

    def _elig(self, prow, P):
        """tracker id -> detection k of the rows that can be cropped (k in P, detection height >= 32 px, any overlap)"""
        e = {}
        for r in prow:
            k = int(r[7])
            if k >= 0 and k in P and P[k][3] - P[k][1] >= MIN_H: e[int(r[4])] = k
        return e

    @staticmethod
    def trace_supported():
        return True

    def step(self, rows, P, embed, frame=None, embed_async=None, *, trace_candidates=None):
        """rows: this frame's tracker rows [x1, y1, x2, y2, id, conf, cls, k, ...]; P: {k: [x1, y1, x2, y2]} person detections (conf >= 0.1)
        keyed like the rows' k; embed(list of k) -> {k: L2-normalised feature}; embed_async(list of k) -> collect() -> {k: feature}.
        Returns the rows (same order, same boxes) with relabelled person IDs."""
        self.t += 1; t = self.t; st = self.stats; st['frames'] += 1
        if self.deferred is not None:                                     # last frame's asynchronous refresh joins its observations now
            recs_d, collect_d = self.deferred; self.deferred = None; got_d = collect_d() or {}
            for o_, k_ in recs_d:
                if k_ in got_d:
                    o_['m1'] = got_d[k_]
                    history = self.hist[o_['tid']]
                    if isinstance(history, _LiveObservationHistory):
                        history.add_feature(o_, 'm1')
                    self.last_emb[o_['tid']] = o_['t']
                    st['crops_embedded'] += 1
        prow = [r for r in rows if int(r[6]) == 0]
        # rows rounded like TRACKS_BASE.jsonl, so the features equal an offline rebuild from the saved rows
        prow_r = [[round(float(v), 2) for v in r[:4]] + [int(r[4]), round(float(r[5]), 5), int(r[6]), int(r[7])] for r in prow]
        now = [int(r[4]) for r in prow_r]; now_set = set(now)
        self.current_raw_ids = tuple(now)
        births = [b for b in now if b not in self.hist]
        # candidates of every birth (frozen at t0): tracker ids output before, not output now, lost <= WINDOW frames
        cands = {}
        velocities = {}
        if births:
            cl = [(c, co) for c, co in self.hist.items() if c not in now_set and t - co[-1]['t'] <= WINDOW]
            for b in births: cands[b] = cl
            if self.pred_gate:   # 11-14: candidate kept only if the birth centre lies in its growing search range around the predicted place
                from .relabel_online import SR_BASE, SR_GROW, SR_MAX, SR_VMAX
                rb = {int(r[4]): r for r in prow_r}
                velocities = {c: cv_velocity(co) for c, co in cl}
                for b in births:
                    bx, by = (rb[b][0] + rb[b][2]) / 2, (rb[b][1] + rb[b][3]) / 2; keep = []
                    for c, co in cl:
                        last = co[-1]; hl = max(last['h'], 1e-6); gap = t - last['t']; vx, vy, _ = velocities[c]; sp = math.hypot(vx, vy)
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
        obs = make_observations(t, prow_r, P, feat)
        # ---- a compact ID is assigned after relink, so an unpublished birth that
        # keeps the old ID does not consume another number.
        pending_public = set()
        for b in now:
            if b in self.retired:
                if self.compact_ids:
                    pending_public.add(b)
                else:
                    self.id_map[b] = self.fresh; self.fresh += 1
                self.retired.discard(b); st['retired_reappearances'] += 1
            elif b not in self.id_map:
                if self.compact_ids:
                    pending_public.add(b)
                else:
                    self.id_map[b] = b
                st['new_ids'] += 1
        out_ids = {self.id_map[x] for x in now if x in self.id_map}
        # ---- judge every birth with candidates (K = 0: features at t_now = t0, nobs = [the birth observation])
        st['births'] += len(births); pairs = []
        summaries = {}
        for b in births:
            cl = cands.get(b)
            if not cl: continue
            st['births_with_candidates'] += 1
            if obs[b]['m1'] is None: st['births_with_candidates_no_feature'] += 1
            else: st['decided_with_feature'] += 1
            t1 = time.perf_counter_ns(); fs = []
            birth = _birth_summary([obs[b]])
            for c, cb in cl:
                if c not in summaries:
                    summaries[c] = _candidate_summary(cb)
                if c not in velocities:
                    velocities[c] = cv_velocity(cb)
                f = pair_features([obs[b]], cb, t, t, prow_r, False,
                                  candidate=summaries[c], birth=birth, velocity=velocities[c])
                f.update(cand_tid=c, n_cand=len(cl), n_births_frame=len(births), n_out_frame=len(now_set)); fs.append(f)
            group_features(fs)
            X = np.asarray([[float(f[c_]) for c_ in self.feats] for f in fs], np.float64)
            t2 = time.perf_counter_ns(); p = predict_json(self.M, X); t3 = time.perf_counter_ns()
            self.feat_ms.append((t2 - t1) / 1e6); self.model_ms.append((t3 - t2) / 1e6); self.cands_per_eval.append(len(cl)); st['pairs_scored'] += len(cl)
            if self.keep_rows: self.rows_log.append((frame, t, b, [c for c, _ in cl], X, p))
            for (c, _), pv in zip(cl, p):
                pv = float(pv)
                eligible = pv >= self.thr and c != b and c in self.id_map and c not in now_set and self.id_map[c] not in out_ids
                if eligible:
                    pairs.append((pv, b, c))
                if trace_candidates is not None and len(trace_candidates) < 17:
                    # Reasons describe eligibility before greedy assignment.
                    why = (0 if eligible else 1 if not pv >= self.thr else 2 if c == b else
                           3 if c not in self.id_map else 4)
                    q = round(pv * 10000) if math.isfinite(pv) and 0 <= pv <= 1 else 0
                    trace_candidates.append([b, c, self.id_map.get(c, 0), q, why, False])
        ranked_pairs = sorted(pairs, key=lambda x: (-x[0], x[1], x[2]))
        if self.joint_candidates:
            # Compare against all eligible choices before transferring any ID.
            # An ambiguous birth is still published with its own number.
            chosen = mutual_best_pairs(ranked_pairs)
            st['joint_withheld_births'] += len({b for _, b, _ in ranked_pairs}) - len(chosen)
            ranked_pairs = chosen
        ub, uc = set(), set()
        chosen_trace = set() if trace_candidates is not None else None
        for pv, b, c in ranked_pairs:
            if b in ub or c in uc or c not in self.id_map or (b not in self.id_map and b not in pending_public): continue
            ub.add(b); uc.add(c); oid = self.id_map[c]; old_new = self.id_map.get(b)
            self.id_map[b] = oid; self.retired.add(c); self.id_map.pop(c, None); st['relabelled'] += 1
            self.events.append(dict(frame=frame, t=t, tracker_id=b, new_id=old_new, old_id=oid, cand_tid=c, p=round(pv, 4), n_cand=len(cands[b])))
            if chosen_trace is not None:
                chosen_trace.add((b, c))
        if chosen_trace is not None:
            for row in trace_candidates:
                row[5] = (row[0], row[1]) in chosen_trace
        for b in now:
            if b in pending_public and b not in self.id_map:
                self.id_map[b] = self._allocate_compact_public_id()
        # ---- histories + deferred refresh
        for b, o in obs.items():
            if b not in self.hist:
                self.hist[b] = self._history_type()
            self.hist[b].append(o)
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
