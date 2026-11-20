"""Per-appearance-model cosine map (2026-11-20, SOLIDER distillation student E).

Some re-ID models put cosines on a different scale than the model the ID rules were fitted on (the learned re-link judge, the KeepID
threshold, the tracker appearance gate, the follow guard). For such a model a fixed table maps its raw cosine c to the scale of
REPVGG_REVIEWED264_20261117: c' = interp(c, x, y), a monotone piecewise-linear quantile map fitted ONCE on the fixed crop set of the
ID-rules audit (same crop pairs for both models: other people in the same frame, the same person 3 s apart, any two crops). No judge
pair and no test data were used to fit it.

Only models listed in TABLES have a map; every other model is untouched (cmap is the identity, the gate threshold stays 0.25).
The map follows the model actually running: activate() is called at start with the start model and by reid_models.reid_swap after a
successful change.

Where it is applied (every place that compares two appearance features of the re-ID model):
  relink_online_judge.cos              (learned judge: every m1/fp cosine feature)
  relabel_online.OnlineRelabel          (KeepID rule threshold)
  bytetrack.py appearance / late merge / lost merge, lostfirst.py, stitch.py, merge_split.py   (Python tracker paths)
  follow_guard.FollowGuard._sims        (follow guard: target and distractor cosines)
  tracker appearance gate (Python botsort.py and the C++ engine, BoT-SORT only): NOT on each cosine; the gate's threshold APP_THR is set
      to the raw-cosine value equivalent to 0.25 on the mapped scale (apply_gate), so the C++ code needs no change. The cost value the gate
      writes stays 0.5 x (1 - raw cosine) in both engines.
e2e_live.py is not covered: it compares the end-to-end model's own embeddings, not the re-ID model's.
"""
import json
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / 'models'
TABLES = {'REPVGG_SOLIDER_DISTILL_E_20261120': 'reid/repvgg_solider_distill_e_20261120_cosmap.json'}
APP_THR_BASE = 0.25          # botsort.BotSortTracker.APP_THR / native_tracker_core.hpp app_thr
_S = dict(key='', x=None, y=None, file='')
_GATED = set()               # id() of trackers whose APP_THR this module changed (so it can be put back)


def load_table(path):
    t = json.loads(Path(path).read_text(encoding='utf-8'))
    x = np.asarray(t['x'], np.float64); y = np.asarray(t['y'], np.float64)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2 or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0):
        raise ValueError('cosine map must be strictly increasing in x and non-decreasing in y: %s' % path)
    return x, y


def activate(key, models_dir=None):
    """the map of model `key` (or none). Raises when a listed table is missing or broken (the caller keeps the old model)."""
    rel = TABLES.get(str(key or ''))
    if rel is None:
        _S.update(key=str(key or ''), x=None, y=None, file='')
        return False
    p = Path(models_dir or MODELS) / rel
    x, y = load_table(p)
    _S.update(key=str(key), x=x, y=y, file=str(p))
    return True


def active():
    return _S['key'] if _S['x'] is not None else ''


def cmap(c):
    """raw cosine of the running model -> cosine on the scale the ID rules were fitted on (identity without a map; None / NaN pass)"""
    if _S['x'] is None or c is None or c != c:
        return c
    return float(np.interp(c, _S['x'], _S['y']))


def inv(c):
    """mapped cosine -> the smallest raw cosine that reaches it (identity without a map)"""
    if _S['x'] is None:
        return c
    x, y = _S['x'], _S['y']
    if c <= y[0]: return float(x[0])
    if c >= y[-1]: return float(x[-1])
    j = int(np.searchsorted(y, c, 'left'))
    if y[j] == c: return float(x[j])
    return float(x[j - 1] + (c - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1]))


def gate_thr(base=APP_THR_BASE):
    """tracker gate 'cosine distance < base' on the mapped scale, written as a limit on the raw cosine distance"""
    return base if _S['x'] is None else 1.0 - inv(1.0 - base)


def apply_gate(tracker):
    """BoT-SORT trackers (Python and C++) carry APP_THR: set it for the running model; a tracker this module changed before gets 0.25
    back when the model has no map. Other trackers are not touched."""
    if tracker is None:
        return tracker
    try:
        has = hasattr(tracker, 'APP_THR')
    except Exception:
        has = False
    if not has:
        return tracker
    if _S['x'] is not None:
        tracker.APP_THR = gate_thr(); _GATED.add(id(tracker))
    elif id(tracker) in _GATED:
        tracker.APP_THR = APP_THR_BASE; _GATED.discard(id(tracker))
    return tracker


def state():
    return dict(model=active(), table=_S['file'], gate_app_thr=gate_thr())
