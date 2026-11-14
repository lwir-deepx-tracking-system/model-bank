"""Train the spatial-spectrum people-proposal model used by src/detect/spec_proposals.py (same feature code).

Usage: python train_spec_proposals.py <out model json> <GT MEASURE dir> ... --empty <empty-scene dir> ...
  GT MEASURE dir: frames.raw.gz + frames.json (16-bit), gt_mot.txt (hand GT, frame 1 = raw 0), optional labelled_frames.txt
  empty-scene dir: frames.raw.gz + frames.json of frames with nobody in them
Samples (every 5th labelled frame): person = central square of each GT box; not person = warm parts outside the GT boxes
(at least as warm as the 25th percentile of the GT pixels, area >= 600 px) and random windows away from the GT boxes.
Model: L2 logistic regression on 128 spectrum bins + warmth, classes balanced.
Decision: the best window's raw model score (absolute). (Within-frame relative scores failed: with many people in view the
median window score rises.) Threshold = 95th percentile of the best-window score over the empty-scene frames, so at most
5 % of empty frames get a proposal.
Held-out report: leave one GT recording out; the fold's own model, threshold set again from the empty scenes; per held-out
recording (every 6th labelled frame): best window on a person and passing, some person window passing, some window off
every person passing. The final model is fitted on all GT recordings."""
import gzip
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.detect.spec_proposals import features, windows   # noqa: E402

args = sys.argv[1:]
out_path = args[0]
ei = args.index('--empty')
dirs, empty_dirs = args[1:ei], args[ei + 1:]
SIZES, STRIDE, STEP_TR, STEP_TE, MINAREA = (64, 96, 128, 160), 32, 5, 6, 600
rng = np.random.default_rng(0)


def load_raw(d):
    meta = json.load(open(os.path.join(d, 'frames.json'), encoding='utf-8'))
    return np.frombuffer(gzip.open(os.path.join(d, 'frames.raw.gz')).read(), dtype='<u2').reshape(meta['frames'], meta['height'], meta['width'])


def load(d):
    raw = load_raw(d)
    gt = {}
    for l in open(os.path.join(d, 'gt_mot.txt')):
        v = l.strip().split(',')
        if len(v) > 5:
            x, y, w, h = [float(t) for t in v[2:6]]
            gt.setdefault(int(float(v[0])) - 1, []).append((int(max(0, x)), int(max(0, y)), int(min(640, x + w)), int(min(480, y + h))))
    lf = os.path.join(d, 'labelled_frames.txt')
    lab = sorted({int(t) - 1 for t in open(lf).read().split()}) if os.path.exists(lf) else sorted(gt)
    return raw, gt, [f for f in lab if f < len(raw)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def sq(x0, y0, x1, y1):
    s = min(x1 - x0, y1 - y0); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return (cx - s // 2, cy - s // 2, cx - s // 2 + s, cy - s // 2 + s)


def samples(raw, gt, frames):
    X, y = [], []
    for f in frames[::STEP_TR]:
        if f not in gt:
            continue
        im = raw[f]; H, W = im.shape
        pm = np.zeros((H, W), bool)
        for b in gt[f]:
            pm[b[1]:b[3], b[0]:b[2]] = True
        pos = [sq(*b) for b in gt[f] if min(b[2] - b[0], b[3] - b[1]) >= 32]
        neg = []
        lab, nl = ndimage.label((im >= np.percentile(im[pm], 25)) & ~ndimage.binary_dilation(pm, iterations=6))
        for sl, i in zip(ndimage.find_objects(lab), range(1, nl + 1)):
            if (lab[sl] == i).sum() >= MINAREA:
                s = sq(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
                if s[2] - s[0] >= 32:
                    neg.append(s)
        for _ in range(6):
            z = int(rng.choice(SIZES)); x0, y0 = int(rng.integers(0, W - z)), int(rng.integers(0, H - z))
            if all(iou((x0, y0, x0 + z, y0 + z), b) < 0.05 for b in gt[f]):
                neg.append((x0, y0, x0 + z, y0 + z))
        if pos + neg:
            X.append(features(im, pos + neg)); y += [1] * len(pos) + [0] * len(neg)
    return np.concatenate(X), np.array(y)


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
    yt = y.astype(float); wv = (1 - yt) * (yt.mean() / (1 - yt.mean())) + yt; lam = 10.0 * Z.shape[1] / 100
    fn = lambda wb: np.sum(wv * (np.logaddexp(0, Z @ wb[1:] + wb[0]) - yt * (Z @ wb[1:] + wb[0]))) + lam * np.sum(wb[1:] ** 2)
    gr = lambda wb: (lambda p: np.r_[np.sum(wv * (p - yt)), Z.T @ (wv * (p - yt)) + 2 * lam * wb[1:]])(1 / (1 + np.exp(-(Z @ wb[1:] + wb[0]))))
    wb = minimize(fn, np.zeros(Z.shape[1] + 1), jac=gr, method='L-BFGS-B', options=dict(maxiter=800)).x
    return mu, sd, wb[1:], wb[0]


def score(F, mu, sd, w, b):
    return ((F - mu) / sd) @ w + b


data = {os.path.basename(d.rstrip('/\\')): load(d) for d in dirs}
S = {k: samples(*v) for k, v in data.items()}
print({k: (int(v[1].sum()), int((1 - v[1]).sum())) for k, v in S.items()}, flush=True)
WINS = windows(480, 640, SIZES, STRIDE)
EF = []
for d in empty_dirs:
    raw = load_raw(d)
    EF += [features(raw[f], WINS) for f in range(0, len(raw), 3)]
print('empty frames', len(EF), flush=True)
rep = {}
for test in data:
    X = np.concatenate([S[k][0] for k in S if k != test]); y = np.concatenate([S[k][1] for k in S if k != test])
    mu, sd, w, b = fit(X, y)
    thr_f = float(np.quantile([score(F, mu, sd, w, b).max() for F in EF], 0.95))
    raw, gt, frames = data[test]
    R = []
    for f in frames[::STEP_TE]:
        if f not in gt:
            continue
        s_ = score(features(raw[f], WINS), mu, sd, w, b)
        on = np.array([any(p[0] <= (q[0] + q[2]) / 2 <= p[2] and p[1] <= (q[1] + q[3]) / 2 <= p[3] for p in gt[f]) for q in WINS])
        off = np.array([all(iou(q, p) < 0.05 for p in gt[f]) for q in WINS])
        i = int(np.argmax(s_))
        R.append(dict(a=bool(on[i] and s_[i] >= thr_f), b=bool(on.any() and s_[on].max() >= thr_f), c=bool(off.any() and s_[off].max() >= thr_f)))
    rep[test] = dict(frames=len(R), threshold_this_fold=round(thr_f, 3),
                     best_window_on_person_and_proposed=round(float(np.mean([r['a'] for r in R])), 3),
                     some_person_window_passes=round(float(np.mean([r['b'] for r in R])), 3),
                     off_person_window_passes=round(float(np.mean([r['c'] for r in R])), 3))
    print(test, json.dumps(rep[test]), flush=True)
X = np.concatenate([S[k][0] for k in S]); y = np.concatenate([S[k][1] for k in S])
mu, sd, w, b = fit(X, y)
eb = np.array([score(F, mu, sd, w, b).max() for F in EF])
thr = float(np.quantile(eb, 0.95))
model = dict(decision='absolute',
             note='spatial-spectrum people proposals; features: src/detect/spec_proposals.py features() (128 log spectrum shares in 16 rings x 8 orientations of the window resized to 64x64, + warmth z)',
             trained_on=list(data), samples=dict(person=int(y.sum()), not_person=int((1 - y).sum())),
             mu=[round(float(v), 6) for v in mu], sd=[round(float(v), 6) for v in sd], w=[round(float(v), 6) for v in w], b=round(float(b), 6),
             threshold=round(thr, 4), threshold_rule='95th percentile of the best-window score over %d empty-scene frames (<= 5 %% of empty frames get a proposal)' % len(eb),
             empty_best_quantiles={str(q): round(float(np.quantile(eb, q)), 3) for q in (0.5, 0.9, 0.95, 0.99)},
             sizes=list(SIZES), stride=STRIDE, top=1, held_out=rep)
json.dump(model, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('FINAL', json.dumps(dict(threshold=model['threshold'], empty=model['empty_best_quantiles']), ensure_ascii=False))
