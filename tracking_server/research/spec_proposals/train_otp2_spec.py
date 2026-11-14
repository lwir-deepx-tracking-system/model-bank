"""Train the spatial-spectrum person model on the public close-range set OpenThermalPose2 only (train split, every 2nd image;
samples as spec_otp2_eval.py) and write it in the src/detect/spec_proposals.py model format, so it can score OUR recordings
that it never saw. Two models: spectrum only (warmth weight 0) and spectrum + warmth.
Threshold = 95th percentile of the best window score over our empty-scene frames (empty/a, empty/b; every 3rd frame), as the
deployed model's rule but without any of our GT recordings.
usage: train_otp2_spec.py <tracking_server dir> <otp2_dataset dir> <empty dir> ... --out <dir>"""
import gzip
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, sys.argv[1])
from src.detect import spec_proposals as sp   # noqa: E402

args = sys.argv[1:]
oi = args.index('--out'); out_dir = args[oi + 1]; args = args[:oi]
TS, ROOT, EMPTY = args[0], args[1], args[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_otp2_eval_lib as lib   # noqa: E402
X, y, kind, n = lib.split(ROOT, 'train', 2)
print('train images', n, 'samples', int(y.sum()), int((1 - y).sum()), flush=True)
WINS = sp.windows(480, 640, (64, 96, 128, 160), 32)
EF = []
for d in EMPTY:
    meta = json.load(open(os.path.join(d, 'frames.json')))
    raw = np.frombuffer(gzip.open(os.path.join(d, 'frames.raw.gz')).read(), dtype='<u2').reshape(meta['frames'], meta['height'], meta['width'])
    EF += [sp.features(raw[f], WINS) for f in range(0, len(raw), 3)]
os.makedirs(out_dir, exist_ok=True)
for name, cols in (('otp2_spectrum', list(range(128))), ('otp2_spectrum_warmth', list(range(129)))):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000)).fit(X[:, cols], y)
    sc, lr = m[0], m[1]
    mu, sd, w = np.zeros(129), np.ones(129), np.zeros(129)
    mu[cols], sd[cols], w[cols] = sc.mean_, sc.scale_, lr.coef_[0]
    b = float(lr.intercept_[0])
    best = np.array([(((F - mu) / sd) @ w + b).max() for F in EF])
    model = dict(decision='absolute', note='trained on OpenThermalPose2 train only (%s); features src/detect/spec_proposals.py features()' % name,
                 trained_on=['OpenThermalPose2 train (every 2nd image)'], samples=dict(person=int(y.sum()), not_person=int((1 - y).sum())),
                 mu=mu.round(6).tolist(), sd=sd.round(6).tolist(), w=w.round(6).tolist(), b=round(b, 6),
                 threshold=round(float(np.quantile(best, 0.95)), 4), threshold_rule='95th percentile of the best window over %d of our empty-scene frames' % len(best),
                 sizes=[64, 96, 128, 160], stride=32, top=3)
    json.dump(model, open(os.path.join(out_dir, name + '.json'), 'w'), indent=1)
    print(name, 'threshold', model['threshold'], flush=True)
