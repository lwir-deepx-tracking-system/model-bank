"""gru_v2/scripts/g2_lutfix.py - the lookup conversion without the two wrong large-box rows (by-eye check 2026-11-15, results/lutcheck_train.json).
data/lut_train.pkl rows 'detector box 170-230 px' and '>= 230 px' shrink the box to 0.86 / 0.64 of its size. They were fitted on 1,659 / 300 DanceTrack
train pairs (no thermal pair that large) whose target = a fixed share of a ground-truth box that is not a whole standing person, while the detector box
there already is the body box: the detectors' own hip keypoints sit at 0.95-0.97 of the detector box height in those rows, as in every other row
(0.93-0.96), but at 1.11 / 1.50 of the target's (below its bottom edge). In the row below (130-170 px, 7,470 pairs) target / detector = 1.00 for both
detectors on both datasets. Fix: rows >= 170 px take the 130-170 px row (identity within 0.8 %). Nothing else changes; no weight is retrained.
Writes data/lut_train_fix.pkl (g2_lib reads this one)."""
import pickle

import numpy as np

from pathlib import Path

D = Path(__file__).resolve().parent.parent / 'data'
L = pickle.loads((D / 'lut_train.pkl').read_bytes()); tab = L['tab'].copy(); edges = np.exp(L['lh_edges']); rows = [i for i in range(tab.shape[0]) if edges[i] >= 170.]
src = min(rows) - 1; assert abs(edges[src] - 130.) < 1e-6 and rows == [9, 10]
print('rows fixed', rows, 'from row', src, 'old h ratio', np.exp(tab[rows, 5, 3]).round(3), 'new', np.exp(tab[src, 5, 3]).round(4), 'max |row %d|' % src, np.abs(tab[src, 4:]).max().round(4), flush=True)
for i in rows: tab[i] = tab[src, 5]      # the one populated aspect cell (0.55-0.65) of that row, as the old rows were filled
(D / 'lut_train_fix.pkl').write_bytes(pickle.dumps(dict(L, tab=tab, fix='rows >= 170 px = row 130-170 px (g2_lutfix.py)'), protocol=4)); print('LUTFIX_DONE', flush=True)
