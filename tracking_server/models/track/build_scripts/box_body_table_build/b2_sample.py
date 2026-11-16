"""body_v2/scripts/b2_sample.py - frame sample of the TRAIN split for the plain whole-person detector (b2_yolo.py imgs).
TMOT train: every 24th frame, DanceTrack train: every 120th frame (start at half a step). Writes data/train_sample.tsv (name, folder, frame index, path)
and data/train_sample_files.txt.   Usage: python b2_sample.py
"""
import collections
import os

import numpy as np

import b2_lib as B
from b2_lib import P


def main():
    out = []
    for r in [r for r in P.registry() if r['split'] == 'train']:
        d = os.path.join(P.PATHS['tmot_train'], r['folder'], 'img1') if r['ds'] == 'TMOT' else os.path.join(P.PATHS['dance'], 'train', r['folder'], 'img1')
        fs = sorted(os.listdir(d)); z = np.load(B.ROOT / 'data' / 'det' / ('near_%s.npz' % r['folder'])); assert len(fs) == int(z['n_frames']), (r['name'], len(fs), int(z['n_frames']))
        step = 24 if r["ds"] == "TMOT" else 120
        for i in range(step // 2, len(fs), step): out.append((r['name'], r['folder'], i, os.path.join(d, fs[i]).replace('\\', '/')))
    print(len(out), 'frames,', len({o[0] for o in out}), 'sequences', collections.Counter('TMOT' if o[0].startswith('seq') else 'DT' for o in out))
    (B.B2 / 'data' / 'train_sample.tsv').write_text('\n'.join('%s\t%s\t%d\t%s' % o for o in out) + '\n', encoding='utf-8')
    (B.B2 / 'data' / 'train_sample_files.txt').write_text('\n'.join(o[3] for o in out) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
