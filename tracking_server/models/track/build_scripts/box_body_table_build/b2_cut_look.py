"""body_v2/scripts/b2_cut_look.py - TRAIN split only: whole-person (GT) boxes cut by the picture bottom against the pose detectors' body boxes.
How many such pairs exist, and what the body box is relative to the visible whole-person box (height share, width share).   Usage: python b2_cut_look.py
"""
import numpy as np

import b2_lib as B
from b2_lib import P


def main():
    d = B.train_pairs(); Z, G, kp = d['Z'], d['G'], d['kp']; hp = B.hip_pos(Z, kp); ds = d['ds']
    bot = np.where(ds == 0, P.Y1, 436.0); gb = G[:, 1] + G[:, 3] / 2; zb = Z[:, 1] + Z[:, 3] / 2
    print('GT bottom max by dataset', gb[ds == 0].max(), gb[ds == 1].max(), 'picture bottom', P.Y1, 436.0)
    cut = gb >= bot - 2.0; res = {}
    for nm, m0 in (('TMOT', ds == 0), ('DT', ds == 1)):
        for cn, mc in (('uncut', ~cut & (gb < bot - 10)), ('cut', cut)):
            m = m0 & mc & (G[:, 3] >= 60)
            if m.sum() < 30: print(nm, cn, 'n', int(m.sum())); continue
            q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 25, 50, 75, 90])]
            hips = np.isfinite(hp[m])
            o = dict(n=int(m.sum()), seqs=int(len(np.unique(d['seq'][m]))), h_over_gt_h=q(Z[m, 3] / G[m, 3]), w_over_gt_w=q(Z[m, 2] / G[m, 2]), h_over_gt_w=q(Z[m, 3] / G[m, 2]), gt_aspect=q(G[m, 2] / G[m, 3]),
                     det_bottom_below_picture=round(float(np.mean(zb[m] > bot[m] + 2)), 3), hips_seen=round(float(hips.mean()), 3), hip_pos=q(hp[m][hips]) if hips.sum() > 10 else None, gt_h=q(G[m, 3]))
            res['%s|%s' % (nm, cn)] = o; print(nm, cn, o)
    (B.B2 / 'results' / 'cut_look.json').write_text(B.jdump(res), encoding='utf-8')


if __name__ == '__main__':
    main()
