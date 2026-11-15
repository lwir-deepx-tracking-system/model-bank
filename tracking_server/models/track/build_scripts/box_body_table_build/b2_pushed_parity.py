"""body_v2/scripts/b2_pushed_parity.py - the 'body_pushed' emulation of b2_eval (current server code: Refiner('body', T1, anchor top, 0.4 Hz beta 0,
no switch transition, family pose)) equals the origin/main 3810b1f server code (Refiner('body', G2Model(box_refine_gru_v2 file))) on val-B streams.
Usage: python b2_pushed_parity.py <copy of origin/main tracking_server/src as package src_om (its parent dir)> <origin/main models/track dir> <out.json> [groups]"""
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from src_om.track import box_refine as OM   # noqa: E402
import b2_eval as E                          # noqa: E402
from b2_lib import BR                        # noqa: E402


def main():
    z = np.load(sys.argv[2] + '/box_refine_gru_v2_tmotdance_20261115.npz', allow_pickle=False); om_model = OM.G2Model({k: z[k] for k in z.files})
    t1, _ = E.tables(); gr = E.groups('valB'); ng = int(sys.argv[4]) if len(sys.argv) > 4 else 6; res = []; worst = 0.0; rows = 0
    for gp in gr[:ng // 2] + gr[-(ng - ng // 2):]:
        a = OM.Refiner('body', om_model); b = BR.Refiner('body', t1, 0.0, None, 0.5, dict(anchor='top', mc_s=0.4, beta_s=0.0, dcut=1.0, sw=False)); b.family_of = lambda k: 'pose'
        for f in range(gp['n']):
            its = gp['by_f'].get(f, ()); H = gp['W'][f] if 0 < f < len(gp['W']) else None; t = f / gp['fps']
            ra = a.step([(tid, zz, None, zr) for tid, j, zz, zr in its], t, H, gp['keys'][f], True); rb = b.step([(tid, zz, zr) for tid, j, zz, zr in its], t, H, gp['keys'][f], True)
            for tid, *_ in its: worst = max(worst, float(np.abs(ra[tid] - rb[tid]).max())); rows += 1
        res.append(dict(name=gp['name'], stream=gp['stream']))
    out = dict(groups=res, rows=rows, max_abs_px=worst, ok=worst < 1e-6); open(sys.argv[3], 'w').write(json.dumps(out, indent=1)); print('PUSHED_PARITY', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
