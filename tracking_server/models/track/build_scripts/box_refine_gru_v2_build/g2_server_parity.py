"""gru_v2/scripts/g2_server_parity.py - the SERVER port (tracking_server/src/track/box_refine.py, mode gru2) against g2_lib on real rows.
1) every val-B stream (near, far, sw of every sequence) and the 3 device-rate recordings: Refiner('gru2') frame by frame vs g2_lib.run_rows ('out_f'),
   with the hand-over switched off (g2_lib has none) -> max |difference| px; then with the hand-over on -> how many hand-overs and rows it changes.
2) a small fixture for the server's unit test: two val-B windows (detector boxes + keypoints, rounded to float32, no ground truth) with g2_lib's output.
Usage: python g2_server_parity.py <tracking_server dir> <fixture.npz> [device rec ...]
"""
import json
import pickle
import sys
from collections import defaultdict

import numpy as np

import g2_lib as L

sys.path.insert(0, sys.argv[1])
from src.track import box_refine as BR   # noqa: E402


def server_rows(model, ids, W, n, times, fps, handover):
    by_f = defaultdict(list)
    for tid in sorted(ids):
        for j, f in enumerate(ids[tid]['f']): by_f[int(f)].append((tid, j))
    BR.HANDOVER_IOU = 0.5 if handover else 2.0; r = BR.Refiner('gru2', model); o = {tid: np.zeros((len(e['f']), 4)) for tid, e in ids.items()}
    for f in range(n):
        its = by_f.get(f, ()); H = W[f] if (W is not None and 0 < f < len(W)) else None; t = float(times[f]) if times is not None else f / fps
        res = r.step([(a, ids[a]['z'][j], ids[a]['kp'][j]) for a, j in its], t, H)
        for a, j in its: o[a][j] = res[a]
    BR.HANDOVER_IOU = 0.5; return o, r.handovers


def main():
    exp = json.loads((L.G2 / 'export' / 'EXPORT.json').read_text()); wp = L.G2 / 'export' / exp['npz']; net = L.load_net(wp); z = np.load(wp, allow_pickle=False)
    model = BR.G2Model({k: z[k] for k in z.files}); size = model.size
    assert np.abs(model.tab - L.TAB).max() == 0 and np.abs(model.lh_cen - L.LH_CEN).max() == 0, 'the exported table is not the one g2_lib reads'
    d = pickle.loads((L.G2 / 'data' / 'rows_valB.pkl').read_bytes()); recs, Wd, nd = d['recs'], d['W'], d['n']; groups = defaultdict(list)
    for r in recs: groups[(r['name'], r['stream'])].append(r)
    worst = 0.; rows = 0; hand = 0; changed = 0; worst_h = 0.; fix = {}
    cases = [(k, {r['id']: dict(f=r['e']['f'], z=r['e']['z'], kp=r['e']['kp']) for r in rr}, Wd[k[0]], nd[k[0]], None, rr[0]['fps']) for k, rr in sorted(groups.items())]
    if len(sys.argv) > 3:
        import g2_device as GD
        for ri, rec in enumerate(sys.argv[3:]):
            S, _ = GD.streams(rec, ri)
            for st, inp in S.items(): cases.append(((rec, st), {i: dict(f=e['f'], z=e['z'], kp=e['kp']) for i, e in inp['ids'].items()}, inp['W'], inp['n'], inp['times'], inp['fps']))
    for key, ids, W, n, times, fps in cases:
        ref = L.run_rows(ids, W, fps, n, net, size, times); o, _ = server_rows(model, ids, W, n, times, fps, False); o2, nh = server_rows(model, ids, W, n, times, fps, True); m = 0.
        for tid in ids:
            dd = np.abs(o[tid] - ref[tid]['out_f']).max() if len(o[tid]) else 0.; m = max(m, float(dd)); rows += len(o[tid])
            d2 = np.abs(o2[tid] - ref[tid]['out_f']).max(1) if len(o2[tid]) else np.zeros(0); changed += int((d2 > 1e-6).sum()); worst_h = max(worst_h, float(d2.max()) if len(d2) else 0.)
        worst = max(worst, m); hand += nh
        print('PARITY %-28s %-4s ids %4d  max |server - g2_lib| %.3g px   hand-overs %d' % (key[0], key[1], len(ids), m, nh), flush=True)
    # fixture: two windows, inputs rounded to float32 first, expected from g2_lib on exactly those inputs
    def busiest(dance, stream, length):   # the window of that many frames with the most rows (thermal: also track starts and gaps inside)
        best = None
        for k in sorted(groups):
            if k[0].startswith('dance') != dance or k[1] != stream: continue
            cnt = np.zeros(nd[k[0]] + 1)
            for r in groups[k]: np.add.at(cnt, r['e']['f'], 1)
            c = np.convolve(cnt, np.ones(length), 'valid'); f0 = int(np.argmax(c))
            if best is None or c[f0] > best[0]: best = (c[f0], k, f0)
        return best[1], best[2], best[2] + length

    for name, (key, f0, f1) in (('a', busiest(False, 'sw', 150)), ('b', busiest(True, 'near', 40))):
        rr = groups[key]; ids = {}
        for r in rr:
            mm = (r['e']['f'] >= f0) & (r['e']['f'] < f1)
            if mm.any(): ids[len(ids) + 1] = dict(f=(r['e']['f'][mm] - f0).astype(np.int64), z=r['e']['z'][mm].astype(np.float32).astype(np.float64), kp=np.round(r['e']['kp'][mm], 2).astype(np.float32).astype(np.float64))
        W = np.asarray(Wd[key[0]][f0:f1], np.float32).astype(np.float64); W[0] = np.eye(2, 3); n = f1 - f0; fps = rr[0]['fps']; ref = L.run_rows(ids, W, fps, n, net, size, None)
        fix[name + '_id'] = np.concatenate([np.full(len(e['f']), i, np.int32) for i, e in ids.items()]); fix[name + '_f'] = np.concatenate([e['f'] for e in ids.values()]).astype(np.int32)
        fix[name + '_z'] = np.concatenate([e['z'] for e in ids.values()]).astype(np.float32); fix[name + '_kp'] = np.concatenate([e['kp'] for e in ids.values()]).astype(np.float32)
        fix[name + '_W'] = W.astype(np.float32); fix[name + '_fps'] = np.float64(fps); fix[name + '_out'] = np.concatenate([ref[i]['out_f'] for i in ids]); fix[name + '_start'] = np.concatenate([ref[i]['start'] for i in ids])
        print('FIXTURE %s: %s rows %d ids %d frames %d starts %d' % (name, 'thermal switch stream' if name == 'a' else 'dance near stream', len(fix[name + '_f']), len(ids), n, int(fix[name + '_start'].sum())), flush=True)
    fix['weights_sha256'] = np.array(exp['sha256']); np.savez_compressed(sys.argv[2], **fix)
    res = dict(weights=exp['npz'], sha256=exp['sha256'], streams=len(cases), rows=rows, max_abs_px_no_handover=worst, handovers=hand, rows_changed_by_handover=changed, max_abs_px_with_handover=worst_h)
    (L.G2 / 'results' / 'server_parity.json').write_text(json.dumps(res, indent=1)); print('SERVER_PARITY_DONE', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
