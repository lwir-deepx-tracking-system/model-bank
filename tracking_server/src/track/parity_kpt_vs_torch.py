"""11-14 one-off check (needs torch + the motion study's m4_model.py and close_range m5_kpt_motion.py on sys.path): the server's numpy keypoint position memory
(pos_predict with PosKpt) gives the same lost-frame boxes as the torch model it was exported from. Run from tracking_server/:
python -m src.track.parity_kpt_vs_torch <pos_*_kpt_*.npz> <the torch _best.pt> <folder with m4_model.py> <folder with m5_kpt_motion.py>"""
import sys
from types import SimpleNamespace
import numpy as np


def main():
    npz, pt, m4dir, m5dir = sys.argv[1:5]; sys.path[:0] = [m4dir, m5dir]
    import torch, m5_kpt_motion as M5
    from . import position_memory as PM
    from ..detect import detect_frame as DF
    from ..common.shared_values import TRACKED
    z = np.load(npz); model = PM.PosKpt({k: z[k] for k in z.files})
    ck = torch.load(pt, map_location='cpu'); arm = ck['arm']; net = M5.Net(arm, ck['nf'], ck.get('hidden', 64)).float(); net.load_state_dict(ck['model']); net.eval()
    PM.POSP.update(key=arm, model=model, gen=PM.POSP['gen'] + 1, err=0, used=0, show=True)

    class KF:
        warp = None
        def multi_predict(self, mean, cov):
            mean = mean.copy(); mean[:, :4] += mean[:, 4:8]; return mean, cov

    rng = np.random.default_rng(0); worst = 0.
    for trial in range(20):
        t = SimpleNamespace(state=TRACKED, cls=0, track_id=1, pp_inn=rng.normal(0, .05, 4)); kf = KF()
        mean = np.array([[rng.uniform(150, 500), rng.uniform(150, 350), rng.uniform(30, 90), rng.uniform(90, 250), rng.normal(0, 3), rng.normal(0, 1), rng.normal(0, .3), rng.normal(0, .3)]]); cov = np.zeros((1, 8, 8))
        nseen = int(rng.integers(5, 25))
        for _ in range(nseen):
            b = mean[0, :4] + rng.normal(0, 2, 4); x1, y1 = b[0] - b[2] / 2, b[1] - b[3] / 2
            kp = np.c_[rng.uniform(x1, x1 + b[2], 17), rng.uniform(y1, y1 + b[3], 17), rng.uniform(0, 1, 17)].astype(np.float32)
            DF.KP['prev'] = dict(boxes=np.array([[x1, y1, x1 + b[2], y1 + b[3]]], np.float32), kpts=kp[None]); t.pp_inn = rng.normal(0, .05, 4)
            mean, cov = PM.pos_predict(kf, [t], mean, cov)
            mean[0, :4] += rng.normal(0, 1, 4)   # the tracker's update moves the state
        p = t.pk; post = p['post']; zl = p['z']; hist = np.stack(list(p['hist']))
        t.state = 2; got = []
        for s in range(PM.KP_STEPS - 1):
            mean, cov = PM.pos_predict(kf, [t], mean, cov); got.append(mean[0, :4].copy())
        # torch: same history, plain Kalman path from the last post state, frame y -> picture y (rows 16..496 of the canvas = 0..480)
        base = np.zeros((M5.STEPS, 4)); c = post[:2].copy(); wh = post[2:4].copy()
        for s in range(M5.STEPS):
            c = c + post[4:6]; wh = wh + (post[6:8] if s == 0 else 0); base[s] = np.r_[c, wh]
        sh = np.array([0, 16., 0, 0])
        hx = np.zeros((1, M5.HIST, ck['nf'])); hm = np.zeros((1, M5.HIST)); hx[0, M5.HIST - len(hist):] = hist; hm[0, M5.HIST - len(hist):] = 1
        T = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None])
        with torch.no_grad():
            q = net(torch.as_tensor(hx, dtype=torch.float32), torch.as_tensor(hm, dtype=torch.float32), T(base - sh), T(zl - sh), T(zl - sh), T(post[4:6]), T([640., 480.]))[0].numpy() + sh
        d = np.abs(np.array(got) - q[1:len(got) + 1]).max(); worst = max(worst, d)
    print('server numpy vs torch, %s, 20 tracks x 14 lost frames: max |diff| px = %.2e' % (arm, worst)); assert worst < 0.05, worst   # float32 torch
    print('PARITY_OK')


if __name__ == '__main__':
    main()
