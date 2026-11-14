"""Does the FastTrackTr NPU-form ID branch carry identity, or only position? (2026-11-14: training id_loss fell to ~0 within one epoch.)
Runs the trained checkpoint frame by frame exactly as the device does (export form NPUNet from ftt_trained_export.py, traj = previous hs),
matches confident detections to the ground truth (IoU >= 0.5), and for frame pairs (t, t+k) asks which person in frame t each person in
frame t+k is, by
  E  cosine of the id_words output (the ID head trained with the circle loss)
  H  cosine of hs (decoder output, the traj handed to the next frame)
  P  nearest box centre (position only: what the IoU linker uses)
  Q  same query slot (the 300 fixed /32 cells) -- how often a person keeps its slot
Accuracy = share of people in t+k whose answer has the right ground-truth id (people absent from frame t are skipped).
"hard" = the cases where P is wrong: if E is right there, the ID branch knows more than position.
Also the shift test: frame t+k moved right by 64 px (two /32 cells); identity features should survive it, a slot shortcut should not.
Usage (Vast): FTT_CODE=<official clone> python ftt_id_check.py <config.yaml> <checkpoint.pth> <out.json> <seq dir> [<seq dir> ...]
(seq dir = MOT layout: img1/*.jpg|png, gt/gt.txt frame,id,x,y,w,h,...; pictures 640x480)"""
import glob, json, os, re, sys
import numpy as np, torch

cfg_path, ckpt, out_path, seqs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'ftt_trained_export.py'), encoding='utf-8').read()
head = src[:src.index('cfg = yaml_to_dict(cfg_path)')]           # imports + sine_pos + MHA + NPUNet, nothing run
sys.argv = [sys.argv[0], cfg_path, ckpt, '.', '.']; g_ = {'__file__': os.path.join(here, 'ftt_trained_export.py'), '__name__': 'ftt_export_head'}
exec(compile(head, 'ftt_trained_export_head', 'exec'), g_)
from utils.utils import yaml_to_dict   # noqa: E402
import models as M                     # noqa: E402
from PIL import Image                  # noqa: E402

cfg = yaml_to_dict(cfg_path); cfg['DEVICE'] = 'cpu'; cfg['pretrained'] = False
model = M.build_rt_model(cfg); sd = torch.load(ckpt, map_location='cpu'); sd = sd.get('model', sd)
mi, un = model.load_state_dict(sd, strict=False); print('checkpoint', os.path.basename(ckpt), 'missing', len(mi), 'unexpected', len(un), flush=True)
model.eval(); net = g_['NPUNet'](model.detr).eval()
dev = 'cuda' if torch.cuda.is_available() and os.environ.get('FTT_CHECK_CPU') != '1' else 'cpu'; net.to(dev)
H, W, SHIFT, GAPS, MAX_FRAMES = 480, 640, 64, (1, 5, 15), int(os.environ.get('FTT_CHECK_FRAMES', '400'))


def iou(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    i_ = ix * iy; ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return i_ / (ar(a)[:, None] + ar(b)[None, :] - i_ + 1e-9)


def run(gray, traj):
    x = torch.from_numpy(np.repeat(gray[None].astype(np.float32), 3, 0))[None].to(dev)
    k4, t4, r4 = net.part_ae(x); kv, tgt, ref = k4.reshape(1, 300, -1), t4.reshape(1, 300, -1), r4.reshape(1, 300, 4)
    t1, r1 = net.part_b(tgt, ref, kv, traj, 0, 1); lg, bx, hs, idw, _ = net.part_b(t1, r1, kv, traj, 1, 2)
    return torch.sigmoid(lg)[0, :, 0].cpu().numpy(), bx[0].cpu().numpy(), hs, hs[0].cpu().numpy(), idw[0].float().cpu().numpy()


def frame_people(sc, bx, hs, idw, gt_boxes, gt_ids):
    """confident detections matched to gt -> list of (gt id, slot, box xyxy px, id_words, hs)"""
    keep = np.flatnonzero(sc >= 0.5)
    if not len(keep) or not len(gt_boxes): return []
    b = bx[keep]; xy = np.c_[(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H, (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H]
    M_ = iou(xy, gt_boxes); used_d, used_g, out = set(), set(), []
    for d, g in sorted(((d, g) for d in range(M_.shape[0]) for g in range(M_.shape[1]) if M_[d, g] >= 0.5), key=lambda p: -M_[p]):
        if d in used_d or g in used_g: continue
        used_d.add(d); used_g.add(g); q = int(keep[d]); out.append((int(gt_ids[g]), q, xy[d], idw[q], hs[q]))
    return out


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9); b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9); return a @ b.T


def score_pair(A, B, res):
    """A = people of frame t, B = of frame t+k; adds to res[method] = [right, total] and hard counts"""
    ida = np.array([p[0] for p in A]); present = [j for j, p in enumerate(B) if p[0] in set(ida.tolist())]
    if len(A) < 2 or not present: return
    E = cos(np.stack([B[j][3] for j in present]), np.stack([p[3] for p in A])); Hm = cos(np.stack([B[j][4] for j in present]), np.stack([p[4] for p in A]))
    ca = np.stack([(p[2][:2] + p[2][2:]) / 2 for p in A]); cb = np.stack([(B[j][2][:2] + B[j][2][2:]) / 2 for j in present])
    Pd = np.linalg.norm(cb[:, None] - ca[None], axis=2); sl_a = np.array([p[1] for p in A])
    for r, j in enumerate(present):
        true = B[j][0]; p_ok = ida[np.argmin(Pd[r])] == true
        for m, ok in (('E', ida[np.argmax(E[r])] == true), ('H', ida[np.argmax(Hm[r])] == true), ('P', p_ok)):
            res[m][0] += int(ok); res[m][1] += 1
            if not p_ok: res[m + '_hard'][0] += int(ok); res[m + '_hard'][1] += 1
        same_slot = np.flatnonzero(sl_a == B[j][1]); res['Q'][1] += 1; res['Q'][0] += int(len(same_slot) == 1 and ida[same_slot[0]] == true)
    # similarity spread of the id embedding: same person vs others
    for r, j in enumerate(present):
        same = ida == B[j][0]; res['sim_same'].extend(E[r][same].tolist()); res['sim_other'].extend(E[r][~same].tolist())


def new_res(): return {k: [0, 0] for k in ('E', 'H', 'P', 'Q', 'E_hard', 'H_hard', 'P_hard')} | {'sim_same': [], 'sim_other': []}


report = {}
with torch.no_grad():
    for sd_ in seqs:
        name = os.path.basename(sd_.rstrip('/')); files = sorted(glob.glob(os.path.join(sd_, 'img1', '*.jpg')) + glob.glob(os.path.join(sd_, 'img1', '*.png')))[:MAX_FRAMES]
        gt = np.loadtxt(os.path.join(sd_, 'gt', 'gt.txt'), delimiter=',', ndmin=2)
        num = lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1])
        traj = torch.zeros(1, 300, 256, device=dev); people, shifted = [], {}
        for f in files:
            gray = np.asarray(Image.open(f).convert('L'), np.uint8); assert gray.shape == (H, W), gray.shape
            rows = gt[gt[:, 0] == num(f)]; gb = np.c_[rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]] if len(rows) else np.zeros((0, 4))
            sc, bx, hs_t, hs, idw = run(gray, traj)
            people.append(frame_people(sc, bx, hs, idw, gb, rows[:, 1] if len(rows) else []))
            # shift test: the same frame moved right by SHIFT px, run with the true previous traj (as if the camera jumped two cells)
            gs = np.zeros_like(gray); gs[:, SHIFT:] = gray[:, :-SHIFT]; gbs = gb.copy(); gbs[:, [0, 2]] += SHIFT
            s2, b2, _, h2, i2 = run(gs, traj); shifted[len(people) - 1] = frame_people(s2, b2, h2, i2, gbs, rows[:, 1] if len(rows) else [])
            traj = hs_t
        rep = {}
        for k in GAPS:
            res = new_res()
            for t in range(len(people) - k): score_pair(people[t], people[t + k], res)
            rep['gap%d' % k] = res
        res = new_res()
        for t in range(len(people) - 1): score_pair(people[t], shifted[t + 1], res)
        rep['shift1'] = res
        for kk, res in rep.items():
            for m in list(res):
                if m.startswith('sim_'): v = np.array(res[m]); res[m] = dict(n=len(v), mean=float(v.mean()) if len(v) else None, p10=float(np.percentile(v, 10)) if len(v) else None, p90=float(np.percentile(v, 90)) if len(v) else None)
                else: right, tot = res[m]; res[m] = dict(acc=round(right / tot, 4) if tot else None, n=tot)
        rep['frames'] = len(files); rep['people_per_frame'] = round(float(np.mean([len(p) for p in people])), 2)
        report[name] = rep; print(name, json.dumps({kk: {m: v.get('acc', v.get('mean')) for m, v in r.items()} if isinstance(r, dict) else r for kk, r in rep.items()}), flush=True)
json.dump(dict(ckpt=ckpt, config=cfg_path, shift_px=SHIFT, max_frames=MAX_FRAMES, seqs=report), open(out_path, 'w'), indent=1)
print('DONE', out_path, flush=True)
