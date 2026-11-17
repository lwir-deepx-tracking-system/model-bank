"""Indoor scorer v2 for the head-to-hip body-box pose model on our hand-labelled indoor recordings (replaces the rough rule of quick_indoor_DE.py).

Labels (tracks_edited_mot.txt: frame,id,x,y,w,h,conf,class,...) are VISIBLE-PART boxes; the model predicts a head-to-hip body box.
Matching rule:
- one-to-one: a predicted box can match at most one labelled person (Hungarian: most matches first, ties go to the higher IoU)
- a pair is allowed when (a) the visible box centre is inside the predicted box AND at least 50% of the visible box area is inside it
  (a partly hidden person: the body box is larger than what is seen), OR (b, 2026-11-15 review) at least 70% of the predicted box lies inside
  the visible box AND it covers at least 25% of the visible area (a close, wide person: the shoulder-wide head-to-hip body box is narrower
  than the visible part; without (b) a correct body box was counted as a miss in 38 of 62 old-found/new-missed cases on dev).
  A box on only the head of a fully seen person stays a miss (covers about a tenth of the visible area).
  (a correct body box around a head-only person counts as found; one big box over two people finds only one of them)
- false boxes = predictions (conf >= 0.25) left unmatched, split into 'extra on a person' (contains some visible-box centre: a duplicate
  or merged box) and 'on nobody'. Class 1 (ignore) label boxes are not people; an unmatched box whose centre is inside one is not counted.
Splits of the labelled people:
- size: visible box height <80, 80-160, 160-320, >=320 px (stands in for distance)
- visibility proxy: min(1, visible height / reference body height). Reference = head-to-hip box of a FIXED reference model (--ref, conf >= 0.05)
  matched to the person with the same rule in the same frame; else the nearest scored frame of the same track within 9 frames; else 'unknown'.
  Bins <0.5 (mostly hidden, e.g. head and shoulders), 0.5-0.8, >=0.8 (torso visible), unknown.
  Use the same --ref for every model you compare, so the bins hold the same people. The bins are partly defined by the reference model: people
  it never sees even weakly go to 'unknown', so the reference model itself has ~0 found share there and optimistic shares in the other bins.
  Predicted boxes are clipped to the image, so a person cut by the image edge looks 'visible' here: see the edge split.
- edge: visible box within 2 px of the image border or not.
Frames: every 3rd image file in name order. The image name is the 1-based frame number of the label file (some recordings have gaps where
frames were removed during labelling, so the number is read from the name, not counted). 16-bit images are min-max normalised to 8 bit like
quick_indoor_DE.py; 8-bit images are used as they are. imgsz 640.
Pooled shares get a recording-level bootstrap 90% interval (resample recordings with replacement, 10000 draws, seed 0).
The old rule (centre inside and IoU >= 0.2, many-to-one) is also reported on the same frames.

Usage: python indoor_score_v2.py --split-file indoor_test_split.json --split dev --data-root <LIG-Data/close_range> --weights E.pt [--ref E.pt]
       --out out.json [--name E]
With --split final the numbers are written to --out only (stdout gets no numbers).
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment

STEP, CONF, REF_CONF, IMGSZ, MIN_INSIDE, OLD_IOU, MAXGAP, EDGE_PX = 3, 0.25, 0.05, 640, 0.5, 0.2, 9, 2
PRED_IN, MIN_COVER = 0.7, 0.25   # rule (b)
SIZE = [('<80', 0, 80), ('80-160', 80, 160), ('160-320', 160, 320), ('>=320', 320, 1e9)]
VIS = [('<0.5', 0, 0.5), ('0.5-0.8', 0.5, 0.8), ('>=0.8', 0.8, 9)]
BOOT, SEED = 10000, 0


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load_img(p):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    if im.dtype != np.uint8: im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def predict(model, im, conf):
    """Person boxes (xyxy) and scores at conf."""
    r = model.predict(im, imgsz=IMGSZ, conf=conf, verbose=False)[0]
    if r.boxes is None or not len(r.boxes): return np.zeros((0, 4)), np.zeros(0)
    b, s, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
    k = c == 0
    return b[k], s[k]


def pairs(v, b):
    """v (n,4) visible boxes, b (m,4) predicted boxes, xyxy -> allowed (n,m), iou (n,m), centre inside (n,m)."""
    n, m = len(v), len(b)
    if not n or not m: z = np.zeros((n, m)); return z.astype(bool), z, z.astype(bool)
    cx, cy = (v[:, 0] + v[:, 2]) / 2, (v[:, 1] + v[:, 3]) / 2
    inside = (b[None, :, 0] <= cx[:, None]) & (cx[:, None] <= b[None, :, 2]) & (b[None, :, 1] <= cy[:, None]) & (cy[:, None] <= b[None, :, 3])
    iw = np.clip(np.minimum(v[:, None, 2], b[None, :, 2]) - np.maximum(v[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(v[:, None, 3], b[None, :, 3]) - np.maximum(v[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    av = (v[:, 2] - v[:, 0]) * (v[:, 3] - v[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (av[:, None] + ab[None] - inter + 1e-9)
    cover = inter / np.maximum(av[:, None], 1e-9); pred_in = inter / np.maximum(ab[None], 1e-9)
    return (inside & (cover >= MIN_INSIDE)) | ((pred_in >= PRED_IN) & (cover >= MIN_COVER)), iou, inside


def assign(ok, iou):
    """One-to-one assignment: most matches, then highest IoU. Returns {person index: box index}."""
    if not ok.any(): return {}
    r, c = linear_sum_assignment(np.where(ok, 1 + 1e-3 * iou, 0.0), maximize=True)
    return {int(i): int(j) for i, j in zip(r, c) if ok[i, j]}


def bin_of(x, bins):
    for name, lo, hi in bins:
        if lo <= x < hi: return name
    return None


def score_recording(rec, d, model, ref_model):
    gt = np.loadtxt(d / 'tracks_edited_mot.txt', delimiter=',', ndmin=2)
    imgs = sorted((d / 'images').glob('*.png'))[::STEP]
    people, frames = [], []
    for f in imgs:
        fn = int(f.stem); im = load_img(f); H, W = im.shape[:2]
        g = gt[gt[:, 0] == fn]
        ign = g[g[:, 7] == 1]; g = g[g[:, 7] != 1]
        v = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]] if len(g) else np.zeros((0, 4))
        if ref_model is model:
            ball, sall = predict(model, im, REF_CONF)
            b, s = ball[sall >= CONF], sall[sall >= CONF]; rb = ball
        else:
            b, s = predict(model, im, CONF); rb, _ = predict(ref_model, im, REF_CONF)
        ok, iou, inside = pairs(v, b)
        m = assign(ok, iou)
        rok, riou, _ = pairs(v, rb)
        rm = assign(rok, riou)
        old = (inside & (iou >= OLD_IOU)).any(1) if len(b) else np.zeros(len(v), bool)
        used = set(m.values()); extra_on_person = on_nobody = ignored = 0
        for j in range(len(b)):
            if j in used: continue
            cx, cy = (b[j, 0] + b[j, 2]) / 2, (b[j, 1] + b[j, 3]) / 2
            if len(ign) and any(r[2] <= cx <= r[2] + r[4] and r[3] <= cy <= r[3] + r[5] for r in ign): ignored += 1
            elif len(v) and inside[:, j].any(): extra_on_person += 1
            else: on_nobody += 1
        frames.append(dict(frame=fn, people=len(v), boxes=len(b), false=extra_on_person + on_nobody, extra_on_person=extra_on_person,
                           on_nobody=on_nobody, ignored=ignored))
        for i in range(len(v)):
            x1, y1, x2, y2 = v[i]
            people.append(dict(rec=rec, frame=fn, id=int(g[i, 1]), h=round(float(y2 - y1), 1),
                               edge=bool(x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= W - EDGE_PX or y2 >= H - EDGE_PX),
                               found=i in m, score=round(float(s[m[i]]), 3) if i in m else None,
                               pred_h=round(float(b[m[i], 3] - b[m[i], 1]), 1) if i in m else None,
                               ref_h=round(float(rb[rm[i], 3] - rb[rm[i], 1]), 1) if i in rm else None, ref_from='same' if i in rm else None,
                               old_found=bool(old[i])))
    # missing reference height: nearest scored frame of the same track within MAXGAP frames
    by_id = {}
    for p in people:
        if p['ref_h'] is not None: by_id.setdefault(p['id'], []).append((p['frame'], p['ref_h']))
    for p in people:
        if p['ref_h'] is None and p['id'] in by_id:
            fr, rh = min(by_id[p['id']], key=lambda t: abs(t[0] - p['frame']))
            if abs(fr - p['frame']) <= MAXGAP: p['ref_h'], p['ref_from'] = rh, f'frame {fr}'
    for p in people:
        p['vis'] = round(min(1.0, p['h'] / p['ref_h']), 3) if p['ref_h'] else None
        p['size_bin'] = bin_of(p['h'], SIZE)
        p['vis_bin'] = bin_of(p['vis'], VIS) if p['vis'] is not None else 'unknown'
        p['edge_bin'] = 'touches edge' if p['edge'] else 'inside'
    return people, frames


def share(ps, key='found'):
    n = len(ps); k = sum(p[key] for p in ps)
    return dict(people=n, found=k, share=round(k / n, 3) if n else None)


def summary(people, frames):
    nf = len(frames)
    out = dict(frames=nf, **share(people), old_rule_share=share(people, 'old_found')['share'],
               false_boxes=sum(f['false'] for f in frames), false_per_frame=round(sum(f['false'] for f in frames) / max(nf, 1), 3),
               extra_on_person_per_frame=round(sum(f['extra_on_person'] for f in frames) / max(nf, 1), 3),
               on_nobody_per_frame=round(sum(f['on_nobody'] for f in frames) / max(nf, 1), 3),
               boxes_per_frame=round(sum(f['boxes'] for f in frames) / max(nf, 1), 3))
    for split, names in (('size_bin', [b[0] for b in SIZE]), ('vis_bin', [b[0] for b in VIS] + ['unknown']), ('edge_bin', ['inside', 'touches edge'])):
        out['by_' + split[:-4]] = {nm: share([p for p in people if p[split] == nm]) for nm in names}
    return out


def boot(per_rec, recs):
    """Recording-level bootstrap 90% interval of the pooled found share (overall and per bin) and of false boxes per frame."""
    rng = np.random.default_rng(SEED); draws = rng.integers(0, len(recs), (BOOT, len(recs)))
    def ci(num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        a, b = num[draws].sum(1), den[draws].sum(1); x = a[b > 0] / b[b > 0]
        return [round(float(np.percentile(x, 5)), 3), round(float(np.percentile(x, 95)), 3)] if len(x) else None
    S = [per_rec[r] for r in recs]
    out = dict(n_recordings=len(recs), found_share=ci([s['found'] for s in S], [s['people'] for s in S]),
               false_per_frame=ci([s['false_boxes'] for s in S], [s['frames'] for s in S]))
    for split in ('by_size', 'by_vis', 'by_edge'):
        out[split] = {nm: ci([s[split][nm]['found'] for s in S], [s[split][nm]['people'] for s in S]) for nm in S[0][split]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-file', required=True); ap.add_argument('--split', required=True, choices=['dev', 'final'])
    ap.add_argument('--data-root', required=True); ap.add_argument('--weights', required=True)
    ap.add_argument('--ref', help='reference model for the visibility proxy (default: --weights)'); ap.add_argument('--name', default='model')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from ultralytics import YOLO
    spl = json.loads(Path(a.split_file).read_text(encoding='utf-8'))
    recs = sorted(r for r, x in spl['recordings'].items() if x['split'] == a.split)
    model = YOLO(a.weights)
    ref_path = a.ref or a.weights
    same = Path(ref_path).resolve() == Path(a.weights).resolve()
    ref_model = model if same else YOLO(ref_path)
    allp, per_rec, frames_all = [], {}, {}
    for r in recs:
        p, f = score_recording(r, Path(a.data_root) / spl['recordings'][r]['dir'], model, ref_model)
        allp += p; frames_all[r] = f; per_rec[r] = summary(p, f)
    pooled = summary(allp, [f for r in recs for f in frames_all[r]])
    pooled['ci90_recording_bootstrap'] = boot(per_rec, recs)
    res = dict(name=a.name, split=a.split, recordings=recs,
               weights=dict(path=str(a.weights), sha256=sha256(a.weights)), ref=dict(path=str(ref_path), sha256=sha256(ref_path), same_as_weights=same),
               settings=dict(every_nth_frame=STEP, conf=CONF, ref_conf=REF_CONF, imgsz=IMGSZ, min_visible_area_inside=MIN_INSIDE, rule_b=dict(pred_inside_visible=PRED_IN, min_cover=MIN_COVER),
                             old_rule_iou=OLD_IOU, ref_max_gap_frames=MAXGAP, edge_px=EDGE_PX, bootstrap=BOOT, seed=SEED,
                             size_bins=[b[0] for b in SIZE], vis_bins=[b[0] for b in VIS] + ['unknown']),
               rule=__doc__.split('Splits of')[0].strip(),
               per_recording=per_rec, pooled=pooled, people=allp, frames=frames_all)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    if a.split == 'final':
        print(f'final split: {len(recs)} recordings, {pooled["frames"]} frames, {pooled["people"]} people scored; numbers written to {a.out} (not printed)')
        return
    print(f'{a.name} split={a.split} recordings={recs}')
    for r in recs + ['POOLED']:
        s = pooled if r == 'POOLED' else per_rec[r]
        print(f'{r}: frames {s["frames"]} people {s["people"]} found {s["share"]} (old rule {s["old_rule_share"]}) false/frame {s["false_per_frame"]} '
              f'(extra on person {s["extra_on_person_per_frame"]}, on nobody {s["on_nobody_per_frame"]})')
        for split in ('by_size', 'by_vis', 'by_edge'):
            ci = pooled['ci90_recording_bootstrap'][split] if r == 'POOLED' else None
            print('   ' + split + ': ' + ', '.join(f'{k} {v["found"]}/{v["people"]}={v["share"]}' + (f' [{ci[k][0]}-{ci[k][1]}]' if ci and ci[k] else '')
                                             for k, v in s[split].items()))
    print('pooled found share 90% interval (recording bootstrap):', pooled['ci90_recording_bootstrap']['found_share'],
          'false/frame:', pooled['ci90_recording_bootstrap']['false_per_frame'])


if __name__ == '__main__':
    main()
