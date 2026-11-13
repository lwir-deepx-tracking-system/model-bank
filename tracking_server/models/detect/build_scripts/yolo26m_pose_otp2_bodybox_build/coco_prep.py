"""COCO person keypoints -> grey YOLO-pose set for experiment D (real occlusions: COCO marks keypoints that are labelled but not visible with v=1).

Usage (Vast): python coco_prep.py <person_keypoints_train2017.json> <person_keypoints_val2017.json> <out_dir> [n_train=30000] [n_val=1000] [threads=48]

Picks images without crowd regions that have at least one person with >= 5 labelled keypoints and a box at least 1/4 of the image height (close or
medium people; this project is about close range). Images that contain an occluded keypoint (v=1) are taken first. Every person of a picked image is
labelled (people without keypoints get a box and all-zero keypoints, so they are not taught as background). Keypoint flags are kept as COCO gives them:
0 = not labelled, 1 = labelled but not visible, 2 = visible (the same meaning as in pose_partial.py).
Each image is fetched one by one from images.cocodataset.org (the full zip does not fit on the machine), converted to grey, longest side 640, JPEG q90.
Writes <out>/{train,val}/{images,labels}, <out>/PREP.json (counts, flag shares, failures).
"""
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def pick(ann_path, n, seed):
    d = json.load(open(ann_path))
    imgs = {i['id']: i for i in d['images']}
    by_img = {}
    for a in d['annotations']: by_img.setdefault(a['image_id'], []).append(a)
    occl, plain = [], []
    for iid, anns in by_img.items():
        if any(a.get('iscrowd', 0) for a in anns): continue
        im = imgs[iid]
        good = [a for a in anns if a['num_keypoints'] >= 5 and a['bbox'][3] >= im['height'] / 4]
        if not good: continue
        has_occl = any((np.array(a['keypoints']).reshape(17, 3)[:, 2] == 1).any() for a in good)
        (occl if has_occl else plain).append(iid)
    rng = random.Random(seed); rng.shuffle(occl); rng.shuffle(plain)
    chosen = (occl + plain)[:n]
    return [(imgs[i], by_img[i]) for i in chosen], dict(candidates_with_occluded=len(occl), candidates_without=len(plain))


def work(item, split, out, sub):
    im, anns = item
    url = 'http://images.cocodataset.org/%s/%s' % (sub, im['file_name'])
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
        g = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if g is None: return ('bad image', None)
        h, w = g.shape; s = 640.0 / max(h, w)
        if s < 1: g = cv2.resize(g, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        stem = Path(im['file_name']).stem
        cv2.imwrite(str(out / split / 'images' / (stem + '.jpg')), g, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lines = []; flags = np.zeros(3, int)
        for a in anns:
            x, y, bw, bh = a['bbox']
            if bw < 2 or bh < 2: continue
            k = np.array(a['keypoints'], float).reshape(17, 3)
            k[:, 0] /= w; k[:, 1] /= h; k[k[:, 2] == 0, :2] = 0
            for v in (0, 1, 2): flags[v] += int((k[:, 2] == v).sum())
            vals = [0, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h] + k.ravel().tolist()
            lines.append(' '.join('%d' % v if i == 0 else ('%.6f' % v) for i, v in enumerate(vals)))
        (out / split / 'labels' / (stem + '.txt')).write_text('\n'.join(lines) + '\n')
        return ('ok', flags)
    except Exception as e:
        return (type(e).__name__, None)


def main():
    tr_ann, va_ann, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n_tr = int(sys.argv[4]) if len(sys.argv) > 4 else 30000; n_va = int(sys.argv[5]) if len(sys.argv) > 5 else 1000; th = int(sys.argv[6]) if len(sys.argv) > 6 else 48
    rep = {}
    for split, ann, n, sub in (('val', va_ann, n_va, 'val2017'), ('train', tr_ann, n_tr, 'train2017')):
        (out / split / 'images').mkdir(parents=True, exist_ok=True); (out / split / 'labels').mkdir(parents=True, exist_ok=True)
        items, info = pick(ann, n, 20261113)
        res = {}; flags = np.zeros(3, int)
        with ThreadPoolExecutor(th) as ex:
            for i, (st, fl) in enumerate(ex.map(lambda it: work(it, split, out, sub), items)):
                res[st] = res.get(st, 0) + 1
                if fl is not None: flags += fl
                if (i + 1) % 2000 == 0: print(split, i + 1, res, flush=True)
        rep[split] = dict(asked=len(items), results=res, keypoint_flags=dict(not_labelled=int(flags[0]), labelled_hidden=int(flags[1]), visible=int(flags[2])), **info)
        print(split, json.dumps(rep[split]), flush=True)
    (out / 'PREP.json').write_text(json.dumps(rep, indent=1)); print('PREP DONE', flush=True)


if __name__ == '__main__':
    main()
