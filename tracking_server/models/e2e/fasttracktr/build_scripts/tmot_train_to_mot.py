"""TMOT (Ahmar et al., ECCV 2024, github.com/wassimea/thermalMOT, Apache-2.0) TRAIN split, thermal only -> FastTrackTr's MOT layout for the
thermal fine-tune (2026-11-15, literature study research_transfer_20261114 section 3: TMOT train + some grey DanceTrack).
Per sequence <out>/train/tmot_<seq>/img1/00000001.jpg ... (8-bit grey 640x480 = rows 16..495 of the 640x512 frame, 3 equal channels),
gt/gt.txt = frame,id,x,y,w,h,1,1,1 (DanceTrack format; id = TMOT track_id + 1; boxes shifted up 16 px, clipped to the picture, dropped when
under 2 px), seqinfo.ini. The val split (our external test set) is never written here.
Usage: python tmot_train_to_mot.py <tmot_dataset.zip> <out dir>"""
import io, json, sys, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

zp, out = sys.argv[1], Path(sys.argv[2]); z = zipfile.ZipFile(zp); names = z.namelist()
seqs = sorted({p.split('/')[3] for p in names if p.startswith('tmot_dataset/images/train/') and p.count('/') >= 4 and p.split('/')[3]})
rep = {}
for s in seqs:
    a = json.loads(z.read('tmot_dataset/annotations/train/%s/thermal/COCO/annotations.json' % s))
    ims = sorted(a['images'], key=lambda i: i['file_name']); fid = {im['id']: k + 1 for k, im in enumerate(ims)}
    d = out / 'train' / ('tmot_' + s); (d / 'img1').mkdir(parents=True, exist_ok=True); (d / 'gt').mkdir(exist_ok=True)
    for k, im in enumerate(ims):
        g = np.asarray(Image.open(io.BytesIO(z.read('tmot_dataset/images/train/%s/thermal/%s' % (s, im['file_name'])))).convert('L'))
        assert g.shape == (512, 640), g.shape
        Image.fromarray(g[16:496]).convert('RGB').save(d / 'img1' / ('%08d.jpg' % (k + 1)), quality=95)
    rows = []
    for an in a['annotations']:
        x, y, w, h = an['bbox']; y -= 16; x1, y1, x2, y2 = max(0, x), max(0, y), min(640, x + w), min(480, y + h)
        if x2 - x1 < 2 or y2 - y1 < 2: continue
        rows.append((fid[an['image_id']], int(an['track_id']) + 1, x1, y1, x2 - x1, y2 - y1))
    rows.sort()
    (d / 'gt' / 'gt.txt').write_text(''.join('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n' % r for r in rows))
    n = len(ims)
    (d / 'seqinfo.ini').write_text('[Sequence]\nname=tmot_%s\nimDir=img1\nframeRate=5\nseqLength=%d\nimWidth=640\nimHeight=480\nimExt=.jpg\n' % (s, n))
    rep[s] = dict(frames=n, boxes=len(rows), people=len({r[1] for r in rows}), frames_with_people=len({r[0] for r in rows})); print(s, rep[s], flush=True)
(out / 'TMOT_TRAIN.json').write_text(json.dumps(dict(source='TMOT train split, thermal, cropped 640x480', seqs=rep,
                                                       total=dict(frames=sum(v['frames'] for v in rep.values()), boxes=sum(v['boxes'] for v in rep.values()))), indent=1))
print('DONE')
