"""DanceTrack train zips -> grey, half size (1280x720 -> 640x360), official folder layout, for FastTrackTr training on the rented GPU
(user 2026-11-14: "응 그렇게해" = download to the PC, shrink to grey 640, send to Vast). The last 4 videos (by number) go to val (held out).
Boxes in gt.txt and the size in seqinfo.ini are scaled by the same factor. Pictures: grey JPEG q90 (the loader turns them into 3 channels).
Usage: python dancetrack_gray.py <out DanceTrack dir> <train1.zip> <train2.zip>"""
import io, re, sys, zipfile
from pathlib import Path
import numpy as np, cv2

out = Path(sys.argv[1]); zips = sys.argv[2:]; S = 0.5
seqs = {}
for zp in zips:
    z = zipfile.ZipFile(zp)
    for n in z.namelist():
        m = re.match(r'[^/]+/(dancetrack\d+)/', n)
        if m: seqs.setdefault(m.group(1), (zp, []))[1].append(n)
names = sorted(seqs); val = set(names[-4:]); total = 0
for sname in names:
    zp, members = seqs[sname]; z = zipfile.ZipFile(zp); split = 'val' if sname in val else 'train'; d = out / split / sname
    (d / 'img1').mkdir(parents=True, exist_ok=True); (d / 'gt').mkdir(parents=True, exist_ok=True)
    for n in members:
        if n.endswith('.jpg'):
            g = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, (int(round(g.shape[1] * S)), int(round(g.shape[0] * S))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / 'img1' / Path(n).name), g, [cv2.IMWRITE_JPEG_QUALITY, 90]); total += 1
        elif n.endswith('gt/gt.txt'):
            a = np.loadtxt(io.BytesIO(z.read(n)), delimiter=','); a[:, 2:6] *= S
            np.savetxt(d / 'gt' / 'gt.txt', a, delimiter=',', fmt=['%d', '%d', '%.2f', '%.2f', '%.2f', '%.2f', '%d', '%d', '%.4g'][:a.shape[1]])
        elif n.endswith('seqinfo.ini'):
            t = z.read(n).decode()
            t = re.sub(r'imWidth=(\d+)', lambda m: 'imWidth=%d' % round(int(m.group(1)) * S), t); t = re.sub(r'imHeight=(\d+)', lambda m: 'imHeight=%d' % round(int(m.group(1)) * S), t)
            (d / 'seqinfo.ini').write_text(t)
    print(split, sname, len(list((d / 'img1').glob('*.jpg'))), flush=True)
print('DONE images', total, 'val', sorted(val))
