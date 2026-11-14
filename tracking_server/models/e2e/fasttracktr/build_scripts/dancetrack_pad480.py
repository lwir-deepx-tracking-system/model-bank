"""Second pass on the grey half-size DanceTrack: every sequence -> 640x480 (scale to width 640, then grey 114 rows top and bottom), so every
training picture has the camera's 4:3 size (the NPU form of FastTrackTr has 300 queries = 15x20 cells only at 480x640). Some sequences were
1920x1080 (half = 960x540), most 1280x720 (half = 640x360). Stored with 3 equal channels (the official loader opens pictures without RGB
conversion). gt.txt boxes scaled and shifted the same way, seqinfo imWidth / imHeight = 640 / 480. In place; a finished sequence is skipped.
Usage: python dancetrack_pad480.py <DanceTrack dir>"""
import re, sys
from pathlib import Path
import numpy as np, cv2

root = Path(sys.argv[1])
for sd in sorted(p for s in ('train', 'val') for p in (root / s).glob('dancetrack*')):
    ini = (sd / 'seqinfo.ini').read_text()
    W = int(re.search(r'imWidth=(\d+)', ini).group(1)); H = int(re.search(r'imHeight=(\d+)', ini).group(1))
    if (W, H) == (640, 480): print('done', sd.name, flush=True); continue
    s = 640.0 / W; nh = int(round(H * s)); top = (480 - nh) // 2; assert 0 <= top, (sd, W, H)
    for f in sorted((sd / 'img1').glob('*.jpg')):
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE); assert g.shape == (H, W), (f, g.shape, H, W)
        if s != 1.0: g = cv2.resize(g, (640, nh), interpolation=cv2.INTER_AREA)
        c = np.full((480, 640), 114, np.uint8); c[top:top + nh] = g
        cv2.imwrite(str(f), cv2.cvtColor(c, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    gt = sd / 'gt' / 'gt.txt'; a = np.loadtxt(gt, delimiter=','); a[:, 2:6] *= s; a[:, 3] += top
    np.savetxt(gt, a, delimiter=',', fmt=['%d', '%d', '%.2f', '%.2f', '%.2f', '%.2f', '%d', '%d', '%.4g'][:a.shape[1]])
    ini = re.sub(r'imWidth=\d+', 'imWidth=640', ini); (sd / 'seqinfo.ini').write_text(re.sub(r'imHeight=\d+', 'imHeight=480', ini))
    print(sd.parent.name, sd.name, W, H, '-> 640x480 top', top, flush=True)
print('PAD DONE')
