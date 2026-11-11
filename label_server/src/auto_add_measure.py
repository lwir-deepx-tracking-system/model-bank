# -*- coding: utf-8 -*-
"""Turn finished live-page measurements (MEASURE_* folders) into label-site sequences. Run by start_label_server.py every 30 s.
A measurement is handled once (auto_added.json), so a sequence that was renamed or sent to the trash does not come back.
Skipped (and tried again later): still recording, over MAX_FRAMES frames, or less than MIN_FREE_GB left on the disk."""
import sys, os, json, gzip, csv, glob, re, time, shutil, argparse, datetime
import numpy as np, cv2
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the server folder (one above this src folder): auto_added.json stays next to start_label_server.py
ap = argparse.ArgumentParser()
ap.add_argument('--src', default='/mnt/lwir_data/results/live_measure'); ap.add_argument('--dst', default='/mnt/lwir_data/label_web/LWIR_labels')
ap.add_argument('--state', default=os.path.join(HERE, 'auto_added.json')); ap.add_argument('--mark', nargs='*', default=[], help='measurement names to mark as already handled')
ap.add_argument('--list', action='store_true'); a = ap.parse_args()
MAX_FRAMES, MIN_FREE_GB = 3000, 1.0
st = json.load(open(a.state, encoding='utf-8')) if os.path.exists(a.state) else {}
def save():
    json.dump(st, open(a.state + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=0); os.replace(a.state + '.tmp', a.state)
for b in a.mark: st[b] = 'already on the site'
if a.mark: save()
for m in sorted(glob.glob(a.src + '/*/MEASURE_*')):
    b = os.path.basename(m)
    if b in st: continue
    fj, fr = m + '/frames.json', m + '/frames.raw.gz'
    if not (os.path.exists(fj) and os.path.exists(fr)): continue                      # no video saved, or still recording
    if time.time() - max(os.stat(m).st_mtime, os.stat(fj).st_mtime) < 15: continue    # just finished: wait for the last files
    j = json.load(open(fj, encoding='utf-8')); n, W, H = j['frames'], j['width'], j['height']
    why = '' if 0 < n <= MAX_FRAMES else '%d frames (limit %d)' % (n, MAX_FRAMES)
    if not why and shutil.disk_usage(a.dst).free / 2**30 < MIN_FREE_GB + n * 200e3 / 2**30: why = 'disk space'
    if a.list or why:
        print('SKIP' if why else 'TODO', b, n, why); continue
    q = b.split('_'); title = re.sub(r'[^\w가-힣 .-]', '', '_'.join(q[4:])).strip(' .')
    name = 'rec_%s_%s' % (q[1], q[2]) + ('_' + title if title else ''); out = os.path.join(a.dst, name); tmp = os.path.join(a.dst, '.tmp_' + name)
    if os.path.exists(out): st[b] = name; save(); continue
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp + '/images')
    raw = np.frombuffer(gzip.open(fr).read(), '<u2')
    if raw.size != n * W * H: print('SKIP', b, 'size mismatch'); shutil.rmtree(tmp); continue
    im8 = (raw.reshape(n, H, W) >> 8).astype(np.uint8)
    for i in range(n): assert cv2.imwrite('%s/images/%06d.png' % (tmp, i + 1), im8[i])
    rows, ids = [], set()
    if os.path.exists(m + '/tracks.csv'):
        for r in csv.DictReader(open(m + '/tracks.csv', encoding='utf-8')):
            f = int(r['raw_frame']) + 1
            if 1 <= f <= n:
                ids.add(r['track_id']); rows.append('%d,%s,%.1f,%.1f,%.1f,%.1f,%.3f,0,-1,-1' % (f, r['track_id'], float(r['x']), float(r['y']), float(r['width']), float(r['height']), float(r['confidence'])))
    open(tmp + '/tracks_mot.txt', 'w').write('\n'.join(rows) + ('\n' if rows else ''))
    th = [cv2.resize(im8[i], (320, 240)) for i in np.linspace(0, n - 1, 20).astype(int)]
    cv2.imwrite(tmp + '/overview_sheet.jpg', np.vstack([np.hstack(th[k:k + 5]) for k in range(0, 20, 5)]))
    kst = datetime.datetime.strptime(q[1] + q[2], '%Y%m%d%H%M%S') + datetime.timedelta(hours=1)
    open(tmp + '/설명.txt', 'w', encoding='utf-8').write(
        '%s (측정 녹화, 손 라벨용, 자동 추가)\n\n- 찍은 시각: %s (한국 시각), 라이브 화면의 측정 버튼으로 녹화. 원본: %s\n- %d장, %.1f초, %dx%d, 8비트 (원본 16비트 값을 256으로 나눈 것, 다른 보정 없음)\n- 프레임 번호 1 = images/000001.png\n'
        '- tracks_mot.txt: 후보 박스 %d개 (추적 번호 %d개). 측정 때 라이브 추적기가 낸 박스 그대로이며 정답이 아닙니다. 틀린 박스는 지우고 빠진 사람은 그려 주세요.\n'
        % (title or '(이름 없음)', kst.strftime('%Y-%m-%d %H:%M'), b, n, j['times_s'][-1] - j['times_s'][0] if n > 1 else 0, W, H, len(rows), len(ids)))
    os.rename(tmp, out); st[b] = name; save(); print('ADDED', b, '->', name, n, len(rows))
