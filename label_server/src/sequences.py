# -*- coding: utf-8 -*-
"""라벨 폴더에서 시퀀스를 찾고, 한 시퀀스의 진행 상태를 읽는다.

scan(root)     tracks_mot.txt 와 images 폴더가 함께 있는 폴더를 시퀀스로 본다 (라벨 폴더 아래 4단계까지).
version(d)     tracks_edited_mot.txt 의 수정 시각. 다른 사람이 먼저 저장했는지 가리는 데 쓴다.
seq_info(rel)  목록 화면 한 줄에 들어가는 값 (프레임 수, 확인한 수, 삭제한 수, 마지막 저장, 지금 열어 둔 사람).
"""
import json, os, re, time

from .state import PRESENCE, SEQS


def scan(root):
    out = {}
    for d, sub, fs in os.walk(root):
        depth = os.path.relpath(d, root).count(os.sep)
        if depth > 3:
            sub[:] = []
            continue
        if 'tracks_mot.txt' in fs and 'images' in sub:
            n = max([int(f[:6]) for f in os.listdir(os.path.join(d, 'images')) if re.fullmatch(r'\d{6}\.png', f)] or [0])   # last frame number (purged frames leave holes)
            if n:
                out[os.path.relpath(d, root).replace(os.sep, '/')] = {'dir': d, 'frames': n}
            sub[:] = []
    return dict(sorted(out.items()))


def version(d):
    p = os.path.join(d, 'tracks_edited_mot.txt')
    return str(os.stat(p).st_mtime_ns) if os.path.exists(p) else '0'


def seq_info(rel):
    s = SEQS[rel]
    d = s['dir']
    info = {'seq': rel, 'frames': s['frames'], 'version': version(d), 'reviewed': 0, 'saved': '', 'by': '', 'done_by': '', 'removed': 0, 'purged': 0, 'kept': s['frames'], 'rev_kept': 0}
    try:
        rv = json.load(open(os.path.join(d, 'review_state.json'), encoding='utf-8'))
        rm = {int(x) for x in rv.get('removed', []) if 1 <= int(x) <= s['frames']}
        info['removed'] = len({int(x) for x in rv.get('removed', [])} | {int(x) for x in rv.get('purged', [])})   # includes purged frames past the last remaining image
        info['kept'] = s['frames'] - len(rm)                      # frames left after deletion
        info['rev_kept'] = len({int(x) for x in rv.get('reviewed', []) if 1 <= int(x) <= s['frames']} - rm)
        info['purged'] = len(set(rv.get('purged', [])))
        info['reviewed'] = len(set(rv.get('reviewed', [])) | rm)
        info['saved'] = rv.get('saved', '')
        info['by'] = rv.get('saved_by', '')
        info['done_by'] = rv.get('completed_by', '') if info['reviewed'] >= s['frames'] else ''
    except Exception:
        pass
    now = time.time()
    info['open_by'] = sorted(u for u, t in PRESENCE.get(rel, {}).items() if now - t < 90)
    return info
