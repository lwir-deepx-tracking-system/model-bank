# -*- coding: utf-8 -*-
"""라벨 폴더에서 시퀀스를 찾고, 한 시퀀스의 진행 상태를 읽는다.

scan(root)     tracks_mot.txt 와 images 폴더가 함께 있는 폴더를 시퀀스로 본다 (라벨 폴더 아래 4단계까지).
version(d, rel) 현재 MOT, 검수 상태, 신체부위 파일의 변경을 가리는 값.
seq_info(rel)  목록 화면 한 줄에 들어가는 값 (프레임 수, 확인한 수, 삭제한 수, 마지막 저장, 지금 열어 둔 사람).
"""
import hashlib, json, os, re, time

from .body_parts import empty_parts
from .state import EDITED_MOT, PARTS_FILE, PRESENCE, REVIEW_FILE, SEQS, SOURCE_MOT
from .transactions import JOURNAL, TransactionError, recover_transaction


EDITOR_CAPABILITIES = {'parts_supported': True, 'editor_schema': 2}


def scan(root):
    out = {}
    for d, sub, fs in os.walk(root):
        depth = os.path.relpath(d, root).count(os.sep)
        if depth > 3:
            sub[:] = []
            continue
        if 'tracks_mot.txt' in fs and 'images' in sub:
            rel = os.path.relpath(d, root).replace(os.sep, '/')
            blocked = False
            if JOURNAL in fs or JOURNAL in sub:
                try:
                    recover_transaction(d, rel)
                except TransactionError:
                    blocked = True  # keep the sequence addressable so its requests report the recovery error
            n = max([int(f[:6]) for f in os.listdir(os.path.join(d, 'images')) if re.fullmatch(r'\d{6}\.png', f)] or [0])   # last frame number (purged frames leave holes)
            if n or blocked:
                out[rel] = {'dir': d, 'frames': n}
            sub[:] = []
    return dict(sorted(out.items()))


def version(d, rel):
    return _version(_annotation_files(d, rel))


def mot_name(d):
    return EDITED_MOT if os.path.isfile(os.path.join(d, EDITED_MOT)) else SOURCE_MOT


def _read_bytes(d, name):
    try:
        with open(os.path.join(d, name), 'rb') as source:
            return source.read()
    except FileNotFoundError:
        return None


def _annotation_files(d, rel):
    recover_transaction(d, rel)
    return [(name, _read_bytes(d, name)) for name in (mot_name(d), REVIEW_FILE, PARTS_FILE)]


def _version(files):
    digest = hashlib.sha256()
    for name, content in files:
        size = len(content) if content is not None else -1
        digest.update(f'{name}:{size}\n'.encode('ascii'))
        if content is not None:
            digest.update(content)
    return digest.hexdigest()


def _invalid_json_number(value):
    raise ValueError(f'invalid JSON number: {value}')


def _json_object(content, name, default):
    if content is None:
        return default
    value = json.loads(content.decode('utf-8'), parse_constant=_invalid_json_number)
    if not isinstance(value, dict):
        raise ValueError(f'{name}: expected a JSON object')
    return value


def read_json(d, name, default):
    return _json_object(_read_bytes(d, name), name, default)


def annotation_snapshot(rel):
    """Call with LOCK held, so the files and version reflect the same save."""
    s = SEQS[rel]
    files = _annotation_files(s['dir'], rel)
    (_, mot), (_, review), (_, parts) = files
    if mot is None:
        raise FileNotFoundError('MOT file is missing')
    return {'version': _version(files), 'mot': mot.decode('utf-8'), 'frames': s['frames'],
            'review': _json_object(review, REVIEW_FILE, {}), 'parts': _json_object(parts, PARTS_FILE, empty_parts()),
            **EDITOR_CAPABILITIES}


def seq_info(rel):
    s = SEQS[rel]
    d = s['dir']
    info = {'seq': rel, 'frames': s['frames'], 'version': version(d, rel), 'reviewed': 0, 'saved': '', 'by': '', 'done_by': '', 'removed': 0, 'purged': 0, 'kept': s['frames'], 'rev_kept': 0, **EDITOR_CAPABILITIES}
    try:
        rv = read_json(d, REVIEW_FILE, {})
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
