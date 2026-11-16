"""녹화 재생의 "재생 추적기" 자리 (2026-11-13).

녹화를 재생할 때 화면에 그리고 정답과 비교하는 상자를 누가 만드는지 고르는 자리다.
  - 'python'          서버의 추적기 그대로 (이 파일을 쓰지 않는다)
  - 'cpp_bytetrack'   C++ ByteTrack 라이브러리 (CppByteTrackBackend: 전과 똑같이 NativeByteTrack 을 부르기만 한다)
  - 'pre:<key>'       저장된 결과 (PrecomputedTracks): PC 에서 미리 만든 <key>.mot.txt 의 상자와 번호를 PNG 번호로 찾아 그대로 돌려준다

서버의 전역 상태(APP)를 읽지 않는다. 그래서 서버 없이 시험할 수 있다 (src/camera/test_replay_backend.py).

모든 재생 추적기가 지키는 약속 (ReplayBackend):
  reset()                    새 회차나 건너뛰기 뒤에 추적 상태를 비운다
  update(tag, detections)    tag = (PNG 번호, 회차, 녹화 안의 칸 번호), detections = 서버 검출기의 [N,6] 캔버스 좌표 행.
                             -> [[x1, y1, x2, y2, 번호, 점수, 클래스, 검출 행 번호]] (640x512 캔버스: y = 그림 y + 16)
  people()                   own_detector 인 추적기만: 방금 update 한 프레임의 사람 [(번호, x1, y1, x2, y2, 점수)] (640x480 그림 좌표, 파일 값 그대로)
  close()                    재생이 끝났다
  meta()                     결과 파일에 남길 설명 (비어 있으면 남기지 않는다)
  own_detector               True = 상자가 서버 검출기에서 나오지 않았다: 검출 후보 점수(AP50 / mAP)를 비우고, 흔들림 보정을 채점에 넣지 않는다
  slow                       True = 한 장에 오래 걸린다 (지금은 쓰는 곳이 없다. 장치에서 모델을 돌리는 추적기를 붙일 때 쓴다)
  errors                     update / reset 에서 "이 추적기의 오류"로 볼 예외 종류

저장된 결과 파일 (<폴더>/<녹화 이름>/<key>.mot.txt 와 <key>.json):
  <key>.mot.txt   한 줄 = frame,id,x,y,w,h,score,-1,-1,-1   frame = 녹화의 PNG 번호 (000123.png = 123), x,y = 왼쪽 위 (640x480 그림 픽셀), 사람만
  <key>.json      만든 기록. 꼭 있어야 하는 것: status ('complete' 또는 'complete_range'), frames_processed, first_processed, last_processed, mot_sha256.
                  있으면 함께 확인하는 것: error, covers_whole_sequence, sequence_total, png_first, png_last, source_wh, rows,
                  frames_dir ('.../<녹화 폴더>/images' 의 녹화 폴더 이름이 이 녹화의 마지막 폴더 이름과 같아야 한다). label 은 화면에 나오는 이름.
빈 프레임과 처리하지 않은 프레임은 MOT 파일에서 똑같이 보인다. 그래서 설명 파일이 "이 녹화의 모든 PNG 를 처음부터 끝까지 처리했다"고 말하지 않으면 읽지 않는다
(Refused). 중간까지만 만든 결과를 끝까지 만든 것처럼 보여 주지 않는다.
"""
import hashlib
import json
import math
import os
import re

from .bytetrack import NativeByteTrack

CANVAS_Y = 16.0                    # 640x480 그림은 640x512 캔버스의 위에서 16 픽셀 아래에 있다
STORED_PREFIX = 'pre:'             # 저장된 결과의 재생 추적기 이름: 'pre:<key>'
STORED_SUFFIX = ' (저장된 결과)'
MOT_EXT, META_EXT = '.mot.txt', '.json'
KEY_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
BAD_KEY = '파일 이름에 쓸 수 없는 글자 (영문·숫자로 시작, 영문·숫자 . _ - 만, 64자까지)'
COMPLETE = ('complete', 'complete_range')
PICTURE_WH = (640, 480)
MOT_MAX_BYTES = 64 * 1024 * 1024
META_MAX_BYTES = 4 * 1024 * 1024
LABEL_MAX = 40
META_KEEP = ('what', 'motip_commit', 'checkpoint', 'checkpoint_sha256', 'settings', 'attention', 'resize', 'input',
             'device', 'threads', 'versions', 'timing_ms', 'written')   # 설명 파일에서 결과 파일로 옮겨 적는 항목
META_ITEM_MAX = 2000


class Refused(ValueError):
    """저장된 결과를 쓰지 않는 이유 (화면에 그대로 보여 주는 한글 글).

    파일에서 읽은 값이 글에 섞이므로 찍을 수 있는 글자만 남긴다 (짝 없는 서러게이트 같은 글자가 있으면 응답을 UTF-8 로 바꾸지 못한다)."""

    def __init__(self, text=''):
        super().__init__(''.join(ch for ch in str(text) if ch.isprintable()))


class ReplayBackend:
    id = ''
    label = ''
    own_detector = False
    slow = False
    errors = (Exception,)

    def reset(self):
        pass

    def update(self, tag, detections):
        raise NotImplementedError

    def people(self):
        return None

    def close(self):
        pass

    def meta(self):
        return {}


class CppByteTrackBackend(ReplayBackend):
    """C++ ByteTrack. 재생 채점이 전에 NativeByteTrack 을 직접 부르던 것과 같은 호출을 같은 순서로 한다."""
    id = 'cpp_bytetrack'
    label = 'C++ ByteTrack'
    errors = (OSError, RuntimeError, ValueError)

    def __init__(self, library):
        self.tracker = NativeByteTrack(library)

    def reset(self):
        self.tracker.reset()

    def update(self, tag, detections):
        return self.tracker.update(detections)

    def close(self):
        self.tracker.close()


def stored_key(backend):
    """'pre:<key>' -> key. 그런 모양이 아니거나 key 가 파일 이름으로 쓸 수 없으면 None."""
    text = str(backend or '')
    if not text.startswith(STORED_PREFIX):
        return None
    key = text[len(STORED_PREFIX):]
    return key if KEY_RE.match(key) else None


def stored_dir(root, name):
    """<root>/<녹화 이름> (root 밖을 가리키면 None). 녹화 이름은 'opi_sequences/seq_001' 처럼 '/' 로 나뉜 상대 경로."""
    parts = str(name or '').replace('\\', '/').split('/')
    if not root or not name or '\x00' in name or ':' in name or any(part in ('', '.', '..') for part in parts):
        return None
    base = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, *parts))
    try:
        inside = os.path.commonpath([base, path]) == base
    except ValueError:
        inside = False
    return path if inside and path != base else None


def stored_paths(root, name, key):
    """-> (<key>.mot.txt, <key>.json) 두 파일이 모두 있을 때. 아니면 None."""
    folder = stored_dir(root, name)
    if folder is None or not KEY_RE.match(str(key or '')):
        return None
    mot, meta = os.path.join(folder, key + MOT_EXT), os.path.join(folder, key + META_EXT)
    return (mot, meta) if os.path.isfile(mot) and os.path.isfile(meta) else None


def png_numbers(images_dir):
    """녹화의 PNG 번호 (파일 이름의 숫자), 작은 것부터."""
    numbers = set()
    for entry in os.listdir(images_dir):
        stem, ext = os.path.splitext(entry)
        if ext.lower() == '.png' and stem.isascii() and stem.isdigit():
            numbers.add(int(stem))
    return sorted(numbers)


def stored_label(meta, key):
    """화면에 나오는 이름. 언제나 '(저장된 결과)' 가 붙는다."""
    raw = meta.get('label') if isinstance(meta, dict) else None
    text = ''.join(ch for ch in str(raw) if ch.isprintable()).strip() if isinstance(raw, str) else ''
    text = (text or key)[:LABEL_MAX]
    return text if '저장된 결과' in text else text + STORED_SUFFIX


def _whole(value):
    """정수로 적힌 값만 정수로 (True/False, 1.5, '3' 은 None)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value)
    return None


def _read(path, limit, what):
    try:
        if os.path.getsize(path) > limit:
            raise Refused('%s이 너무 큼' % what)
        with open(path, 'rb') as handle:
            return handle.read()
    except OSError as error:
        raise Refused('%s을 읽지 못함 (%s)' % (what, type(error).__name__))


def _check_meta(meta, pngs):
    status = meta.get('status')
    if status not in COMPLETE:
        raise Refused('끝까지 만든 결과가 아님 (status %s)' % (str(status)[:30] if status is not None else '없음'))
    if meta.get('error'):
        raise Refused('만들다가 오류가 난 결과')
    if meta.get('covers_whole_sequence') is False:
        raise Refused('녹화 전체를 처리한 결과가 아님')
    total, first, last = len(pngs), pngs[0], pngs[-1]
    for field, want, text in (('frames_processed', total, '처리한 장 수'), ('first_processed', first, '첫 PNG 번호'),
                              ('last_processed', last, '마지막 PNG 번호')):
        got = _whole(meta.get(field))
        if got is None:
            raise Refused('설명 파일에 %s 가 없음' % field)
        if got != want:
            raise Refused('%s가 녹화와 다름 (결과 %d, 녹화 %d)' % (text, got, want))
    for field, want, text in (('sequence_total', total, '전체 장 수'), ('png_first', first, '첫 PNG 번호'),
                              ('png_last', last, '마지막 PNG 번호')):
        if field in meta and _whole(meta[field]) != want:
            raise Refused('%s가 녹화와 다름 (%s %s, 녹화 %d)' % (text, field, str(meta[field])[:20], want))
    if 'source_wh' in meta and tuple(meta['source_wh'] if isinstance(meta['source_wh'], (list, tuple)) else ()) != PICTURE_WH:
        raise Refused('그림 크기가 640x480 이 아님')


def _source_name(meta):
    """설명 파일의 frames_dir ('.../<녹화 폴더>/images') -> 녹화 폴더 이름. 알 수 없으면 ''."""
    raw = meta.get('frames_dir')
    parts = [part for part in raw.replace('\\', '/').split('/') if part] if isinstance(raw, str) else []
    name = parts[-2] if len(parts) >= 2 and parts[-1].lower() == 'images' else ''
    return ''.join(ch for ch in name if ch.isprintable())[:80]


def _parse_mot(raw, pngs):
    """-> ({PNG 번호: [(번호, x1, y1, x2, y2, 점수)]}, 줄 수). 읽지 못하는 줄이 하나라도 있으면 Refused."""
    known = set(pngs)
    people, seen, count = {}, set(), 0
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        raise Refused('결과 파일이 글자 파일이 아님')
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.strip().split(',')
        try:
            if len(fields) < 7:
                raise ValueError
            frame_value, id_value = float(fields[0]), float(fields[1])
            x, y, w, h, score = (float(v) for v in fields[2:7])
            if not all(math.isfinite(v) for v in (frame_value, id_value, x, y, w, h, score)):
                raise ValueError
            frame, track = int(frame_value), int(id_value)
            if frame != frame_value or track != id_value or track < 0 or w <= 0 or h <= 0:
                raise ValueError
        except ValueError:
            raise Refused('결과 파일 %d번째 줄을 읽지 못함' % number)
        if frame not in known:
            raise Refused('결과 파일 %d번째 줄: 녹화에 없는 PNG 번호 %d' % (number, frame))
        if (frame, track) in seen:
            raise Refused('결과 파일 %d번째 줄: 한 장에 같은 번호 %d 가 두 번' % (number, track))
        seen.add((frame, track))
        people.setdefault(frame, []).append((track, x, y, x + w, y + h, score))
        count += 1
    return people, count


class PrecomputedTracks(ReplayBackend):
    """저장된 결과 하나. 만들 때 파일을 모두 읽고 확인한다 (맞지 않으면 Refused). update 는 PNG 번호로 찾기만 한다."""
    own_detector = True

    def __init__(self, key, mot_path, meta_path, pngs, recording=''):
        if not KEY_RE.match(str(key or '')):
            raise Refused('이름이 올바르지 않음')
        pngs = sorted({int(v) for v in pngs})
        if not pngs:
            raise Refused('녹화에 PNG 가 없음')
        meta_raw = _read(meta_path, META_MAX_BYTES, '설명 파일')   # try 밖에서 읽는다: Refused 도 ValueError 라서 안에 두면 "읽지 못함"이 "JSON 이 아님"으로 바뀐다
        try:   # utf-8-sig: Windows 에서 손으로 고쳐 맨 앞에 BOM 이 붙은 파일도 읽는다. RecursionError: 괄호가 아주 깊게 겹친 파일
            meta = json.loads(meta_raw.decode('utf-8-sig'))
        except (ValueError, RecursionError):
            raise Refused('설명 파일이 올바른 JSON 이 아님')
        if not isinstance(meta, dict):
            raise Refused('설명 파일이 올바른 JSON 이 아님')
        _check_meta(meta, pngs)
        # 다른 녹화로 만든 결과가 장 수와 PNG 번호가 같은 녹화 폴더에 들어온 경우. 마지막 폴더 이름만 견준다
        # (윗폴더만 다른 같은 이름의 녹화는 가려내지 못한다. 설명 파일에 frames_dir 이 없으면 확인하지 않는다).
        source_name = _source_name(meta)
        if recording and source_name and source_name != str(recording).replace('\\', '/').split('/')[-1]:
            raise Refused('다른 녹화로 만든 결과 (결과 %s, 녹화 %s)' % (source_name[:40], str(recording)[:60]))
        raw = _read(mot_path, MOT_MAX_BYTES, '결과 파일')
        sha256 = hashlib.sha256(raw).hexdigest()
        if not isinstance(meta.get('mot_sha256'), str):
            raise Refused('설명 파일에 mot_sha256 이 없음')
        if meta['mot_sha256'].lower() != sha256:
            raise Refused('결과 파일이 설명 파일과 맞지 않음 (sha256 다름)')
        self.real, rows = _parse_mot(raw, pngs)
        if 'rows' in meta and _whole(meta['rows']) != rows:
            raise Refused('결과 파일의 줄 수가 설명 파일과 다름 (파일 %d, 설명 %s)' % (rows, str(meta['rows'])[:20]))
        self.id = STORED_PREFIX + key
        self.key = key
        self.label = stored_label(meta, key)
        self.pngs = frozenset(pngs)
        self.canvas = {png: [[p[1], p[2] + CANVAS_Y, p[3], p[4] + CANVAS_Y, p[0], p[5], 0, -1] for p in found]
                       for png, found in self.real.items()}
        self._last = []
        source = {}
        for field in META_KEEP:
            if field in meta:
                try:
                    text = json.dumps(meta[field], ensure_ascii=False)
                    text.encode('utf-8')   # 결과 파일(UTF-8)에 적지 못하는 값은 옮기지 않는다
                    if len(text) <= META_ITEM_MAX:
                        source[field] = meta[field]
                except (TypeError, ValueError, RecursionError):
                    pass
        self._meta = dict(kind='stored result', key=key, label=self.label, recording=recording, source_recording=source_name,
                          mot_file=os.path.basename(mot_path), mot_sha256=sha256, meta_file=os.path.basename(meta_path),
                          status=meta['status'], frames=len(pngs), png_first=pngs[0], png_last=pngs[-1], rows=rows,
                          frames_with_boxes=len(self.real), ids=len({p[0] for found in self.real.values() for p in found}),
                          note='Boxes and IDs were computed offline and are only looked up by PNG number. '
                               'Nothing here measures the device: no model ran on it. offline_run describes the computer that made the file.',
                          offline_run=source)

    def update(self, tag, detections):
        png = int(tag[0])
        if png not in self.pngs:
            raise ValueError('PNG %d 는 저장된 결과를 만든 녹화에 없음' % png)
        self._last = self.real.get(png, [])
        return list(self.canvas.get(png, ()))

    def people(self):
        return self._last

    def meta(self):
        return self._meta


def scan_stored(folder, images_dir, name=''):
    """한 녹화의 저장된 결과 폴더를 본다. -> (쓸 수 있는 것 [dict(id, key, label)], 못 쓰는 것 [dict(key, why)]), 이름 순."""
    usable, refused = [], []
    try:
        keys = sorted(entry[:-len(MOT_EXT)] for entry in os.listdir(folder) if entry.endswith(MOT_EXT))
    except OSError:
        return usable, refused
    if not keys:
        return usable, refused
    try:
        pngs = png_numbers(images_dir)
    except OSError:
        pngs = []
    for key in keys:
        if not key or key.startswith('.'):
            continue   # 숨김 파일 (맥이 복사하며 만드는 '._이름' 등)은 결과가 아니다
        if not KEY_RE.match(key):   # 조용히 건너뛰지 않는다: 파일을 넣었는데 왜 안 나오는지 화면에 이유가 나와야 한다
            shown = ''.join(ch for ch in key if ch.isprintable())[:LABEL_MAX] or '(빈 이름)'
            refused.append(dict(key=shown, why=BAD_KEY))
            continue
        try:
            if not os.path.isfile(os.path.join(folder, key + META_EXT)):
                raise Refused('설명 파일(%s%s)이 없음' % (key, META_EXT))
            item = PrecomputedTracks(key, os.path.join(folder, key + MOT_EXT), os.path.join(folder, key + META_EXT), pngs, name)
            usable.append(dict(id=item.id, key=key, label=item.label))
        except Refused as why:
            refused.append(dict(key=key, why=str(why)))
        except Exception as why:   # 미처 생각하지 못한 깨진 파일 하나가 다른 녹화와 다른 추적기의 목록까지 막지 않게 한다
            refused.append(dict(key=key, why='읽지 못함 (%s)' % type(why).__name__))
    return usable, refused


def make(tgt):
    """재생 목표(tgt) -> 재생 추적기. 'python' 은 여기로 오지 않는다. 만들지 못하면 예외 (재생을 시작하지 않는다)."""
    backend = tgt.get('backend', 'python')
    if backend == 'cpp_bytetrack':
        return CppByteTrackBackend(tgt['tracker_library'])
    key = stored_key(backend)
    tracks = tgt.get('tracks')
    if key is not None and isinstance(tracks, dict) and tracks.get('key') == key:
        return PrecomputedTracks(key, tracks['mot'], tracks['meta'], png_numbers(tgt['dir']), tgt.get('name', ''))
    raise ValueError('unknown replay backend')
