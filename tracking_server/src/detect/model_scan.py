"""검출 모델 파일 찾기: 검출 모델 폴더에 새로 들어온 모델 파일을 읽어 보고, 쓸 수 있으면 화면의 모델 목록에 올린다."""
import json
import time
import subprocess
import re
from pathlib import Path
from ..common.state import APP
from ..common.shared_values import HERE, CORE, DET, sha, anchors


# ---- v28 (user 11-10): models put into dxnn/ are found by themselves ----
W480_DETS = {'w480', 's480', 'ft1', 'ft9'}; S6_DETS = {'m640', 's6'}; BANK = {}; BANK_SEEN = {}; BANK_T = [0.0]
BANK_REJ = {}; BANK_HEADS = ['box_s8', 'box_s16', 'box_s32', 'cls_s8', 'cls_s16', 'cls_s32']
E2E_DETS = set(); E2E_KEYS = set()   # v30 (user 11-10): models whose one output is the finished [1,300,6] list (x1,y1,x2,y2,score,class); no host decode, no host NMS
# 11-13 (user): models whose one output is the already decoded box list of all N anchors, before NMS (host NMS only, no host box decode).
# 'v8' = [1,20,N]: rows cx,cy,w,h (model pixels) + 16 class scores 0..1 (YOLOv8 export); 'yolox' = [1,N,21]: per anchor cx,cy,w,h (model pixels), objectness 0..1, 16 class scores 0..1 (YOLOX export)
RAW_DETS = {}; RAW_KEYS = {}   # det ('bank:<stem>') -> kind for _detect; model key (<stem>) -> kind for Engine
def raw_shape(kind, n): return (1, 20, n) if kind == 'v8' else (1, n, 21)
def bank_probe(p):
    """v29 (user 11-10: any file name): read the input size and the output heads from the file itself with parse_model (no NPU use).
    -> ((H, W, kind), None) for a detector with a 640x480 or 640x640 uint8 input, else (None, reason).
    kind (v30) picks the post-processing path: 'six' = six split heads (host decode + NMS), 'e2e' = one [1,300,6] end-to-end output,
    'v8' / 'yolox' (11-13) = one decoded output over all N anchors of the input size, [1,20,N] / [1,N,21] (host NMS only; see RAW_DETS)."""
    try: r = subprocess.run(['parse_model', '-m', str(p), '-v'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    except OSError: return None, 'parse_model missing'
    except subprocess.SubprocessError as e: return None, 'parse_model error: ' + type(e).__name__
    t = re.sub(r'\x1b\[[0-9;]*m', '', r.stdout.decode('utf-8', 'replace'))
    sh = {m.group(1): (tuple(int(v) for v in m.group(2).split(',')), m.group(3)) for m in re.finditer(r'\+- (\S+) \{shape: \[([0-9, ]+)\], dtype: (\w+)', t)}
    m = re.search(r'Model Output Tensors:\s*\n((?:\s+- \S+\s*\n)+)', t)
    outs = re.findall(r'- (\S+)', m.group(1)) if m else []
    if 'images' not in sh: return None, 'not a readable model file'
    ishape, idt = sh['images']
    if idt != 'uint8' or len(ishape) != 4 or (ishape[0], ishape[3]) != (1, 3) or ishape[1:3] not in ((480, 640), (640, 640)):
        return None, 'input %s %s is not 640x480 / 640x640 uint8' % (list(ishape), idt)
    H, W = ishape[1:3]
    if len(outs) == 1 and sh.get(outs[0]) == ((1, 300, 6), 'float32'): return (H, W, 'e2e'), None
    if len(outs) == 1:   # 11-13: one output that is not the end-to-end list: only the two decoded forms with 4 box values + 16 classes over exactly the N anchors of this input
        n = anchors(H, W)
        for kind in ('v8', 'yolox'):
            if sh.get(outs[0]) == (raw_shape(kind, n), 'float32'): return (H, W, kind), None
        got = sh.get(outs[0])
        return None, 'the one output %s is %s; accepted are float32 [1,300,6] (end-to-end), [1,20,%d] (YOLOv8 form) or [1,%d,21] (YOLOX form)' % (
            outs[0], '%s %s' % (list(got[0]), got[1]) if got else 'unreadable', n, n)
    if outs != BANK_HEADS: return None, 'outputs %s are neither the six detector heads nor one output of an accepted form ([1,300,6], [1,20,N], [1,N,21])' % outs
    for c, pre in ((4, 'box_s'), (16, 'cls_s')):
        for st in (8, 16, 32):
            if sh.get(pre + str(st)) != ((1, c, H // st, W // st), 'float32'): return None, 'head %s%d has shape %s' % (pre, st, sh.get(pre + str(st)))
    return (H, W, 'six'), None
def bank_scan(folder, skip, now):
    """-> [(det, key, name, path, sha, (H, W), yoff, kind)] for new, fully written detector files (v29: any file name; size and outputs read from the file; kind as in bank_probe).
    A file is taken only after its size and time stayed the same over two scans and it is older than 10 s (no half-copied upload)."""
    new = []
    try: files = sorted(list(Path(folder).glob('*.dxnn')) + list(Path(folder).glob('*/*.dxnn')))   # layout 11-11: models/detect/<kind>/ sub-folders are scanned too (one level)
    except OSError: return new
    for p in files:
        stem = p.stem
        if stem in BANK: continue
        try:
            if p.resolve() in skip: continue
            st = p.stat()
        except OSError: continue
        sig = (st.st_size, st.st_mtime)
        if BANK_SEEN.get(stem) != sig or now - st.st_mtime < 10.0: BANK_SEEN[stem] = sig; continue
        if BANK_REJ.get(stem) == sig: continue
        hw, why = bank_probe(p)
        if hw is None and why == 'parse_model missing':   # tool missing: fall back to the old file-name rule
            hw = (480, 640, 'six') if stem.endswith('_640x480') else (640, 640, 'six') if stem.endswith('_640x640') else None
        if hw is None: BANK_REJ[stem] = sig; print('BANK file not listed: %s (%s)' % (p.name, why), flush=True); continue
        name = stem
        for side in (p.with_name(stem + '_compile_metadata.json'), p.with_suffix('.json')):
            try:
                v = json.loads(side.read_text(encoding='utf-8')).get('display_name')
                if isinstance(v, str) and v.strip(): name = v.strip()[:80]; break
            except (OSError, ValueError, AttributeError): pass
        try: h = sha(p)
        except OSError: continue
        if '%dx%d' % (hw[1], hw[0]) not in name: name += ' %dx%d' % (hw[1], hw[0])
        if hw[2] == 'e2e': name += ' [NMS-free]'
        new.append(('bank:' + stem, stem, name + ' (자동 등록)', p, h, hw[:2], 16.0 if hw[0] == 480 else -64.0, hw[2]))
    return new


def bank_refresh(force=False):
    """v28: add newly placed dxnn/ files to the catalog (ids only grow, nothing is removed while the server runs). Called at start and from GET /model, at most every 5 s."""
    now_ = time.time()
    if not APP.MODEL['catalog'] or APP.args.no_bank_auto or (not force and now_ - BANK_T[0] < 5.0): return
    if not APP.BANK_LOCK.acquire(False): return   # another request is scanning
    try: BANK_T[0] = now_; APP.bank_add_(now_)
    finally: APP.BANK_LOCK.release()


def bank_add_(now_):
    skip_ = set()
    for q_ in (HERE / 'models' / 'links').glob('*.dxnn'):
        try: skip_.add(q_.resolve())
        except OSError: pass
    for det_, key_, name_, p_, h_, hw_, yoff_, kind_ in bank_scan(APP.BANK_DIR, skip_, now_):
        with APP.MODEL['lock']:
            BANK[key_] = p_; APP.man[key_] = h_; APP.det_shape[key_] = hw_; CORE[key_] = 0; DET[det_] = dict(names=[key_], yoff=yoff_)
            (W480_DETS if hw_[0] == 480 else S6_DETS).add(det_)   # input canvas (frame preparation)
            if kind_ == 'e2e': E2E_DETS.add(det_); E2E_KEYS.add(key_)   # v30: output path
            elif kind_ in ('v8', 'yolox'): RAW_DETS[det_] = kind_; RAW_KEYS[key_] = kind_   # 11-13: output path
            APP.MODEL['catalog'].append(dict(id=len(APP.MODEL['catalog']) + 1, det=det_, key=key_, name=name_))
        print('BANK model listed: id %d %s (%s)' % (len(APP.MODEL['catalog']), key_, h_[:12]), flush=True)
