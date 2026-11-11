"""여러 부분이 함께 쓰는 고정 값과 작은 함수: 폴더 위치(HERE, ROOT), 검출 모델 표, 사람 클래스 번호, 추적 상태 번호, 파일 확인값(sha), 앵커 수.

HERE 는 실행 파일(start_tracking_server.py)이 있는 폴더(tracking_server)다. 모델·페이지·저장 파일을 이 폴더 기준으로 찾는다."""
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent.parent
ROOT = HERE.parent
SHAPE = {'DRT512A': (512, 384), 'DRT512B': (512, 384)}                       # --det rt512: two 512x384 column tiles (default)
SHAPE_TT = {'TTA75': (480, 640), 'TTB75': (480, 640)}                          # --det tt75: two full-frame towers on rows 16:496 (claude_tt75_deploy_01)
SHAPE_RT480 = {'NEWMRT480A': (480, 384), 'NEWMRT480B': (480, 384)}            # --det rt480: new m RT480 column tiles on rows 16:496 (claude_ft_deploy_o1_01/NEWM)
SHAPE_W480 = {'NEWMRW480': (480, 640), 'S480': (480, 640), 'FT1': (480, 640), 'FT9': (480, 640), 'S640': (640, 640), 'M640': (640, 640), 'S6': (640, 640)}                                         # --det w480 (v16): one model on the whole camera band, rows 16:496
CORE = {'DRT512A': 0, 'DRT512B': 1, 'TTA75': 0, 'TTB75': 1, 'NEWMRT480A': 0, 'NEWMRT480B': 1, 'NEWMRW480': 0, 'S640': 0, 'M640': 0, 'S6': 0, 'S480': 0, 'FT1': 0, 'FT9': 0}
DET = {'rt512': dict(names=['DRT512A', 'DRT512B'], yoff=0.0), 'tt75': dict(names=['TTA75', 'TTB75'], yoff=16.0),
       'rt480': dict(names=['NEWMRT480A', 'NEWMRT480B'], yoff=16.0), 'w480': dict(names=['NEWMRW480'], yoff=16.0), 's640': dict(names=['S640'], yoff=-64.0),
       'm640': dict(names=['M640'], yoff=-64.0),
       's6': dict(names=['S6'], yoff=-64.0),
       's480': dict(names=['S480'], yoff=16.0),
       'ft1': dict(names=['FT1'], yoff=16.0),
       'ft9': dict(names=['FT9'], yoff=16.0)}   # v24e (11-10): YOLO26m_(FLIR) + room/close-range labels + real empty-office negatives + FLIR rehearsal (m_rehearsal_v9); PC gate: 0 wrong boxes at conf >= .4   # v22i: YOLO26m_(FLIR) fine-tuned on our camera labels   # v22h: YOLO26s 640x480 on the camera band, as w480   # v22g: YOLO26s 640x640, six split-head outputs (model bank yolo26s_nms_640x640)   # v22: YOLO26m_(FLIR) 640x640, six split-head outputs (tensor_requirements.md)   # yoff: detector y -> frame y
PERSON_CLS = 0
TARGET = {0, 2, 6}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def anchors(h, w): return sum((h // s) * (w // s) for s in (8, 16, 32))


TRACKED, LOST, REMOVED = 1, 2, 3
