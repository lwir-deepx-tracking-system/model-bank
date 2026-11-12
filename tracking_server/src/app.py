"""프로그램 본체 main(): 옵션 읽기 -> 준비 -> 프레임 반복 -> 결과 저장.

main() 안에 정의돼 있던 함수와 클래스는 일별 파일로 옮겼다. 원래 정의하던 자리에는 그 이름을 이어 주는 한 줄이 있다
(예: APP.gb_send = gimbal_link.gb_send). 옮긴 함수들이 함께 쓰는 값은 APP.이름 으로, 실행 중에 바뀌는 설정 네 가지는 settings.이름 으로 읽고 쓴다.
crop_into, vit_sync, sync_embed, budget_embed 는 옮기지 않고 여기 그대로 있다 (main() 의 지역 변수 vit 를 직접 써야 해서)."""
import os
import sys
import json
import time
import traceback
import argparse
from pathlib import Path
import numpy as np
from .common import settings
from .common.state import APP
from .common.shared_values import (
    HERE, ROOT, SHAPE, SHAPE_TT, SHAPE_RT480, SHAPE_W480, CORE, DET, PERSON_CLS, TARGET, sha, anchors, LOST, REMOVED)
from .detect.model_scan import BANK_T
from .common.device_wait import wait_m1_free
from .track.kalman import gmc_warp, GKF
from .track.position_memory import pos_select
from .track.camera_motion import GMC
from .track.stabilizer import STAB_M, stab_shift, stab_margins
from .track.bytetrack import ByteTrackTracker, CameraByteTrack, crop_requests
from .track.botsort import BotSortTracker
from .track.lostfirst import LostFirstTracker
from .track.gate import GateTracker, NumpyGRUGate
from .track.confirm_time import ConfirmByTime, apply as confirm_apply
from .track.warm_p95 import p95 as warm_p95, MODE as WARM_P95_MODE   # 11-12: same number as np.percentile, lighter
from .common.research_synth import load_synth_plan, apply_synth_plan
from .detect.zoom_redetect import ZOOM_HW, zoom_candidates, zoom_windows, zoom_pack, zoom_merge
from .reid.reid_models import reid_catalog, reid_swap
from .common.pc_dry_run import make_mock_dx
from .common.research_prepare import prepare
from .detect import model_scan, npu_engine, detect_frame
from .web import system_buttons, page_handler, page_settings, measure
from .track import camera_motion
from .reid import reid_embed
from .camera import camera_link, replay_eval
from .gimbal import gimbal_link
from .common import result_summary
from .common import stage_timing   # 2026-11-11: 단계별 시간 재기 (LWIR_STAGE_TIMING=1 일 때만, 기본 꺼짐)
from .detect import frame_feed     # 2026-11-11: 도착하는 대로 검출기에 넣기 (LWIR_NPU_FEED=arrival 일 때만, 기본 꺼짐)


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    APP.__dict__.clear()
    # v24f, v33
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', choices=['bytetrack', 'rebind'], default='rebind'); ap.add_argument('--no-vit', action='store_true')
    ap.add_argument('--vit', default='VIT_REID1', help='ViT DXNN name in models/ (batch 1)'); ap.add_argument('--vit-every', type=int, default=2)
    ap.add_argument('--k', type=int, default=2, help='max crops queued per ViT frame'); ap.add_argument('--sim', type=float, default=0.5)
    ap.add_argument('--vit-conf', type=float, default=0.25); ap.add_argument('--vit-min-h', type=float, default=32.0)
    ap.add_argument('--vit-stale', type=int, default=6, help='drop queued crop requests older than this many frames'); ap.add_argument('--late-window', type=int, default=10)
    ap.add_argument('--app-first', action='store_true', help='appearance-first re-association of lost tracks (wide gate), motion fallback afterwards')
    ap.add_argument('--gate', type=float, default=3.5, help='appearance-first spatial gate in box heights'); ap.add_argument('--no-motion-fallback', action='store_true')
    ap.add_argument('--vit-sync', action='store_true', help='embed up to --sync-k re-identification candidates SYNCHRONOUSLY (batch 1) on candidate frames and use them in that '
                    'frame\'s association (hybrid rule); the asynchronous round-robin gallery refresh stays'); ap.add_argument('--sync-k', type=int, default=2)
    ap.add_argument('--hybrid-fallback', action='store_true', help='with --vit-sync --app-first: a featured detection with no appearance match may take the '
                    'motion fallback, but only against lost tracks that have NO gallery feature (the appearance veto stays where both sides have one); default off')
    ap.add_argument('--det', choices=['rt512', 'tt75', 'rt480', 'w480', 's640', 'm640', 's6', 's480', 'ft1', 'ft9'], default='rt512', help="detector front-end: rt512 = DRT512A/B column tiles (default, unchanged); tt75 = two full-frame towers "
                    "TTA75/TTB75 on rows 16:496, host average of the raw outputs -> one Decoder(480,640) -> fast_nms (claude_tt75_deploy_01)")
    ap.add_argument('--mock-tt-onnx', default='')
    ap.add_argument('--mock-rt480-onnx', default='',
                    help='kit folder with NEWMRT480A|B/<name>.onnx (mock) and <name>/compiler/<name>.dxnn (prepare --det-src NEWM)')
    ap.add_argument('--mock-vit-src', default='',
                    help='claude_vit_reid_01: vit_student.py (mock), <vit>/compiler/<vit>.dxnn and tracker_ab/SYNTH_PLAN.json (prepare)')
    ap.add_argument('--out', default=None); ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--prepare', action='store_true', help='PC: build upload_track/ (script + DRT512A/B + ViT DXNN + MANIFEST)')
    ap.add_argument('--det-src', default='DRT512', help="prepare only: 'DRT512' (original YOLO26m-RT, opt0) or a deploy-kit tag such as 'E08' "
                    "(claude_ft_deploy_o1_01/<tag>/upload/models/<tag>RT512A/B.dxnn, fine-tuned, opt_level 1) copied into the DRT512A/B slots -> upload_track_<tag>/")
    ap.add_argument('--data', default='', help='v4 (11-02): frame folder with coco.json (images: id, file_name, extra_info.video_id); default = the FLIR video test')
    ap.add_argument('--mock', action='store_true', help='PC dry run with a fake dx_engine'); ap.add_argument('--mock-data', default='')
    ap.add_argument('--mock-trace', default='')
    ap.add_argument('--mock-student', default=str(HERE / 'run_kd_02' / 'student_folded.pt')); ap.add_argument('--mock-device', default='cpu')
    ap.add_argument('--pipeline', action='store_true', help='software pipelining: submit frame i+1 tiles before post-processing frame i (period/latency timing)')
    ap.add_argument('--synth-plan', default=None, help='synthetic occlusion plan (SYNTH_PLAN.json from make_synth_plan.py): inside the planned gaps every person '
                    'detection with IoU >= .5 to the planned GT box is hidden (conf 0) before the tracker and the ViT crop selection; default off')
    ap.add_argument('--vit-bench-n', type=int, default=0, help='COST BENCH (default 0 = off): every frame, after NMS and before tracker.update, run exactly N '
                    'synchronous batch-1 inferences of --vit on NPU_2 (same crop preprocessing as the --vit-sync path) on the frame\'s person detections '
                    '(conf >= .10, h >= 32, highest conf first, repeated cyclically; a fixed centre crop when none); the embeddings are discarded. Requires --no-vit '
                    '(tracker without appearance); the ViT engine is loaded for the bench only. RESULT.json gets "vit_bench"')
    ap.add_argument('--vit-budget', type=int, default=0, help='LOST-FIRST ViT rule (default 0 = off): at most N synchronous batch-1 crops per frame on NPU_2 for ALL '
                    'priorities together (P1 gated re-identification candidates inside update, then P2 at-risk / P3 just-lost / P4 round-robin gallery refresh), '
                    'LostFirstTracker = current hybrid association + EMA gallery (use --sim 0.6 --gate 3.5 for the chosen G1_A1_g3.5_s0.6_N2); no asynchronous queue. '
                    'Requires the ViT (no --no-vit), --tracker rebind; not with --vit-sync / --vit-bench-n')
    ap.add_argument('--vit-priority', choices=['lostfirst'], default='lostfirst', help='with --vit-budget: crop priority (only lostfirst = P1 > P2 > P3 > P4)')
    ap.add_argument('--warm-max', type=int, default=0, help='v33: drop a person detection whose box raw 95th percentile is above this (0 = no upper limit)')
    ap.add_argument('--warm-min', type=int, default=0, help='v24f: drop a person detection whose box raw 16-bit 95th percentile is below this (0 = off; 45000 measured safe for this camera: lights <= 43397)')
    ap.add_argument('--edge-birth', type=float, default=0.0, help='v24f: score a detection touching a frame edge (--edge-px) needs to start a new track (0 = off; existing tracks keep matching at lower scores)')
    ap.add_argument('--edge-px', type=float, default=4.0, help='v24f: how close to the frame edge counts as touching')
    ap.add_argument('--n-init', type=int, default=2, help='matched frames, the birth frame included, before a NEW track is output (default 2 = unchanged: '
                    'birth frame + 1 hit; 1 behaves like 2; tracks born on a video\'s first frame are output at once). Offline choice: 3 '
                    '(idsw_fix/RESULT_KO.md, verified with --tracker rebind --no-vit only)')
    ap.add_argument('--confirm-ms', type=float, default=0.0, help='11-11: the confirmation rule of a NEW track written as a time. 0 (default) = off: --n-init frames as before. '
                    'N > 0, live server only: frames = round(N x measured frames per second / 1000), never below --n-init, re-computed while running '
                    '(src/track/confirm_time.py). 130 = 3 frames at 22-27 fps, 4 frames at 30 fps')
    ap.add_argument('--radius', type=float, default=2.0, help='motion re-bind radius in lost-track heights (default 2.0 = unchanged). Offline choice: 1.5 '
                    '(with --n-init 3, idsw_fix/RESULT_KO.md, verified with --tracker rebind --no-vit only)')
    ap.add_argument('--gmc', action='store_true', help='BoT-SORT camera-motion compensation (Ultralytics GMC sparseOptFlow, downscale 2) on the full gray '
                    'frame before each tracker.update (default off = unchanged)')
    ap.add_argument('--class-gate', action='store_true', help='class gate (default off = unchanged): person tracks only with person detections, '
                    'car / truck tracks only with car / truck detections, in every association (claude_review_fixes_01/classgate)')
    ap.add_argument('--relabel', default='', help='v5: re-ID DXNN name in models/ for the lost-track ID relabel on top of the unchanged tracker (relabel_online.py, '
                    'PREREG_RELABEL_M1.json); requires --tracker bytetrack --no-vit')
    ap.add_argument('--rl-every', type=int, default=2); ap.add_argument('--rl-k', type=int, default=2); ap.add_argument('--rl-new-k', type=int, default=2)
    ap.add_argument('--gmc-ds', type=int, default=2); ap.add_argument('--gmc-corners', type=int, default=1000); ap.add_argument('--gmc-mindist', type=float, default=1.0)
    ap.add_argument('--gmc-thread', action='store_true'); ap.add_argument('--npu-depth', type=int, default=1); ap.add_argument('--rl-async', action='store_true')
    ap.add_argument('--ort-threads', type=int, default=1); ap.add_argument('--preload', action='store_true')
    ap.add_argument('--zoom', action='store_true', help='v9: zoom re-detection of small ambiguous person outputs on NPU_2 (same NEWMRT480A DXNN, one 480x384 '
                    'canvas per frame; candidates from the previous frame, windows cut from the current one); requires --det rt480 --pipeline --no-vit; default off = unchanged')
    ap.add_argument('--zoom-lo', type=float, default=0.03); ap.add_argument('--zoom-hi', type=float, default=0.40); ap.add_argument('--zoom-maxh', type=float, default=32.0)
    ap.add_argument('--zoom-max', type=int, default=12); ap.add_argument('--zoom-ctx', type=float, default=3.0); ap.add_argument('--zoom-min', type=int, default=48)
    ap.add_argument('--zoom-target', type=float, default=80.0); ap.add_argument('--zoom-maxscale', type=float, default=4.0)
    ap.add_argument('--zoom-floor', type=float, default=0.0, help='v9: with more than 2 candidates scale to this person height instead of --zoom-target (0 = off; '
                    'PREREG_ZOOM.json used 64 so that four 192-px windows fit one canvas)')
    ap.add_argument('--zoom-acc', type=float, default=0.50); ap.add_argument('--zoom-acch', type=float, default=40.0)
    ap.add_argument('--zoom-lost', action='store_true', help='v9: also zoom the predicted boxes of LOST person tracks (height < --zoom-maxh) of the previous frame')
    ap.add_argument('--zoom-cls', default='', help="v9 hook point (default '' = accept all): 'module:function' importable from the script folder, called once per frame on the "
                    'accepted zoom boxes BEFORE they are appended: fn(boxes [m,7] frame x1,y1,x2,y2,conf,cls,window, gray frame, canvas uint8 480x384, windows) -> bool mask [m] '
                    '(True = keep); a classifier gate against false zoom boxes plugs in here')
    ap.add_argument('--judge', default='', help='v10: path of RELINK_JUDGE_FROZEN.json = the learned relink judge replaces the KeepID rule in the --relabel path '
                    '(relink_online_judge.OnlineJudge; same re-ID engine / crops on NPU_2); requires --relabel; default off = unchanged')
    ap.add_argument('--split6', action=argparse.BooleanOptionalAction, default=True, help='the detector DXNNs have the six split-head outputs (box_s8/16/32, cls_s8/16/32) of tensor_requirements.md and the host merges them into boxes/scores (default on); --no-split6 = DXNNs with the two outputs boxes/scores')
    ap.add_argument('--bank-dir', default='', help='v28: folder watched for new six-head detector DXNN files (default: models/detect and its sub-folders)')
    ap.add_argument('--no-bank-auto', action='store_true', help='v28: do not list new dxnn/ files by themselves')
    ap.add_argument('--min-person-h', type=float, default=0.0, help='v27: drop person detections under this height in px before tracking (user 11-10: 32 = smallest hand-labelled person 33 px); 0 = off = unchanged')
    ap.add_argument('--live-port', type=int, default=0, help='v14: live camera mode, serve the annotated MJPEG stream on this port (0 = off = normal file run)')
    ap.add_argument('--live-bind', default='127.0.0.1', help='address of the web page. The page has no password; the default is this device only, give the address of a trusted network to view it from another computer')
    ap.add_argument('--live-seconds', type=float, default=600, help='v14: live run length in seconds')
    ap.add_argument('--measure-dir', default='/mnt/lwir_data/results/live_measure', help='v22c: folder for the measurement button files (a YYYYMMDD subfolder per day); the run folder when it cannot be written')
    ap.add_argument('--live-wait', type=float, default=90, help='v21: seconds to wait for the first camera frame; 0 = never give up: the page and the gimbal link stay up with a placeholder picture and the camera is reopened every 180 s until --live-seconds ends')
    ap.add_argument('--replay-bytetrack-library', default=os.environ.get('LWIR_REPLAY_BYTETRACK_LIBRARY', ''),
                    help='Optional path override for the C++ ByteTrack library used by camera preview and playback')
    ap.add_argument('--replay-root', default='', help='11-12: folder of the label site recordings (<root>/<name>/images/000001.png ...); the web page can play one of them instead of the camera (GET /replay, POST /replay/start, /replay/stop). Empty or missing folder = replay unavailable')
    ap.add_argument('--live-cam', default='', help='v14: path of the cam_pipe helper (default: src/camera/cam_pipe)')
    ap.add_argument('--live-map', choices=['fixed', 'pct'], default='pct', help='v14: 16 -> 8 bit mapping: fixed = linear 0..65535 (team runtime), pct = smoothed 1..99 percentile window')
    ap.add_argument('--live-scale', type=float, default=1.5, help='v14: display scale of the streamed picture')
    ap.add_argument('--mamba', default='', help='v18: folder of the Mamba relabeler (replaces --judge); needs --relabel and --gmc')
    ap.add_argument('--mamba-crops', type=int, default=3, help='v18: appearance crops per frame for the Mamba relabeler')
    ap.add_argument('--gimbal', default='', help='v19: serial device of the gimbal (e.g. /dev/ttyUSB0). Live server only: a click on a person on the web page makes the gimbal follow that ID')
    ap.add_argument('--gimbal-bridge', default='', help='v19: path of the gimbal_bridge program (default: src/gimbal/gimbal_bridge_v25)')
    ap.add_argument('--gimbal-speed', type=int, default=30, help='v19: manual speed and highest following speed, in device units')
    ap.add_argument('--gimbal-aim-y', type=float, default=0.3, help='v22e: the point of the person box the gimbal brings to the image centre, as a fraction of the box height from its top (0.3 = chest; 0.5 = box centre, the team rule)')
    ap.add_argument('--gimbal-kp', type=float, default=30.0, help='v19: proportional gain of the following control')
    ap.add_argument('--gimbal-x-sign', type=int, default=1, choices=[-1, 1], help='v19: -1 if the pan axis runs the wrong way')
    ap.add_argument('--gimbal-y-sign', type=int, default=1, choices=[-1, 1], help='v19: -1 if the tilt axis runs the wrong way')
    ap.add_argument('--gimbal-opts', default='', help='v19: further options of gimbal_bridge, as one string, for example: --deadband 0.06 --input-timeout-ms 400')
    ap.add_argument('--gui-host', default='', help="v20: address of the PC that runs the team's GUI (pc-application). Sends it the camera video (H.264/RTP, UDP 5000) and the track boxes (UDP 5001) and takes its commands (TCP 5002) through the team's own code in gimbal_bridge. Needs --gimbal")
    ap.add_argument('--gui-bind', default='0.0.0.0', help='v20: address the GUI command listener binds to (the board address on the GUI network, or 0.0.0.0)')
    ap.add_argument('--gui-video-port', type=int, default=5000); ap.add_argument('--gui-metadata-port', type=int, default=5001)
    ap.add_argument('--gui-command-port', type=int, default=5002, help='v20: 0 = send only, no command listener')
    ap.add_argument('--gui-encoder', default='mpph264enc', help='v20: GStreamer encoder: mpph264enc (board hardware), x264enc (software, tests), mppjpegenc / jpegenc with --gui-codec jpeg')
    ap.add_argument('--gui-codec', choices=['h264', 'jpeg'], default='h264'); ap.add_argument('--gui-bitrate', type=int, default=3000, help='v20: kbit/s'); ap.add_argument('--gui-fps', type=int, default=30)
    ap.add_argument('--gui-map', choices=['team', 'pct'], default='team', help="v20: picture sent to the GUI. team = the raw 16-bit frame, mapped like the team runtime (fixed 0..65535); pct = the same 1..99 percent mapping as the web page")
    ap.add_argument('--gui-opts', default='', help='v20: further --gui-* options of gimbal_bridge, as one string')
    ap.add_argument('--mamba-link-without-appearance', action='store_true', help='v19: also link when a track has no appearance. Off by default: on TMOT train and the FLIR test nearly all wrong links are of this kind, but half of the known correct links are too')
    ap.add_argument('--mamba-duplicates', action='store_true', help='v18: hide the younger of two person boxes that overlap heavily and look alike (one person boxed twice)')
    ap.add_argument('--confirm-gate', default='', help='v11: path of GATE_GRU32_FROZEN.json = the learned confirmation gate replaces 3FrameConfirm (GateTracker); requires --tracker bytetrack --class-gate --no-vit; default off = unchanged')
    ap.add_argument('--judge-crops', type=int, default=2, help='v10: with --judge, crops per frame on NPU_2 (new IDs first, then refresh); 0 = every eligible crop')
    APP.args = ap.parse_args()
    APP.FEED_MODE = frame_feed.load_mode()
    APP.FEED = None
    APP.FEED_READY = False
    ready_prefetch_compatible = frame_feed.ready_prefetch_compatible(APP.args)
    APP.FEED_HOT_SWITCH = ready_prefetch_compatible and os.environ.get('LWIR_NPU_HOT_SWITCH') == '1'
    try:
        ready_prefetch = frame_feed.ready_prefetch_enabled(
            APP.FEED_MODE, os.environ.get('LWIR_LIVE_READY_PREFETCH', ''), ready_prefetch_compatible)
    except ValueError as exc:
        ap.error(str(exc))
    settings.WARM_MIN, settings.EDGE_BIRTH, settings.EDGE_PX = int(APP.args.warm_min), float(APP.args.edge_birth), float(APP.args.edge_px)   # v24f
    settings.WARM_MAX = max(0, int(APP.args.warm_max))   # v33
    if APP.args.prepare: return prepare(APP.args)
    if APP.args.relabel and (APP.args.tracker != 'bytetrack' or not APP.args.no_vit or APP.args.vit_budget or APP.args.vit_bench_n):
        raise SystemExit('--relabel requires --tracker bytetrack --no-vit (no --vit-budget / --vit-bench-n)')
    if APP.args.gmc_thread and not (APP.args.gmc and APP.args.pipeline): raise SystemExit('--gmc-thread requires --gmc --pipeline')
    if APP.args.npu_depth < 1 or (APP.args.npu_depth > 1 and not (APP.args.pipeline and APP.args.no_vit)): raise SystemExit('--npu-depth > 1 requires --pipeline --no-vit')
    if APP.args.rl_async and not APP.args.relabel: raise SystemExit('--rl-async requires --relabel')
    if APP.args.det in ('m640', 's6', 's480', 'ft1', 'ft9') and (APP.args.zoom or not APP.args.split6): raise SystemExit('--det m640: six split-head outputs only, no --zoom (the zoom canvas is a 480x384 tile model)')   # v22
    if APP.args.det == 's640': APP.args.split6 = False   # YOLO26s build with the two outputs boxes + scores
    if APP.args.zoom and not (APP.args.det == 'rt480' and APP.args.pipeline and APP.args.no_vit and not APP.args.vit_bench_n and not APP.args.vit_budget):
        raise SystemExit('--zoom requires --det rt480 --pipeline --no-vit (no --vit-bench-n / --vit-budget)')
    if APP.args.zoom_lost and not APP.args.zoom: raise SystemExit('--zoom-lost requires --zoom')
    if APP.args.gimbal and not APP.args.live_port: raise SystemExit('--gimbal needs --live-port')
    if APP.args.gui_host and not APP.args.gimbal: raise SystemExit('--gui-host needs --gimbal (the GUI commands go to the gimbal executor; for a test without a gimbal give a pseudo terminal)')
    if APP.args.mamba and (not APP.args.relabel or not APP.args.gmc or APP.args.judge or not Path(APP.args.mamba).is_dir()): raise SystemExit('--mamba needs an existing folder, --relabel and --gmc, and replaces --judge')
    if APP.args.judge and not APP.args.relabel: raise SystemExit('--judge requires --relabel <re-ID DXNN> (the crops run on the same engine)')
    if APP.args.judge and (APP.args.judge_crops < 0 or not Path(APP.args.judge).is_file()): raise SystemExit('--judge needs an existing RELINK_JUDGE_FROZEN.json and --judge-crops >= 0')
    if APP.args.confirm_gate and (APP.args.tracker != 'bytetrack' or not APP.args.class_gate or not APP.args.no_vit or not Path(APP.args.confirm_gate).is_file()): raise SystemExit('--confirm-gate needs an existing GATE_GRU32_FROZEN.json, --tracker bytetrack --class-gate --no-vit')
    GATE_FN = NumpyGRUGate(APP.args.confirm_gate) if APP.args.confirm_gate else None; gate_trackers = []   # v11
    if APP.args.zoom_cls and (not APP.args.zoom or ':' not in APP.args.zoom_cls): raise SystemExit('--zoom-cls module:function requires --zoom')
    if APP.args.zoom and not (0 <= APP.args.zoom_lo < APP.args.zoom_hi <= 1 and APP.args.zoom_maxh > 0 and APP.args.zoom_max >= 1 and APP.args.zoom_ctx > 0 and APP.args.zoom_min >= 8
                          and APP.args.zoom_target > 0 and APP.args.zoom_maxscale >= 1 and APP.args.zoom_floor >= 0 and 0 < APP.args.zoom_acc <= 1 and APP.args.zoom_acch > 0):
        raise SystemExit('bad --zoom-* parameters')
    if APP.args.class_gate and (not APP.args.no_vit or APP.args.vit_budget):
        raise SystemExit('--class-gate requires --no-vit (no --vit-budget): only the ViT-free trackers were verified')
    if APP.args.n_init < 1 or not APP.args.radius > 0: raise SystemExit('--n-init must be >= 1 and --radius > 0')
    if APP.args.confirm_ms < 0: raise SystemExit('--confirm-ms must be >= 0')
    print('WARM p95: %s' % WARM_P95_MODE, flush=True)
    APP.CONFIRM = ConfirmByTime(APP.args.confirm_ms) if APP.args.confirm_ms > 0 else None   # 11-11: confirmation rule in ms; used by the live loop only (offline runs have no arrival clock)
    trk_extra = APP.args.n_init != 2 or APP.args.radius != 2.0             # False = every tracker code path below is the pre-idsw_fix one
    if trk_extra and (APP.args.tracker != 'rebind' or not APP.args.no_vit):
        print('NOTE --n-init %d --radius %g: verified offline only with --tracker rebind --no-vit' % (APP.args.n_init, APP.args.radius), flush=True)
    if APP.args.vit_bench_n < 0 or (APP.args.vit_bench_n > 0 and not APP.args.no_vit): raise SystemExit('--vit-bench-n N > 0 requires --no-vit (cost bench only)')
    if APP.args.vit_budget < 0 or (APP.args.vit_budget > 0 and (APP.args.no_vit or APP.args.tracker != 'rebind' or APP.args.vit_sync or APP.args.vit_bench_n)):
        raise SystemExit('--vit-budget N > 0 requires the ViT (no --no-vit) and --tracker rebind, and excludes --vit-sync / --vit-bench-n')
    budget = APP.args.vit_budget                                   # 0 = off: every code path below is the pre-lostfirst one
    t_start = time.monotonic()
    use_vit = not APP.args.no_vit
    tag = APP.args.tracker if APP.args.tracker == 'bytetrack' else ('rebind_novit' if not use_vit else 'rebind_%s_e%d_k%d_s%s' % (APP.args.vit, APP.args.vit_every, APP.args.k, APP.args.sim))
    if use_vit and APP.args.app_first: tag += '_af%g%s' % (APP.args.gate, '_nofb' if APP.args.no_motion_fallback else '')
    if use_vit and APP.args.vit_sync: tag += '_sync%d' % APP.args.sync_k
    if use_vit and APP.args.app_first and APP.args.hybrid_fallback: tag += '_hfb'
    if APP.args.det != 'rt512': tag += '_' + APP.args.det
    if APP.args.pipeline: tag += '_pipe'
    if APP.args.synth_plan: tag += '_synth'
    if APP.args.vit_bench_n: tag += '_vb%d' % APP.args.vit_bench_n
    if budget: tag = 'rebind_%s_%s%d_s%s_g%g' % (APP.args.vit, 'lf', budget, APP.args.sim, APP.args.gate) + ('_' + APP.args.det if APP.args.det != 'rt512' else '') + ('_pipe' if APP.args.pipeline else '') + ('_synth' if APP.args.synth_plan else '')
    if trk_extra: tag += ('_n%d' % APP.args.n_init if APP.args.n_init != 2 else '') + ('_r%g' % APP.args.radius if APP.args.radius != 2.0 else '')
    if APP.args.gmc: tag += '_gmc' + ('' if (APP.args.gmc_ds, APP.args.gmc_corners, APP.args.gmc_mindist) == (2, 1000, 1.0) else '_ds%d_c%d_md%g' % (APP.args.gmc_ds, APP.args.gmc_corners, APP.args.gmc_mindist))
    if APP.args.class_gate: tag += '_cg'
    if APP.args.relabel: tag += '_rl'
    tag += ('_gt' if APP.args.gmc_thread else '') + ('_d%d' % APP.args.npu_depth if APP.args.npu_depth > 1 else '') + ('_ra' if APP.args.rl_async else '') + ('_ot%d' % APP.args.ort_threads if APP.args.ort_threads != 1 else '') + ('_pl' if APP.args.preload else '')
    if APP.args.zoom:   # v9: '_zoom' + the non-default parameters
        zd_ = (('lo', APP.args.zoom_lo, 0.03), ('hi', APP.args.zoom_hi, 0.40), ('h', APP.args.zoom_maxh, 32.0), ('n', APP.args.zoom_max, 12), ('c', APP.args.zoom_ctx, 3.0), ('m', APP.args.zoom_min, 48),
               ('t', APP.args.zoom_target, 80.0), ('s', APP.args.zoom_maxscale, 4.0), ('f', APP.args.zoom_floor, 0.0), ('a', APP.args.zoom_acc, 0.50), ('ah', APP.args.zoom_acch, 40.0))
        tag += '_zoom' + ''.join('_%s%g' % (k_, v_) for k_, v_, d_ in zd_ if v_ != d_) + ('_zl' if APP.args.zoom_lost else '') + ('_zc' if APP.args.zoom_cls else '')
    if APP.args.judge: tag += '_judge' + ('_jc%d' % APP.args.judge_crops if APP.args.judge_crops != 2 else '')   # v10
    if APP.args.confirm_gate: tag += '_cgate'   # v11
    APP.det_names, APP.det_yoff = DET[APP.args.det]['names'], DET[APP.args.det]['yoff']; APP.det_shape = {**SHAPE, **SHAPE_TT, **SHAPE_RT480, **SHAPE_W480}
    if APP.args.mock:
        make_mock_dx(APP.args); APP.DATA = Path(APP.args.mock_data); APP.out = HERE / 'tmp' / 'mock_track' / tag; APP.out.mkdir(parents=True, exist_ok=True); APP.man = {}
        if not hasattr(os, 'sched_setaffinity'): os.sched_setaffinity = lambda *a: None
    else:
        APP.man = json.loads((HERE / 'models' / 'MANIFEST.json').read_text())
        for n in APP.det_names + ([APP.args.vit] if (use_vit or APP.args.vit_bench_n) else []) + ([APP.args.relabel] if APP.args.relabel else []): assert sha(HERE / 'models' / 'links' / (n + '.dxnn')) == APP.man[n], n
        if not APP.args.data and not APP.args.live_port: raise SystemExit('--data <frame folder with coco.json> is required')
        APP.DATA = Path(APP.args.data); APP.out = HERE / (APP.args.out or ('track_' + tag)); APP.out.mkdir(exist_ok=False)
        waited = wait_m1_free()
        sys.path.insert(0, str(HERE / 'src' / 'detect'))   # native_post.py + post.so (built by setup.sh) sit in src/detect
    import cv2, dx_engine as dx, native_post; APP.cv2 = cv2; APP.dx = dx; APP.native_post = native_post
    APP.join_tiles = npu_engine.join_tiles
    APP.fast_nms = APP.native_post.fast_nms
    Decoder = npu_engine.Decoder
    APP.cv2.setNumThreads(1); os.sched_setaffinity(0, {4, 5, 6, 7})
    if not APP.args.mock:
        st = APP.dx.DeviceStatus.get_current_status(0)
        assert st.get_memory_used() < 1024 ** 2 and all(st.get_core_utilization(i) < 1 for i in range(3)), 'NPU not idle'
    cfg = APP.dx.Configuration()
    for item, attr in [('CUSTOM_INTRA_OP_THREADS', 'CUSTOM_INTRA_OP_THREADS_NUM'), ('CUSTOM_INTER_OP_THREADS', 'CUSTOM_INTER_OP_THREADS_NUM')]:
        cfg.set_attribute(getattr(APP.dx.Configuration.ITEM, item), getattr(APP.dx.Configuration.ATTRIBUTE, attr), str(APP.args.ort_threads) if item == 'CUSTOM_INTRA_OP_THREADS' else '1'); cfg.set_enable(getattr(APP.dx.Configuration.ITEM, item), True)
    APP.LIVE = None
    if APP.args.live_port:
        LiveImages = camera_link.LiveImages
        APP.images = LiveImages()
    else:
        coco = json.loads((APP.DATA / 'coco.json').read_text(encoding='utf-8'))
        APP.images = sorted(coco['images'], key=lambda x: x['file_name'])
        if APP.args.max_frames: APP.images = APP.images[:APP.args.max_frames]
    synth_hide = None; synth_st = None
    if APP.args.synth_plan:
        synth_plan, synth_hide = load_synth_plan(APP.args.synth_plan); names_ = {Path(im['file_name']).name for im in APP.images}
        if not APP.args.max_frames: assert set(synth_hide) <= names_, 'SYNTH_PLAN frames missing from coco.json: %d' % len(set(synth_hide) - names_)
        synth_st = dict(plan=synth_plan['_path'], plan_sha256=synth_plan['_sha256'], n_events=synth_plan['n_events'], iou_thr=synth_plan['iou_thr'],
                        planned_frames_in_run=len(set(synth_hide) & names_), planned_boxes_in_run=int(sum(len(synth_hide[k]) for k in set(synth_hide) & names_)),
                        frames_applied=0, hidden_dets=0, planned_boxes_without_det=0)
        print('SYNTH_PLAN', json.dumps(synth_st), flush=True)
    # ---- v23: live model change (catalog, request state) ----
    import threading as md_th
    MODEL_CAT = [('w480', 'NEWMRW480', 'YOLO26m_(FLIR) 640x480'), ('s480', 'S480', 'YOLO26s 640x480'), ('m640', 'M640', 'YOLO26m_(FLIR) 640x640'),
                 ('s6', 'S6', 'YOLO26s 640x640')]   # v34 (11-10): the old own-camera fine-tune entry was taken off the list on the user request (models/FT1.dxnn kept); the new indoor file is listed from dxnn/   # v32 (11-10): the v9 trial entry was taken off the list on the user request (file kept in dxnn/)
    APP.MODEL = dict(lock=md_th.Lock(), req=None, busy=False, last=None, history=[], catalog=[])
    if APP.args.det in [c_[0] for c_ in MODEL_CAT]:   # only for the single-engine detectors; the two-tile builds are not in the catalog
        APP.MODEL['catalog'] = [dict(id=0, det=c_[0], key=c_[1], name=c_[2]) for c_ in MODEL_CAT if APP.args.mock or (c_[1] in APP.man and (HERE / 'models' / 'links' / (c_[1] + '.dxnn')).exists())]
        for i_, c_ in enumerate(APP.MODEL['catalog']): c_['id'] = i_ + 1
    APP.BANK_DIR = Path(APP.args.bank_dir) if APP.args.bank_dir else HERE / 'models' / 'detect'
    APP.BANK_LOCK = md_th.Lock()
    APP.bank_refresh = model_scan.bank_refresh
    APP.bank_add_ = model_scan.bank_add_
    APP.bank_refresh(force=True); BANK_T[0] = 0.0
    APP.SEL_FILE = HERE / 'last_selection.json'; APP.SEL = {}; APP.SEL_ON = bool(APP.args.live_port) and not APP.args.mock; APP.SEL_LOCK = md_th.Lock()   # v36
    APP.sel_save = page_settings.sel_save
    APP.RULES = dict(lock=md_th.Lock(), pending=None, tracker='rebind' if APP.args.tracker == 'rebind' else 'confirm3', relabel=bool(APP.args.relabel), last=None, changes=0)   # v24g. v40: the GRU confirmation choice is gone (user 11-10)
    # ---- v40 (user 11-10): tree page. app = BoT-SORT appearance, lead = 예측 제어 (default OFF, not kept over a restart), warm_prev = Filter numbers kept while 온도 Filter is off
    APP.CTL = dict(app='repvgg', lead=False, lead_ms=150, lv=None, lead_px=0.0, warm_prev=(45000, 0), lead_motion=None)
    APP.REID = dict(lock=md_th.Lock(), req=None, cur=APP.args.relabel or '', last=None, history=[])   # v41: appearance model picked on the page; req is taken by the frame loop between two frames
    mk_tracker_ = page_settings.mk_tracker_
    APP.ctl_state = page_settings.ctl_state
    APP.RULES_PRESETS = [('confirm3', 'ByteTrack'), ('rebind', 'ByteTrack (위치 재결합 켜짐)'),
                     ('ocsort', 'OC-SORT (가려졌다 나온 사람 위치 보정, 시험 안 함)'), ('botsort', 'BoT-SORT (ByteTrack + 카메라 움직임 보정 + RepVGG_(TMOT,CAMEL) 외형 비교, 시험 안 함)'), ('sort', 'SORT (가장 단순: Kalman + 겹침만, 놓치면 바로 새 번호, 시험 안 함)')]   # v31: AltTracker. v36: BoT-SORT and SORT in, C-BIoU off the list (user 11-10)
    APP.rules_state = page_settings.rules_state
    APP.rules_state0_ = APP.rules_state   # stab: stab / stab_px next to the v24g fields
    APP.rules_state = page_settings.rules_state_2
    APP.rules_request = page_settings.rules_request
    APP.rules_request0_ = APP.rules_request   # stab: the 'stab' key is taken here, every other key goes to the v24g parser unchanged
    APP.rules_request = page_settings.rules_request_2
    APP.MODEL_TEXT = {0: '바뀜', 1: '없는 모델 번호', 2: '다른 변경이 진행 중', 3: '모델을 만들지 못함', 4: '실패해서 이전 모델로 되돌림', 5: '되돌리기도 실패', 6: '준비 안 됨'}
    APP.model_state = page_settings.model_state
    APP.model_request = page_settings.model_request
    # ---- v36 (user 11-10): the last choices made on the page (detector model, tracker, Re-ID switch, position predictor, filter numbers) are kept in a small file
    # and taken again at the next start. Live runs only. A missing / broken file or a model that is no longer listed = the normal defaults.
    if APP.SEL_ON:
        try: sv_ = json.loads(APP.SEL_FILE.read_text(encoding='utf-8'))
        except (OSError, ValueError): sv_ = {}
        if isinstance(sv_, dict): APP.SEL.update({k_: sv_[k_] for k_ in ('det', 'tracker', 'relabel', 'predictor', 'pred_show', 'warm_min', 'warm_max', 'edge_birth', 'edge_px', 'app', 'lead_ms', 'reid', 'confirm_ms') if k_ in sv_})
        try:   # v40
            if isinstance(sv_, dict) and isinstance(sv_.get('warm_prev'), list) and len(sv_['warm_prev']) == 2: APP.CTL['warm_prev'] = (int(sv_['warm_prev'][0]), int(sv_['warm_prev'][1])); APP.SEL['warm_prev'] = list(APP.CTL['warm_prev'])
        except (ValueError, TypeError): pass
        qs_ = [{k_: [str(APP.SEL[k_])] for k_ in ('warm_min', 'warm_max', 'edge_birth', 'edge_px') if k_ in APP.SEL}]
        if 'tracker' in APP.SEL: qs_.append({'tracker': [str(APP.SEL['tracker'])]})
        if 'relabel' in APP.SEL: qs_.append({'relabel': ['1' if APP.SEL['relabel'] else '0']})
        if 'predictor' in APP.SEL: qs_.append({'predictor': [str(APP.SEL['predictor'])]})
        if 'pred_show' in APP.SEL: qs_.append({'pred_show': ['1' if APP.SEL['pred_show'] else '0']})
        if 'app' in APP.SEL: qs_.append({'app': [str(APP.SEL['app'])]})   # v40
        if 'lead_ms' in APP.SEL: qs_.append({'lead_ms': [str(APP.SEL['lead_ms'])]})   # v40
        if 'confirm_ms' in APP.SEL: qs_.append({'confirm_ms': [str(APP.SEL['confirm_ms'])]})   # 11-11: a value set on the page wins over --confirm-ms, like the other Filter numbers
        if isinstance(sv_, dict) and 'stab' in sv_: APP.SEL['stab'] = bool(sv_['stab']); qs_.append({'stab': ['1' if APP.SEL['stab'] else '0']})   # stab
        for q_ in qs_:
            if not q_: continue
            try: ok_, tx_ = APP.rules_request(q_)
            except Exception as e_: ok_, tx_ = False, repr(e_)[:120]
            if not ok_:
                for k_ in q_: APP.SEL.pop(k_, None)
            print('LAST selection %s -> %s (%s)' % (sorted(q_), 'taken' if ok_ else 'dropped', tx_), flush=True)
        dv_ = APP.SEL.pop('det', None)
        APP.sel_save()   # the model name is written back only after the change worked: a model that breaks the start is not tried again
        if dv_ and dv_ != APP.args.det:
            c_ = next((x_ for x_ in APP.MODEL['catalog'] if x_['det'] == dv_), None)
            if c_ is None and isinstance(dv_, str) and dv_.startswith('bank:') and APP.MODEL['catalog'] and not APP.args.no_bank_auto:   # fix 11-11: a model file the scan lists by itself is not in the list yet at this point (listed at the second scan): wait for it beside the frame loop
                md_th.Thread(target=page_settings.sel_det_late, args=(dv_,), daemon=True).start(); print('LAST selection model %s -> waiting for the model scan' % dv_, flush=True)
            else:
                r_ = APP.model_request(c_['id']) if c_ is not None else 1
                print('LAST selection model %s -> %s' % (dv_, 'queued' if r_ == -1 else 'not taken (%s)' % APP.MODEL_TEXT.get(r_, '?')), flush=True)
    APP.pending, APP.eng = {}, {}
    APP.Engine = npu_engine.Engine
    failed = False; APP.rows = []; APP.preds = []; util = []; APP.tracks_out = []; rebind_log = []
    vit_queue = []           # pending crop requests: dict(kind, track_id, frame, idx, box(frame xyxy), src('cur'|'prev'))
    inflight = None          # (job meta, submit_ns)
    delivered = 0; vit_wait_total = 0.0; vit_lat = []; crops_queued = 0; dropped = 0; reb_tot = dict(motion=0, appearance=0, cand_frames=0, cand_crops=0, sync_frames=0, sync_crops=0)
    sync_lat = []
    lf_crop_ms = []; lf_npu = []; lf_frame_ms = [0.0]; lf_trackers = []          # --vit-budget (lost-first): per-crop ms, NPU ms, this frame's ms, trackers
    APP.bench_eng = None; APP.bench_inf = []; APP.bench_npu = []; APP.bench_st = dict(crops=0, real=0, centre=0, frames_no_person=0)   # --vit-bench-n
    APP.rl_eng = None; APP.relabeler = None; APP.rl_crop_ms = []; APP.rl_npu = []; APP.tracks_base = []                                     # --relabel (v5)
    APP.rl_async = None; APP.gmc_pool = None; APP.gmc_futs = {}; APP.gmc_objs = []; gmc_comp_ms = []                                  # v7 speed options
    zoom_eng = None; zoom_inflight = None; zoom_log = []; zoom_fates = {}                                            # v9 --zoom
    zoom_st = dict(candidates=0, windows=0, windows_skipped=0, canvases=0, raw_canvas_boxes=0, accepted=0, frames_with_candidates=0)
    try:
        for n in APP.det_names: APP.eng[n] = APP.Engine(n, CORE[n], APP.args.npu_depth)
        if APP.args.zoom:   # v9: the SAME tile-A DXNN loaded again, bound to NPU_2 (depth 1); one 480x384 canvas per frame; own Decoder (dec480[0] is busy in detect)
            zoom_eng = APP.Engine('ZOOM', 2, 1, model=APP.det_names[0]); zoom_x = np.zeros((1,) + ZOOM_HW + (3,), np.uint8); zoom_canvas = np.zeros(ZOOM_HW, np.uint8); dec_zoom = Decoder(*ZOOM_HW)
            zoom_gate = None
            if APP.args.zoom_cls:   # --zoom-cls hook: module:function from the script folder (classifier gate on the accepted zoom boxes)
                import importlib; mod_, fn_ = APP.args.zoom_cls.split(':', 1); sys.path.insert(0, str(HERE)); zoom_gate = getattr(importlib.import_module(mod_), fn_)
        vit = APP.Engine(APP.args.vit, 2) if use_vit else None
        if APP.args.relabel:
            from .reid.relabel_online import OnlineRelabel
            APP.rl_eng = APP.Engine(APP.args.relabel, 2); APP.rl_x = np.zeros((1, APP.rl_eng.H, APP.rl_eng.W, 3), np.uint8)
            if APP.args.mamba:   # v18
                sys.path.insert(0, str(Path(APP.args.mamba).resolve()))
                from mamba_relabel import MambaRelabel
                APP.relabeler = MambaRelabel(APP.args.mamba, crops=APP.args.mamba_crops, duplicates=APP.args.mamba_duplicates, need_appearance=not APP.args.mamba_link_without_appearance)
                print('MAMBA', json.dumps(dict(folder=str(APP.args.mamba), crops=APP.args.mamba_crops, veto=APP.relabeler.veto, duplicates=APP.args.mamba_duplicates)), flush=True)
            elif APP.args.judge:   # v10: the learned judge takes OnlineRelabel's place (same step / reset / stats / events / per_frame interface)
                from .reid.relink_online_judge import OnlineJudge
                APP.relabeler = OnlineJudge(APP.args.judge, crops=APP.args.judge_crops)
                print('JUDGE', json.dumps(dict(spec=str(APP.args.judge), sha256=APP.relabeler.spec_sha, config=APP.relabeler.spec['config'], threshold=APP.relabeler.thr, features=len(APP.relabeler.feats), crops=APP.args.judge_crops)), flush=True)
            else:
                APP.relabeler = OnlineRelabel(budget=True, every=APP.args.rl_every, k=APP.args.rl_k, new_k=APP.args.rl_new_k)
            APP.rl_embed = reid_embed.rl_embed
            if APP.args.rl_async:
                APP.AsyncEmbedder = npu_engine.AsyncEmbedder
                APP.rl_async = APP.AsyncEmbedder(APP.args.relabel)
        if APP.rl_eng is not None:   # v41: appearance model change while running
            reid_make_ = reid_embed.reid_make_
            if APP.SEL.get('reid') and APP.SEL['reid'] != APP.REID['cur']:
                rs_ = str(APP.SEL.pop('reid')); APP.sel_save()   # like the detector: the name is written back only after the change worked, so a model that kills the process while it loads is not tried again at the next start
                if any(x_['key'] == rs_ and x_['available'] for x_ in reid_catalog(HERE / 'models' / 'links', APP.man, APP.args.relabel or '')):
                    APP.REID['req'] = dict(key=rs_, t=time.time()); print('LAST selection reid %s -> queued' % rs_, flush=True)
                else:
                    print('LAST selection reid %s -> dropped (not installed)' % rs_, flush=True)
        if APP.args.gmc_thread:      # --gmc-thread: one worker, frames in submission order, fresh GMC per video (same warps as the serial path)
            from concurrent.futures import ThreadPoolExecutor
            APP.gmc_pool = ThreadPoolExecutor(max_workers=1)
            APP.gmc_job = camera_motion.gmc_job
            gmc_submit = camera_motion.gmc_submit
        else:
            gmc_submit = camera_motion.gmc_submit_2
        if APP.args.vit_bench_n > 0:
            APP.bench_eng = APP.Engine(APP.args.vit, 2); APP.bench_x = np.zeros((1, APP.bench_eng.H, APP.bench_eng.W, 3), np.uint8)
            APP.BENCH_CENTRE = np.array([288.0, 192.0, 352.0, 320.0])     # 64x128 person-sized box at the frame centre (no eligible person crop)
            APP.bench_crop = reid_embed.bench_crop
            vit_bench = reid_embed.vit_bench
        APP.dec = [Decoder(512, 384), Decoder(512, 384)]; APP.jo = np.empty((1, 20, 2 * anchors(512, 384)), np.float32)
        APP.dec480 = [Decoder(480, 384), Decoder(480, 384)]; APP.jo480 = np.empty((1, 20, 2 * anchors(480, 384)), np.float32)   # --det rt480
        APP.dec_s = Decoder(640, 640); APP.dec_tt = Decoder(480, 640); n_tt = anchors(480, 640); APP.mrg_out = [np.empty((1, 4, n_tt), np.float32), np.empty((1, 16, n_tt), np.float32)]; APP.half = np.float32(.5)
        APP.avg_merge = detect_frame.avg_merge
        APP._detect = detect_frame._detect
        detect = detect_frame.detect
        vit_x = np.zeros((1, vit.H, vit.W, 3), np.uint8) if vit else None
        APP.tracker = None; prev_vid = None; prev_gray = None; tr_seq = 0; rr_ptr = 0; gmc = None; gmc_fail = 0
        previous_motion_frame = previous_motion_time = None
        def crop_into(g, box):
            x1, y1, x2, y2 = box; a, b_, c, d = int(max(0, x1)), int(max(0, y1)), int(min(640, x2)), int(min(512, y2))
            if c - a < 2 or d - b_ < 2: return False
            vit_x[0] = APP.cv2.cvtColor(APP.cv2.resize(g[b_:d, a:c], (vit.W, vit.H), interpolation=APP.cv2.INTER_LINEAR), APP.cv2.COLOR_GRAY2RGB); return True
        def vit_sync(g_cur, g_prev, now_frame):
            """Collect a finished ViT job (blocking wait) and submit the next queued crop; returns ms spent waiting."""
            nonlocal inflight, delivered, vit_wait_total
            w = 0.0
            if inflight is not None:
                meta, t_sub = inflight; t0 = time.perf_counter_ns(); e = vit.wait()[0].ravel(); t1 = time.perf_counter_ns()
                w = (t1 - t0) / 1e6; vit_wait_total += w; vit_lat.append((t1 - t_sub) / 1e6); delivered += 1; inflight = None
                e = e / max(np.linalg.norm(e), 1e-12)
                r = APP.tracker.attach_embedding(meta['track_id'], meta['frame'], e, meta['kind']) if APP.tracker is not None else None
                if r: rebind_log.append(dict(frame=now_frame, new_id=r[0], lost_id=r[1], sim=round(r[2], 4), crop_frame=meta['frame']))
            # submit only on every-N frames (--vit-every), one job in flight, batch 1
            while inflight is None and vit_queue and (APP.tracker is None or APP.tracker.frame_id % APP.args.vit_every == 0 or APP.args.vit_every <= 1):
                meta = vit_queue.pop(0); g = g_cur if meta['src'] == 'cur' else g_prev
                if g is None or not crop_into(g, meta['box']): continue
                vit.submit(vit_x, now_frame); inflight = (meta, time.perf_counter_ns())
            return w
        APP.a = APP.eng[APP.det_names[0]]; APP.b = APP.eng[APP.det_names[1]] if len(APP.det_names) > 1 else None   # v16: --det w480 has one engine
        APP.FRAME_DY = [0.0]; APP.FRAME_ROWS = [None]   # FRAME_DY: output y shift (16.0 for 640x480 inputs, 0.0 for 640x512 frames)
        APP.to_canvas = detect_frame.to_canvas
        APP.mem = None
        if APP.args.live_port:
            import threading as lv_th, subprocess as lv_sp, http.server as lv_http, collections as lv_col, socket as lv_sock; APP.lv_th = lv_th; APP.lv_sp = lv_sp; APP.lv_http = lv_http; APP.lv_col = lv_col; APP.lv_sock = lv_sock
            APP.TILE_BUFS = [[np.zeros((1, 480, 384, 3), np.uint8), np.zeros((1, 480, 384, 3), np.uint8)] for _ in range(APP.args.npu_depth + 3)]   # ring of input buffers: the DXRT binding keeps every submitted input array alive (about 0.55 MB per tile and frame leaked), so live mode reuses fixed buffers
            APP.WHOLE_BUFS = [np.zeros((1, 480, 640, 3), np.uint8) for _ in range(APP.args.npu_depth + 3)]   # v16: --det w480
            APP.S640_BUFS = [np.full((1, 640, 640, 3), 114, np.uint8) for _ in range(APP.args.npu_depth + 3)]   # v16: --det s640
            _Live = camera_link._Live
            APP.LIVE = _Live(); APP.LIVE.stopped = False; APP.LIVE.cam_lost = False; APP.LIVE.q = APP.lv_col.deque(maxlen=3); APP.LIVE.cond = APP.lv_th.Condition(); APP.LIVE.arr = APP.lv_col.deque()
            APP.LIVE.camera_bytetrack = CameraByteTrack(APP.args)
            APP.LIVE.restart_supported = ready_prefetch_compatible and os.environ.get('LWIR_REAL_GUI_LAUNCH') == '1'
            APP.LIVE.restart_pending = False
            APP.LIVE.restart_stop = False
            APP.LIVE.instance = str(time.monotonic_ns())
            APP.LIVE.jpeg = b''; APP.LIVE.seq = 0; APP.LIVE.viewers = 0; APP.LIVE.pic_t = -1e9; APP.LIVE.pic_n = 0; APP.LIVE.snap_t = -1e9; APP.LIVE.jcond = APP.lv_th.Condition(); APP.LIVE.t0 = time.monotonic(); APP.LIVE.cam_fps = 0.0; APP.LIVE.done_ts = APP.lv_col.deque(maxlen=40)
            APP.LIVE.meas = None; APP.LIVE.meas_n = 0; APP.LIVE.meas_last = None; APP.LIVE.meas_lock = APP.lv_th.Lock(); APP.LIVE.meas_hist = []; APP.LIVE.rec = None   # v22: measurement button (v22b: history, v22d: video)
            APP.LIVE.lo = None; APP.LIVE.hi = None; APP.LIVE.last = None; APP.LIVE.log = open(APP.out / 'cam_pipe.log', 'w'); APP.LIVE.raws = APP.lv_col.deque(); APP.LIVE.rawq = APP.lv_col.deque(); APP.LIVE.warm_dropped = 0; APP.LIVE.last_frame_t = time.monotonic(); APP.LIVE.cam_fail_n = 0; APP.LIVE.cam_unrec_n = 0; APP.LIVE.cam_next_spawn = None; APP.LIVE.cam_hold = False; APP.LIVE.cam_reset = None; APP.LIVE.sys_lock = APP.lv_th.Lock(); APP.LIVE.cam_lock = APP.lv_th.Lock(); APP.LIVE.frames_in = 0; APP.LIVE.q_drop = 0; APP.LIVE.prev_t = None; APP.LIVE.upd_t = None; APP.LIVE.gap_n = 0; APP.LIVE.delays = APP.lv_col.deque(maxlen=2000); APP.LIVE.dlock = APP.lv_th.Lock(); APP.LIVE.perf = APP.lv_col.deque(maxlen=900); APP.LIVE.perf_acc = [time.perf_counter(), 0, 0.0, 0.0, 0.0]   # v26b: q_drop = camera frames thrown away because the tracker was 3 frames behind, gap_n = input gaps over 1 s, delays = (time, capture-to-picture ms) of the last frames. v24: last_frame_t = when the last camera frame arrived; v24d: cam_fail_n = cam_pipe exits in a row without a frame (0 = camera fine or repairing itself), cam_next_spawn = earliest next cam_pipe start (backoff)
            APP.LIVE_W, APP.LIVE_H = 640, 480
            page_handler.init_preview(APP.LIVE)
            APP.LIVE.stabq = APP.lv_col.deque()   # stab: correction (cx, cy) or None of every frame read, paired with LIVE.arr
            APP.lv_reader = camera_link.lv_reader
            APP.lv_spawn = camera_link.lv_spawn
            APP.lv_spawn()
            APP.lv_cam_wait = camera_link.lv_cam_wait
            APP.GB = None
            if APP.args.gimbal:   # v19
                import os as lv_os, shlex, glob as lv_glob; APP.lv_os = lv_os; APP.shlex = shlex; APP.lv_glob = lv_glob
                if abs(time.perf_counter() - time.monotonic()) > 0.01: raise SystemExit('perf_counter is not CLOCK_MONOTONIC here: the capture times sent to the gimbal bridge would be wrong')
                APP.gb_device = gimbal_link.gb_device
                APP.gb_dev_sig = gimbal_link.gb_dev_sig
                APP.gb_in_use = gimbal_link.gb_in_use
                _Gimbal = gimbal_link._Gimbal
                APP.GB = _Gimbal(); APP.GB.lock = APP.lv_th.Lock(); APP.GB.people = []; APP.GB.frame = 0; APP.GB.skipped = 0; APP.GB.state = dict(running=False, ready=False, auto=0, lost=0, moving=0, error=0, id=0, answer='')
                APP.GB.gui = bool(APP.args.gui_host); APP.GB.state['gui'] = dict(on=APP.GB.gui, client=0, given=0, sent=0, encoded=0, rtp_bytes=0, failed=0)
                APP.GB.log = open(APP.out / 'gimbal_bridge.log', 'w')
                APP.gui_opts_ = ['--gui-host', APP.args.gui_host, '--gui-bind', APP.args.gui_bind, '--gui-video-port', str(APP.args.gui_video_port), '--gui-metadata-port', str(APP.args.gui_metadata_port), '--gui-command-port', str(APP.args.gui_command_port),
                             '--gui-encoder', APP.args.gui_encoder, '--gui-codec', APP.args.gui_codec, '--gui-bitrate', str(APP.args.gui_bitrate), '--gui-fps', str(APP.args.gui_fps), '--gui-clip', '0,65535',
                             *APP.shlex.split(APP.args.gui_opts)] if APP.GB.gui else []   # v20. --gui-map pct sends the 8-bit picture scaled to 16 bit, so the clip is the same
                # v20: one writer thread owns the pipe. Short lines queue up (at most 64); a frame with its picture (about 0.6 MB) replaces the previous one that was not sent yet,
                # so the tracker never waits for the bridge and the bridge always gets the newest frame.
                # v26: the bridge can be started again while the server runs (gb_spawn + gb_watch below). Every bridge process has its own writer and reader thread; GB.proc is the current one.
                APP.GB.wq = APP.lv_col.deque(); APP.GB.wframe = None; APP.GB.wcond = APP.lv_th.Condition(); APP.GB.wopen = False; APP.GB.proc = None; APP.GB.dev = APP.args.gimbal; APP.GB.dev_sig = None; APP.GB.spawn_t = 0.0; APP.GB.restarts = -1; APP.GB.closing = False; APP.GB.busy = ''; APP.GB.mover = None; APP.GB.mover_t = {}
                APP.gb_send = gimbal_link.gb_send
                APP.gb_send_frame = gimbal_link.gb_send_frame
                APP.gb_writer = gimbal_link.gb_writer
                APP.gb_reader = gimbal_link.gb_reader
                APP.gb_spawn = gimbal_link.gb_spawn
                APP.gb_end = gimbal_link.gb_end
                gb_watch = gimbal_link.gb_watch
                if not APP.gb_spawn(): print('GIMBAL device %s is %s at the start: the tracker runs without it and connects when it can' % (APP.args.gimbal, 'in use by process ' + APP.GB.busy if APP.GB.busy else 'not there'), flush=True)
                APP.lv_th.Thread(target=gb_watch, daemon=True).start()
                gb_mover_watch = gimbal_link.gb_mover_watch
                APP.lv_th.Thread(target=gb_mover_watch, daemon=True).start()
            APP.lv_to8 = camera_link.lv_to8
            APP.live_next = camera_link.live_next
            live_publish = page_handler.live_publish
            APP.GB_PAGE = ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>LWIR live</title><style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0;padding:10px}"
                       "img{max-width:100%;display:block;margin:auto;cursor:crosshair}button{font-size:18px;padding:8px 24px;margin:6px 0;background:#b00020;color:#fff;border:0;border-radius:6px}#s{margin-left:12px}"
                       "#pad{display:inline-grid;grid-template-columns:repeat(3,64px);gap:4px;margin:6px 16px 6px 0;vertical-align:middle}#pad button{background:#333;padding:10px 0;margin:0;touch-action:none;user-select:none;-webkit-user-select:none}</style></head><body>"
                       "<div>사람을 클릭하면 짐벌이 그 사람을 따라갑니다. 수동: 키보드 방향키 또는 아래 화살표 버튼(누르는 동안 움직임). 정지: 스페이스 또는 정지 버튼. 이 페이지를 닫으면 짐벌이 멈춥니다.</div>"
                       "<div><div id='pad'><span></span><button data-x='0' data-y='1'>&#9650;</button><span></span><button data-x='-1' data-y='0'>&#9664;</button><button data-x='0' data-y='-1'>&#9660;</button><button data-x='1' data-y='0'>&#9654;</button></div>"
                       "<button id='b'>정지</button><span id='s'></span></div><div style='margin:6px 0'><button id='mst' style='background:#0b6'>측정 시작</button> <button id='msp' style='background:#555'>측정 끝</button> <span id='mi'></span></div><div id='mr' style='font-family:monospace;white-space:pre;font-size:14px;margin:4px 0'></div><img id='v' src='/stream'><script>"
                       "const v=document.getElementById('v'),S=document.getElementById('s');"
                       "function show(t){if(!t.running)S.textContent='짐벌 프로그램이 꺼져 있습니다';else if(t.error)S.textContent='짐벌 오류: 서버를 다시 켜야 합니다';"
                       "else if(t.lost)S.textContent=t.id+'번 놓침, 잠시 기다립니다';else if(t.auto)S.textContent=t.id+'번을 따라가는 중';else if(t.moving)S.textContent='수동으로 움직이는 중';else S.textContent='대기 중';"
                       "if(t.answer)S.textContent+=' ('+t.answer+')';}"
                       "function post(u){return fetch(u,{method:'POST',headers:{'X-Gimbal':'1'}}).then(r=>r.json()).then(show).catch(()=>{S.textContent='서버 연결 끊김'});}"
                       "v.addEventListener('click',e=>{const r=v.getBoundingClientRect();post('/gimbal/click?x='+((e.clientX-r.left)/r.width)+'&y='+((e.clientY-r.top)/r.height));});"
                       "document.getElementById('b').addEventListener('click',()=>post('/gimbal/stop'));"
                       "const K={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1]};let held=null;"
                       "addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();held=null;post('/gimbal/stop');return;}const m=K[e.key];if(!m)return;e.preventDefault();if(e.repeat||held===e.key)return;held=e.key;post('/gimbal/move?x='+m[0]+'&y='+m[1]);});"
                       "addEventListener('keyup',e=>{if(e.key===held){held=null;post('/gimbal/move?x=0&y=0');}});"
                       "addEventListener('blur',()=>{if(held){held=null;post('/gimbal/move?x=0&y=0');}});"
                       "let padDown=false;for(const b of document.querySelectorAll('#pad button')){const go=e=>{e.preventDefault();if(padDown)return;padDown=true;b.setPointerCapture&&b.setPointerCapture(e.pointerId);post('/gimbal/move?x='+b.dataset.x+'&y='+b.dataset.y);};"
                       "const stop=e=>{if(!padDown)return;padDown=false;post('/gimbal/move?x=0&y=0');};b.addEventListener('pointerdown',go);for(const n of ['pointerup','pointercancel','pointerleave'])b.addEventListener(n,stop);b.addEventListener('contextmenu',e=>e.preventDefault());}"
                       "setInterval(()=>post('/gimbal/ping'),400);post('/gimbal/ping');const MI=document.getElementById('mi'),MR=document.getElementById('mr');function f1(x){return x==null?'-':x.toFixed(1);}function mshow(m){if(m.on){MI.textContent='측정 중: '+m.frames+'프레임, '+m.seconds.toFixed(0)+'초';}else MI.textContent=m.last?'측정 '+m.last.n+'회차 끝, 저장: '+m.last.file:'측정 대기';const r=m.last;if(!r||!r.frames){MR.textContent=r?'프레임이 없습니다':'';return;}let t='측정 '+r.n+'회차: '+r.frames+'프레임, '+r.seconds.toFixed(1)+'초, 출력 '+f1(r.out_fps)+' fps, 카메라 '+f1(r.cam_fps)+' fps\n';t+='항목                    p50     p95     p99     최대   33ms 초과\n';for(const [k,lab] of [['delay','입력→출력 지연 (ms)'],['interval','출력 간격 (ms)'],['compute','처리 시간 (ms)'],['npu_wait','NPU 대기 (ms)'],['tracker','트래커 (ms)']]){const s=r[k];if(!s)continue;t+=(lab+'                        ').slice(0,22)+[s.p50,s.p95,s.p99,s.max].map(v=>f1(v).padStart(8)).join('')+(f1(s.over33)+'%').padStart(10)+'\n';}MR.textContent=t;}function mpost(u){return fetch(u,{method:'POST',headers:{'X-Gimbal':'1'}}).then(r=>r.json()).then(mshow).catch(()=>{MI.textContent='서버 연결 끊김';});}document.getElementById('mst').addEventListener('click',()=>mpost('/measure/start'));document.getElementById('msp').addEventListener('click',()=>mpost('/measure/stop'));setInterval(()=>fetch('/measure').then(r=>r.json()).then(mshow).catch(()=>{}),1000);"
                       "</script></body></html>").encode()
            if (HERE / 'page' / 'live_page.html').exists(): APP.GB_PAGE = (HERE / 'page' / 'live_page.html').read_bytes()   # v22: the designed page (gimbal pad + measurement button) next to this script; the inline page above is the fallback
            APP.meas_stats = measure.meas_stats
            APP.meas_dir = measure.meas_dir
            APP.meas_clean = measure.meas_clean
            APP.meas_start = measure.meas_start
            APP.meas_stop = measure.meas_stop
            APP.meas_state = measure.meas_state
            APP.meas_note = measure.meas_note
            APP.MEAS_FIELDS = {
                'name, memo': '사용자가 적은 측정 이름과 메모',
                'delay': '입력→출력 지연(ms): 카메라 프레임이 서버에 도착한 순간부터 그 프레임의 추적 결과가 나온 순간까지. 카메라 안에서 걸리는 시간은 빠짐',
                'interval': '출력 간격(ms): 추적 결과가 나오는 간격. 카메라 프레임 간격보다 짧아질 수 없음',
                'compute': '처리 시간(ms): 한 프레임을 넣고 결과를 받기까지 서버 안에서 걸린 시간. 파이프라인을 켜면 다음 프레임을 기다리는 시간이 들어감',
                'npu_wait': 'NPU 대기(ms): 검출 모델(NPU) 결과를 받을 때까지 기다린 시간, NPU 계산 포함',
                'tracker': '트래커(ms): 박스를 이전 프레임과 이어 붙이는 추적 계산 시간',
                'p50, p95, p99, max, mean': '프레임의 50/95/99%가 이 값 이하 / 가장 느린 프레임 / 평균',
                'over33': '33 ms(30 fps 한 프레임 시간)를 넘긴 프레임 비율(%)',
                'out_fps': '초당 추적 결과 수', 'cam_fps': '카메라가 보낸 초당 프레임 수(측정 구간 평균)',
                'rows': '프레임별 기록 t(측정 시작 후 초), delay_ms, compute_ms, npu_wait_ms, tracker_ms, cam_fps. 같은 내용이 같은 이름의 .csv에 한글 열 이름으로 있음'}
            APP.cam_usb_seen = camera_link.cam_usb_seen
            APP.cam_usb_gone = camera_link.cam_usb_gone
            APP.cam_usb_power = camera_link.cam_usb_power
            APP.cam_reset_work = camera_link.cam_reset_work
            APP.cam_reset_request = camera_link.cam_reset_request
            APP.sys_reboot = system_buttons.sys_reboot
            lv_Handler = page_handler.make_lv_Handler()
            lv_Server = page_handler.make_lv_Server()
            APP.lv_srv = lv_Server((APP.args.live_bind, APP.args.live_port), lv_Handler); APP.lv_srv.daemon_threads = True
            APP.lv_th.Thread(target=APP.lv_srv.serve_forever, daemon=True).start()
            APP.lv_srv_local = None
            if APP.args.live_bind not in ('', '0.0.0.0', '127.0.0.1', 'localhost'):   # v26b: the same pages on 127.0.0.1, so the keep-alive on the board can read /healthz while the page address (Tailscale) is away
                try:
                    APP.lv_srv_local = lv_Server(('127.0.0.1', APP.args.live_port), lv_Handler); APP.lv_srv_local.daemon_threads = True
                    APP.lv_th.Thread(target=APP.lv_srv_local.serve_forever, daemon=True).start()
                except OSError as e_: APP.lv_srv_local = None; print('LIVE no listener on 127.0.0.1:%d (%s)' % (APP.args.live_port, e_), flush=True)
            live_close = page_handler.live_close
            print('LIVE http://%s:%d/ for %d s' % (APP.args.live_bind, APP.args.live_port, APP.args.live_seconds), flush=True)
        if APP.args.preload:         # --preload: decode once, outside every timed interval
            t_pl = time.perf_counter(); APP.mem = [APP.to_canvas(APP.cv2.imread(str(APP.DATA / im_['file_name']), APP.cv2.IMREAD_GRAYSCALE | APP.cv2.IMREAD_ANYDEPTH), im_['file_name']) for im_ in APP.images]
            print('PRELOAD %d frames %.1f s' % (len(APP.mem), time.perf_counter() - t_pl), flush=True)
        read_frame = detect_frame.read_frame
        submit_tiles = detect_frame.submit_tiles
        APP.TIMING = stage_timing.start()   # None unless LWIR_STAGE_TIMING=1
        APP.FEED_INCOMPATIBLE = frame_feed.refuse_reason()
        APP.FEED = frame_feed.start()
        APP.FEED_READY = True
        if APP.FEED is not None:
            ready_prefetch = False
            read_frame, submit_tiles = APP.FEED.read_frame, APP.FEED.submit_time
        elif APP.LIVE is not None and APP.LIVE.restart_supported:
            def read_frame(_):
                return APP.live_next(fresh_only=True), 0.0
        def sync_embed(idxs, g_cur, arr_cur, frame_no):
            """--vit-sync: embed the candidate detections (arr rows) now, batch 1, one after another; a pending asynchronous job is collected first."""
            nonlocal inflight, delivered, vit_wait_total
            t0_ = time.perf_counter_ns(); got = {}
            if inflight is not None:
                meta, t_sub = inflight; e = vit.wait()[0].ravel(); vit_lat.append((time.perf_counter_ns() - t_sub) / 1e6); delivered += 1; inflight = None
                e = e / max(np.linalg.norm(e), 1e-12)
                r = APP.tracker.attach_embedding(meta['track_id'], meta['frame'], e, meta['kind'])
                if r: rebind_log.append(dict(frame=frame_no, new_id=r[0], lost_id=r[1], sim=round(r[2], 4), crop_frame=meta['frame']))
            for i_ in idxs:
                if not crop_into(g_cur, arr_cur[i_, :4]): continue
                vit.submit(vit_x, frame_no); e = vit.wait()[0].ravel(); got[i_] = e / max(np.linalg.norm(e), 1e-12)
            sync_lat.append((time.perf_counter_ns() - t0_) / 1e6); return got
        def budget_embed(idxs, g_, arr_, frame_no):
            """--vit-budget: embed the given detection rows of (g_, arr_) = this frame or the previous one NOW, batch 1, one after another
            (crop -> submit -> wait, as sync_embed / the --vit-bench-n cost bench). No asynchronous job exists in this mode."""
            got = {}
            for i_ in idxs:
                t0_ = time.perf_counter_ns()
                if g_ is None or arr_ is None or not crop_into(g_, arr_[i_, :4]): continue
                vit.submit(vit_x, frame_no); e = vit.wait()[0].ravel(); got[i_] = e / max(np.linalg.norm(e), 1e-12)
                dt_ = (time.perf_counter_ns() - t0_) / 1e6; lf_crop_ms.append(dt_); lf_frame_ms[0] += dt_
                if not APP.args.mock: lf_npu.append(vit.b.get_npu_inference_time() / 1000)
            return got
        g_next = None; t_sub_next = None; t_prev_emit = None; ahead = []     # v7: frames read + submitted ahead [(gray, submit ns)], --npu-depth of them
        if ready_prefetch:
            print('LIVE camera prefetch: only take the next frame before publication if it is queued', flush=True)
        if APP.args.pipeline and APP.images and not ready_prefetch:
            for k_ in range(min(APP.args.npu_depth, len(APP.images))):
                g_k, _ = read_frame(k_)
                if g_k is None:
                    break
                ahead.append((g_k, submit_tiles(g_k, k_))); gmc_submit(k_, g_k)
        do_model_swap = page_settings.do_model_swap
        for i, im in enumerate(APP.images):
            if APP.LIVE is not None and APP.LIVE.restart_stop:
                break
            vid = im['extra_info']['video_id']; reset = vid != prev_vid
            if APP.MODEL['req'] is not None:   # v23: a model change was asked for; taken here, between two frames
                rq_ = APP.MODEL['req']; APP.MODEL['busy'] = True
                for _ in ahead: APP.a.wait()   # frames already submitted to the old model: collected and dropped, they go in again below
                if APP.FEED is not None: APP.FEED.pause()
                code_ = do_model_swap(rq_['det'])
                if APP.FEED is not None: APP.FEED.resume(code_)
                if APP.args.pipeline and code_ != 5: ahead[:] = [(g_, submit_tiles(g_, i + n_)) for n_, (g_, _) in enumerate(ahead)]
                with APP.MODEL['lock']:
                    APP.MODEL['req'] = None; APP.MODEL['busy'] = False
                    APP.MODEL['last'] = dict(id=rq_['id'], result=code_, text=APP.MODEL_TEXT.get(code_, '?'), at=time.strftime('%H:%M:%S'), seconds=round(time.time() - rq_['t'], 1))
                    APP.MODEL['history'].append(dict(APP.MODEL['last'], det=rq_['det'], frame=i))
                if code_ == 0: APP.sel_save(det=APP.args.det)   # v36
                if rq_['req'] is not None and APP.GB is not None: APP.gb_send('N %d %d' % (rq_['req'], code_))
                if code_ == 5: raise RuntimeError('detector model could not be restored')
            if APP.RULES['pending'] is not None and APP.tracker is not None:   # v24g: tracker preset / relabel switch between two frames
                with APP.RULES['lock']: pd_ = APP.RULES['pending']; APP.RULES['pending'] = None
                if 'relabel' in pd_: APP.RULES['relabel'] = pd_['relabel']
                if 'predictor' in pd_: pos_select(pd_['predictor'])   # v32
                if 'tracker' in pd_ and pd_['tracker'] != APP.RULES['tracker']:
                    old_ = APP.RULES['tracker']; APP.RULES['tracker'] = pd_['tracker']
                    for k_, a_ in (('motion', 'rebinds_motion'), ('appearance', 'rebinds_app'), ('cand_frames', 'cand_frames'), ('cand_crops', 'cand_crops'), ('sync_frames', 'sync_frames'), ('sync_crops', 'sync_crops')):
                        reb_tot[k_] += getattr(APP.tracker, a_, 0)
                    if APP.relabeler is not None: APP.relabeler.reset()
                    APP.tracker = mk_tracker_(pd_['tracker'])   # v40: one factory for the page switch and the per-video reset
                    if APP.args.gmc: APP.tracker.kf = GKF()
                    print('RULES tracker %s -> %s at frame %d' % (old_, pd_['tracker'], i), flush=True)
                with APP.RULES['lock']: APP.RULES['last'] = dict(APP.RULES['last'] or {}, applied=time.strftime('%H:%M:%S'), tracker=APP.RULES['tracker'], relabel=APP.RULES['relabel'])
            if APP.REID['req'] is not None and APP.rl_eng is not None:   # v41: appearance model change asked for on the page
                import gc
                with APP.REID['lock']: rr_ = APP.REID['req']; APP.REID['req'] = None
                tr_ = time.perf_counter(); rold_ = APP.REID['cur']
                reid_drain_ = reid_embed.reid_drain_
                rok_, rnew_, rtx_ = reid_swap(rr_['key'], (APP.rl_eng, APP.rl_async, APP.rl_x), reid_make_, reid_drain_)
                if rok_:
                    APP.rl_eng, APP.rl_async, APP.rl_x = rnew_; rnew_ = None; gc.collect(); APP.REID['cur'] = rr_['key']; APP.sel_save(reid=rr_['key'])
                elif APP.SEL.get('reid') == rr_['key']: APP.SEL.pop('reid', None); APP.sel_save()   # a saved model that does not load is not tried again at the next start
                APP.REID['last'] = dict(key=rr_['key'], ok=bool(rok_), text=rtx_, at=time.strftime('%H:%M:%S'), seconds=round(time.perf_counter() - tr_, 1), frame=i)
                APP.REID['history'].append(dict(APP.REID['last'], old=rold_))
                print('REID %s -> %s at frame %d: %s (%.1f s)' % (rold_, rr_['key'], i, 'changed' if rok_ else 'FAILED, old model kept: ' + rtx_.encode('ascii', 'backslashreplace').decode(), time.perf_counter() - tr_), flush=True)
            if reset:
                if vit is not None and inflight is not None: vit_sync(None, None, i)   # drain before the reset
                if APP.tracker is not None:
                    for k_, a_ in (('motion', 'rebinds_motion'), ('appearance', 'rebinds_app'), ('cand_frames', 'cand_frames'), ('cand_crops', 'cand_crops'), ('sync_frames', 'sync_frames'), ('sync_crops', 'sync_crops')):
                        reb_tot[k_] += getattr(APP.tracker, a_)
                vit_queue.clear(); prev_vid = vid; prev_gray = None; rr_ptr = 0; prev_arr = None
                if APP.relabeler is not None: APP.relabeler.reset()
                if budget:
                    APP.tracker = LostFirstTracker(budget, sim=APP.args.sim, gate=APP.args.gate, vit_conf=APP.args.vit_conf, vit_min_h=APP.args.vit_min_h,
                                               radius=APP.args.radius, n_init=APP.args.n_init); lf_trackers.append(APP.tracker)
                else:
                    APP.tracker = ByteTrackTracker(APP.args.tracker, sim=APP.args.sim, app=use_vit, late_window=APP.args.late_window, app_first=APP.args.app_first, gate=APP.args.gate,
                                      motion_fallback=not APP.args.no_motion_fallback, vit_conf=APP.args.vit_conf, vit_min_h=APP.args.vit_min_h, hybrid=True, sync_k=APP.args.sync_k,
                                      hybrid_fallback=bool(use_vit and APP.args.app_first and APP.args.hybrid_fallback), radius=APP.args.radius, n_init=APP.args.n_init,
                                      class_gate=APP.args.class_gate)
                    if APP.LIVE is not None: APP.tracker = mk_tracker_(APP.RULES['tracker'])   # v40: live = the tracker chosen on the page survives a stream restart (before: back to the GRU gate tracker)
                    elif GATE_FN is not None:   # v11: the learned confirmation gate replaces 3FrameConfirm (GateTracker has its own confirm rule). v40: offline runs only
                        APP.tracker = GateTracker(gate=GATE_FN, thr_out=GATE_FN.thr_out, thr_low=GATE_FN.thr_low, n_init=APP.args.n_init, class_gate=APP.args.class_gate, radius=APP.args.radius); gate_trackers.append(APP.tracker)
                if APP.args.gmc:   # --gmc: fresh camera-motion state per video
                    previous_motion_frame = previous_motion_time = None
                    APP.tracker.kf = GKF()
                    if APP.args.gmc_thread: gmc = True
                    else: gmc_fail += gmc.n_fail if gmc is not None else 0; gmc = GMC(APP.args.gmc_ds); gmc.fp = dict(gmc.fp, maxCorners=APP.args.gmc_corners, minDistance=APP.args.gmc_mindist)
            async_vit = vit is not None and not budget                  # --vit-budget: every crop is synchronous inside / right after update()
            if APP.args.pipeline:
                if (ready_prefetch or replay_eval.lossless()) and not ahead:
                    # The last result was already published before we waited here.
                    g = APP.live_next(fresh_only=True)
                    if g is None:
                        break
                    t0 = submit_tiles(g, i)
                    gmc_submit(i, g)
                else:
                    g, t0 = ahead.pop(0)
                ra = APP.a.wait(); rb = APP.b.wait() if APP.b is not None else None; t1 = time.perf_counter_ns()
                read_ms = 0.0
                next_i = i + APP.args.npu_depth
                if next_i < len(APP.images):
                    if ready_prefetch or replay_eval.lossless():
                        # An empty queue must never delay this result, including during replay transitions.
                        g_n = APP.live_next(wait=False, fresh_only=True)
                    else:
                        g_n, read_ms = read_frame(next_i)
                    if g_n is not None:
                        ahead.append((g_n, submit_tiles(g_n, next_i)))
                        gmc_submit(next_i, g_n)
                vw = vit_sync(g, prev_gray, i) if async_vit else 0.0                                       # one sync per frame; the ViT job overlaps detection of i+1
                t_nms0 = time.perf_counter_ns(); p = detect(ra, rb); t2 = time.perf_counter_ns(); t3 = t2
            else:
                g, _ = read_frame(i); read_ms = 0.0
                if g is None:
                    break
                t0 = submit_tiles(g, i)
                vw = vit_sync(g, prev_gray, i) if async_vit else 0.0                                          # sync point 1
                ra = APP.a.wait(); rb = APP.b.wait() if APP.b is not None else None; t1 = time.perf_counter_ns(); t_nms0 = t1
                p = detect(ra, rb); t2 = time.perf_counter_ns()
                if async_vit: vw += vit_sync(g, prev_gray, i)                                                   # sync point 2
                t3 = time.perf_counter_ns()
            if APP.LIVE is not None and APP.LIVE.restart_stop:
                break
            arr = p[np.isin(p[:, 5].astype(int), list(TARGET))].astype(np.float32)                        # frame coords
            zoom_flags = None; z_ms = (0.0, 0.0); n_zacc = 0
            if zoom_inflight is not None:   # v9: collect + merge THIS frame's zoom canvas (submitted at the end of the previous iteration) before the tracker
                assert zoom_inflight[1] == i, (zoom_inflight[1], i)
                tz0_ = time.perf_counter_ns(); rz_ = zoom_eng.wait(); tz1_ = time.perf_counter_ns(); wins_, _, zlog_ = zoom_inflight; zoom_inflight = None
                pz_ = APP.fast_nms(dec_zoom(rz_)); normal_ = p[(p[:, 4] >= 0.10) & np.isin(p[:, 5].astype(int), list(TARGET)), :4]
                zacc_ = zoom_merge(pz_, wins_, normal_, APP.args.zoom_acc, APP.args.zoom_acch, zoom_fates)
                if zoom_gate is not None and len(zacc_):   # --zoom-cls hook: keep only the boxes the gate accepts (dropped ones counted as fate 'gated')
                    keep_ = np.asarray(zoom_gate(zacc_, g, zoom_canvas, wins_), bool).reshape(-1); assert keep_.shape == (len(zacc_),), keep_.shape
                    zoom_fates['gated'] = zoom_fates.get('gated', 0) + int((~keep_).sum()); zoom_fates['accepted'] -= int((~keep_).sum()); zacc_ = zacc_[keep_]
                n_zacc = len(zacc_)
                zoom_st['raw_canvas_boxes'] += len(pz_); zoom_st['accepted'] += n_zacc
                zlog_['raw_canvas_boxes'] = int(len(pz_)); zlog_['accepted'] = [[round(float(v_), 2) for v_ in r_[:4]] + [round(float(r_[4]), 5), int(r_[6])] for r_ in zacc_]
                if n_zacc:   # appended AFTER the normal rows (their indices are unchanged); PRED gets a 'zoom' 0/1 flag per row
                    zoom_flags = [0] * len(p) + [1] * n_zacc; p = np.concatenate([p, zacc_[:, :6]]).astype(np.float32); arr = np.concatenate([arr, zacc_[:, :6]]).astype(np.float32)
                z_ms = ((tz1_ - tz0_) / 1e6, (time.perf_counter_ns() - tz1_) / 1e6)
            if synth_hide is not None:     # --synth-plan: hide the planned person BEFORE the tracker and the ViT (its time is excluded from tracker_ms)
                ts_ = time.perf_counter_ns(); hb_ = synth_hide.get(Path(im['file_name']).name); n_hid = 0
                if hb_ is not None:
                    arr, n_hid, n_miss = apply_synth_plan(arr, hb_, synth_st['iou_thr'])
                    synth_st['frames_applied'] += 1; synth_st['hidden_dets'] += n_hid; synth_st['planned_boxes_without_det'] += n_miss
                synth_ms = (time.perf_counter_ns() - ts_) / 1e6
            bench_ms = vit_bench(g, arr, i) if APP.bench_eng is not None else 0.0   # --vit-bench-n: after NMS, before tracker.update (excluded from tracker_ms)
            frame_time = APP.LIVE.arr[0] if APP.LIVE is not None and APP.LIVE.arr else None
            eval_pos = replay_eval.first_pass_position(frame_time) if frame_time is not None else None
            if eval_pos == 0:
                if APP.relabeler is not None:
                    APP.relabeler.reset()
                APP.tracker = mk_tracker_(APP.RULES['tracker'])
                if APP.args.gmc:
                    previous_motion_frame = previous_motion_time = None
                    APP.tracker.kf = GKF()
            gmc_ms = 0.0
            APP.CTL['lead_motion'] = None
            if gmc is not None:   # --gmc: warp prev -> current, consumed by the next multi_predict (pool) and applied to unconfirmed tracks now
                tg_ = time.perf_counter_ns()
                if APP.args.gmc_thread: motion, c_ms_ = APP.gmc_futs.pop(i).result(); gmc_comp_ms.append(c_ms_)
                else:
                    if eval_pos == 0:
                        gmc.prev = None
                    motion = gmc.estimate(g)
                W_ = motion.warp
                APP.tracker.kf.warp = W_
                if frame_time is not None:
                    APP.CTL['lead_motion'] = motion.paired(previous_motion_frame, g, previous_motion_time, frame_time)
                previous_motion_frame, previous_motion_time = (g, frame_time) if frame_time is not None else (None, None)
                for t_ in APP.tracker.tracked:
                    if not t_.is_activated and t_.mean is not None:
                        m_, c_ = gmc_warp(t_.mean[None].copy(), t_.cov[None], W_); t_.mean, t_.cov = m_[0], c_[0]
                gmc_ms = (time.perf_counter_ns() - tg_) / 1e6
            stab_c_ = APP.LIVE.stabq[0] if APP.LIVE is not None and APP.LIVE.stabq else None; STAB_M[0] = stab_margins(stab_c_)   # stab: correction of frame i (same pairing as LIVE.rawq[0])
            if APP.LIVE is not None and (settings.WARM_MIN > 0 or settings.WARM_MAX > 0) and len(arr) and APP.LIVE.rawq:   # v24f warmth veto: a person box must contain pixels as warm as a person (raw p95 >= WARM_MIN)
                raw_i_ = APP.LIVE.rawq[0]; keep_ = np.ones(len(arr), bool)
                if stab_c_ is not None: raw_i_ = stab_shift(raw_i_, stab_c_)   # stab: a box in stabilized coordinates reads the 16-bit pixels under it
                for k_ in range(len(arr)):
                    if int(arr[k_, 5]) != PERSON_CLS: continue
                    a_, b_, c_, d_ = int(max(0, arr[k_, 0])), int(max(0, arr[k_, 1] - 16)), int(min(640, arr[k_, 2])), int(min(480, arr[k_, 3] - 16))
                    if c_ - a_ < 4 or d_ - b_ < 4: continue
                    p95_ = warm_p95(raw_i_[b_:d_:2, a_:c_:2])   # 11-12: was float(np.percentile(..., 95)); same value (src/track/warm_p95.py)
                    if p95_ < settings.WARM_MIN or (settings.WARM_MAX > 0 and p95_ > settings.WARM_MAX): keep_[k_] = False   # v33: range
                if not keep_.all(): APP.LIVE.warm_dropped += int((~keep_).sum()); arr = arr[keep_]
            if APP.LIVE is not None and APP.LIVE.arr and APP.tracker is not None:   # v26b: first frame after an input gap of over 1 s: no track is continued by position alone (the tracker counts frames, not seconds, so it would not notice the gap).
                ta_ = APP.LIVE.arr[0]                                         # every track ends here; people get new tracks through the normal confirmation, and KeepID may give a returning person the old number by appearance only.
                if eval_pos is None and APP.LIVE.upd_t is not None and ta_ - APP.LIVE.upd_t > 1.0:
                    for t_ in APP.tracker.tracked + APP.tracker.lost: t_.state = REMOVED
                    print('LIVE input gap: %d tracks ended at frame %d' % (len(APP.tracker.tracked) + len(APP.tracker.lost), i), flush=True); APP.tracker.tracked = []; APP.tracker.lost = []
                APP.LIVE.upd_t = ta_
                if APP.CONFIRM is not None:   # 11-11 --confirm-ms: the tracker's confirmation count follows the measured frame rate (arrival times of the frames the tracker gets)
                    ch_ = confirm_apply(APP.CONFIRM, APP.tracker, ta_, fps=replay_eval.initial_fps() if eval_pos is not None else None)
                    if ch_: print('CONFIRM %g ms at %.1f fps: %d -> %d frames (frame %d)' % (APP.CONFIRM.ms, APP.CONFIRM.fps or 0.0, ch_[0], ch_[1], i), flush=True)
            if APP.TIMING is not None: APP.TIMING.mark()
            n_sync0 = len(sync_lat)
            if budget:   # lost-first: P1 through feat_fn inside update, P2/P4 on this frame and P3 on the previous frame right after the association
                lf_frame_ms[0] = 0.0
                feat_fn = lambda idxs, g_=g, arr_=arr, f_=i: budget_embed(idxs, g_, arr_, f_)
                feat_fn_prev = (lambda idxs, g_=prev_gray, arr_=prev_arr, f_=i: budget_embed(idxs, g_, arr_, f_)) if prev_gray is not None else None
                tr_rows = APP.tracker.update(arr, None, feat_fn, feat_fn_prev); t4 = time.perf_counter_ns()
                vs = lf_frame_ms[0]
            else:
                feat_fn = (lambda idxs, g_=g, arr_=arr, f_=i: sync_embed(idxs, g_, arr_, f_)) if (vit is not None and APP.args.vit_sync and APP.args.app_first) else None
                if GATE_FN is not None: APP.tracker.gray = g   # v11: the gate's local-contrast / texture features read the current gray frame
                bs_feats = None   # v38: BoT-SORT appearance = RepVGG_(TMOT,CAMEL) (KeepID engine, NPU core 2) on the 4 highest-score people >= 32 px. v40: app = off | repvgg
                is_botsort = isinstance(APP.tracker, BotSortTracker) or (APP.LIVE is not None and APP.RULES['tracker'] == 'botsort')
                if is_botsort and APP.rl_eng is not None and len(arr) and APP.CTL['app'] == 'repvgg':
                    bk_ = [int(j) for j in np.argsort(-arr[:, 4]) if int(arr[j, 5]) == PERSON_CLS and arr[j, 4] >= 0.25 and arr[j, 3] - arr[j, 1] >= 32.0][:4]
                    try:
                        if bk_: bs_feats = APP.rl_embed(bk_, g, {j: arr[j, :4].tolist() for j in bk_})
                    except Exception as e_:
                        bs_feats = None; print('BOTSORT embed failed', repr(e_)[:120], flush=True)
                tr_rows = APP.tracker.update(arr, bs_feats, feat_fn); t4 = time.perf_counter_ns()
                vs = sync_lat[-1] if len(sync_lat) > n_sync0 else 0.0
            base_rows = tr_rows; rl_ms = 0.0
            if APP.relabeler is not None and APP.RULES['relabel']:   # --relabel: P and crop boxes in the saved-PRED form (v24g: live on/off) (canvas y + 64 rounded to 0.01, minus 64; conf rounded to 1e-5)
                tr0_ = time.perf_counter_ns(); pc_ = arr[:, :4].copy(); pc_[:, 1] += 64.0; pc_[:, 3] += 64.0; pr_ = np.round(pc_, 2).tolist(); cr_ = np.round(arr[:, 4], 5).tolist()
                P_ = {j: [pr_[j][0], pr_[j][1] - 64.0, pr_[j][2], pr_[j][3] - 64.0] for j in range(len(arr)) if int(arr[j, 5]) == PERSON_CLS and cr_[j] >= 0.1}
                mamba_kw_ = dict(warp=W_, ts_ns=None if APP.args.live_port else (i + 1) * 33333333) if APP.args.mamba else {}
                tr_rows = APP.relabeler.step(tr_rows, P_, lambda ks, g_=g, b_=P_: APP.rl_embed(ks, g_, b_), frame=Path(im['file_name']).name,
                                         embed_async=(lambda ks, g_=g, b_=P_: APP.rl_async.submit(ks, g_, b_)) if APP.rl_async is not None else None, **mamba_kw_)
                rl_ms = (time.perf_counter_ns() - tr0_) / 1e6
            zc_ms = 0.0; n_zc = 0; n_zw = 0
            if zoom_eng is not None and ahead and i + 1 < len(APP.images) and APP.images[i + 1]['extra_info']['video_id'] == vid:
                # v9: candidates of THIS frame (jo480 = its decoded pre-NMS tiles, still intact until detect(i+1)) -> windows cut from the NEXT frame's gray
                # (ahead[0], already read for the ahead submit) -> one canvas submitted on NPU_2 now, collected at the next iteration right after detect(i+1)
                tz_ = time.perf_counter_ns()
                extra_ = ([[*t_.xyxy.tolist(), min(float(t_.score), APP.args.zoom_hi)] for t_ in APP.tracker.lost if t_.state == LOST and t_.cls == PERSON_CLS and 0 < float(t_.xywh[3]) < APP.args.zoom_maxh]
                          if APP.args.zoom_lost else None)
                cz_ = zoom_candidates(APP.jo480, APP.det_yoff, APP.args.zoom_lo, APP.args.zoom_hi, APP.args.zoom_maxh, APP.args.zoom_max, extra_); n_zc = len(cz_)
                if n_zc:
                    g1_ = ahead[0][0]; wins_ = zoom_windows(cz_, APP.args.zoom_ctx, APP.args.zoom_min, APP.args.zoom_target, APP.args.zoom_maxscale, APP.args.zoom_floor); n_zw = len(wins_)
                    if wins_:
                        zoom_canvas.fill(int(round(APP.cv2.mean(g1_)[0]))); zoom_pack(g1_, wins_, zoom_canvas); zoom_x[0] = APP.cv2.cvtColor(zoom_canvas, APP.cv2.COLOR_GRAY2RGB)
                        zlog_ = dict(image_id=Path(APP.images[i + 1]['file_name']).name, from_image_id=Path(im['file_name']).name, n_candidates=n_zc,
                                     candidates=[[round(float(v_), 2) for v_ in r_[:4]] + [round(float(r_[4]), 5)] for r_ in cz_],
                                     windows=[dict(x0=w_['x0'], y0=w_['y0'], side=w_['side'], S=w_['S'], px=w_['px'], py=w_['py']) for w_ in wins_], raw_canvas_boxes=0, accepted=[])
                        zoom_log.append(zlog_); zoom_eng.submit(zoom_x, i + 1); zoom_inflight = (wins_, i + 1, zlog_)
                    zoom_st['frames_with_candidates'] += 1
                zoom_st['candidates'] += n_zc; zoom_st['windows'] += n_zw; zoom_st['windows_skipped'] += n_zc - n_zw; zoom_st['canvases'] += int(zoom_inflight is not None)
                zc_ms = (time.perf_counter_ns() - tz_) / 1e6
            # ViT crop requests: new/lost every frame, refresh every N frames (crop_requests); stale requests (> --vit-stale frames) are dropped
            nq = 0
            if async_vit:
                keep = [q for q in vit_queue if APP.tracker.frame_id - q['frame'] <= APP.args.vit_stale]; dropped += len(vit_queue) - len(keep); vit_queue[:] = keep
                req = crop_requests(APP.tracker, APP.args.vit_every, APP.args.k, APP.args.vit_conf, APP.args.vit_min_h, prev_gray is not None)
                if APP.args.vit_sync and APP.args.app_first: req = [q for q in req if q['kind'] != 'new']       # candidates were embedded synchronously
                vit_queue.extend(req); nq = len(req); crops_queued += nq
            t5 = time.perf_counter_ns()
            row = dict(image_id=Path(im['file_name']).name, det_total_ms=(t2 - t0) / 1e6, pre_model_ms=(t1 - t0) / 1e6, nms_ms=(t2 - t_nms0) / 1e6, n=len(p), tracker_ms=(t4 - t3) / 1e6 - vs,
                       vit_wait_ms=vw, vit_sync_ms=vs, vit_queued=nq, frame_total_ms=(t5 - t0) / 1e6, n_tracks=len(tr_rows), latency_ms=(t5 - t0) / 1e6, read_ms=read_ms)
            if synth_hide is not None: row['tracker_ms'] -= synth_ms; row['synth_ms'] = synth_ms; row['synth_hidden'] = n_hid
            if APP.bench_eng is not None: row['tracker_ms'] -= bench_ms; row['vit_bench_ms'] = bench_ms
            if gmc is not None: row['tracker_ms'] -= gmc_ms; row['gmc_ms'] = gmc_ms
            if budget: row['vit_crops'] = int(APP.tracker.per_frame[-1][0]); row['vit_demand'] = list(APP.tracker.per_frame[-1][1:])   # vit_sync_ms = this frame's crop ms
            if APP.relabeler is not None: row['relabel_ms'] = rl_ms; row['relabel_crops'] = int(APP.relabeler.per_frame[-1][0]) if (APP.RULES['relabel'] and APP.relabeler.per_frame) else 0
            if zoom_eng is not None:   # v9: wait + merge are between t3 and t4 (excluded from tracker_ms); the canvas build is before t5 (inside frame_total / latency / period)
                row['tracker_ms'] -= z_ms[0] + z_ms[1]; row['zoom_wait_ms'] = z_ms[0]; row['zoom_merge_ms'] = z_ms[1]; row['zoom_canvas_ms'] = zc_ms
                row['zoom_cands'] = n_zc; row['zoom_wins'] = n_zw; row['zoom_acc'] = n_zacc
            if APP.args.pipeline:
                # period = time between consecutive frame completions minus the JPEG read done inside the interval (a camera delivers frames in memory)
                row['period_ms'] = ((t5 - t_prev_emit) / 1e6 - read_ms) if t_prev_emit is not None else None
                t_prev_emit = t5
            APP.rows.append(row)
            pc = p.copy(); pc[:, 1] += 64.0 - APP.FRAME_DY[0]; pc[:, 3] += 64.0 - APP.FRAME_DY[0]
            APP.preds.append(dict(image_id=row['image_id'], xyxy=np.round(pc[:, :4], 2).tolist(), conf=np.round(pc[:, 4], 5).tolist(), cls=pc[:, 5].astype(int).tolist()))
            if zoom_eng is not None: APP.preds[-1]['zoom'] = zoom_flags if zoom_flags is not None else [0] * len(p)   # v9: 1 = zoom re-detection (appended rows)
            APP.tracks_out.append(dict(image_id=row['image_id'], coco_image_id=im['id'], video_id=vid, tracker_reset=reset,
                                   tracks_xyxy_id_conf_class_detectionindex=[[round(float(v) - (APP.FRAME_DY[0] if j_ in (1, 3) else 0.0), 2) for j_, v in enumerate(r[:4])] + [int(r[4]), round(float(r[5]), 5), int(r[6]), int(r[7])] for r in tr_rows]))
            if APP.relabeler is not None:
                APP.tracks_base.append(dict(image_id=row['image_id'], coco_image_id=im['id'], video_id=vid, tracker_reset=reset,
                                        tracks_xyxy_id_conf_class_detectionindex=[[round(float(v) - (APP.FRAME_DY[0] if j_ in (1, 3) else 0.0), 2) for j_, v in enumerate(r[:4])] + [int(r[4]), round(float(r[5]), 5), int(r[6]), int(r[7])] for r in base_rows]))
            if APP.LIVE is not None: live_publish(g, tr_rows, row, detections=p, tracker_detections=arr)
            if APP.TIMING is not None: APP.TIMING.frame(i, t0, t1, t2, t4, rl_ms, t5)
            prev_gray = g
            if budget: prev_arr = arr
            if i % 50 == 0 and not APP.args.mock:
                s_ = APP.dx.DeviceStatus.get_current_status(0); util.append(dict(frame=i, core_util=[s_.get_core_utilization(k) for k in range(3)], mem=s_.get_memory_used()))
            if (i + 1) % 500 == 0: print('FRAMES', i + 1, round(time.monotonic() - t_start, 1), 'vit delivered', delivered, flush=True)
        if APP.LIVE is not None and APP.LIVE.restart_pending:
            for _ in ahead:
                APP.a.wait()
                if APP.b is not None:
                    APP.b.wait()
        if APP.FEED is not None: APP.FEED.close()
        if APP.TIMING is not None: APP.TIMING.close()
        if APP.LIVE is not None: live_close()
        if vit is not None and inflight is not None: vit_sync(None, None, len(APP.images))
        if APP.relabeler is not None: APP.relabeler.reset()          # v7: drains an asynchronous refresh still in flight
        if APP.gmc_pool is not None: APP.gmc_pool.shutdown(wait=True)
        with (APP.out / 'TRACKS.jsonl').open('w', encoding='utf-8', newline='\n') as fh:
            for r in APP.tracks_out: fh.write(json.dumps(r, separators=(',', ':')) + '\n')
        with (APP.out / ({'rt480': 'PRED_NEWMRT480.jsonl', 'w480': 'PRED_NEWMRW480.jsonl', 's640': 'PRED_S640.jsonl', 'm640': 'PRED_M640.jsonl', 's6': 'PRED_S6.jsonl', 's480': 'PRED_S480.jsonl', 'ft1': 'PRED_FT1.jsonl', 'ft9': 'PRED_FT9.jsonl'}.get(APP.args.det, 'PRED_DRT512.jsonl'))).open('w', encoding='utf-8', newline='\n') as fh:
            for r in APP.preds: fh.write(json.dumps(r, separators=(',', ':')) + '\n')
        (APP.out / 'RAW_TIMINGS.json').write_text(json.dumps(APP.rows)); (APP.out / 'UTIL_SAMPLES.json').write_text(json.dumps(util)); (APP.out / 'REBIND_LOG.json').write_text(json.dumps(rebind_log))
        if APP.relabeler is not None:
            with (APP.out / 'TRACKS_BASE.jsonl').open('w', encoding='utf-8', newline='\n') as fh:
                for r in APP.tracks_base: fh.write(json.dumps(r, separators=(',', ':')) + '\n')
            (APP.out / 'RELABEL_LOG.json').write_text(json.dumps(APP.relabeler.events))
        summ = result_summary.summ
        secs = time.monotonic() - t_start
        if APP.tracker is not None:
            for k_, a_ in (('motion', 'rebinds_motion'), ('appearance', 'rebinds_app'), ('cand_frames', 'cand_frames'), ('cand_crops', 'cand_crops'), ('sync_frames', 'sync_frames'), ('sync_crops', 'sync_crops')):
                reb_tot[k_] += getattr(APP.tracker, a_)
        res = dict(status='DONE', mode=tag, det=APP.args.det, det_models=APP.det_names, pipeline=APP.args.pipeline, tracker=APP.args.tracker, vit_model=(APP.args.vit if use_vit else None), vit_every=APP.args.vit_every, k=APP.args.k, sim=APP.args.sim,
                   app_first=APP.args.app_first, vit_sync=APP.args.vit_sync, sync_k=APP.args.sync_k, hybrid_fallback=APP.args.hybrid_fallback, frames=len(APP.rows), mock=APP.args.mock,
                   period_ms=summ('period_ms'), latency_ms=summ('latency_ms'), det_wait_ms=summ('pre_model_ms'), nms_ms=summ('nms_ms'), read_ms=summ('read_ms'),
                   vit_sync_ms=dict(frames_with_sync=int(len(sync_lat)), p50=float(np.percentile(sync_lat, 50)) if sync_lat else None, p95=float(np.percentile(sync_lat, 95)) if sync_lat else None,
                                    max=float(max(sync_lat)) if sync_lat else None, candidate_frames=reb_tot['cand_frames'], candidate_crops=reb_tot['cand_crops'],
                                    sync_frames=reb_tot['sync_frames'], sync_crops=reb_tot['sync_crops']) if (use_vit and APP.args.vit_sync) else None,
                   det_total_ms=summ('det_total_ms'), tracker_ms=summ('tracker_ms'), vit_wait_ms=summ('vit_wait_ms'), frame_total_ms=summ('frame_total_ms'),
                   vit=dict(crops_queued=crops_queued, delivered=delivered, dropped_stale=dropped, delivered_per_s=delivered / max(secs, 1e-9), delivered_per_frame=delivered / max(1, len(APP.rows)),
                            submit_to_result_ms_p50=float(np.percentile(vit_lat, 50)) if vit_lat else None, submit_to_result_ms_p95=float(np.percentile(vit_lat, 95)) if vit_lat else None,
                            npu_ms_per_crop=(vit.b.get_npu_inference_time() / 1000 / max(1, delivered)) if (vit is not None and not APP.args.mock and not budget) else None) if use_vit else None,
                   rebinds=dict(motion=reb_tot['motion'], appearance=reb_tot['appearance'], late_merges=len(rebind_log),
                                note='summed over all videos; appearance = pairs decided by cosine (association-time or late merge)'),
                   npu_ms={n: (APP.eng[n].b.get_npu_inference_time() / 1000 if not APP.args.mock else None) for n in APP.eng},
                   core_util_mean=[float(np.mean([u['core_util'][k] for u in util])) for k in range(3)] if util else None,
                   models={n: dict(core=CORE.get(n, 2), sha=APP.man.get(n)) for n in list(APP.eng) + ([APP.args.vit] if use_vit else [])}, SDK=cfg.get_version(), seconds=secs,
                   waited_for_other_m1_eval_s=(None if APP.args.mock else waited))
        if synth_st is not None: res['synth'] = dict(synth_st, synth_ms=summ('synth_ms'))
        if APP.args.class_gate: res['class_gate'] = dict(on=True, rule='person tracks <-> person detections only; car/truck among themselves', source='claude_review_fixes_01/classgate')
        if APP.args.gmc: res['gmc'] = dict(method='Ultralytics GMC sparseOptFlow (BoT-SORT default)', downscale=APP.args.gmc_ds, corners=APP.args.gmc_corners, min_distance=APP.args.gmc_mindist, gmc_ms=summ('gmc_ms'),
                                       failed_frames=(sum(o_.n_fail for o_ in APP.gmc_objs) if APP.args.gmc_thread else gmc_fail + (gmc.n_fail if gmc is not None else 0)),
                                       threaded=APP.args.gmc_thread, worker_compute_ms=(dict(p50=float(np.percentile(gmc_comp_ms, 50)), p95=float(np.percentile(gmc_comp_ms, 95))) if gmc_comp_ms else None),
                                       note='gmc_ms = main-thread time (with --gmc-thread: waiting for the worker + applying the warp)')
        res['speed_options'] = dict(gmc_thread=APP.args.gmc_thread, npu_depth=APP.args.npu_depth, rl_async=APP.args.rl_async, ort_threads=APP.args.ort_threads, preload=APP.args.preload)
        if APP.rl_async is not None: res['rl_async'] = dict(slots=APP.rl_async.n, submit_ms_p95=float(np.percentile(APP.rl_async.sub_ms, 95)) if APP.rl_async.sub_ms else None,
                                                        collect_ms_p50=float(np.percentile(APP.rl_async.col_ms, 50)) if APP.rl_async.col_ms else None,
                                                        collect_ms_p95=float(np.percentile(APP.rl_async.col_ms, 95)) if APP.rl_async.col_ms else None)
        if zoom_eng is not None:
            dist_z = result_summary.dist_z
            zc_, zw_, za_ = ([r.get(k_, 0) for r in APP.rows] for k_ in ('zoom_cands', 'zoom_wins', 'zoom_acc'))
            res['zoom'] = dict(on=True, model=APP.det_names[0], engine_key='ZOOM', core=2, sha=APP.man.get(APP.det_names[0]),
                               params=dict(lo=APP.args.zoom_lo, hi=APP.args.zoom_hi, maxh=APP.args.zoom_maxh, max=APP.args.zoom_max, ctx=APP.args.zoom_ctx, min=APP.args.zoom_min, target=APP.args.zoom_target,
                                           maxscale=APP.args.zoom_maxscale, floor=APP.args.zoom_floor, acc=APP.args.zoom_acc, acch=APP.args.zoom_acch, lost=APP.args.zoom_lost, cls_hook=(APP.args.zoom_cls or None), canvas_hw=list(ZOOM_HW)),
                               frames=len(APP.rows), **zoom_st, fates=zoom_fates, frames_with_accepted=int(sum(1 for v in za_ if v > 0)),
                               candidates_per_frame=dist_z(zc_), windows_per_frame=dist_z(zw_), accepted_per_frame=dist_z(za_),
                               canvas_ms=summ('zoom_canvas_ms'), wait_ms=summ('zoom_wait_ms'), merge_ms=summ('zoom_merge_ms'),
                               npu_ms_total=(zoom_eng.b.get_npu_inference_time() / 1000 if not APP.args.mock else None),
                               npu_ms_per_canvas=(zoom_eng.b.get_npu_inference_time() / 1000 / max(1, zoom_st['canvases']) if not APP.args.mock else None),
                               core2_util_mean=(res['core_util_mean'][2] if res['core_util_mean'] else None),
                               rule='candidates = previous frame pre-NMS person anchors (top class person, lo <= score < hi, h < maxh), NMS .5, top max; window = max(ctx x h, min) square, '
                                    'scale = clamp(target / h, 1, maxscale), shelf-packed into one 480x384 canvas cut from the CURRENT frame, pad = mean gray; same DXNN on NPU_2; accept = person, '
                                    'conf >= acc, centre inside its window, mapped h < acch, IoU <= .5 vs normal dets (conf >= .10, cls 0/2/6), NMS .5; appended to the detections before the '
                                    'tracker; wait_ms + merge_ms excluded from tracker_ms, canvas_ms inside frame_total / latency / period; core2_util_mean shares NPU_2 with KeepID')
            res['models']['ZOOM'] = dict(core=2, sha=APP.man.get(APP.det_names[0]), role='zoom', model=APP.det_names[0])
            (APP.out / 'ZOOM_LOG.json').write_text(json.dumps(zoom_log, separators=(',', ':')))
        if trk_extra: res['tracker_params'] = dict(n_init=APP.args.n_init, radius=APP.args.radius, source='idsw_fix/ix_tracker.py FixTracker F4 n_init + F5 radius')
        if budget:
            pf = np.asarray([x for t in lf_trackers for x in t.per_frame], np.int64).reshape(-1, 5); u = pf[:, 0]
            dist_ = result_summary.dist_
            res['vit_budget'] = dict(n=budget, priority=APP.args.vit_priority, sim=APP.args.sim, gate=APP.args.gate, vit_conf=APP.args.vit_conf, vit_min_h=APP.args.vit_min_h, model=APP.args.vit, core=2,
                                     crops=int(u.sum()), crops_per_frame=dict(mean=float(u.mean()) if len(u) else None, p50=float(np.percentile(u, 50)) if len(u) else None,
                                                                             p95=float(np.percentile(u, 95)) if len(u) else None, max=int(u.max()) if len(u) else None,
                                                                             hist={int(v): int((u == v).sum()) for v in np.unique(u)}, frames_at_budget=int((u >= budget).sum()),
                                                                             frames_with_crops=int((u > 0).sum())),
                                     use={k: int(sum(t.use[k] for t in lf_trackers)) for k in ('P1', 'P2', 'P3', 'P4')},
                                     demand={k: int(sum(t.demand[k] for t in lf_trackers)) for k in ('P1', 'P2', 'P3', 'P4')},
                                     p1_candidate_frames=int(sum(t.cand_frames for t in lf_trackers)), p1_candidate_crops=int(sum(t.cand_crops for t in lf_trackers)),
                                     ms_per_frame=summ('vit_sync_ms'), ms_per_crop=dist_(lf_crop_ms), npu_ms_per_crop=dist_(lf_npu),
                                     rule='LostFirstTracker (G1 EMA gallery, A1 hybrid, P1 gated): <= N synchronous batch-1 crops per frame for P1+P2+P3+P4 together; '
                                          'ms_per_frame = this frame\'s crop ms (excluded from tracker_ms, included in period/latency); no asynchronous queue')
        if APP.relabeler is not None:
            dist_r = result_summary.dist_r
            u = np.asarray([x[0] for x in APP.relabeler.per_frame], np.int64)
            res['relabel'] = dict(model=APP.args.relabel, core=2, sha=APP.man.get(APP.args.relabel), every=APP.args.rl_every, k=APP.args.rl_k, new_k=APP.args.rl_new_k, stats=APP.relabeler.stats,
                                  crops_per_frame=dict(mean=float(u.mean()) if len(u) else None, max=int(u.max()) if len(u) else None, hist={int(v): int((u == v).sum()) for v in np.unique(u)}),
                                  ms_per_frame=summ('relabel_ms'), ms_per_crop=dist_r(APP.rl_crop_ms), npu_ms_per_crop=dist_r(APP.rl_npu),
                                  rule=('relink_online_judge.OnlineJudge (v10 --judge: learned relink judge, RESULT.json judge); crops synchronous after the tracker update; '
                                        'relabel_ms excluded from tracker_ms, included in period/latency; TRACKS_BASE.jsonl = tracker rows' if APP.args.judge else
                                        'relabel_online.OnlineRelabel (PREREG_RELABEL.json rule, PREREG_RELABEL_M1.json budget); crops synchronous after the tracker update; '
                                        'relabel_ms excluded from tracker_ms, included in period/latency; TRACKS_BASE.jsonl = tracker rows'))
            res['models'][APP.args.relabel] = dict(core=2, sha=APP.man.get(APP.args.relabel), role='relabel')
            if APP.args.judge:   # v10
                jt_ = APP.relabeler.timing(); js_ = APP.relabeler.stats; jsp_ = APP.relabeler.spec
                res['judge'] = dict(on=True, spec=str(APP.args.judge), spec_sha256=APP.relabeler.spec_sha, config=jsp_['config'], model_kind=jsp_['model']['kind'], feature_set=jsp_['feature_set'],
                                    n_features=len(APP.relabeler.feats), threshold=APP.relabeler.thr, max_wait_frames=jsp_['decision']['max_wait_frames'], candidate_window_frames=jsp_['decision']['candidate_window_frames'],
                                    crops_budget=APP.args.judge_crops, rl_async=APP.args.rl_async, frames=js_['frames'], births=js_['births'], evaluated=js_['births_with_candidates'], relabels=js_['relabelled'],
                                    pairs_scored=js_['pairs_scored'], evaluated_with_own_feature=js_['decided_with_feature'], evaluated_without_own_feature=js_['births_with_candidates_no_feature'],
                                    retired_reappearances=js_['retired_reappearances'], ms_per_decision=jt_['ms_per_decision'], feature_ms=jt_['feature_ms'], model_ms=jt_['model_ms'],
                                    candidates_per_decision=jt_['candidates_per_decision'], ms_per_frame=summ('relabel_ms'),
                                    crops=dict(new=js_['crops_new'], refresh=js_['crops_refresh'], embedded=js_['crops_embedded'], new_for_births_with_candidates=js_['crops_new_births_with_candidates'],
                                               per_frame=res['relabel']['crops_per_frame']),
                                    core2=dict(sync_crops=len(APP.rl_crop_ms), sync_crop_ms_total=float(sum(APP.rl_crop_ms)), sync_crop_ms_per_crop=dist_r(APP.rl_crop_ms), npu_ms_per_crop=dist_r(APP.rl_npu),
                                               npu_ms_total_sync=(float(sum(APP.rl_npu)) if APP.rl_npu else None),
                                               async_submit_ms_total=(float(sum(APP.rl_async.sub_ms)) if APP.rl_async is not None else None), async_collect_ms_total=(float(sum(APP.rl_async.col_ms)) if APP.rl_async is not None else None),
                                               util_mean=(res['core_util_mean'][2] if res['core_util_mean'] else None),
                                               note='sync = the births\' crops (+ every crop without --rl-async): crop -> submit -> wait on NPU_2, wall ms and get_npu_inference_time per crop; '
                                                    'async = the refresh crops under --rl-async (submit + collect wall ms, NPU time not read); util_mean = NPU_2 utilisation samples'),
                                    rule='relink_online_judge.OnlineJudge: at every frame each NEW tracker person ID (first output in the sequence) with >= 1 candidate (person ID output '
                                         'before, not output now, lost <= 90 frames) is judged AT ITS BIRTH FRAME (K = 0): relink_data.pair_features + rank / min-other columns of every '
                                         'candidate, frozen model p >= threshold, greedy by p, one-to-one, forward only; the new ID takes the candidate output ID, the candidate tracker ID '
                                         'is retired (fresh ID if output again); boxes / confidences / classes unchanged. Crops: eligible = output person row with detection k in P and '
                                         'detection height >= 32 px (any overlap); per frame <= --judge-crops: births with candidates, other births (synchronous, before the decision), '
                                         'then refresh (fewest features of the last 5 observations, oldest last feature, smallest ID; asynchronous with --rl-async, attached to that '
                                         'observation at the next frame). Missing feature = availability flag 0 + NaN (training median + missing flag inside the model).')
        if APP.bench_eng is not None:
            dist = result_summary.dist
            res['vit_bench'] = dict(n=APP.args.vit_bench_n, model=APP.args.vit, core=2, sha=APP.man.get(APP.args.vit), frames=len(APP.rows), crops=APP.bench_st['crops'], crops_person=APP.bench_st['real'],
                                    crops_centre=APP.bench_st['centre'], frames_no_person=APP.bench_st['frames_no_person'], ms_per_frame=summ('vit_bench_ms'), ms_per_crop=dist(APP.bench_inf),
                                    npu_ms_per_crop=dist(APP.bench_npu), rule='every frame after NMS, before tracker.update: N x (crop -> submit -> wait) batch 1 on NPU_2, '
                                    'person dets conf >= .10 h >= 32 by conf desc, cyclic; centre 64x128 crop when none; embeddings discarded; ms_per_frame excluded from tracker_ms, '
                                    'included in period/latency; npu_ms_per_crop = get_npu_inference_time() after each wait')
            res['models'][APP.args.vit] = dict(core=2, sha=APP.man.get(APP.args.vit), role='vit_bench')
            res['npu_ms'][APP.args.vit + '_bench_last'] = (APP.bench_eng.b.get_npu_inference_time() / 1000 if not APP.args.mock else None)
        if GATE_FN is not None:   # v11
            gs_ = {}
            for t_ in gate_trackers:
                for k_, v_ in t_.stats.items():
                    if isinstance(v_, dict):
                        d_ = gs_.setdefault(k_, {})
                        for kk_, vv_ in v_.items(): d_[str(kk_)] = d_.get(str(kk_), 0) + vv_
                    else: gs_[k_] = gs_.get(k_, 0) + v_
            res['gate'] = dict(on=True, spec=str(APP.args.confirm_gate), spec_sha256=GATE_FN.sha, source=GATE_FN.spec['source'], source_sha256=GATE_FN.spec['source_sha256'], hidden=GATE_FN.H,
                               thr_out=GATE_FN.thr_out, thr_low=GATE_FN.thr_low, stats=gs_, gate_seconds_total=float(sum(t_.gate_time for t_ in gate_trackers)),
                               rule='GateTracker (failure_analysis_20261102/track_gate): a tentative person track matched this frame (incl. its birth) is output iff GRU p >= thr_out; sticky; frame-1 births output at once; lifecycle unchanged; gate time is inside tracker_ms')
        (APP.out / 'RESULT.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ('mode', 'period_ms', 'latency_ms', 'det_wait_ms', 'tracker_ms', 'frame_total_ms', 'vit', 'rebinds')}), flush=True)
        if budget: print(json.dumps({'vit_budget': {k: res['vit_budget'][k] for k in ('n', 'crops', 'crops_per_frame', 'use', 'demand', 'ms_per_frame', 'ms_per_crop', 'npu_ms_per_crop')}}), flush=True)
        if APP.bench_eng is not None: print(json.dumps({'vit_bench': {k: res['vit_bench'][k] for k in ('n', 'crops', 'crops_person', 'crops_centre', 'ms_per_frame', 'ms_per_crop', 'npu_ms_per_crop')}}), flush=True)
        if APP.relabeler is not None: print(json.dumps({'relabel': {k: res['relabel'][k] for k in ('stats', 'crops_per_frame', 'ms_per_frame', 'ms_per_crop', 'npu_ms_per_crop')}}), flush=True)
        if APP.args.judge: print(json.dumps({'judge': {k: res['judge'][k] for k in ('births', 'evaluated', 'relabels', 'pairs_scored', 'evaluated_without_own_feature', 'ms_per_decision', 'feature_ms', 'model_ms', 'crops', 'core2')}}), flush=True)
        if zoom_eng is not None: print(json.dumps({'zoom': {k: res['zoom'][k] for k in ('candidates', 'windows', 'windows_skipped', 'canvases', 'raw_canvas_boxes', 'accepted', 'fates', 'canvas_ms', 'wait_ms', 'merge_ms', 'npu_ms_per_canvas')}}), flush=True)
    except BaseException:
        failed = True
        if APP.FEED is not None:
            try:
                APP.FEED.close()
            except Exception as exc:
                print('FEED cleanup failed: %r' % exc, flush=True)
        worker_ = getattr(APP.LIVE, 'preview_worker', None)
        if worker_ is not None: worker_.close()
        for e_ in list(APP.eng.values()):
            while getattr(e_, 'q', None):
                try: e_.b.wait(e_.q.pop(0)[0])
                except BaseException: pass
        for n in list(APP.pending):
            try: (APP.eng.get(n) or (zoom_eng if (zoom_eng is not None and n == zoom_eng.name) else None) or (APP.rl_eng if (APP.rl_eng is not None and n == APP.rl_eng.name) else None) or (APP.bench_eng if (APP.bench_eng is not None and n == APP.bench_eng.name) else vit)).wait()
            except BaseException: pass
        (APP.out / 'FAILURE.txt').write_text(traceback.format_exc()); raise
    finally:
        if not failed and not APP.pending:
            for e in list(APP.eng.values()) + ([vit] if (use_vit and 'vit' in dir()) and vit is not None else []) + ([APP.bench_eng] if APP.bench_eng is not None else []) + ([APP.rl_eng] if APP.rl_eng is not None else []) + ([APP.rl_async] if APP.rl_async is not None else []) + ([zoom_eng] if zoom_eng is not None else []): e.b.dispose()

    if APP.LIVE is not None and APP.LIVE.restart_pending:
        return system_buttons.RESTART_REQUESTED
