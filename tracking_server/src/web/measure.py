"""측정 버튼: 측정 시작·끝, 지연 시간 통계와 원본 프레임·추적 결과 파일 저장."""
import json
import time
import zlib
from pathlib import Path
import numpy as np
from ..common.state import APP


def meas_stats(v_):   # v22
    if not v_: return None
    a_ = np.asarray(v_, np.float64)
    return dict(p50=float(np.percentile(a_, 50)), p95=float(np.percentile(a_, 95)), p99=float(np.percentile(a_, 99)), max=float(a_.max()), mean=float(a_.mean()), over33=float(100.0 * (a_ > 33.0).mean()))


def meas_dir(day_):   # v22c: the data card first, the run folder when it cannot be written
    try:
        d_ = Path(APP.args.measure_dir) / day_; d_.mkdir(parents=True, exist_ok=True)
        t_ = d_ / '.write_test'; t_.write_text('x'); t_.unlink(); return d_
    except OSError as e_: print('MEASURE dir not writable, using the run folder:', e_, flush=True); return APP.out


def meas_clean(t_, n_):   # v22b: user text, bounded
    return str(t_ or '').replace('\r', '').strip()[:n_]


def meas_start(name_='', memo_='', video_=True):
    with APP.LIVE.meas_lock:
        if getattr(APP.LIVE, 'restart_pending', False):
            return
        if APP.LIVE.meas is not None: return
        APP.LIVE.meas_n += 1; rp_ = getattr(APP.LIVE, 'replay', None); m_ = dict(n=APP.LIVE.meas_n, t0=time.perf_counter(), wall=time.strftime('%Y-%m-%d %H:%M:%S'), stamp=time.strftime('%Y%m%d_%H%M%S'), rows=[], boxes=[], name=APP.meas_clean(name_, 80), memo=APP.meas_clean(memo_, 2000), video=bool(video_), **(dict(replay=[rp_['name']]) if rp_ is not None else {}))   # 11-12: replay = recordings played during this measurement (src/camera/replay_link.py adds the later ones)
        safe_ = ''.join(c_ if (c_.isalnum() or c_ in '-_') else '_' for c_ in m_['name'])[:40].strip('_')
        m_['dir'] = APP.meas_dir(m_['stamp'][:8]) / ('MEASURE_%s_%d' % (m_['stamp'], m_['n']) + ('_' + safe_ if safe_ else '')); m_['dir'].mkdir(parents=True, exist_ok=True)
        if m_['video']:
            rc_ = dict(q=APP.lv_col.deque(), cond=APP.lv_th.Condition(), f=open(m_['dir'] / 'frames.raw.gz', 'wb'), z=zlib.compressobj(1, zlib.DEFLATED, 31), times=[], dropped=0, stop=False)   # v22f: gzip stream, lossless
            def rec_writer(rc_=rc_):
                while True:
                    with rc_['cond']:
                        while not rc_['q'] and not rc_['stop']: rc_['cond'].wait(0.5)
                        if not rc_['q'] and rc_['stop']: return
                        t_, r_ = rc_['q'].popleft()
                    rc_['f'].write(rc_['z'].compress(r_.astype('<u2', copy=False).tobytes())); rc_['times'].append(t_)
            rc_['th'] = APP.lv_th.Thread(target=rec_writer, daemon=True); rc_['th'].start(); m_['rec'] = rc_; APP.LIVE.rec = rc_
        APP.LIVE.meas = m_


def meas_stop(memo_=None):
    with APP.LIVE.meas_lock: m_ = APP.LIVE.meas; APP.LIVE.meas = None; APP.LIVE.rec = None
    if m_ is None: return
    rc_ = m_.get('rec'); vinfo_ = None
    if rc_ is not None:   # v22d: let the writer finish the queued frames
        with rc_['cond']: rc_['stop'] = True; rc_['cond'].notify_all()
        rc_['th'].join(60); rc_['f'].write(rc_['z'].flush()); rc_['f'].close()
        vinfo_ = dict(file='frames.raw.gz', width=APP.LIVE_W, height=APP.LIVE_H, dtype='uint16 little endian', frames=len(rc_['times']), dropped=rc_['dropped'],
                      bytes=len(rc_['times']) * APP.LIVE_W * APP.LIVE_H * 2, gz_bytes=(m_['dir'] / 'frames.raw.gz').stat().st_size,
                      note='lossless gzip of the raw frames: after decompression frame i = bytes [i*W*H*2, (i+1)*W*H*2); raw camera values, not mapped to 8 bit',
                      times_s=[round(t_ - m_['t0'], 4) for t_ in rc_['times']])
        (m_['dir'] / 'frames.json').write_text(json.dumps(vinfo_, ensure_ascii=False), encoding='utf-8')
    if memo_ is not None: m_['memo'] = APP.meas_clean(memo_, 2000)
    rw_ = m_['rows']; secs_ = time.perf_counter() - m_['t0']
    ts_ = [r_[0] for r_ in rw_]; iv_ = [(b2_ - a2_) * 1000.0 for a2_, b2_ in zip(ts_, ts_[1:])]
    res_ = dict(n=m_['n'], name=m_['name'], memo=m_['memo'], started=m_['wall'], seconds=secs_, frames=len(rw_), out_fps=(len(rw_) - 1) / (ts_[-1] - ts_[0]) if len(rw_) > 2 and ts_[-1] > ts_[0] else None,
                cam_fps=float(np.mean([r_[5] for r_ in rw_])) if rw_ else None, delay=APP.meas_stats([r_[1] for r_ in rw_]), interval=APP.meas_stats(iv_),
                compute=APP.meas_stats([r_[2] for r_ in rw_]), npu_wait=APP.meas_stats([r_[3] for r_ in rw_]), tracker=APP.meas_stats([r_[4] for r_ in rw_]),
                config=dict(det=APP.args.det, pipeline=APP.args.pipeline, npu_depth=APP.args.npu_depth, gmc=APP.args.gmc, gimbal=bool(APP.args.gimbal), gui=bool(getattr(APP.args, 'gui_host', None))),
                note='delay = camera frame received by the server -> result published (the camera\'s own sensor-to-USB delay is not included); interval = between published results; compute = frame_total_ms')
    if m_.get('replay'): res_['replay'] = list(m_['replay']); res_['replay_note'] = '이 측정에는 카메라가 아니라 저장된 녹화를 재생한 구간이 있음: ' + ', '.join(m_['replay'])   # 11-12
    res_['fields'] = APP.MEAS_FIELDS
    md_ = m_['dir']; fn_ = 'summary.json'; res_['folder'] = str(md_); res_['file'] = str(md_ / fn_); res_['csv'] = str(md_ / 'timing.csv')
    res_['server_run'] = str(APP.out); res_['video'] = {k_: v_ for k_, v_ in vinfo_.items() if k_ != 'times_s'} if vinfo_ else None
    if vinfo_:   # our tracker's boxes, tied to frames.raw by the arrival time
        ix_ = {t_: i_ for i_, t_ in enumerate(rc_['times'])}
        with open(md_ / 'tracks.csv', 'w', encoding='utf-8', newline='') as ft_:
            ft_.write('raw_frame,track_id,class_id,confidence,x,y,width,height\n')
            for t_, bs_ in m_['boxes']:
                i_ = ix_.get(t_)
                if i_ is None: continue
                for x1_, y1_, x2_, y2_, tid_, cf_, cl_ in bs_: ft_.write('%d,%d,%d,%.4f,%.2f,%.2f,%.2f,%.2f\n' % (i_, tid_, cl_, cf_, x1_, y1_, x2_ - x1_, y2_ - y1_))
    (md_ / 'README_KO.txt').write_text('측정 %d: %s\n시작 %s, %.1f초\n메모: %s\n\n' % (m_['n'], m_['name'], m_['wall'], secs_, m_['memo']) + (('주의: ' + res_['replay_note'] + '\n\n') if m_.get('replay') else '') +
        'summary.json  요약(지연·처리 시간 p50/p95/p99), 이름·메모, 항목 설명, 프레임별 시간\n'
        'timing.csv    프레임별 시간, 한글 열 이름 (엑셀)\n'
        + ('frames.raw.gz 측정 동안 카메라가 보낸 모든 프레임. 16비트 원본값, 640x480, 리틀 엔디언, 앞에서부터 이어 붙임 (%d장, 놓친 프레임 %d장)\n' % (vinfo_['frames'], vinfo_['dropped']) +
           '              (frames.raw.gz = 무손실 압축. 풀면 원본과 똑같음: gzip -dk frames.raw.gz 또는 파이썬 gzip.open)\n'
           'frames.json   프레임 크기·형식, 각 프레임 도착 시각(측정 시작 후 초)\n'
           'tracks.csv    우리 추적기 출력: raw_frame(영상 안의 프레임 번호, 0부터), 추적 번호, 클래스(0=사람), 점수, 박스 x,y,w,h (640x480 좌표)\n' if vinfo_ else '영상 저장 안 함\n'), encoding='utf-8')
    with open(md_ / 'timing.csv', 'w', encoding='utf-8-sig', newline='') as fc_:   # utf-8-sig: Excel reads the Korean header
        fc_.write('# 측정 ' + str(m_['n']) + ': ' + m_['name'].replace(',', ' ') + ' | 시작 ' + m_['wall'] + ' | 메모: ' + m_['memo'].replace('\n', ' / ').replace(',', ' ') + '\n')
        fc_.write('프레임순번,측정시작후_초,입력출력지연_ms,처리시간_ms,NPU대기_ms,트래커_ms,카메라_fps\n')
        for i_, r_ in enumerate(rw_): fc_.write('%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.2f\n' % (i_ + 1, r_[0] - m_['t0'], r_[1], r_[2], r_[3], r_[4], r_[5]))
    res_['rows_saved'] = len(rw_)
    (md_ / fn_).write_text(json.dumps(dict(res_, rows=[dict(t=round(r_[0] - m_['t0'], 4), delay_ms=round(r_[1], 3), compute_ms=round(r_[2], 3), npu_wait_ms=round(r_[3], 3), tracker_ms=round(r_[4], 3), cam_fps=round(r_[5], 2)) for r_ in rw_]), ensure_ascii=False), encoding='utf-8')
    print('MEASURE', json.dumps({k_: res_[k_] for k_ in ('n', 'frames', 'seconds', 'out_fps', 'cam_fps', 'delay', 'interval', 'file')}), flush=True)
    APP.LIVE.meas_last = res_; APP.LIVE.meas_hist.append(res_)


def meas_state():
    with APP.LIVE.meas_lock: m_ = APP.LIVE.meas; nf_ = len(m_['rows']) if m_ else 0; rc_ = m_.get('rec') if m_ else None
    return dict(on=m_ is not None, frames=nf_, video_frames=len(rc_['times']) if rc_ else None, video_dropped=rc_['dropped'] if rc_ else None, seconds=(time.perf_counter() - m_['t0']) if m_ else 0.0, last=APP.LIVE.meas_last,
                cur=dict(n=m_['n'], name=m_['name'], memo=m_['memo']) if m_ else None,
                history=[dict(n=h_['n'], name=h_['name'], started=h_['started'], frames=h_['frames'], delay_p50=(h_['delay'] or {}).get('p50'), delay_p95=(h_['delay'] or {}).get('p95')) for h_ in APP.LIVE.meas_hist])


def meas_note(n_, name_, memo_):   # v22b: change the name/memo of a finished measurement; rewrites its JSON (the CSV keeps its first line)
    h_ = next((x_ for x_ in APP.LIVE.meas_hist if x_['n'] == n_), None)
    if h_ is None: return False
    if name_ is not None: h_['name'] = APP.meas_clean(name_, 80)
    if memo_ is not None: h_['memo'] = APP.meas_clean(memo_, 2000)
    fp_ = Path(h_['file'])
    try: d_ = json.loads(fp_.read_text(encoding='utf-8')); d_.update(name=h_['name'], memo=h_['memo']); fp_.write_text(json.dumps(d_, ensure_ascii=False), encoding='utf-8')
    except Exception as e_: print('MEASURE note failed', e_, flush=True); return False
    return True
