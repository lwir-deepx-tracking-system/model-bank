"""Native Tk desktop GUI for model-bank over one direct Ethernet cable.

No browser, WebKit, HTTP service, or Runtime repository is used.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import queue
import re
import threading
import time
import tkinter as tk
from collections import Counter, deque
from tkinter import messagebox, ttk
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageTk

from link import Link, LinkError

BG = '#171b20'
FG = '#e8edf1'
ORANGE = '#e7a24c'


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


class App:
    def __init__(self, root, link, on_close=None):
        self.root, self.link = root, link
        self.on_close = on_close
        self.stop = threading.Event()
        self.events = queue.Queue(maxsize=12)
        self.frame_lock = threading.Lock()
        self.frame_slot = None
        self.workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix='modelbank-pc')
        self.controls = ThreadPoolExecutor(max_workers=1, thread_name_prefix='modelbank-control')
        self.rules = {}
        self.model = {}
        self.replay = {}
        self.measure = {}
        self.perf = {}
        self.frame_seq = None
        self.frame_token = None
        self.image = None
        self.image_box = (0, 0, 640, 480)
        self.jpeg_size = (640, 480)
        self.overlay = None
        self.overlay_boxes = []
        self.overlay_history = {}
        self.overlay_time = None
        self.quality_mode = 'default'
        self.replay_names = []
        self.seek_dragging = False
        self.model_ids = []
        self.reid_keys = []
        self.predictor_keys = []
        self.move_held = None
        self.eval_override = False
        self.measure_override = False
        self.measure_ids = []
        self.connected = False
        self.following = False
        self.aim_screen_y = 0.5
        self.clock_samples = deque(maxlen=30)
        self.clock_offset = None
        self.shown_times = deque(maxlen=120)
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<KeyPress>', self.key_down)
        root.bind('<KeyRelease>', self.key_up)
        root.bind('<FocusOut>', lambda _e: self.release())
        root.after(25, self.pump)
        threading.Thread(target=self.video_loop, daemon=True).start()
        threading.Thread(target=self.state_loop, daemon=True).start()
        threading.Thread(target=self.ping_loop, daemon=True).start()

    def _build(self):
        self.root.title('ModelBank · 직결 LAN')
        self.root.geometry('1480x900')
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        if 'clam' in style.theme_names(): style.theme_use('clam')
        style.configure('.', font=('Noto Sans CJK KR', 10))
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG)
        style.configure('TNotebook', background=BG)
        style.configure('TNotebook.Tab', padding=(12, 8))
        style.configure('TLabelframe', background=BG, foreground=ORANGE)
        style.configure('TLabelframe.Label', background=BG, foreground=ORANGE)
        style.configure('TCheckbutton', background=BG, foreground=FG)
        style.configure('TButton', padding=(7, 5))
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        ttk.Label(top, text='ModelBank · PC 전용 직결 화면', font=('Noto Sans CJK KR', 16, 'bold')).pack(side='left')
        self.status = tk.StringVar(value='Orange Pi 연결 중…')
        ttk.Label(top, textvariable=self.status).pack(side='right')
        area = ttk.PanedWindow(self.root, orient='horizontal')
        area.pack(fill='both', expand=True, padx=8, pady=4)
        left = ttk.Frame(area)
        right = ttk.Frame(area)
        area.add(left, weight=3)
        area.add(right, weight=2)
        self.video = tk.Canvas(left, bg='#090b0d', highlightthickness=0, width=800, height=600)
        self.video.pack(fill='both', expand=True)
        self.video.bind('<Button-1>', self.on_video_click)
        self.video.bind('<Configure>', lambda _e: self.draw_frame())
        bar = ttk.Frame(left, padding=(0, 6))
        bar.pack(fill='x')
        self._button(bar, '원점 복귀 (Space)', lambda: self.command('/gimbal/home')).pack(side='left', padx=3)
        self._button(bar, '따라가기 정지', lambda: self.command('/gimbal/stop')).pack(side='left', padx=3)
        self._button(bar, '카메라로', lambda: self.command('/replay/stop')).pack(side='left', padx=3)
        self.frame_text = tk.StringVar(value='영상 대기 중')
        ttk.Label(bar, textvariable=self.frame_text).pack(side='right')
        self.quality = tk.StringVar(value='기본 화질')
        quality_box = ttk.Combobox(bar, textvariable=self.quality, values=['기본 화질', '고화질'],
                                   state='readonly', width=10)
        quality_box.bind('<<ComboboxSelected>>', lambda _e: setattr(
            self, 'quality_mode', 'high' if self.quality.get() == '고화질' else 'default'))
        quality_box.pack(side='right', padx=8)
        self.smooth_overlay = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text='상자 떨림 보정', variable=self.smooth_overlay,
                        command=self.reset_overlay).pack(side='right', padx=8)
        self.follow_text = tk.StringVar(value='짐벌 대기 중')
        ttk.Label(left, textvariable=self.follow_text, wraplength=750).pack(anchor='w', pady=4)
        self.aim_text = tk.StringVar(value='가로 오차 – · 세로 오차 –')
        ttk.Label(left, textvariable=self.aim_text).pack(anchor='w', pady=2)
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill='both', expand=True)
        self._operation_tab()
        self._model_tab()
        self._replay_tab()
        self._measure_tab()
        self._performance_tab()
        self._detail_tab()
        self.message = tk.StringVar(value='')
        ttk.Label(self.root, textvariable=self.message, padding=(10, 5)).pack(fill='x')

    def _tab(self, title):
        outer = ttk.Frame(self.tabs)
        self.tabs.add(outer, text=title)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        frame = ttk.Frame(canvas, padding=10)
        frame.bind('<Configure>', lambda _e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=frame, anchor='nw', tags='body')
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure('body', width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        return frame

    def _section(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=9)
        frame.pack(fill='x', pady=5)
        return frame

    def _button(self, parent, label, action):
        return ttk.Button(parent, text=label, command=action)

    def _row(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=3)
        return frame

    def _combo(self, parent, label, values, callback, width=25):
        row = self._row(parent)
        ttk.Label(row, text=label, width=17).pack(side='left')
        var = tk.StringVar()
        box = ttk.Combobox(row, textvariable=var, values=values, state='readonly', width=width)
        box.pack(side='left', fill='x', expand=True)
        box.bind('<<ComboboxSelected>>', lambda _e: callback(var.get()))
        return var, box

    def _switch(self, parent, label, key, source=None):
        var = tk.BooleanVar()
        box = ttk.Checkbutton(parent, text=label, variable=var,
                              command=lambda: self.rule(key, '1' if var.get() else '0'))
        box.pack(anchor='w', pady=2)
        self.switches.append((var, key, source))
        return var

    def _entry(self, parent, label, default=''):
        row = self._row(parent)
        ttk.Label(row, text=label, width=17).pack(side='left')
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var).pack(side='left', fill='x', expand=True)
        return var

    def _text(self, parent, height=8):
        box = tk.Text(parent, height=height, wrap='word', bg='#101419', fg=FG,
                      insertbackground=FG, relief='flat', padx=8, pady=8)
        box.pack(fill='both', expand=True, pady=3)
        return box

    def _set_text(self, widget, text):
        if widget.get('1.0', 'end-1c') == text:
            return
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)

    def _operation_tab(self):
        p = self._tab('운영')
        self.switches = []
        box = self._section(p, '연결 · 짐벌')
        self.gimbal_label = tk.StringVar(value='연결 대기 중')
        ttk.Label(box, textvariable=self.gimbal_label, wraplength=440).pack(anchor='w')
        row = self._row(box)
        for title, x, y in [('◀', -1, 0), ('▲', 0, 1), ('▼', 0, -1), ('▶', 1, 0)]:
            btn = ttk.Button(row, text=title)
            btn.bind('<ButtonPress-1>', lambda _e, x=x, y=y: self.command('/gimbal/move', x=x, y=y))
            btn.bind('<ButtonRelease-1>', lambda _e: self.command('/gimbal/move', x=0, y=0))
            btn.bind('<Leave>', lambda _e: self.command('/gimbal/move', x=0, y=0))
            btn.pack(side='left', padx=4)
        ttk.Label(box, text='방향키 또는 W A S D: 누르는 동안 이동 · Space: 원점 복귀').pack(anchor='w')
        box = self._section(p, 'NPU 검출 투입')
        self.feed_var, self.feed_box = self._combo(box, '모드', ['normal', 'arrival', 'auto', 'tile'],
                                                   lambda v: self.command('/system/npu-feed', mode=v))
        self.feed_label = tk.StringVar(value='상태 대기 중')
        ttk.Label(box, textvariable=self.feed_label, wraplength=440).pack(anchor='w')
        box = self._section(p, '표시 · 제어')
        for label, key in [('사진 먼저', 'picture_first'), ('예측 상자 표시', 'pred_show'),
                           ('예측 제어', 'lead'), ('KeepID', 'relabel'),
                           ('KeepID 찾는 범위 확장', 'relink_pred')]:
            self._switch(box, label, key)
        self.lead_ms = self._entry(box, '예측 제어 ms', '150')
        self._button(box, '예측 제어 시간 적용', lambda: self.rule('lead_ms', self.lead_ms.get())).pack(anchor='e')
        box = self._section(p, 'Orange Pi 관리')
        row = self._row(box)
        self.camera_button = self._button(row, '카메라 초기화', lambda: self.confirm('카메라를 다시 열까요?', '/camera/reset'))
        self.camera_button.pack(side='left', padx=3)
        self.restart_button = self._button(row, '추적 프로그램 재시작', lambda: self.confirm('추적 프로그램을 다시 시작할까요?', '/system/restart'))
        self.restart_button.pack(side='left', padx=3)
        self.reboot_button = self._button(row, 'Orange Pi 재부팅', lambda: self.confirm('Orange Pi를 재부팅할까요?', '/system/reboot?confirm=REBOOT'))
        self.reboot_button.pack(side='left', padx=3)
        ttk.Label(box, text='직결 모드에서 화면은 이 PC에서만 실행됩니다.', wraplength=440).pack(anchor='w')

    def _model_tab(self):
        p = self._tab('모델 · 추적')
        box = self._section(p, '검출 모델')
        self.model_var, self.model_box = self._combo(box, '모델', [], self.select_model)
        self.model_note = tk.StringVar(value='모델 목록 대기 중')
        ttk.Label(box, textvariable=self.model_note, wraplength=440).pack(anchor='w')
        self._switch(box, '사람 크기에 따른 모델 자동 선택', 'auto_det', 'auto_det')
        self._switch(box, '두 검출 모델 상주', 'resident_det', 'resident_det')
        self.e2e_var, self.e2e_box = self._combo(box, '엔드투엔드', ['off', 'ftt'], lambda v: self.rule('e2e_live', v))
        self.e2e_link_var, self.e2e_link_box = self._combo(
            box, '번호 잇기', ['auto', 'iou', 'id'], lambda v: self.rule('e2e_link', v))
        box = self._section(p, 'Tracker · 외형 · 위치')
        self.tracker_var, self.tracker_box = self._combo(box, 'Tracker', [], self.select_tracker)
        self.engine_var, self.engine_box = self._combo(box, '실행 엔진', ['auto', 'python'],
                                                       lambda v: self.rule_multi(tracker=self.tracker_var.get(), engine=v))
        row = self._row(box)
        self._button(row, '카메라 C++ ByteTrack', lambda: self.command('/tracker/camera', backend='cpp_bytetrack')).pack(side='left', padx=3)
        self._button(row, 'Python ByteTrack', lambda: self.rule_multi(tracker='confirm3', engine='python')).pack(side='left', padx=3)
        self.app_var, self.app_box = self._combo(box, '외형 비교', ['off', 'repvgg', 'vit'], lambda v: self.rule('app', v))
        self.reid_var, self.reid_box = self._combo(box, '외형 모델', [], self.select_reid)
        self.predictor_var, self.predictor_box = self._combo(box, '위치 기억', [], self.select_predictor)
        self.box_var, self.box_box = self._combo(box, '상자 다듬기', ['off', 'size', 'body'],
                                                lambda v: self.rule('box_refine', v))
        self._switch(box, '놓친 상자를 보일 때와 동일하게', 'lost_same', 'box_refine')
        self.blend_var, self.blend_box = self._combo(box, '모델 바뀔 때 섞기', ['0', '0.3', '0.5', '1.0'],
                                                    lambda v: self.rule('box_blend', v))
        box = self._section(p, '검출 뒤 Filter · ID 확인')
        self.warm_min = self._entry(box, 'Warmth 최소')
        self.warm_max = self._entry(box, 'Warmth 최대')
        self.edge_birth = self._entry(box, '가장자리 점수')
        self.confirm_ms = self._entry(box, '새 ID 확인 ms')
        self._button(box, '값 적용', self.apply_filters).pack(anchor='e')
        box = self._section(p, '추가 화면 옵션')
        for label, key in [('온도 Filter', 'warm_on'), ('사람 후보 표시', 'prop_show'),
                           ('피부 온도 표시', 'temp_show'), ('흔들림 보정', 'stab')]:
            self._switch(box, label, key)

    def _replay_tab(self):
        p = self._tab('녹화 재생')
        box = self._section(p, '녹화 고르기')
        self.replay_var, self.replay_box = self._combo(box, '녹화', [], lambda _v: None)
        self.backend_var, self.backend_box = self._combo(box, '재생 Tracker', ['python'], lambda _v: None)
        self.fps_var = self._entry(box, '초당 장 수', '25')
        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text='반복', variable=self.loop_var).pack(anchor='w')
        row = self._row(box)
        self.replay_start_button = self._button(row, '재생 시작', self.start_replay)
        self.replay_start_button.pack(side='left', padx=3)
        self.replay_stop_button = self._button(row, '카메라로 돌아가기', lambda: self.command('/replay/stop'))
        self.replay_stop_button.pack(side='left', padx=3)
        self.replay_state = tk.StringVar(value='녹화 목록 대기 중')
        ttk.Label(box, textvariable=self.replay_state, wraplength=440).pack(anchor='w')
        box = self._section(p, '재생 조작')
        row = self._row(box)
        self._button(row, '일시정지', lambda: self.command('/replay/pause')).pack(side='left', padx=2)
        self._button(row, '계속', lambda: self.command('/replay/play')).pack(side='left', padx=2)
        self._button(row, '−10초', lambda: self.skip_replay(-10)).pack(side='left', padx=2)
        self._button(row, '+10초', lambda: self.skip_replay(10)).pack(side='left', padx=2)
        self._button(row, '+1장', lambda: self.command('/replay/step')).pack(side='left', padx=2)
        self._button(row, '처음부터 다시 채점', lambda: self.confirm('처음부터 다시 채점할까요?', '/replay/eval/reset')).pack(side='left', padx=2)
        self.seek_position = tk.DoubleVar(value=0)
        self.seek_scale = ttk.Scale(box, from_=0, to=1, variable=self.seek_position)
        self.seek_scale.pack(fill='x', pady=3)
        self.seek_scale.bind('<ButtonPress-1>', lambda _e: setattr(self, 'seek_dragging', True))
        self.seek_scale.bind('<ButtonRelease-1>', self.seek_from_scale)
        self.seek_var = self._entry(box, '장 번호 (1부터)', '1')
        row = self._row(box)
        self._button(row, '위치 이동', self.seek).pack(side='left', padx=2)
        self._button(row, '속도 적용', lambda: self.command('/replay/speed', fps=self.fps_var.get())).pack(side='left', padx=2)
        self.draw_gt = tk.BooleanVar()
        ttk.Checkbutton(box, text='정답 상자 그리기', variable=self.draw_gt,
                        command=lambda: self.command('/replay/eval/draw', on=int(self.draw_gt.get()))).pack(anchor='w')
        box = self._section(p, '정답 비교')
        row = self._row(box)
        self._button(row, '현재 진행', lambda: self.eval_view('/replay/eval')).pack(side='left', padx=2)
        self._button(row, '이 위치까지', lambda: self.eval_view('/replay/eval?pos=' + str(max(0, int(self.seek_var.get()) - 1)))).pack(side='left', padx=2)
        self._button(row, '최종 결과', lambda: self.eval_view('/replay/eval?view=final')).pack(side='left', padx=2)
        self.eval_text = self._text(box, 14)

    def _measure_tab(self):
        p = self._tab('측정')
        box = self._section(p, '실시간 측정 · 16비트 영상')
        self.measure_name = self._entry(box, '이름')
        self.measure_memo = self._entry(box, '메모')
        self.measure_video = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text='16비트 원본 영상 저장', variable=self.measure_video).pack(anchor='w')
        row = self._row(box)
        self._button(row, '측정 시작', lambda: self.command('/measure/start',
                    body={'name': self.measure_name.get(), 'memo': self.measure_memo.get(),
                          'video': self.measure_video.get()})).pack(side='left', padx=3)
        self._button(row, '측정 끝', lambda: self.command('/measure/stop',
                    body={'memo': self.measure_memo.get()})).pack(side='left', padx=3)
        self.measure_status = tk.StringVar(value='측정 대기 중')
        ttk.Label(box, textvariable=self.measure_status, wraplength=440).pack(anchor='w')
        box = self._section(p, '결과 · 지난 측정')
        self.measure_history_var, self.measure_history_box = self._combo(
            box, '지난 측정', [], self.show_measure_record)
        self.measure_text = self._text(box, 18)
        self.measure_n = self._entry(box, '회차 번호')
        self._button(box, '이름·메모 수정', lambda: self.command('/measure/note',
                    body={'n': self.measure_n.get(), 'name': self.measure_name.get(),
                          'memo': self.measure_memo.get()})).pack(anchor='e')

    def _performance_tab(self):
        p = self._tab('성능')
        box = self._section(p, '최근 성능')
        self.perf_summary = tk.StringVar(value='성능 자료 대기 중')
        ttk.Label(box, textvariable=self.perf_summary, wraplength=440).pack(anchor='w')
        self.graph_span = tk.StringVar(value='1분')
        span_box = ttk.Combobox(box, textvariable=self.graph_span,
                                values=['1분', '5분', '15분'], state='readonly', width=8)
        span_box.pack(anchor='e')
        span_box.bind('<<ComboboxSelected>>', lambda _e: self.draw_graph((self.perf.get('rows') or [])))
        self.graph = tk.Canvas(box, height=220, bg='#101419', highlightthickness=0)
        self.graph.pack(fill='x', pady=5)
        self.perf_text = self._text(box, 15)

    def _detail_tab(self):
        p = self._tab('상태 상세')
        box = self._section(p, '현재 적용된 값 · 장치 상태')
        self.detail_text = self._text(box, 30)
        row = self._row(box)
        for title, path in [('장치 상태', '/healthz'), ('짐벌·사람', '/gimbal/people'),
                            ('자동 모델 기록', '/rules/auto_det_log'), ('현재 모델', '/model')]:
            self._button(row, title, lambda path=path: self.detail_route(path)).pack(side='left', padx=2)
        ttk.Label(box, text='실제 적용 상태를 그대로 표시합니다.').pack(anchor='w')

    def command(self, path, body=None, **query):
        if path in ('/measure/start', '/measure/stop'):
            self.measure_override = False
        if path in ('/replay/start', '/replay/stop', '/replay/eval/reset'):
            self.eval_override = False
        if query:
            from urllib.parse import urlencode
            path += '?' + urlencode(query)
        def work():
            try:
                result = self.link.post(path, body)
                self.events.put(('command', result), timeout=1)
            except Exception as exc:
                self.events.put(('error', str(exc)), timeout=1)
        self.controls.submit(work)

    def rule(self, key, value):
        self.command('/rules/set', **{key: value})

    def rule_multi(self, **values):
        self.command('/rules/set', **values)

    def select_model(self, label):
        try:
            index = self.model_box['values'].index(label)
            self.command('/model/set', id=self.model_ids[index])
        except (ValueError, IndexError): pass

    def select_tracker(self, key):
        self.rule_multi(tracker=key, engine=self.engine_var.get() or 'auto')

    def select_reid(self, label):
        try:
            index = self.reid_box['values'].index(label)
            self.rule('reid', self.reid_keys[index])
        except (ValueError, IndexError): pass

    def select_predictor(self, label):
        try:
            index = self.predictor_box['values'].index(label)
            self.rule('predictor', self.predictor_keys[index])
        except (ValueError, IndexError): pass

    def apply_filters(self):
        self.rule_multi(warm_min=self.warm_min.get(), warm_max=self.warm_max.get(),
                        edge_birth=self.edge_birth.get(), confirm_ms=self.confirm_ms.get())

    def confirm(self, question, path):
        if messagebox.askyesno('ModelBank', question, parent=self.root):
            self.command(path)

    def start_replay(self):
        self.eval_override = False
        self.command('/replay/start', seq=self.replay_var.get(), fps=self.fps_var.get(),
                     loop=int(self.loop_var.get()), backend=self.backend_var.get() or 'python')

    def seek(self):
        try: pos = max(0, int(self.seek_var.get()) - 1)
        except ValueError:
            self.message.set('장 번호를 숫자로 입력하세요')
            return
        self.command('/replay/seek', pos=pos, pause=1)

    def skip_replay(self, seconds):
        state = self.replay.get('state') or {}
        if not state.get('on'): return
        fps = float(state.get('fps') or 25)
        pos = int(state.get('pos') or 0) + round(seconds * fps)
        pos = min(max(0, pos), max(0, int(state.get('total') or 1) - 1))
        self.command('/replay/seek', pos=pos, pause=1)

    def seek_from_scale(self, _event):
        self.seek_dragging = False
        self.seek_var.set(str(int(round(self.seek_position.get())) + 1))
        self.seek()

    def eval_view(self, path):
        self.eval_override = True
        def work():
            try: self.events.put(('eval', self.link.get(path)), timeout=1)
            except Exception as exc: self.events.put(('error', str(exc)), timeout=1)
        self.workers.submit(work)

    def show_measure_record(self, label):
        try: index = self.measure_history_box['values'].index(label)
        except ValueError: return
        number = self.measure_ids[index]
        self.measure_n.set(str(number))
        self.measure_override = True
        def work():
            try: self.events.put(('measure_record', self.link.get('/measure/' + str(number))), timeout=1)
            except Exception as exc: self.events.put(('error', str(exc)), timeout=1)
        self.workers.submit(work)

    def detail_route(self, path):
        def work():
            try: self.events.put(('detail', self.link.get(path)), timeout=1)
            except Exception as exc: self.events.put(('error', str(exc)), timeout=1)
        self.workers.submit(work)

    def video_loop(self):
        last = None
        while not self.stop.is_set():
            try:
                with self.link.session() as session:
                    while not self.stop.is_set():
                        head, jpeg = self.link.snapshot(self.quality_mode, session=session)
                        if head['status'] == 200 and jpeg and head.get('token') != last:
                            last = head.get('token')
                            with self.frame_lock:
                                self.frame_slot = (head, jpeg)
                        self.stop.wait(0.03)
            except Exception as exc:
                try: self.events.put(('error', str(exc)), timeout=0.2)
                except queue.Full: pass
                self.stop.wait(1)

    def state_loop(self):
        while not self.stop.is_set():
            try:
                start = time.time()
                clock = self.link.get('/time')
                end = time.time()
                data = {name: self.link.get(path) for name, path in
                        [('rules', '/rules'), ('model', '/model'), ('replay', '/replay'),
                         ('measure', '/measure'), ('perf', '/perf'),
                         ('health', '/healthz'), ('people', '/gimbal/people')]}
                data['_clock'] = (end - start, clock.get('t', end) - (start + end) / 2)
                try: self.events.put(('state', data), timeout=0.5)
                except queue.Full: pass
            except Exception as exc:
                try: self.events.put(('error', str(exc)), timeout=0.2)
                except queue.Full: pass
            self.stop.wait(1)

    def ping_loop(self):
        while not self.stop.is_set():
            try:
                data = self.link.post('/gimbal/ping')
                try: self.events.put(('ping', data), timeout=0.1)
                except queue.Full: pass
            except Exception: pass
            self.stop.wait(0.4)

    def pump(self):
        for _ in range(8):
            try: kind, data = self.events.get_nowait()
            except queue.Empty: break
            if kind == 'state': self.show_state(data)
            elif kind == 'ping': self.show_ping(data)
            elif kind == 'command':
                self.message.set(data.get('text') or data.get('answer') or '적용되었습니다')
            elif kind == 'eval': self._set_text(self.eval_text, pretty(data.get('eval', data)))
            elif kind == 'measure_record': self._set_text(self.measure_text, pretty(data))
            elif kind == 'detail': self._set_text(self.detail_text, pretty(data))
            elif kind == 'error':
                self.status.set('Orange Pi 연결 대기 중')
                self.message.set(data[:160])
                self.connected = False
        with self.frame_lock:
            frame = self.frame_slot
            self.frame_slot = None
        if frame is not None:
            self.show_frame(*frame)
        if not self.stop.is_set(): self.root.after(25, self.pump)

    def show_frame(self, header, jpeg):
        try:
            image = Image.open(io.BytesIO(jpeg)).convert('RGB')
            self.image = image
            self.jpeg_size = image.size
            self.frame_seq = header.get('seq')
            self.frame_token = header.get('token')
            self.overlay = header.get('overlay')
            self.overlay_boxes = self.smooth_boxes(self.overlay) if self.smooth_overlay.get() else (self.overlay or {}).get('b', [])
            self.draw_frame()
            now = time.time()
            self.shown_times.append(now)
            while self.shown_times and now - self.shown_times[0] > 2:
                self.shown_times.popleft()
            fps = len(self.shown_times) / 2
            capture = header.get('capture_unix')
            delay = (now + self.clock_offset - capture) * 1000 if capture is not None and self.clock_offset is not None else None
            self.frame_text.set(f"표시 프레임 {self.frame_seq} · {image.width}×{image.height}"
                                + (f' · 지연 {max(0, delay):.0f} ms' if delay is not None and 0 <= delay < 5000 else '')
                                + f' · 이 화면 {fps:.0f} FPS')
        except Exception as exc:
            self.message.set(f'영상 해독 실패: {exc}')

    def draw_frame(self):
        if self.image is None: return
        w = max(1, self.video.winfo_width())
        h = max(1, self.video.winfo_height())
        iw, ih = self.image.size
        scale = min(w / iw, h / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        x0, y0 = (w - dw) // 2, (h - dh) // 2
        self.image_box = (x0, y0, dw, dh)
        resized = self.image.resize((dw, dh), getattr(Image, 'Resampling', Image).BILINEAR)
        self.photo = ImageTk.PhotoImage(resized)
        self.video.delete('all')
        self.video.create_image(x0, y0, image=self.photo, anchor='nw')
        if self.following and not (self.replay.get('state') or {}).get('on'):
            cx, cy = x0 + dw / 2, y0 + dh * self.aim_screen_y
            self.video.create_line(cx - 12, cy, cx + 12, cy, fill=ORANGE, width=2)
            self.video.create_line(cx, cy - 12, cx, cy + 12, fill=ORANGE, width=2)
        overlay = self.overlay or {}
        if overlay.get('pf') == 1:
            for b in self.overlay_boxes:
                bx1, by1, bx2, by2, tid, followed = b
                color = ORANGE if followed else '#55e5a1'
                self.video.create_rectangle(x0 + bx1 * dw / overlay['w'], y0 + by1 * dh / overlay['h'],
                                            x0 + bx2 * dw / overlay['w'], y0 + by2 * dh / overlay['h'],
                                            outline=color, width=2)
                self.video.create_text(x0 + bx1 * dw / overlay['w'] + 4,
                                       y0 + by1 * dh / overlay['h'] + 4,
                                       text=str(tid), fill=color, anchor='nw')
            for p in overlay.get('p', []):
                if len(p) < 6: continue
                bx1, by1, bx2, by2, tid, label = p[:6]
                self.video.create_rectangle(x0 + bx1 * dw / overlay['w'],
                                            y0 + by1 * dh / overlay['h'],
                                            x0 + bx2 * dw / overlay['w'],
                                            y0 + by2 * dh / overlay['h'],
                                            outline=ORANGE, dash=(6, 4), width=2)
                self.video.create_text(x0 + bx1 * dw / overlay['w'] + 4,
                                       y0 + by1 * dh / overlay['h'] + 4,
                                       text=f'{tid} · {label}', fill=ORANGE, anchor='nw')
            search = overlay.get('s')
            if search and len(search) > 4:
                _, cx, cy, radius = search[:4]
                if radius and cx is not None and cy is not None:
                    self.video.create_oval(x0 + (cx - radius) * dw / overlay['w'],
                                           y0 + (cy - radius) * dh / overlay['h'],
                                           x0 + (cx + radius) * dw / overlay['w'],
                                           y0 + (cy + radius) * dh / overlay['h'],
                                           outline=ORANGE, dash=(4, 4))

    def reset_overlay(self):
        self.overlay_history = {}
        self.overlay_time = None
        self.overlay_boxes = self.smooth_boxes(self.overlay) if self.smooth_overlay.get() else (self.overlay or {}).get('b', [])
        self.draw_frame()

    def smooth_boxes(self, overlay):
        if not overlay or overlay.get('pf') != 1:
            self.overlay_history = {}
            self.overlay_time = None
            return []
        boxes = overlay.get('b') or []
        stamp = overlay.get('t')
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            self.overlay_history = {}
            self.overlay_time = None
            return boxes
        dt = stamp - self.overlay_time if self.overlay_time is not None else 0
        if dt <= 0 or dt > 0.120:
            self.overlay_history = {}
        counts = Counter(b[4] for b in boxes)
        alpha = 1 - math.exp(-max(dt, 0) / 0.065)
        next_history, out = {}, []
        for box in boxes:
            left, top, right, bottom, tid, followed = box
            width, height = right - left, bottom - top
            if counts[tid] != 1 or width <= 0 or height <= 0:
                out.append(box)
                continue
            x, y = (left + right) / 2, (top + bottom) / 2
            sx, sy = x, y
            prior = self.overlay_history.get(tid)
            if prior and prior['followed'] == followed and max(
                    width / prior['width'], prior['width'] / width,
                    height / prior['height'], prior['height'] / height) <= 1.35 and math.hypot(
                        x - prior['x'], y - prior['y']) < min(4, max(2, min(width, height) * 0.12)):
                dx = (prior['sx'] - x) * (1 - alpha)
                dy = (prior['sy'] - y) * (1 - alpha)
                distance = math.hypot(dx, dy)
                limit = min(2, min(width, height) * 0.06)
                factor = limit / distance if distance > limit else 1
                sx += dx * factor
                sy += dy * factor
            next_history[tid] = dict(x=x, y=y, sx=sx, sy=sy,
                                     width=width, height=height, followed=followed)
            out.append([left + sx - x, top + sy - y, right + sx - x,
                        bottom + sy - y, tid, followed])
        self.overlay_history = next_history
        self.overlay_time = stamp
        return out

    def on_video_click(self, event):
        if self.frame_seq is None: return
        x0, y0, w, h = self.image_box
        x, y = (event.x - x0) / w, (event.y - y0) / h
        if not 0 <= x <= 1 or not 0 <= y <= 1: return
        token = self.frame_token
        def work():
            try: self.events.put(('command', self.link.click(token, x, y)), timeout=1)
            except Exception as exc: self.events.put(('error', str(exc)), timeout=1)
        self.controls.submit(work)

    def show_ping(self, data):
        self.following = bool(data.get('auto'))
        if data.get('auto'):
            text = f"{data.get('id', 0)}번 사람을 따라가는 중"
        elif data.get('lost'):
            text = f"{data.get('id', 0)}번 사람을 놓침 · 다시 찾는 중"
        elif data.get('moving'):
            text = '수동 이동 중'
        else:
            text = '짐벌 대기 중'
        self.follow_text.set(text + (' · ' + data['answer'] if data.get('answer') else ''))
        self.gimbal_label.set(pretty({k: data.get(k) for k in ('running', 'ready', 'auto', 'moving', 'error', 'id', 'cam_idle')}))
        pose = data.get('pose') or data.get('position_estimate')
        if pose:
            self.gimbal_label.set(self.gimbal_label.get() + '\n짐벌 위치: ' + pretty(pose)[:170])

    def show_state(self, data):
        self.connected = True
        self.clock_samples.append(data['_clock'])
        self.clock_offset = min(self.clock_samples, key=lambda x: x[0])[1]
        health = data['health']
        self.status.set(f"Orange Pi 직결됨 · 카메라 {health.get('cam_fps', '–')} / 추적 {health.get('track_fps', '–')} FPS")
        self.rules, self.model, self.replay, self.measure, self.perf = (
            data['rules'], data['model'], data['replay'], data['measure'], data['perf'])
        r, m, rp, ms, pf = self.rules, self.model, self.replay, self.measure, self.perf
        extra = str((r.get('gimbal_cfg') or {}).get('extra') or '')
        aim_match = re.search(r'--aim-screen-y\s+([0-9.]+)', extra)
        self.aim_screen_y = float(aim_match.group(1)) if aim_match and 0 < float(aim_match.group(1)) < 1 else 0.5
        people = data['people']
        followed = (people.get('state') or {}).get('id')
        person = next((p for p in people.get('people', []) if p.get('id') == followed), None)
        if person and (people.get('state') or {}).get('auto'):
            aim = float((r.get('gimbal_cfg') or {}).get('aim_y') or 0.5)
            ex = (person['x1'] + person['x2']) / 2 - people.get('width', 640) / 2
            ey = person['y1'] + aim * (person['y2'] - person['y1']) - people.get('height', 480) / 2
            self.aim_text.set(f'가로 오차 {ex:+.0f} px · 세로 오차 {ey:+.0f} px')
        else:
            self.aim_text.set('가로 오차 – · 세로 오차 –')
        catalog = m.get('catalog') or []
        labels = [c['name'] for c in catalog]
        self.model_ids = [c['id'] for c in catalog]
        if tuple(self.model_box['values']) != tuple(labels): self.model_box['values'] = labels
        self.model_box.configure(state='disabled' if m.get('busy') or not catalog else 'readonly')
        for c in catalog:
            if c['id'] == (m.get('pending') if m.get('busy') else m.get('current')):
                self.model_var.set(c['name'])
                break
        self.model_note.set('모델 변경 중…' if m.get('busy') else
                            (m.get('last') or {}).get('text', '현재 모델 사용 중'))
        presets = r.get('presets') or []
        keys = [p['key'] if isinstance(p, dict) else p[0] for p in presets]
        if tuple(self.tracker_box['values']) != tuple(keys): self.tracker_box['values'] = keys
        self.tracker_var.set(r.get('tracker') or '')
        self.engine_var.set(r.get('engine') or 'auto')
        self.app_var.set(r.get('app') or 'off')
        self.e2e_var.set((r.get('e2e_live') or {}).get('want') or 'off')
        reids = r.get('reids') or []
        self.reid_keys = [x['key'] for x in reids]
        self.reid_box['values'] = [x['name'] for x in reids]
        for x in reids:
            if x['key'] == r.get('reid'): self.reid_var.set(x['name'])
        predictors = r.get('predictors') or []
        self.predictor_keys = [x['key'] for x in predictors]
        self.predictor_box['values'] = [x.get('name', x['key']) for x in predictors]
        for x in predictors:
            if x['key'] == r.get('predictor'):
                self.predictor_var.set(x.get('name', x['key']))
        self.box_var.set((r.get('box_refine') or {}).get('mode') or 'off')
        self.blend_var.set(str((r.get('box_refine') or {}).get('blend_s', 0)))
        self.e2e_link_var.set((r.get('e2e_live') or {}).get('link') or 'auto')
        for var, key, source in self.switches:
            value = r.get(source or key)
            if source and isinstance(value, dict): value = value.get(key if key in value else 'on')
            var.set(bool(value))
        typing = isinstance(self.root.focus_get(), (ttk.Entry, ttk.Combobox, tk.Entry))
        for var, key in [(self.warm_min, 'warm_min'), (self.warm_max, 'warm_max'),
                         (self.edge_birth, 'edge_birth')]:
            if not typing:
                var.set(str(r.get(key, '')))
        confirm = r.get('confirm') or {}
        if not typing:
            self.confirm_ms.set(str(confirm.get('ms', 0)))
        feed = r.get('npu_feed') or {}
        self.feed_box.configure(state='readonly' if feed.get('can_toggle', True) else 'disabled')
        mode = feed.get('requested') or feed.get('mode') or ''
        if mode in self.feed_box['values']: self.feed_var.set(mode)
        self.feed_label.set(pretty(feed)[:700])
        items = [it for it in rp.get('list', []) if it.get('usable')]
        self.replay_names = [it['name'] for it in items]
        self.replay_box['values'] = self.replay_names
        if not self.replay_var.get() and self.replay_names: self.replay_var.set(self.replay_names[0])
        backends = [b for b in rp.get('backends', []) if not b.get('stored')]
        backend_ids = [b['id'] for b in backends]
        self.backend_box['values'] = backend_ids
        if not self.backend_var.get() and backend_ids: self.backend_var.set(backend_ids[0])
        replay_state = rp.get('state') or {}
        self.replay_start_button.configure(state='normal' if rp.get('available') and self.replay_names and not replay_state.get('busy') else 'disabled')
        self.replay_stop_button.configure(state='normal' if replay_state.get('on') and not replay_state.get('busy') else 'disabled')
        self.camera_button.configure(state='normal' if r.get('sys_buttons', True) else 'disabled')
        self.restart_button.configure(state='normal' if r.get('server_restart') and not r.get('restart_pending') else 'disabled')
        self.reboot_button.configure(state='normal' if r.get('sys_buttons', True) else 'disabled')
        total = max(1, int(replay_state.get('total') or 1))
        self.seek_scale.configure(to=total - 1 if total > 1 else 1)
        if not self.seek_dragging and replay_state.get('pos') is not None:
            self.seek_position.set(float(replay_state['pos']))
        self.replay_state.set(pretty({k: replay_state.get(k) for k in
                              ('on', 'name', 'pos', 'total', 'paused', 'busy', 'returning', 'fps', 'backend')})[:650])
        if not self.eval_override:
            self._set_text(self.eval_text, pretty(rp.get('eval') or rp.get('eval_last') or {}))
        self.measure_status.set(f"{'측정 중' if ms.get('on') else '측정 대기'} · {ms.get('frames', 0)}장 · {ms.get('seconds', 0):.1f}초")
        history = ms.get('history') or []
        self.measure_ids = [item['n'] for item in history]
        self.measure_history_box['values'] = [f"{item['n']}회 · {item.get('name') or '이름 없음'} · {item.get('frames', 0)}장" for item in history]
        if not self.measure_override:
            self._set_text(self.measure_text, pretty({'last': ms.get('last'), 'history': history}))
        rows = pf.get('rows') or []
        pipeline = pf.get('pipeline') or {}
        self.perf_summary.set(
            f"최근 5초: 입력 {pipeline.get('camera_fps_5s') or '–'} FPS · "
            f"출력 {pipeline.get('track_fps_5s') or '–'} FPS · "
            f"출력 간격 p95 {pipeline.get('output_interval_p95_ms') or '–'} ms\n"
            f"최근 1분: 수신→추적 p95 {pipeline.get('track_p95_ms') or '–'} ms "
            f"(목표 33 ms) · 큐 버림 {pipeline.get('queue_drops_5s')}장")
        self._set_text(self.perf_text, pretty({'pipeline': pf.get('pipeline'), 'latest': rows[-3:]}))
        self.draw_graph(rows)

    def draw_graph(self, rows):
        c = self.graph
        c.delete('all')
        if len(rows) < 2: return
        w, h = max(300, c.winfo_width()), max(100, c.winfo_height())
        span = {'1분': 60, '5분': 300, '15분': 900}[self.graph_span.get()]
        recent = rows[-span:]
        values = [float(row[1] or 0) for row in recent]
        points = []
        for i, v in enumerate(values):
            points.extend((15 + i * (w - 30) / max(1, len(values) - 1), h - 15 - min(40, v) * (h - 30) / 40))
        c.create_line(*points, fill=ORANGE, width=2)
        c.create_text(12, 12, text='추적 출력 FPS', fill=FG, anchor='nw')

    def key_down(self, event):
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox, tk.Text)): return
        if event.keysym == 'space':
            self.command('/gimbal/home')
            return
        if event.keysym == 'Escape':
            self.command('/replay/stop')
            return
        directions = {'Left': (-1, 0), 'a': (-1, 0), 'Right': (1, 0), 'd': (1, 0),
                      'Up': (0, 1), 'w': (0, 1), 'Down': (0, -1), 's': (0, -1)}
        if event.keysym in directions and self.move_held is None:
            self.move_held = event.keysym
            x, y = directions[event.keysym]
            self.command('/gimbal/move', x=x, y=y)

    def key_up(self, event):
        if event.keysym == self.move_held: self.release()

    def release(self):
        if self.move_held:
            self.move_held = None
            self.command('/gimbal/move', x=0, y=0)

    def close(self):
        if self.stop.is_set():
            return
        self.stop.set()
        self.controls.shutdown(wait=True, cancel_futures=True)
        try: self.link.post('/gimbal/stop')
        except Exception: pass
        self.workers.shutdown(wait=False, cancel_futures=True)
        if self.on_close:
            self.on_close()
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='ModelBank native PC GUI over direct Ethernet')
    parser.add_argument('--board', default='192.168.77.2', help='Orange Pi direct Ethernet IPv4')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = tk.Tk()
    App(root, Link(args.board, args.port))
    root.mainloop()


if __name__ == '__main__':
    main()
