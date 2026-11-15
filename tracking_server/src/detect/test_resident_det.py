"""두 검출 모델 상주와 즉시 전환 (resident_det.py) 시험. NPU 없이 돈다: 엔진은 흉내(Engine), 프레임은 0 으로 채운 그림, 시각은 지어낸 값이다.

- 켜 두면 바꾸기 요청이 다음 프레임 경계에서 엔진을 만들지도 놓지도 않고 적용된다. 옛 짝에 이미 넣은 프레임(넣는 중이던 것, 넣어 둔 것)은
  옛 모델 결과로 쓰이고, 프레임이 빠지거나 두 번 들어가거나 순서가 바뀌지 않는다.
- 다른 짝을 못 만들면(NPU 메모리, 입출력 모양 다름, 켤 수 없는 실행) 한 줄을 찍고 옛 길(do_model_swap)로 바꾼다. 다시 켜면 다시 해 본다.
- 두 모델이 아닌 셋째 모델은 옛 길: 다른 짝을 먼저 놓고, 옛 엔진이 남지 않고, 두 모델로 돌아오면 다시 만든다.
- /rules 의 resident_det 는 JSON 으로 바뀐다. /rules/set?resident_det= 는 나머지가 받아들여질 때만 바뀌고, 끄면 다른 짝을 놓는다.
- 자동 선택(auto_det)은 즉시 바꾸기를 한 번만 센다.
Run from tracking_server/: python -m unittest src.detect.test_resident_det"""
from collections import deque
import gc
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import weakref

import numpy as np

from ..common.state import APP
from ..common.shared_values import CORE, DET, sha
from ..track import e2e_live
from ..web import page_settings
from .model_scan import BANK, W480_DETS, KPT_DETS
from . import auto_det, frame_feed, resident_det

NEAR_KEY, FAR_KEY, THIRD_KEY = auto_det.NEAR_KEY, auto_det.FAR_KEY, 'NEWMRW480'
NEAR_DET, FAR_DET, THIRD_DET = 'bank:' + NEAR_KEY, 'bank:' + FAR_KEY, 'w480'
SECOND = frame_feed.SECOND
MID = {THIRD_KEY: 0, NEAR_KEY: 1, FAR_KEY: 2}   # model number in a result
N, F = MID[NEAR_KEY], MID[FAR_KEY]
FPS = 30.0


class Engine:
    """Stand-in for npu_engine.Engine (the parts the feed and resident_det read). A frame's result says which model and core ran it: [frame, model, core]."""
    log = []      # (name, core) of every engine made, in order
    gone = []     # names of the disposed engines
    shape = {}    # model key -> input shape (default 1x480x640x3)

    def __init__(self, name, core, depth=1, model=None):
        self.name = name; self.core = core; self.model = model or name; self.b = self; self.packed = np.empty(3, np.float32)
        self.in_shape = Engine.shape.get(self.model, (1, 480, 640, 3)); self.names = ['box_s8', 'cls_s8', 'kpt_s8']
        self.slots = [np.empty((1, 4, 60, 80), np.float32), np.empty((1, 16, 60, 80), np.float32), np.empty((1, 51, 60, 80), np.float32)]
        self.split6 = True; self.kpt = True; self.runs = []
        Engine.log.append((name, core))

    def run_async(self, image, *, user_arg, output_buffer):
        self.runs.append(user_arg['f']); return (self.name, MID[self.model], self.core, user_arg['f'])   # the job: which engine ran it (names only, no reference)

    def wait(self):
        name, model, core, f = APP.pending.pop(self.name)
        if name != self.name: raise RuntimeError('job of %s waited on %s' % (name, self.name))   # the SDK cannot wait a job on another engine
        return [np.array([f, model, core], np.float32)]

    def dispose(self):
        Engine.gone.append(self.name)


def person(tid, h, x=100.0, top=50.0):
    return [x, top, x + h * 0.4, top + h, tid, 0.9, 0, 0]


class Base(unittest.TestCase):
    def setUp(self):
        self.saved = dict(vars(APP)); self.core = dict(CORE); self.det = dict(DET); self.bank = dict(BANK)
        self.w480, self.kpt = set(W480_DETS), set(KPT_DETS)
        auto_det.reset(); resident_det.reset()
        Engine.log = []; Engine.gone = []; Engine.shape = {}
        self.tmp = tempfile.TemporaryDirectory()
        man = {}
        for key in (THIRD_KEY, NEAR_KEY, FAR_KEY):   # small stand-in files: the sha check is the real one
            p = Path(self.tmp.name) / (key + '.dxnn'); p.write_bytes(b'model %s' % key.encode()); BANK[key] = p; man[key] = sha(p)
        CORE[NEAR_KEY] = CORE[FAR_KEY] = 0
        DET[NEAR_DET] = dict(names=[NEAR_KEY], yoff=16.0); DET[FAR_DET] = dict(names=[FAR_KEY], yoff=16.0)
        W480_DETS.update((NEAR_DET, FAR_DET)); KPT_DETS.update((NEAR_DET, FAR_DET))
        cat = [dict(id=1, det=THIRD_DET, key=THIRD_KEY, name='third'), dict(id=2, det=NEAR_DET, key=NEAR_KEY, name='near'), dict(id=3, det=FAR_DET, key=FAR_KEY, name='far')]
        self.live = SimpleNamespace(cond=threading.Condition(), q=deque(), stopped=False, cam_lost=False, prev_t=None, last=None, stabq=deque(), arr=deque(), rawq=deque())
        self.sel = []; self.lines = []; self.codes = []; self.n = 0; self.hold = {}; self.entered = threading.Event()
        APP.LIVE = self.live; APP.GB = None; APP.FEED_MODE = 'arrival'; APP.pending = {}; APP.lv_to8 = lambda raw: raw; APP.FRAME_DY = [0.0]
        APP.args = SimpleNamespace(pipeline=True, gmc_thread=False, live_wait=0, det=NEAR_DET, mock=False, npu_depth=1, gimbal='')
        APP.MODEL = dict(lock=threading.Lock(), req=None, busy=False, last=None, history=[], catalog=cat)
        APP.MODEL_TEXT = {0: '바뀜', 1: '없는 모델 번호', 2: '다른 변경이 진행 중', 3: '모델을 만들지 못함', 4: '실패해서 이전 모델로 되돌림', 5: '되돌리기도 실패', 6: '준비 안 됨'}
        APP.man = man; APP.det_shape = {THIRD_KEY: (480, 640), NEAR_KEY: (480, 640), FAR_KEY: (480, 640)}
        APP.SEL = {}; APP.SEL_ON = False; APP.sel_save = lambda **kw: self.sel.append(kw)
        APP.model_request = page_settings.model_request
        APP.Engine = Engine; APP.b = None
        APP.eng = {NEAR_KEY: Engine(NEAR_KEY, 0, 1)}; APP.det_names = [NEAR_KEY]; APP.det_yoff = 16.0; APP.a = APP.eng[NEAR_KEY]   # as app.py at start
        self.patches = [patch('builtins.print', side_effect=lambda *a, **k: self.lines.append(' '.join(str(x) for x in a))),
                        patch.dict(frame_feed.STAB, on=False), patch.object(frame_feed, 'stab_off', return_value=None),
                        patch.object(frame_feed.camera_extra.replay_eval, 'lossless', return_value=True),   # nothing dropped: every frame must come out
                        patch.object(frame_feed.camera_extra.replay_eval, 'reset_first_input', return_value=False),
                        patch.object(frame_feed.detect_frame, 'submit_tiles', side_effect=self.submit_tiles)]
        for p_ in self.patches: p_.start()
        APP.FEED = frame_feed.Feed(frame_feed.make_second()); APP.FEED_READY = True   # the arrival feed: near on cores 0 and 1

    def tearDown(self):
        feed = vars(APP).get('FEED')
        if feed is not None: feed._stop_thread()
        for p_ in reversed(self.patches): p_.stop()
        APP.__dict__.clear(); APP.__dict__.update(self.saved)
        CORE.clear(); CORE.update(self.core); DET.clear(); DET.update(self.det); BANK.clear(); BANK.update(self.bank)
        W480_DETS.clear(); W480_DETS.update(self.w480); KPT_DETS.clear(); KPT_DETS.update(self.kpt)
        auto_det.reset(); resident_det.reset(); self.tmp.cleanup()

    def submit_tiles(self, image, seq):   # the feeding thread, after it chose the engine under the lock; hold[frame] = Event keeps it here, before it submits
        ev = self.hold.get(seq)
        if ev is not None: self.entered.set(); ev.wait(5)
        APP.a.submit(image, seq); return seq

    def turn_on(self, v='1'):
        ok, text, apply = resident_det.parse_request({'resident_det': [v]})
        self.assertTrue(ok, text); apply()

    def push(self, n):   # n camera frames into the arrival queue, 30 a second
        with self.live.cond:
            for _ in range(n): self.live.q.append((1.0 + self.n / FPS, np.zeros((480, 640), np.uint8))); self.n += 1
            self.live.cond.notify_all()

    def take(self, k):   # the frame loop: read the next frame, wait for its detector result
        g, _ = APP.FEED.read_frame(k); self.assertIsNotNone(g)
        return tuple(int(v) for v in APP.FEED.wait()[0])

    def boundary(self, i):
        """Stand-in for the top of the frame loop in app.py: the v23 model change block (the instant switch first, else the old path) and resident_det.frame_step."""
        rq = APP.MODEL['req']
        if rq is not None:
            APP.MODEL['busy'] = True
            code = resident_det.instant(rq)
            if code is None:
                APP.FEED.pause(); resident_det.before_swap(rq['det']); code = page_settings.do_model_swap(rq['det']); APP.FEED.resume(code)
            with APP.MODEL['lock']:
                APP.MODEL['req'] = None; APP.MODEL['busy'] = False; APP.MODEL['last'] = dict(id=rq['id'], result=code)
            self.codes.append(code)
        resident_det.frame_step(i)

    def said(self, start):
        return [l for l in self.lines if l.startswith(start)]

    def wait_out(self, n):   # until the feeding thread has n frames submitted and not yet taken
        with self.live.cond:
            self.assertTrue(self.live.cond.wait_for(lambda: len(APP.FEED.out) >= n and not APP.FEED.working, timeout=2))


class InstantSwitchTest(Base):
    def test_switch_at_the_next_frame_without_engines_and_without_losing_frames(self):
        self.turn_on()
        self.boundary(0)   # frame_step makes the far pair next to the near pair the feed already has
        st = resident_det.state()
        self.assertEqual((st['on'], st['active'], st['mode'], st['cores']), (True, True, 'on', [0, 1]))
        self.assertEqual(sorted((p['det'], p['in_use']) for p in st['resident']), sorted([(NEAR_DET, True), (FAR_DET, False)]))
        self.assertEqual(Engine.log[-2:], [(FAR_KEY, 0), (FAR_KEY + SECOND, 1)])
        made = len(Engine.log)
        far0 = resident_det.RES.pairs[FAR_DET]['eng'][0]
        self.hold[1] = threading.Event()   # frame 1: near core 1 already chosen, not yet submitted, when the switch comes
        self.push(8)
        got = [self.take(0)]
        self.assertTrue(self.entered.wait(2))
        self.assertEqual(APP.model_request(3), -1)   # far, as the page or auto_det asks
        self.boundary(1)
        self.assertEqual(self.codes, [0])
        self.assertEqual((APP.args.det, APP.det_names, APP.det_yoff, APP.b), (FAR_DET, [FAR_KEY], 16.0, None))
        self.assertEqual(APP.eng, {FAR_KEY: far0}); self.assertIs(APP.a, APP.FEED)
        self.assertEqual(self.said('MODEL instant'), ['MODEL instant %s -> %s (resident, cores 0/1)' % (NEAR_DET, FAR_DET)])
        self.hold[1].set()
        got += [self.take(k) for k in (1, 2, 3)]
        self.wait_out(1)   # frame 4 already submitted to the far pair when the next switch comes
        self.assertEqual(APP.model_request(2), -1)
        self.boundary(4)
        got += [self.take(k) for k in (4, 5, 6, 7)]
        self.assertEqual(got, [(0, N, 0), (1, N, 1), (2, F, 0), (3, F, 1), (4, F, 0), (5, N, 1), (6, N, 0), (7, N, 1)])   # old frames from the old model, order kept
        self.assertEqual(len(Engine.log), made); self.assertEqual(Engine.gone, [])   # no engine made or released by a switch
        ran = sorted(f for e in [APP.FEED.eng[0], APP.FEED.eng[1]] + resident_det.RES.pairs[FAR_DET]['eng'] for f in e.runs if f >= 0)
        self.assertEqual(ran, list(range(8)))   # every frame went to the detector once (nothing submitted again)
        self.assertEqual((APP.FEED.st['taken'], APP.FEED.st['order_bad'], APP.FEED.st['redo']), (8, 0, 0))
        self.assertEqual(self.said('MODEL changed'), [])   # the old path was not used
        st = resident_det.state()
        self.assertEqual((st['switches'], st['current_det'], st['last_switch']['from'], st['last_switch']['to'], st['last_switch']['old_frames']), (2, NEAR_DET, FAR_DET, NEAR_DET, 1))
        self.assertGreaterEqual(st['last_switch']['ms'], 0.0); self.assertGreaterEqual(st['last_switch']['ms_switch'], 0.0)
        self.assertEqual(APP.MODEL['last']['result'], 0)
        json.dumps(st)

    def test_auto_det_counts_the_instant_switch_once(self):
        self.turn_on(); self.boundary(0)
        ok, text, apply = auto_det.parse_request({'auto_det': ['1']}); self.assertTrue(ok, text); apply()
        asked = []
        for k in range(int(2 * FPS)):   # a small whole person: far after 1 s
            r = auto_det.on_frame(k / FPS, [person(4, 70)], frame=k)
            if r is not None: asked.append(r)
        self.assertEqual(asked, ['far']); self.assertEqual(APP.MODEL['req']['det'], FAR_DET)
        self.boundary(60)
        self.assertEqual((self.codes, APP.args.det), ([0], FAR_DET))
        for k in range(60, 90): auto_det.on_frame(k / FPS, [person(4, 70)], frame=k)
        st = auto_det.state()
        self.assertEqual((st['current'], st['pending'], st['switches'], st['failures']), ('far', None, 1, 0))
        self.assertEqual(auto_det.AUTO.ch.recent(3.0), 1)   # counted when asked, not again when the detector changed
        self.assertEqual(resident_det.state()['switches'], 1)
        json.dumps(st)


class FeedCoreTest(unittest.TestCase):
    def test_a_core_still_running_the_other_models_frame_takes_no_new_frame(self):
        feed = object.__new__(frame_feed.Feed)
        feed.mode = 'arrival'; feed.eng = [object(), object()]; feed.busy = [False, False]; feed.last_e = 1
        self.assertEqual(feed._free(), 0)
        feed.held = [True, False]   # the old pair's core 0 has not given its result yet
        self.assertEqual(feed._free(), 1)
        feed.held = [True, True]
        self.assertIsNone(feed._free())
        feed.held = [False, False]; feed.busy = [False, True]
        self.assertEqual(feed._free(), 0)
        feed.mode = 'normal'; feed.busy = [False, False]; feed.held = [True, False]
        self.assertIsNone(feed._free())
        feed.held = None
        self.assertEqual(feed._free(), 0)


class FallbackTest(Base):
    def test_failed_extra_pair_falls_back_to_the_old_path_with_one_line(self):
        def engine(name, core, depth=1, model=None):
            if name == FAR_KEY + SECOND: raise RuntimeError('NPU memory full')
            return Engine(name, core, depth, model)
        APP.Engine = MagicMock(side_effect=engine)
        self.turn_on(); self.boundary(0)
        st = resident_det.state()
        self.assertEqual((st['active'], st['mode'], st['resident']), (False, 'failed', []))
        self.assertIn('NPU memory full', st['failed'])
        self.assertEqual(Engine.gone, [FAR_KEY])   # the half-made pair is let go
        self.assertEqual(len(self.said('RESIDENT')), 1); self.assertIn('old path', self.said('RESIDENT')[0])
        calls = APP.Engine.call_count
        self.boundary(1); self.boundary(2)
        self.assertEqual(APP.Engine.call_count, calls)   # no new try by itself
        APP.Engine.side_effect = lambda name, core, depth=1, model=None: Engine(name, core, depth, model)
        self.assertEqual(APP.model_request(3), -1); self.boundary(3)
        self.assertEqual((self.codes, APP.args.det), ([0], FAR_DET))
        self.assertEqual(len(self.said('MODEL changed')), 1); self.assertEqual(self.said('MODEL instant'), [])   # today's path
        self.assertEqual(len(self.said('RESIDENT')), 1)   # still the one line
        self.assertEqual(resident_det.state()['fallbacks'], 1)
        json.dumps(resident_det.state())
        self.turn_on(); self.boundary(4)   # turned on again: one more try (now near is the other model)
        self.assertTrue(resident_det.state()['active'])
        self.assertEqual(Engine.log[-2:], [(NEAR_KEY, 0), (NEAR_KEY + SECOND, 1)])

    def test_different_input_output_form_is_not_made(self):
        Engine.shape[FAR_KEY] = (1, 640, 640, 3)
        self.turn_on(); self.boundary(0)
        st = resident_det.state()
        self.assertEqual((st['active'], st['mode']), (False, 'failed'))
        self.assertEqual(sorted(Engine.gone), sorted([FAR_KEY, FAR_KEY + SECOND]))
        self.assertEqual(len(self.said('RESIDENT refused')), 1)

    def test_runs_that_cannot_hold_both_models_print_one_line_each(self):
        self.turn_on()
        APP.args.mock = True
        for i in range(3): self.boundary(i)
        self.assertEqual((resident_det.state()['mode'], len(self.said('RESIDENT refused: not with --mock'))), ('refused', 1))
        APP.args.mock = False; APP.FEED.mode = 'normal'
        for i in range(3): self.boundary(i)
        self.assertEqual(len(self.said('RESIDENT refused: it needs the arrival feed')), 1)
        APP.FEED.mode = 'arrival'
        with patch.dict(e2e_live.E2E, want='ftt'):
            for i in range(3): self.boundary(i)
            self.assertEqual(len(self.said('RESIDENT waiting: the end-to-end model is on')), 1)
            self.assertFalse(resident_det.state()['active'])
        self.assertEqual(len(Engine.log), 2)   # nothing made so far (only the feed's near pair)
        self.boundary(9)   # end-to-end off again: made now
        self.assertTrue(resident_det.state()['active'])
        with patch.dict(e2e_live.E2E, want='ftt'):
            self.boundary(10)   # on again: the other pair goes, before the end-to-end model is loaded in this frame
        self.assertEqual(sorted(Engine.gone), sorted([FAR_KEY, FAR_KEY + SECOND]))
        self.assertEqual(len(self.said('RESIDENT off: the end-to-end model is on')), 1)
        self.assertEqual(len(self.said('RESIDENT waiting: the end-to-end model is on')), 1)


class ThirdModelTest(Base):
    def test_third_model_takes_the_old_path_and_the_pair_comes_back(self):
        self.turn_on(); self.boundary(0)
        old0 = weakref.ref(APP.FEED.eng[0]); far0 = weakref.ref(resident_det.RES.pairs[FAR_DET]['eng'][0])
        alive = []

        def engine(name, core, depth=1, model=None):
            if name == THIRD_KEY: gc.collect(); alive.append((old0() is not None, far0() is not None))
            return Engine(name, core, depth, model)
        APP.Engine = engine
        self.push(12)
        got = [self.take(0), self.take(1)]
        self.wait_out(1)   # frame 2 is on near core 0 when the third model is asked for
        self.assertEqual(APP.model_request(1), -1); self.boundary(2)   # the third model, chosen by hand
        self.assertEqual((self.codes, APP.args.det), ([0], THIRD_DET))
        self.assertEqual(alive, [(False, False)])   # near core 0 and the far pair were gone when the third model's engine was made (nothing kept them, not even the frame on the NPU)
        self.assertEqual(sorted(Engine.gone), sorted([FAR_KEY, FAR_KEY + SECOND, NEAR_KEY + SECOND]))   # the other pair went first; Feed.pause released near core 1 as before
        self.assertEqual(list(APP.eng), [THIRD_KEY])
        st = resident_det.state()
        self.assertEqual((st['active'], st['mode'], st['resident'], st['on']), (False, 'waiting', [], True))
        self.assertEqual(len(self.said('MODEL changed')), 1); self.assertEqual(self.said('MODEL instant'), [])
        got += [self.take(2), self.take(3)]   # frame 2 went in again for the new model (the old path, as before)
        n = len(Engine.log); self.boundary(3); self.boundary(4)
        self.assertEqual(len(Engine.log), n)   # not made while the third model runs
        self.assertEqual(APP.model_request(2), -1); self.boundary(5)   # back to near (old path), then the far pair again
        self.assertEqual(APP.args.det, NEAR_DET); self.assertTrue(resident_det.state()['active'])
        self.assertEqual(Engine.log[n:], [(NEAR_KEY, 0), (NEAR_KEY + SECOND, 1), (FAR_KEY, 0), (FAR_KEY + SECOND, 1)])
        got += [self.take(4), self.take(5)]
        n = len(Engine.log)
        self.assertEqual(APP.model_request(3), -1); self.boundary(6)
        self.assertEqual((len(Engine.log), APP.args.det, self.codes[-1]), (n, FAR_DET, 0))
        self.assertEqual(len(self.said('MODEL instant')), 1)
        got += [self.take(k) for k in (6, 7, 8)]
        self.assertEqual([g[0] for g in got], list(range(9)))
        self.assertEqual([g[1] for g in got[:6]] + [got[8][1]], [N, N, MID[THIRD_KEY], MID[THIRD_KEY], N, N, F])
        self.assertEqual(APP.FEED.st['order_bad'], 0)


class RulesTest(Base):
    def test_rules_key_off_by_default_saved_and_released_when_off(self):
        self.boundary(0)
        self.assertEqual((len(Engine.log), resident_det.state()['mode'], resident_det.state()['on']), (2, 'off', False))   # off by default: nothing made
        APP.rules_request0_ = MagicMock(return_value=(True, '적용됨'))
        self.assertFalse(page_settings.rules_request_2({'resident_det': ['2']})[0])
        APP.rules_request0_.return_value = (False, 'bad')
        self.assertFalse(page_settings.rules_request_2({'resident_det': ['1'], 'warm_min': ['x']})[0])
        self.assertFalse(resident_det.RES.want)   # the rest failed: nothing changed
        APP.rules_request0_.return_value = (True, '적용됨')
        ok, _ = page_settings.rules_request_2({'resident_det': ['1'], 'warm_min': ['0']})
        self.assertTrue(ok); self.assertTrue(resident_det.RES.want); self.assertEqual(self.sel[-1], dict(resident_det=True))
        APP.rules_request0_.assert_called_with({'warm_min': ['0']})   # the rest goes to the old parser without the resident_det key
        self.boundary(1); self.assertTrue(resident_det.state()['active'])
        far = resident_det.RES.pairs[FAR_DET]
        far['busy'][0] = True   # its last frame is still on the NPU
        self.assertTrue(page_settings.rules_request_2({'resident_det': ['0']})[0]); self.assertEqual(self.sel[-1], dict(resident_det=False))
        self.boundary(2)
        st = resident_det.state()
        self.assertEqual((Engine.gone, st['active'], st['release_waiting']), ([], False, 'turned off'))   # released at the next frame, not under a running job
        far['busy'][0] = False; self.boundary(3)
        self.assertEqual(sorted(Engine.gone), sorted([FAR_KEY, FAR_KEY + SECOND]))
        st = resident_det.state()
        self.assertEqual((st['on'], st['active'], st['mode'], st['resident'], st['release_waiting']), (False, False, 'off', [], None))
        self.assertEqual(APP.model_request(3), -1); self.boundary(4)
        self.assertEqual((len(self.said('MODEL changed')), self.said('MODEL instant')), (1, []))   # off: today's path

    def test_rules_state_is_json(self):
        json.dumps(resident_det.state())
        self.turn_on(); self.boundary(0); json.dumps(resident_det.state())
        self.assertEqual(APP.model_request(3), -1); self.boundary(1)
        APP.rules_state0_ = lambda: {}
        out = page_settings.rules_state_2()
        st = json.loads(json.dumps(out))['resident_det']
        self.assertEqual((st['on'], st['active'], st['cores'], st['switches'], st['current_det']), (True, True, [0, 1], 1, FAR_DET))
        self.assertEqual(sorted(p['key'] for p in st['resident']), sorted([NEAR_KEY, FAR_KEY]))
        self.assertEqual(st['keys'], dict(near=NEAR_KEY, far=FAR_KEY))
        for k in ('reason', 'why', 'last_switch', 'made', 'fallbacks', 'failed', 'source'): self.assertIn(k, st)

    def test_environment_turns_it_on(self):
        with patch.dict(os.environ, {resident_det.ENV: '1'}):
            resident_det.reset()
        self.assertEqual((resident_det.RES.want, resident_det.state()['source']), (True, 'env'))
        self.boundary(0); self.assertTrue(resident_det.state()['active'])
        resident_det.close()   # server end: the other pair goes
        self.assertEqual(sorted(Engine.gone), sorted([FAR_KEY, FAR_KEY + SECOND]))


if __name__ == '__main__':
    unittest.main()
