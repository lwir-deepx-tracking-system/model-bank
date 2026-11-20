"""The selectable tracker score limits (threshold_set.py) and the page factory that hands them to both engines.

Run from the tracking_server folder:  python -m unittest src.track.test_threshold_set

The Python engine is the real tracker. The C++ engine is stood in for by a class with the same attribute behaviour as the
extension type (fields that are properties of the type, plus a __dict__ that swallows any other name), so the test runs on a
PC without the built extension; test_native_trackers.py runs the same sets on the real extension where it is built.
"""
import os
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from ..web import id_rule_switches, page_settings
from . import native_backend, threshold_set
from .botsort import BotSortTracker
from .bytetrack import ByteTrackTracker
from .ocsort import OCSortTracker
from .sort import SortTracker

PY = dict(new=0.50, match=0.9, unconfirmed=0.6)
CPP = dict(new=0.25, match=0.8, unconfirmed=0.7)
ARGS = NS(n_init=2, class_gate=False, radius=2.0, sim=0.5, late_window=10, gate=3.5, no_motion_fallback=False,
          vit_conf=0.25, vit_min_h=32.0, sync_k=2)


def cpp_field(name):
    return property(lambda self: self.fields[name], lambda self, value: self.fields.__setitem__(name, value))


class OldCppTracker:
    """Extension built before 2026-11-20: `new` and `match` are fields, the unconfirmed limit is a literal 0.7 in the code."""
    new, match = cpp_field('new'), cpp_field('match')

    def __init__(self, mode, kwargs):
        vars(self).update(fields=dict(new=0.25, match=0.8, unconfirmed=0.7), mode=mode, tracked=[], lost=[], frame_id=0, next_id=0,
                          n_init=kwargs.get('n_init', 2), kf=None, new_this_frame=[], lost_this_frame=[], rebinds_motion=0,
                          rebinds_app=0, rebinds_fallback=0)

    def update(self, arr, det_feats=None, feat_fn=None):
        return []

    def limits(self):
        """what the C++ update would use"""
        return dict(self.fields)


class CppTracker(OldCppTracker):
    """Extension built from the 2026-11-20 source: `unconfirmed` is a field too."""
    unconfirmed = cpp_field('unconfirmed')


def limits(tracker):
    inner = threshold_set._inner(tracker)
    if isinstance(inner, OldCppTracker):
        return inner.limits()
    return {name: getattr(tracker, name) for name in ('new', 'match', 'unconfirmed')}


class Fixture(unittest.TestCase):
    def setUp(self):
        saved = dict(threshold_set.CHOICE)
        self.addCleanup(lambda: (threshold_set.CHOICE.clear(), threshold_set.CHOICE.update(saved)))
        threshold_set.CHOICE['key'] = 'engine_default'
        self.rules = dict(tracker='botsort', engine='auto')
        self.app = NS(args=ARGS, RULES=self.rules, tracker=None)
        patcher = patch.object(page_settings, 'APP', self.app)
        patcher.start(); self.addCleanup(patcher.stop)
        self.cpp_class = CppTracker
        self.make = threshold_set.factory(page_settings.mk_tracker_)   # as app.py builds it

    def tracker(self, kind, engine):
        """one tracker from the page factory, on the Python engine or on the stand-in C++ engine"""
        self.rules.update(tracker=kind, engine='python' if engine == 'Python' else 'auto')
        module = ModuleType('_native_' + kind)
        module.API_VERSION = 1
        module.create_tracker = lambda mode, kwargs: self.cpp_class(mode, kwargs)
        loaded = (module, None) if engine == 'C++' and kind != 'confirm3' else (None, 'matching extension not built')
        with patch.object(native_backend, '_load_module', return_value=loaded), patch.object(native_backend, '_warn_once'):
            made = self.make(kind)
        self.assertEqual(native_backend.backend_of(made), engine)
        return made


class FactoryTests(Fixture):
    def test_default_leaves_both_engines_exactly_as_before(self):
        py = self.tracker('botsort', 'Python'); cpp = self.tracker('botsort', 'C++')
        self.assertIsInstance(py, BotSortTracker)
        self.assertEqual(limits(py), PY)
        self.assertEqual(limits(cpp), CPP)
        plain = BotSortTracker('bytetrack', n_init=2)
        self.assertEqual({k: v for k, v in vars(py).items() if k not in ('kf', 'backend_label')},
                         {k: v for k, v in vars(plain).items() if k not in ('kf', 'backend_label')})   # nothing written on it
        self.assertEqual(set(vars(cpp)) - set(vars(CppTracker('botsort', {}))), {'backend_label'})
        for kind, cls in (('ocsort', OCSortTracker), ('sort', SortTracker)):
            py = self.tracker(kind, 'Python'); cpp = self.tracker(kind, 'C++')
            self.assertIsInstance(py, cls)
            self.assertEqual((py.new, limits(cpp)['new']), (0.50, 0.25), kind)
        self.assertEqual(threshold_set.state()['track_thr'], 'engine_default')

    def test_one_set_gives_both_engines_the_same_numbers(self):
        for key, want in (('py', PY), ('cpp', CPP)):
            threshold_set.set_key(key)
            for engine in ('Python', 'C++'):
                self.assertEqual(limits(self.tracker('botsort', engine)), want, (key, engine))
                for kind in ('ocsort', 'sort'):   # only the new-track score differs between their engines
                    made = self.tracker(kind, engine)
                    self.assertEqual(limits(made)['new'], want['new'], (key, kind, engine))
                    self.assertEqual(limits(made)['unconfirmed'], 0.7, (key, kind, engine))
                    if engine == 'C++':
                        self.assertEqual(limits(made)['match'], 0.8)

    def test_bytetrack_rows_are_not_touched_by_any_set(self):
        for key in threshold_set.KEYS:
            threshold_set.set_key(key)
            made = self.tracker('confirm3', 'Python')
            self.assertIsInstance(made, ByteTrackTracker)
            self.assertEqual(limits(made), dict(new=0.50, match=0.9, unconfirmed=0.7), key)
            self.assertNotIn(threshold_set.OWN, vars(made))

    def test_the_numbers_change_what_the_python_tracker_does(self):
        """A lone box with score 0.40 starts a track only when the new-track score is 0.25."""
        frame = np.array([[100, 100, 140, 220, 0.40, 0]], np.float32)
        born = {}
        for key in ('py', 'cpp'):
            threshold_set.set_key(key)
            made = self.tracker('botsort', 'Python')
            made.update(frame); made.update(frame)
            born[key] = made.next_id
        self.assertEqual(born, {'py': 0, 'cpp': 1})

    def test_old_cpp_build_keeps_its_fixed_unconfirmed_limit_and_says_so(self):
        self.cpp_class = OldCppTracker
        threshold_set.set_key('py')
        made = self.tracker('botsort', 'C++')
        self.assertEqual(limits(made), dict(new=0.50, match=0.9, unconfirmed=0.7))
        self.assertNotIn('unconfirmed', vars(threshold_set._inner(made)))   # not parked in the object's __dict__ as if it were set
        now = threshold_set.state(made, 'botsort')['track_thr_now']
        self.assertEqual((now['backend'], now['unconfirmed'], now['note']), ('C++', 0.7, threshold_set.REBUILD))
        threshold_set.set_key('cpp', made, 'botsort')
        self.assertEqual(threshold_set.state(made, 'botsort')['track_thr_now']['note'], '')   # 0.7 is what the cpp set asks for
        self.cpp_class = CppTracker
        threshold_set.set_key('py')
        self.assertEqual(threshold_set.state(self.tracker('botsort', 'C++'), 'botsort')['track_thr_now']['note'], '')


class LiveSwitchTests(Fixture):
    def test_running_tracker_gets_the_set_at_once_and_its_own_numbers_back(self):
        for engine, own in (('Python', PY), ('C++', CPP)):
            made = self.tracker('botsort', engine)
            for key, want in (('py', PY), ('cpp', CPP), ('engine_default', own), ('cpp', CPP), ('engine_default', own)):
                self.assertTrue(threshold_set.set_key(key, made, 'botsort'))
                self.assertEqual(limits(made), want, (engine, key))
            threshold_set.set_key('engine_default')

    def test_bad_value_changes_nothing(self):
        made = self.tracker('botsort', 'Python')
        self.assertFalse(threshold_set.set_key('fast', made, 'botsort'))
        self.assertEqual((threshold_set.CHOICE['key'], limits(made)), ('engine_default', PY))

    def test_start_value_from_the_environment(self):
        for value, key in (('py', 'py'), ('CPP', 'cpp'), ('', 'engine_default'), ('0.5', 'engine_default')):
            with patch.dict(os.environ, {'LWIR_TRACK_THR': value}):
                self.assertEqual(threshold_set._start_key(), key)
        with patch.dict(os.environ):
            os.environ.pop('LWIR_TRACK_THR', None)
            self.assertEqual(threshold_set._start_key(), 'engine_default')

    def test_rules_request_and_state(self):
        """/rules/set?track_thr=... through the installed switch: checked first, applied only with the rest of the query."""
        from ..detect import pose_duplicate
        saved = dict(pose_duplicate.RULE)
        self.addCleanup(lambda: (pose_duplicate.RULE.clear(), pose_duplicate.RULE.update(saved)))
        pose_duplicate.RULE.update(mode='named', dropped=0)
        seen = []

        def old_request(query):
            seen.append(dict(query))
            return ('warm_min' not in query or query['warm_min'] != ['x']), 'old parser'

        app = NS(args=NS(det='bank:some_pose_kpt_640x480'), RULES=dict(tracker='botsort'), tracker=self.tracker('botsort', 'C++'),
                 rules_request=old_request, rules_state=lambda: dict(tracker='botsort', old_key=1))
        id_rule_switches.install(app)
        state = app.rules_state()
        self.assertEqual((state['old_key'], state['track_thr'], state['nested_dup'], state['nested_dup_active']), (1, 'engine_default', 'named', False))
        self.assertEqual({k: state['track_thr_now'][k] for k in ('kind', 'backend', 'new', 'match', 'unconfirmed')},
                         dict(kind='botsort', backend='C++', **CPP))
        self.assertEqual(app.rules_request({'warm_min': ['0']}), (True, 'old parser'))            # other keys: untouched path
        self.assertEqual(app.rules_request({'track_thr': ['py']}), (True, '적용됨'))
        self.assertEqual(seen, [{'warm_min': ['0']}])                                                # our key alone: the old parser is not called
        self.assertEqual(limits(app.tracker), PY)
        self.assertFalse(app.rules_request({'track_thr': ['both']})[0])
        self.assertFalse(app.rules_request({'nested_dup': ['on']})[0])
        self.assertFalse(app.rules_request({'track_thr': ['cpp'], 'nested_dup': ['kpt'], 'warm_min': ['x']})[0])   # the rest refused: nothing changes
        self.assertEqual((threshold_set.CHOICE['key'], pose_duplicate.RULE['mode'], limits(app.tracker)), ('py', 'named', PY))
        self.assertTrue(app.rules_request({'track_thr': ['CPP'], 'nested_dup': ['kpt'], 'warm_min': ['0']})[0])
        self.assertEqual(seen[-1], {'warm_min': ['0']})
        state = app.rules_state()
        self.assertEqual((state['track_thr'], state['nested_dup'], limits(app.tracker)), ('cpp', 'kpt', CPP))
        self.assertEqual(state['track_thr_sets'], dict(py=PY, cpp=CPP))
        app.tracker = None                                                                           # before the first frame: no tracker yet
        self.assertTrue(app.rules_request({'track_thr': ['engine_default']})[0])
        self.assertIsNone(app.rules_state()['track_thr_now'])
        with patch.object(threshold_set, 'state', side_effect=RuntimeError('boom')):                # GET /rules must survive a fault in the extra keys
            state = app.rules_state()
        self.assertEqual((state['old_key'], 'track_thr' in state), (1, False))
        self.assertIn('boom', state['id_rule_switches_error'])


if __name__ == '__main__':
    unittest.main()
