"""2026-11-20: per-model cosine map (cos_map.py) - identity for models without a table, the shipped table for
REPVGG_SOLIDER_DISTILL_E_20261120, the switch to that model and back (reid_models.reid_swap), every covered cosine site, the BoT-SORT gate.
Run: python -m unittest src.reid.test_cos_map (from tracking_server/)"""
import json
import unittest

import numpy as np

from . import cos_map
from . import relink_online_judge as RJ
from . import relabel_online as RO
from .reid_models import REID_INFO, reid_swap
from ..track.botsort import BotSortTracker
from ..gimbal import follow_guard as FG

KEY = 'REPVGG_SOLIDER_DISTILL_E_20261120'


class CosMapTest(unittest.TestCase):
    def tearDown(self):
        cos_map.activate('')

    def test_table_shipped_and_monotone(self):
        x, y = cos_map.load_table(cos_map.MODELS / cos_map.TABLES[KEY])
        self.assertTrue(np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0))
        self.assertEqual((x[0], x[-1], y[0], y[-1]), (-1.0, 1.0, -1.0, 1.0))
        man = json.loads((cos_map.MODELS / 'MANIFEST.json').read_text(encoding='utf-8'))
        self.assertIn(KEY, man); self.assertIn(KEY, man['_links']); self.assertIn(man['_links'][KEY], man['_files'])
        self.assertIn(KEY, REID_INFO)

    def test_identity_without_map(self):
        for k in ('', 'RVGGA0_REID2C256', 'REPVGG_REVIEWED264_20261117'):
            self.assertFalse(cos_map.activate(k))
            for c in (-1.0, 0.0, 0.33, 0.596, 1.0):
                self.assertEqual(cos_map.cmap(c), c)
            self.assertEqual(cos_map.gate_thr(), 0.25)
            self.assertIsNone(cos_map.cmap(None))

    def test_map_scale_and_inverse(self):
        self.assertTrue(cos_map.activate(KEY))
        # fitted on the fixed crop set: student E's medians 0.43 / 0.76 / 0.18 land near REVIEWED264's 0.29 / 0.68 / 0.09
        self.assertAlmostEqual(cos_map.cmap(0.43), 0.28, delta=0.03)
        self.assertAlmostEqual(cos_map.cmap(0.762), 0.663, delta=0.03)
        for c in np.linspace(-0.5, 0.99, 40):
            self.assertAlmostEqual(cos_map.cmap(cos_map.inv(cos_map.cmap(c))), cos_map.cmap(c), places=6)
        g = cos_map.gate_thr()
        self.assertAlmostEqual(cos_map.cmap(1.0 - g), 0.75, places=6)   # raw gate limit = the raw cosine that maps to 0.75
        self.assertLess(g, 0.25)                                       # this model's cosines run high, so the raw limit is stricter
        x = cos_map.cmap(float('nan')); self.assertTrue(x != x)

    def test_switch_to_model_and_back(self):
        drained = []
        ok, new, txt = reid_swap(KEY, 'old', lambda k: ('engine', k), lambda: drained.append(1))
        self.assertTrue(ok); self.assertEqual(cos_map.active(), KEY); self.assertEqual(new, ('engine', KEY))
        ok, new, txt = reid_swap('REPVGG_REVIEWED264_20261117', new, lambda k: ('engine', k), lambda: drained.append(1))
        self.assertTrue(ok); self.assertEqual(cos_map.active(), ''); self.assertEqual(cos_map.cmap(0.5), 0.5)
        # a failed change keeps the old model AND its map
        cos_map.activate(KEY)
        def bad(k): raise RuntimeError('no file')
        ok, kept, txt = reid_swap('REPVGG_REVIEWED264_20261117', 'cur', bad, lambda: None)
        self.assertFalse(ok); self.assertEqual(kept, 'cur'); self.assertEqual(cos_map.active(), KEY)
        def bad_drain(): raise RuntimeError('busy')
        ok, kept, txt = reid_swap('RVGGA0_REID2C256', 'cur', lambda k: 'x', bad_drain)
        self.assertFalse(ok); self.assertEqual(cos_map.active(), KEY)
        self.assertEqual(len(drained), 2)

    def test_sites_use_the_map(self):
        a = np.zeros(128); a[0] = 1.0; b = np.zeros(128); b[0] = 0.76; b[1] = np.sqrt(1 - 0.76 ** 2)
        raw = float(a @ b)
        cos_map.activate(KEY)
        self.assertAlmostEqual(RJ.cos(a, b), cos_map.cmap(raw), places=9)           # judge
        self.assertLess(RJ.cos(a, b), raw)
        g = FG.FollowGuardV2.__new__(FG.FollowGuardV2); g.tmpl = [b]; g.distract = {7: b}   # follow guard
        st_, sd = g._sims(a)
        self.assertAlmostEqual(st_, cos_map.cmap(raw), places=9); self.assertAlmostEqual(sd, cos_map.cmap(raw), places=9)
        cos_map.activate('')
        self.assertAlmostEqual(RJ.cos(a, b), raw, places=12)
        st_, sd = g._sims(a); self.assertAlmostEqual(st_, raw, places=12)
        self.assertTrue(hasattr(RO, '_cmap'))

    def test_gate_python_tracker(self):
        t = BotSortTracker('bytetrack')
        cos_map.apply_gate(t); self.assertEqual(t.APP_THR, 0.25)                    # no map: untouched
        cos_map.activate(KEY); cos_map.apply_gate(t); self.assertAlmostEqual(t.APP_THR, cos_map.gate_thr())
        cos_map.activate(''); cos_map.apply_gate(t); self.assertEqual(t.APP_THR, 0.25)   # back to the class value
        self.assertIs(cos_map.apply_gate(None), None)

    def test_gate_native_tracker_if_built(self):
        try:
            from ..track import native_backend
            t = native_backend.create('botsort', 'botsort', dict(n_init=3), lambda: None, native=True)
        except Exception:
            self.skipTest('C++ tracker extension not built here')
        if t is None or type(t).__module__.startswith('src.track.botsort'):
            self.skipTest('C++ tracker extension not built here')
        cos_map.activate(KEY); cos_map.apply_gate(t); self.assertAlmostEqual(t.APP_THR, cos_map.gate_thr())
        cos_map.activate(''); cos_map.apply_gate(t); self.assertEqual(t.APP_THR, 0.25)

    # ---- 2026-11-21 (review): server START with E, through the real page tracker factory (page_settings.mk_tracker_, Python engine)
    def _factory_tracker(self):
        import types
        from ..common.state import APP
        from ..web import page_settings
        saved = {k: vars(APP).get(k) for k in ('args', 'RULES')}
        APP.args = types.SimpleNamespace(sim=0.5, late_window=10, gate=3.5, no_motion_fallback=False, vit_conf=0.25, vit_min_h=32.0, sync_k=2, radius=2.0,
                                         n_init=3, class_gate=True)
        APP.RULES = dict(engine='python')
        try:
            return page_settings.mk_tracker_('botsort')
        finally:
            for k, v in saved.items():
                if v is None: vars(APP).pop(k, None)
                else: setattr(APP, k, v)

    def test_start_with_E_as_relabel(self):
        on, err = cos_map.follow(KEY)                      # app.py start: APP.REID['cur'] = --relabel
        self.assertTrue(on); self.assertEqual(err, '')
        t = self._factory_tracker()                        # first tracker of the frame loop
        self.assertAlmostEqual(t.APP_THR, cos_map.gate_thr()); self.assertLess(t.APP_THR, 0.25)

    def test_start_with_E_saved_choice(self):
        cos_map.follow('RVGGA0_REID2C256')                 # start model of the start command
        t = self._factory_tracker(); self.assertEqual(t.APP_THR, 0.25)
        ok, new, txt = reid_swap(KEY, 'old', lambda k: ('engine', k), lambda: None)   # the saved choice is loaded by the frame loop
        self.assertTrue(ok)
        cos_map.apply_gate(t)                              # app.py: the running tracker after a successful change
        self.assertAlmostEqual(t.APP_THR, cos_map.gate_thr())
        t2 = self._factory_tracker(); self.assertAlmostEqual(t2.APP_THR, cos_map.gate_thr())   # every later tracker (new video, page change)
        ok, new, txt = reid_swap('RVGGA0_REID2C256', new, lambda k: ('engine', k), lambda: None)
        cos_map.apply_gate(t2); self.assertEqual(t2.APP_THR, 0.25); self.assertEqual(self._factory_tracker().APP_THR, 0.25)

    def test_start_with_broken_table_does_not_stop_the_server(self):
        old = cos_map.TABLES[KEY]
        cos_map.TABLES[KEY] = 'reid/missing_cosmap.json'
        try:
            on, err = cos_map.follow(KEY)
            self.assertFalse(on); self.assertIn('WITHOUT', err); self.assertEqual(cos_map.active(), '')
            ok, kept, txt = reid_swap(KEY, 'cur', lambda k: 'x', lambda: None)   # a live change to a model with a broken table is refused
            self.assertFalse(ok); self.assertEqual(kept, 'cur')
        finally:
            cos_map.TABLES[KEY] = old


if __name__ == '__main__':
    unittest.main()
