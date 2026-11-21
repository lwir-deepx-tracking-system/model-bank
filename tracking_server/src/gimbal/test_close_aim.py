"""PC-only tests of the close-person aim (src/gimbal/close_aim.py). No camera or serial port is opened."""
import math
import os
import time
import unittest
from unittest.mock import patch

from . import close_aim as CA
from types import SimpleNamespace as NS

from . import test_live_tracks as TL

H = 480.0


def row(top, h, x1=200., w=120.):
    return (7, x1, top, x1 + w, top + h, .9)


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(CA.STATE); CA.STATE['on'] = True; CA.reset()

    def tearDown(self):
        CA.STATE.clear(); CA.STATE.update(self.saved)

    def aim_after(self, top, h, base=.5, frames=60, dt=1 / 30.):
        """the bridge aim point fraction of the real box after `frames` steady frames"""
        out = None
        for i in range(frames):
            out = CA.apply([row(top, h)], 7, base, 100. + i * dt, H)[0]
        return (out[2] + base * (out[4] - out[2]) - top) / h, out

    def test_small_box_keeps_the_bridge_aim(self):
        for frac in (.1, .3, .45):
            CA.reset()
            aim, out = self.aim_after(60., frac * H)
            self.assertAlmostEqual(aim, .5, places=6)
            self.assertEqual(out, row(60., frac * H))   # the row is not touched at all

    def test_large_box_aims_higher(self):
        aim, _ = self.aim_after(40., .80 * H)
        self.assertAlmostEqual(aim, CA.CLOSE_AIM, places=3)
        CA.reset(); aim_mid, _ = self.aim_after(40., .625 * H)   # half way: smoothstep 0.5
        self.assertAlmostEqual(aim_mid, .375, places=3)
        CA.reset(); aim_big, _ = self.aim_after(10., .97 * H)
        self.assertAlmostEqual(aim_big, CA.CLOSE_AIM, places=3)

    def test_mapping_is_monotone_and_continuous(self):
        prev = .5
        for i in range(101):
            a = CA.aim_for(.5, CA.closeness(i / 100.))
            self.assertLessEqual(a, prev + 1e-12); self.assertLess(abs(a - prev), .02); prev = a

    def test_top_touching_box_is_close_and_held(self):
        aim, _ = self.aim_after(0., .35 * H, frames=30)   # small but its top is at the picture edge
        self.assertAlmostEqual(aim, CA.CLOSE_AIM, places=3)
        t = 100. + 29 / 30.
        out = CA.apply([row(80., .35 * H)], 7, .5, t + 1.0, H)[0]   # 1 s later the top left the edge: still close (hold 2 s)
        self.assertAlmostEqual((out[2] - 80.) / (.35 * H) + .5, CA.CLOSE_AIM, places=3)
        a = []
        for i in range(1, 200):   # after the hold the aim goes back down smoothly
            out = CA.apply([row(80., .35 * H)], 7, .5, t + 1.0 + i / 30., H)[0]; a.append((out[2] - 80.) / (.35 * H) + .5)
        self.assertAlmostEqual(a[-1], .5, places=2)
        self.assertTrue(all(y >= x - 1e-12 for x, y in zip(a, a[1:])))
        self.assertLess(max(y - x for x, y in zip(a, a[1:])), .02)

    def test_smooth_over_size_flicker(self):
        """box height jumping 40 % <-> 85 % every frame: the aim moves by less than 0.02 of the box per frame."""
        aims = []
        for i in range(90):
            h = (.40 if i % 2 else .85) * H
            out = CA.apply([row(60., h)], 7, .5, 100. + i / 30., H)[0]
            aims.append((out[2] - 60.) / h + .5)
        self.assertLess(max(abs(b - a) for a, b in zip(aims, aims[1:])), .04)
        self.assertTrue(CA.CLOSE_AIM < aims[-1] < .5)

    def test_horizontal_and_size_unchanged_and_others_untouched(self):
        other = (9, 10., 0., 60., 470., .8)
        out = CA.apply([other, row(5., .9 * H)], 7, .5, 100., H)
        self.assertEqual(out[0], other)
        r, o = out[1], row(5., .9 * H)
        self.assertEqual((r[0], r[1], r[3], r[5]), (o[0], o[1], o[3], o[5]))
        self.assertAlmostEqual(r[4] - r[2], o[4] - o[2], places=9)
        self.assertLess(r[2], o[2])   # moved up = the bridge aims higher

    def test_never_lower_than_a_head_aim(self):
        aim, out = self.aim_after(0., .95 * H, base=.08)
        self.assertAlmostEqual(aim, .08, places=6); self.assertEqual(out, row(0., .95 * H))

    def test_off_not_following_or_lost(self):
        CA.set_on(False)
        self.assertEqual(CA.apply([row(0., .9 * H)], 7, .5, 100., H), [row(0., .9 * H)])
        CA.set_on(True)
        self.assertEqual(CA.apply([row(0., .9 * H)], 7, .5, 100., H, following=False), [row(0., .9 * H)])
        self.assertEqual(CA.apply([row(0., .9 * H)], 0, .5, 100., H), [row(0., .9 * H)])
        self.assertEqual(CA.apply([row(0., .9 * H)], 8, .5, 100., H), [row(0., .9 * H)])   # the followed id is not in the rows
        self.assertIsNone(CA.state()['close_aim_now'])

    def test_env_start_value(self):
        for v, on in (('0', False), ('1', True), ('', True), (' 0 ', False)):
            with patch.dict(os.environ, {'LWIR_CLOSE_AIM': v}):
                self.assertEqual(CA._start(), on)
        with patch.dict(os.environ):
            os.environ.pop('LWIR_CLOSE_AIM', None); self.assertTrue(CA._start())


class PublishTests(unittest.TestCase):
    """the moved row is what the bridge gets; the click rows (APP.GB.people) stay the drawn boxes. Same rig as test_live_tracks."""
    setUp_live, tearDown_live, publish = TL.LiveTracksTests.setUp, TL.LiveTracksTests.tearDown, TL.LiveTracksTests.publish

    def setUp(self):
        self.setUp_live(); self.saved_ca = dict(CA.STATE); CA.STATE['on'] = True; CA.reset()

    def tearDown(self):
        CA.STATE.clear(); CA.STATE.update(self.saved_ca); self.tearDown_live()

    def test_close_followed_row_is_sent_higher(self):
        from ..common.state import APP
        self.people = [(17, 100., 0., 220., 440., .9)]
        APP.GB.proc = NS(args=['bridge', '--aim-y', '0.5', '--aim-screen-y', '0.5'])
        self.publish()
        self.assertIn(b' 17 100.0 0.0 120.0 440.0 0.900', APP.GB.wframe)   # not followed yet: unchanged
        APP.GB.state.update(id=17, auto=1)
        self.publish(acknowledge=False)
        self.assertEqual(APP.GB.people, self.people)
        self.assertIn(b' 17 100.0 -110.0 120.0 440.0 0.900', APP.GB.wframe)   # (0.25 - 0.5) x 440 = -110: aim point at 110 px = top + 0.25 h
        self.assertAlmostEqual(CA.state()['close_aim_now'], .25)

    def test_switch_on_rules(self):
        from ..web import id_rule_switches
        saved = []
        app = NS(args=NS(det='x'), RULES={}, tracker=None, rules_request=lambda q: (True, 'old'),
                  rules_state=lambda: {}, sel_save=lambda **kw: saved.append(kw))
        id_rule_switches.install(app)
        self.assertTrue(app.rules_request({'close_aim': ['off']})[0])
        self.assertEqual((CA.STATE['on'], saved[-1]), (False, {'close_aim': '0'}))
        self.assertFalse(app.rules_request({'close_aim': ['maybe']})[0])
        self.assertTrue(app.rules_request({'close_aim': ['1']})[0])
        self.assertEqual((CA.STATE['on'], saved[-1], app.rules_state()['close_aim']), (True, {'close_aim': '1'}, True))

if __name__ == '__main__':
    unittest.main()
