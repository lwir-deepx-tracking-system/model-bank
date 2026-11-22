"""11-22: follow-loss recorder (follow_log.py). Fake gimbal state, no server."""
import os, tempfile, unittest
from types import SimpleNamespace as NS
from . import follow_log as fl


def gb_(people=None):
    return NS(state={'auto': 1, 'lost': 0, 'error': 0, 'id': 7, 'guard': [1, 0]}, people=people or [])


class Rec(unittest.TestCase):
    def setUp(self):
        fl._ring.clear(); fl._events.clear(); fl._cause[0] = None
        self.d = tempfile.mkdtemp(); fl._path[0] = os.path.join(self.d, 'follows.log')
        self.gb = gb_()

    def feed(self, n, present=True, box=(300, 100, 340, 220), others=(), t0=0.0):
        for i in range(n):
            ppl = ([[7, *box, .9]] if present else []) + [list(o) for o in others]
            self.gb.people = ppl
            fl.frame(self.gb, t0 + i / 30, 7, ppl)

    def end(self, new=(0, 0, 0, 0)):
        fl.state(self.gb, (1, 0, 0, 7), new)
        return fl.events()[-1]

    def test_lost_age(self):
        self.feed(10); self.feed(20, present=False, t0=1)
        e = self.end((1, 1, 0, 7)); self.assertEqual((e['kind'], e['reason']), ('lost', 'track_ended_lost_age'))
        self.assertEqual(e['n_missing'], 20); self.assertTrue(e['last3s']); self.assertTrue(os.path.exists(fl._path[0]))

    def test_short_gap_edge_cover_switch_visible(self):
        self.feed(10); self.feed(5, present=False); self.assertTrue(self.end()['reason'].startswith('detection_missing_5'))
        fl._ring.clear(); self.feed(10, box=(2, 100, 40, 220)); self.feed(3, present=False); self.assertEqual(self.end()['reason'], 'target_left_picture')
        fl._ring.clear(); self.feed(10, others=[[9, 310, 110, 350, 230, .8]]); self.feed(2, present=False, others=[[9, 310, 110, 350, 230, .8]])
        e = self.end(); self.assertEqual(e['reason'], 'covered_by_other_person'); self.assertEqual(e['near'][0][0], 9)
        fl._ring.clear(); self.feed(10); self.gb.people = [[11, 301, 101, 341, 221, .9]]
        self.assertEqual(self.end((1, 0, 0, 11))['reason'], 'id_switched_relink')
        fl._ring.clear(); self.feed(10); self.assertEqual(self.end()['reason'], 'bridge_dropped_target_visible')

    def test_causes_and_off(self):
        self.feed(5); fl.cause('stop'); self.assertEqual(self.end()['reason'], 'user_released')
        self.feed(5); fl.cause('camera_gap'); self.assertEqual(self.end()['reason'], 'camera_stall')
        self.feed(5); fl.cause('boundary'); self.assertEqual(self.end()['reason'], 'tracker_or_model_switch')
        n = len(fl.events()); fl.state(self.gb, (0, 0, 0, 0), (0, 0, 0, 0)); fl.state(self.gb, (1, 0, 0, 7), (1, 0, 0, 7)); self.assertEqual(len(fl.events()), n)
        os.environ['LWIR_FOLLOW_LOG'] = '0'
        try: fl.state(self.gb, (1, 0, 0, 7), (0, 0, 0, 0)); self.assertEqual(len(fl.events()), n)
        finally: del os.environ['LWIR_FOLLOW_LOG']
        self.assertEqual(len(fl.health()), 3); fl.start(); self.assertEqual(fl.events()[-1]['kind'], 'server_start')


if __name__ == '__main__':
    unittest.main()
