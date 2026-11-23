"""Virtual clicks are bound to the displayed replay, never to a physical gimbal."""
import unittest
from .replay_virtual_follow import ReplayFollow


class ReplayFollowTests(unittest.TestCase):
    def setUp(self):
        self.follow = ReplayFollow()
        self.boxes = (((100, 100), (180, 300), (0, 200, 0), 7),)

    def frame(self, stamp=1., run=4, pos=0):
        return self.follow.update(stamp, (run, pos, 0, 0), 30, self.boxes, 640, 480)

    def test_click_selects_person_and_empty_click_restores_full_view(self):
        self.frame()
        self.assertTrue(self.follow.click(1., 4, 140/640, 200/480)[0])
        crop, selected, *_ = self.frame(2., pos=1)
        self.assertEqual(selected, 7)
        self.assertEqual(crop[2:], (320, 240))
        self.assertTrue(self.follow.click(2., 4, 1., 1.)[0])
        self.assertIsNone(self.follow.selected)

    def test_old_run_and_overlapping_people_cannot_select(self):
        self.frame()
        self.assertFalse(self.follow.click(1., 3, .2, .4)[0])
        self.boxes += (((100, 100), (180, 300), (0, 200, 0), 8),)
        self.frame(2., pos=1)
        self.assertFalse(self.follow.click(2., 4, 140/640, 200/480)[0])

    def test_seek_and_new_run_clear_previous_selection(self):
        self.frame(pos=10)
        self.follow.click(1., 4, 140/640, 200/480)
        self.frame(2., pos=0)
        self.assertIsNone(self.follow.selected)
        self.follow.click(2., 4, 140/640, 200/480)
        self.frame(3., run=5, pos=1)
        self.assertIsNone(self.follow.selected)

    def test_zoom_rejects_nonfinite_and_out_of_range_values(self):
        for v in [float('nan'), float('inf'), .9, 4.1]:
            self.assertFalse(self.follow.set_zoom(v)[0])
        self.assertTrue(self.follow.set_zoom(3.)[0])
