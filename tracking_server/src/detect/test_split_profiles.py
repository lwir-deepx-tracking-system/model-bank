"""Split-model eligibility and rebuilding across model changes, without DX-RT."""
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import Mock, patch

from ..common.state import APP
from . import dual_tile, frame_feed


class SplitProfileTests(unittest.TestCase):
    def test_each_registered_model_resolves_its_own_weights(self):
        profiles = [(dual_tile.KEY, dual_tile.SOURCE_SHA, dual_tile.MODEL),
                    (dual_tile.NEW_KEY, dual_tile.NEW_SOURCE_SHA, dual_tile.NEW_MODEL),
                    (dual_tile.NEW_650_KEY, dual_tile.NEW_650_SOURCE_SHA, dual_tile.NEW_650_MODEL),
                    (dual_tile.TEMPORAL_KEY, dual_tile.TEMPORAL_SOURCE_SHA, dual_tile.TEMPORAL_MODEL)]
        for key, sha, model in profiles:
            with self.subTest(key=key), patch.dict(vars(APP), args=NS(det='bank:'+key), man={key: sha}):
                self.assertEqual(dual_tile.selected_profile()[:2], (key, model))
                self.assertTrue(dual_tile.selected())
                APP.man[key] = 'changed'
                self.assertFalse(dual_tile.selected())
                self.assertIn('지원하지', dual_tile.assets_reason())

    def test_reprepare_replaces_the_old_model_and_rejects_unsupported_model(self):
        feed = frame_feed.Feed.__new__(frame_feed.Feed)
        feed.regular_eng = [object(), object()]
        feed.tile_engine = feed.tile_tone = None
        first, second = NS(b=Mock()), NS(b=Mock())
        with patch.dict(vars(APP), args=NS(mock=False)), \
             patch.object(dual_tile, 'assets_reason', return_value=None), \
             patch.object(dual_tile, 'Tone'), patch.object(dual_tile, 'Engine', side_effect=[first, second]):
            feed._prepare_tile()
            self.assertIs(feed.tile_engine, first)
            feed._dispose_tile()
            first.b.dispose.assert_called_once()
            feed._prepare_tile()
            self.assertIs(feed.tile_engine, second)
            feed._dispose_tile()
        with patch.dict(vars(APP), args=NS(mock=False)), patch.object(dual_tile, 'assets_reason', return_value='unsupported'):
            feed._prepare_tile()
            self.assertIsNone(feed.tile_engine)
            self.assertEqual(feed.tile_error, 'unsupported')

    def test_thread_switch_interval_is_restored_when_leaving_split_mode(self):
        feed = frame_feed.Feed.__new__(frame_feed.Feed)
        feed.mode = 'normal'
        previous = sys.getswitchinterval()
        try:
            feed._set_mode('tile')
            self.assertAlmostEqual(sys.getswitchinterval(), .001)
            feed._set_mode('arrival')
            self.assertEqual(sys.getswitchinterval(), previous)
        finally:
            sys.setswitchinterval(previous)
