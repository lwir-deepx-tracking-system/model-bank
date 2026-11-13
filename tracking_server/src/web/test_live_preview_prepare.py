"""Check annotation preparation at the live frame publication boundary."""
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from ..common.state import APP
from . import page_handler as page
from .test_preview_pipeline import PublishFixture


def tracks(*rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 7)


class LivePreviewPrepareTests(PublishFixture):
    def test_recent_preview_without_viewer_or_snapshot_skips_formatting(self):
        rows = tracks((100, 116, 140, 276, 7, .9, 0))
        self.live.viewers = 0
        self.live.pic_t = time.perf_counter()

        with patch.object(page, 'track_color') as color, \
             patch.object(page, '_preview_snapshot') as snapshot:
            self.publish(rows)
            APP.GB = None
            self.publish(rows)

        color.assert_not_called()
        snapshot.assert_not_called()
        self.assertEqual(self.live.seq, 0)
        self.assertEqual(self.controls, [((7, 100., 100., 140., 260., float(rows[0, 5])),)])
        self.assertEqual(len(self.evaluations), 2)
        for evaluated in self.evaluations:
            np.testing.assert_array_equal(evaluated, rows)

    def test_selected_id_uses_control_result_even_past_control_cap(self):
        rows = tracks((0, 0, 10, 10, 99, .9, 2),
                      *((100, 116, 140, 276, tid, .9, 0) for tid in range(1, 62)))
        APP.GB.state['id'] = 1
        stages = []

        def publish_control(people, correction, stamp, raw, source, blocked, *, healthy_live):
            stages.append('control')
            self.assertTrue(healthy_live)
            self.assertFalse(blocked)
            self.assertEqual(source, ('fixture',))
            self.assertEqual([p[0] for p in people], list(range(1, 61)))
            self.assertEqual(people[0], (1, 100., 100., 140., 260., float(rows[1, 5])))
            APP.GB.state['id'] = 61
            return 61

        def evaluate(stamp, evaluated, *args):
            stages.append('eval')
            self.assertIs(evaluated, rows)
            return None, None

        def click_people(stamp, ids):
            self.assertEqual(ids, {61})
            return (), 12

        def submit(token, frame, **kwargs):
            stages.append('preview')
            self.assertEqual(frame.boxes, (((100, 100), (140, 260), (0, 220, 255), 61),))
            self.assertIn('persons 61', frame.top_text)
            self.assertEqual((frame.click_people, frame.click_ticket), ((), 12))

        worker = SimpleNamespace(async_live=True, token=Mock(return_value=object()), submit=Mock(side_effect=submit))
        self.live.preview_worker = worker
        with patch.object(page.live_tracks, 'publish', side_effect=publish_control), \
             patch.object(page.replay_eval, 'on_frame', side_effect=evaluate), \
             patch.object(page.live_tracks, 'preview_people', side_effect=click_people), \
             patch.object(page, 'track_color') as color:
            self.publish(rows)

        self.assertEqual(stages, ['control', 'eval', 'preview'])
        color.assert_not_called()

    def test_replay_overlay_replaces_live_display_in_its_own_order(self):
        self.install_replay()
        APP.GB.state['id'] = 7
        rows = tracks((100, 116, 140, 276, 7, .9, 0))
        overlay = tracks((44, 56, 66, 76, 9, .8, 0), (1, 2, 3, 4, 8, .9, 1),
                         (12, 25, 32, 49, 7, .7, 0))
        colors = {9: (1, 2, 3), 7: (4, 5, 6)}
        for overlay_rows, expected_draw in (
            (overlay, (((44, 40), (66, 60), colors[9], 9), ((12, 9), (32, 33), colors[7], 7))),
            (overlay[:0], ()),
        ):
            with self.subTest(overlay_rows=len(overlay_rows)), \
                 patch.object(page.replay_link, 'top_text', return_value='REPLAY fixture'), \
                 patch.object(page.replay_eval, 'on_frame', return_value=(None, overlay_rows)) as evaluate, \
                 patch.object(page, 'track_color', side_effect=colors.__getitem__) as color, \
                 patch.object(page.preview_worker, 'render_preview', return_value=b'jpeg') as render:
                self.publish(rows)

            self.assertIs(evaluate.call_args.args[1], rows)
            self.assertEqual(self.controls[-1][0][0], 7)
            frame = render.call_args.args[1]
            self.assertEqual(frame.boxes, expected_draw)
            self.assertEqual([c.args[0] for c in color.call_args_list], [box[3] for box in expected_draw])
            self.assertIn(f'persons {len(expected_draw)}', frame.top_text)

    def test_snapshot_request_with_no_gimbal_preserves_coordinate_rounding(self):
        APP.GB = None
        APP.args.live_scale = 10.
        self.live.viewers = 0
        self.live.pic_t = self.live.snap_t = time.perf_counter()
        rows = tracks((.7, 16.75, 1.7, 17.75, 7, .9, 0))
        with patch.object(page.preview_worker, 'render_preview', return_value=b'jpeg') as render:
            self.publish(rows)

        frame = render.call_args.args[1]
        self.assertEqual(frame.boxes, (((6, 7), (17, 17), page.track_color(7), 7),))
        self.assertIn('persons 1', frame.top_text)
