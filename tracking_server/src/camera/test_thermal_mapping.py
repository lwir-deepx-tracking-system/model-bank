"""Detector mapping contract: original RAW16 stays untouched, switches are isolated."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from . import replay_cam, thermal_mapping
from ..common.state import APP
from ..web import page_settings


RAW = np.arange(480 * 640, dtype=np.uint32).reshape(480, 640).astype(np.uint16)


def baseline(raw, mode):
    if mode == 'fixed':
        return cv2.convertScaleAbs(raw, alpha=255.0 / 65535.0)
    return (raw >> 8).astype(np.uint8)


class FakeField:
    def __init__(self):
        self.resets = 0
        self.frames = 0

    def reset_temporal_state(self):
        self.resets += 1

    def process(self, raw):
        self.frames += 1
        return np.full(raw.shape, self.frames, np.uint8)


class ThermalMappingTest(unittest.TestCase):
    def test_fieldscale_replay_accepts_only_raw16_png(self):
        with tempfile.TemporaryDirectory() as folder:
            for bits in (8, 16):
                header = (b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR'
                          + (640).to_bytes(4, 'big') + (480).to_bytes(4, 'big')
                          + bytes((bits, 0, 0, 0, 0)) + b'\x00\x00\x00\x00')
                (Path(folder) / f'{bits}.png').write_bytes(header)
            self.assertEqual(replay_cam.PngSource(folder).total, 2)
            source = replay_cam.PngSource(folder, raw16_only=True)
            self.assertEqual(source.total, 1)
            self.assertEqual(source.bad, 1)
            self.assertEqual(source.kind, '16-bit')

    def test_baseline_and_clahe_do_not_modify_raw(self):
        original = RAW.copy()
        mapper = thermal_mapping.ThermalMapping()
        source = (object(), None, None, False, 1.0)
        self.assertTrue(np.array_equal(mapper.process(RAW, baseline, source), baseline(RAW, 'pct')))
        mapper.set_mode('fixed')
        self.assertTrue(np.array_equal(mapper.process(RAW, baseline, source), baseline(RAW, 'fixed')))
        mapper.set_mode('clahe')
        expected = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(baseline(RAW, 'pct'))
        self.assertTrue(np.array_equal(mapper.process(RAW, baseline, source), expected))
        self.assertTrue(np.array_equal(RAW, original))

    def test_fieldscale_source_seek_gap_and_repeat(self):
        fake = FakeField()
        mapper = thermal_mapping.ThermalMapping()
        with patch.object(thermal_mapping, '_field_processor', return_value=fake):
            mapper.set_mode(thermal_mapping.FIELD)
        producer = object()
        def process(pass_no, position, t, repeat=False):
            return mapper.process(RAW, baseline, (producer, pass_no, position, repeat, t))
        a = process(0, 0, 1.0)
        self.assertTrue(np.array_equal(a, process(0, 0, 1.1, True)))
        self.assertEqual(fake.frames, 1)
        process(0, 1, 1.2)
        process(0, 0, 1.3)
        process(1, 0, 1.4)
        process(None, None, 2.0)
        process(None, None, 3.2)
        self.assertEqual(fake.resets, 6)

    def test_failed_selection_keeps_previous_mode(self):
        mapper = thermal_mapping.ThermalMapping('clahe')
        with patch.object(thermal_mapping, '_field_processor', side_effect=RuntimeError('missing')):
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                mapper.set_mode(thermal_mapping.FIELD)
        self.assertEqual(mapper.status()['mode'], 'clahe')

    def test_queue_metadata_only_for_fieldscale(self):
        mapper = thermal_mapping.ThermalMapping()
        producer = object()
        self.assertEqual(len(thermal_mapping.queued_frame(mapper, 1.0, RAW, producer)), 2)
        with patch.object(thermal_mapping, '_field_processor', return_value=FakeField()):
            mapper.set_mode(thermal_mapping.FIELD)
        queued = thermal_mapping.queued_frame(mapper, 2.0, RAW, producer, replay_pass=2, position=7)
        self.assertEqual(queued[2], (producer, 2, 7, False, 2.0))
        self.assertEqual(thermal_mapping.detector_gray(mapper, baseline, queued, producer).shape, RAW.shape)

    def test_gui_rule_switch_and_invalid_mode(self):
        mapper = thermal_mapping.ThermalMapping()
        saved = []
        values = dict(THERMAL=mapper, args=SimpleNamespace(live_port=8091), LIVE=None,
                      sel_save=lambda **kwargs: saved.append(kwargs))
        with patch.dict(vars(APP), values):
            ok, _ = page_settings.rules_request_2({'thermal_map': ['clahe']})
            self.assertTrue(ok)
            self.assertEqual(mapper.status()['mode'], 'clahe')
            self.assertEqual(saved[-1], {'thermal_map': 'clahe'})
            ok, _ = page_settings.rules_request_2({'thermal_map': ['unknown']})
            self.assertFalse(ok)
            self.assertEqual(mapper.status()['mode'], 'clahe')


if __name__ == '__main__':
    unittest.main()
