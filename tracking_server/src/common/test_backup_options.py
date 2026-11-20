"""New diagnostic modes remain explicit opt-ins."""
import os
import unittest
from unittest.mock import patch
from . import stage_timing
from .identity_trace import IdentityTrace, diagnostic_payload


class BackupOptionsTests(unittest.TestCase):
    def test_stage_auto_is_opt_in_and_explicit_off_wins(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(stage_timing.wanted())
            os.environ['LWIR_REAL_GUI_LAUNCH'] = '1'
            self.assertFalse(stage_timing.wanted())
            os.environ['LWIR_STAGE_TIMING_AUTO'] = '1'
            self.assertTrue(stage_timing.wanted())
            os.environ['LWIR_STAGE_TIMING'] = '0'
            self.assertFalse(stage_timing.wanted())

    def test_identity_diagnostics_require_opt_in_and_loopback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(diagnostic_payload(IdentityTrace.from_env(), '127.0.0.1', 'localhost'))
        trace = IdentityTrace(enabled=True)
        self.assertIsNotNone(diagnostic_payload(trace, '127.0.0.1', 'localhost'))
        self.assertIsNone(diagnostic_payload(trace, '192.0.2.1', 'localhost'))
        self.assertIsNone(diagnostic_payload(trace, '127.0.0.1', 'example.test'))
