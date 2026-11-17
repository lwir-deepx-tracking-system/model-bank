"""Native library installation tests; no compiled tracker or device required."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import bytetrack


class LibraryReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / 'bytetrack.so'
        self.path.write_bytes(b'first build')
        self.signatures = patch.dict(bytetrack._LOADED_LIBRARY_SIGNATURES, {}, clear=True)
        self.signatures.start()

    def tearDown(self):
        self.signatures.stop()
        self.temp.cleanup()

    def replace(self):
        next_path = self.path.with_suffix('.next')
        next_path.write_bytes(b'second build')
        next_path.replace(self.path)

    def test_atomic_replacement_requires_restart(self):
        lib = object()
        with patch.object(bytetrack, '_bind_bytetrack_library', return_value=lib) as bind:
            self.assertIs(bytetrack.load_bytetrack_library(self.path), lib)
            self.replace()
            with self.assertRaisesRegex(RuntimeError, 'restart required'):
                bytetrack.load_bytetrack_library(self.path)
            self.assertEqual(bind.call_count, 1)

    def test_unchanged_file_stays_available(self):
        lib = object()
        with patch.object(bytetrack, '_bind_bytetrack_library', return_value=lib):
            self.assertIs(bytetrack.load_bytetrack_library(self.path), lib)
            self.assertIs(bytetrack.load_bytetrack_library(self.path), lib)

    def test_replacement_during_binding_is_not_reported_as_current(self):
        def bind(*_):
            self.replace()
            return object()
        with patch.object(bytetrack, '_bind_bytetrack_library', side_effect=bind):
            with self.assertRaisesRegex(RuntimeError, 'changed while loading'):
                bytetrack.load_bytetrack_library(self.path)
