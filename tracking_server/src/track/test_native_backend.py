from contextlib import ExitStack
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

from . import native_backend as backend


class FakeTracker:
    def __init__(self):
        self.mode = "bytetrack"
        self.tracked = []
        self.lost = []
        self.frame_id = 0
        self.next_id = 0
        self.n_init = 2
        self.kf = object()
        self.new_this_frame = []
        self.lost_this_frame = []
        self.rebinds_motion = 0
        self.rebinds_app = 0
        self.rebinds_fallback = 0

    def update(self, arr):
        self.frame_id += 1


class NativeBackendTests(unittest.TestCase):
    def setUp(self):
        backend._WARNED.clear()
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.contexts.enter_context(patch.dict(backend._IMPORTS, clear=True))

    def extension_files(self, stem):
        source_dir = Path(self.contexts.enter_context(TemporaryDirectory()))
        native_dir = source_dir / "_native"
        native_dir.mkdir()
        source = source_dir / f"{stem}.cpp"
        source.write_text("// test source", encoding="utf-8")
        binary = native_dir / f"_native_{stem}.so"
        binary.write_bytes(b"old build")
        self.contexts.enter_context(patch.object(backend, "_SOURCE_DIR", source_dir))
        self.contexts.enter_context(patch.object(backend, "_NATIVE_DIR", native_dir))
        self.contexts.enter_context(patch.object(backend.importlib.machinery, "EXTENSION_SUFFIXES", [".so"]))
        self.contexts.enter_context(patch.dict(sys.modules))
        sys.modules.pop(f"_native_{stem}", None)
        return source, binary

    @staticmethod
    def install(stem, modes=None, factory=None, api_version=1):
        module = types.ModuleType("_native_" + stem)
        module.API_VERSION = api_version
        if modes is not None:
            module.SUPPORTED_MODES = modes
        module.create_tracker = factory or (lambda mode, kwargs: FakeTracker())
        return module

    def test_same_name_native_selected_for_confirm3_alias(self):
        seen = []
        module = self.install("bytetrack", {"bytetrack", "rebind"},
                              lambda mode, kwargs: seen.append((mode, kwargs)) or FakeTracker())
        with patch.object(backend, "_load_module", return_value=(module, None)):
            obj = backend.create("confirm3", "confirm3", {"n_init": 3}, FakeTracker)
            label = backend.backend_label("confirm3")
        self.assertEqual(seen, [("bytetrack", {"n_init": 3})])
        self.assertEqual(label, "C++")
        self.assertEqual(backend.backend_of(obj), "C++")
        obj.frame_id = 5
        self.assertEqual(obj.frame_id, 5)

    def test_rebind_uses_same_bytetrack_module_and_rebind_mode(self):
        seen = []
        module = self.install("bytetrack", {"bytetrack", "rebind"},
                              lambda mode, kwargs: seen.append(mode) or FakeTracker())
        with patch.object(backend, "_load_module", return_value=(module, None)):
            backend.create("rebind", "rebind", {}, FakeTracker)
        self.assertEqual(seen, ["rebind"])

    def test_unsupported_mode_falls_back(self):
        module = self.install("ocsort", {"ocsort"})
        with patch.object(backend, "_load_module", return_value=(module, None)):
            obj = backend.create("ocsort", "other", {}, FakeTracker)
            label = backend.backend_label("ocsort", "other")
        self.assertEqual(backend.backend_of(obj), "Python")
        # The extension exists, though create() correctly rejects this mode.
        self.assertEqual(label, "Python")

    def fake_load(self, module):
        spec = types.SimpleNamespace(loader=Mock())
        self.contexts.enter_context(patch.object(backend.importlib.util, "spec_from_file_location", return_value=spec))
        self.contexts.enter_context(patch.object(backend.importlib.util, "module_from_spec", return_value=module))

    def test_environment_switch_keeps_python(self):
        self.extension_files("ocsort")
        self.fake_load(self.install("ocsort"))
        with patch.dict(os.environ, {"LWIR_NATIVE_TRACKERS": "0"}):
            tracker = backend.create("ocsort", "ocsort", {}, FakeTracker)
            self.assertEqual(backend.backend_of(tracker), "Python")
            self.assertEqual(backend.backend_status("ocsort")["reason"], "disabled by LWIR_NATIVE_TRACKERS=0")
        with patch.dict(os.environ):
            os.environ.pop("LWIR_NATIVE_TRACKERS", None)
            self.assertEqual(backend.backend_label("ocsort"), "C++")

    def test_newer_shared_header_requires_rebuild(self):
        source, binary = self.extension_files("sort")
        self.fake_load(self.install("sort"))
        header = source.parent / "native_tracker_core.hpp"
        header.write_text("// newer core", encoding="utf-8")
        newer = binary.stat().st_mtime_ns + 5_000_000_000
        os.utime(header, ns=(newer, newer))
        status = backend.backend_status("sort")
        self.assertEqual(status["backend"], "Python")
        self.assertIn("rebuild needed", status["reason"])
        os.utime(binary, ns=(newer, newer))
        self.assertEqual(backend.backend_label("sort"), "C++")

    def test_missing_extension_uses_python(self):
        _, binary = self.extension_files("sort")
        binary.unlink()
        obj = backend.create("sort", "sort", {}, FakeTracker)
        self.assertEqual(backend.backend_label("sort"), "Python")
        self.assertEqual(backend.backend_of(obj), "Python")

    def test_invalid_native_api_falls_back(self):
        with patch.object(backend, "_load_module", return_value=(None, "API_VERSION is not 1")):
            obj = backend.create("botsort", "botsort", {}, FakeTracker)
        self.assertEqual(backend.backend_of(obj), "Python")

    def test_unwritable_native_is_reliably_labeled_with_proxy(self):
        class Slotted:
            __slots__ = (
                "tracked", "lost", "frame_id", "next_id", "n_init", "kf",
                "new_this_frame", "lost_this_frame", "rebinds_motion", "rebinds_app",
                "rebinds_fallback",
            )

            def __init__(self):
                self.tracked, self.lost = [], []
                self.frame_id = 1
                self.next_id, self.n_init, self.kf = 0, 2, object()
                self.new_this_frame, self.lost_this_frame = [], []
                self.rebinds_motion = self.rebinds_app = self.rebinds_fallback = 0

            def update(self, arr):
                self.frame_id += 1

        module = self.install("sort", factory=lambda mode, kwargs: Slotted())
        with patch.object(backend, "_load_module", return_value=(module, None)):
            obj = backend.create("sort", "sort", {}, FakeTracker)
        self.assertEqual(backend.backend_of(obj), "C++")
        obj.frame_id = 3
        self.assertEqual(obj.frame_id, 3)

    def test_native_missing_required_state_falls_back(self):
        module = self.install("sort", factory=lambda mode, kwargs: object())
        with patch.object(backend, "_load_module", return_value=(module, None)):
            obj = backend.create("sort", "sort", {}, FakeTracker)
        self.assertEqual(backend.backend_of(obj), "Python")

    def test_supported_kinds_select_matching_native_module(self):
        modes = {"bytetrack": "bytetrack", "confirm3": "bytetrack", "rebind": "rebind",
                 "botsort": "botsort", "ocsort": "ocsort", "sort": "sort"}
        for kind, native_mode in modes.items():
            with self.subTest(kind=kind):
                stem = "bytetrack" if kind in {"confirm3", "rebind"} else kind
                self.assertEqual(backend._module_stem(kind), stem)
                factory = Mock(return_value=FakeTracker())
                module = self.install(stem, {native_mode}, factory)
                with patch.object(backend, "_load_module", return_value=(module, None)):
                    tracker = backend.create(kind, kind, {"n_init": 3}, FakeTracker)
                    self.assertEqual(backend.backend_label(kind), "C++")
                factory.assert_called_once_with(native_mode, {"n_init": 3})
                self.assertEqual(backend.backend_of(tracker), "C++")

    def test_same_path_replacement_keeps_active_tracker_and_requires_restart(self):
        _, binary = self.extension_files("bytetrack")
        factory = Mock(side_effect=lambda mode, kwargs: FakeTracker())
        old_module = self.install("bytetrack", {"bytetrack", "rebind"}, factory)
        loader = Mock()
        spec = types.SimpleNamespace(loader=loader)
        with patch.object(backend.importlib.util, "spec_from_file_location", return_value=spec), \
                patch.object(backend.importlib.util, "module_from_spec", return_value=old_module) as load:
            active = backend.create("confirm3", "confirm3", {}, FakeTracker)
            self.assertEqual(backend.backend_label("rebind"), "C++")
            active.update(None)
            original = binary.stat()
            staged = binary.with_suffix(".tmp")
            staged.write_bytes(b"new build")
            os.utime(staged, ns=(original.st_atime_ns, original.st_mtime_ns))
            staged.replace(binary)

            status = backend.backend_status("rebind")
            self.assertEqual(status["backend"], "Python")
            self.assertTrue(status["restart_required"])
            next_tracker = backend.create("rebind", "rebind", {}, FakeTracker)

            self.assertEqual(backend.backend_of(next_tracker), "Python")
            self.assertEqual(backend.backend_of(active), "C++")
            self.assertEqual(active.frame_id, 1)
            self.assertIs(sys.modules["_native_bytetrack"], old_module)
            load.assert_called_once_with(spec)
            loader.exec_module.assert_called_once_with(old_module)
            factory.assert_called_once_with("bytetrack", {})

    def test_replacement_during_first_import_is_never_selected(self):
        _, binary = self.extension_files("sort")
        module = self.install("sort")
        loader = Mock()

        def replace_during_load(_):
            staged = binary.with_suffix(".tmp")
            staged.write_bytes(b"replacement build")
            staged.replace(binary)

        loader.exec_module.side_effect = replace_during_load
        spec = types.SimpleNamespace(loader=loader)
        with patch.object(backend.importlib.util, "spec_from_file_location", return_value=spec), \
                patch.object(backend.importlib.util, "module_from_spec", return_value=module) as load:
            self.assertTrue(backend.backend_status("sort")["restart_required"])
            self.assertEqual(backend.backend_of(backend.create("sort", "sort", {}, FakeTracker)), "Python")
            load.assert_called_once_with(spec)

    def test_new_binary_is_available_without_restart_before_first_import(self):
        _, binary = self.extension_files("botsort")
        binary.unlink()
        self.assertFalse(backend.backend_status("botsort")["restart_required"])
        self.assertEqual(backend.backend_label("botsort"), "Python")

        binary.write_bytes(b"first build")
        module = self.install("botsort")
        spec = types.SimpleNamespace(loader=Mock())
        with patch.object(backend.importlib.util, "spec_from_file_location", return_value=spec), \
                patch.object(backend.importlib.util, "module_from_spec", return_value=module):
            tracker = backend.create("botsort", "botsort", {}, FakeTracker)
            self.assertEqual(backend.backend_of(tracker), "C++")
            self.assertFalse(backend.backend_status("botsort")["restart_required"])


if __name__ == "__main__":
    unittest.main()
