import types
import unittest
from unittest.mock import patch

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

    def test_missing_extension_uses_python(self):
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


if __name__ == "__main__":
    unittest.main()
