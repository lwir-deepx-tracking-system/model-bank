"""The ByteTrack menu row asks for `engine=auto`: C++ only when it can fully stand in for Python on the camera."""
import types
import unittest
from unittest.mock import patch

from ..camera import replay_link
from ..common.state import APP
from . import page_settings


class FakeCameraTracker:
    def __init__(self, available=True):
        self.available = available

    def state(self):
        return dict(available=self.available)


def gimbal(auto=0, control=True):
    return types.SimpleNamespace(state=dict(auto=auto), track_control_supported=control)


class EngineChoiceTests(unittest.TestCase):
    def choose(self, query, key='confirm3', tracker=None, gb=None, replay=False):
        try: before = (APP.GB,)
        except NameError: before = ()
        APP.GB = gb
        try:
            with patch.object(replay_link, 'blocking', return_value=replay):
                return page_settings.prefer_cpp_bytetrack(query, key, tracker or FakeCameraTracker())
        finally:
            if before: APP.GB = before[0]
            else: delattr(APP, 'GB')

    def test_auto_prefers_cpp_when_library_and_bridge_are_ready(self):
        self.assertTrue(self.choose({'engine': ['auto']}))
        self.assertTrue(self.choose({'engine': ['auto']}, gb=gimbal()))

    def test_explicit_python_and_other_trackers_stay_python(self):
        self.assertFalse(self.choose({}))
        self.assertFalse(self.choose({'engine': ['python']}))
        self.assertFalse(self.choose({'engine': ['auto']}, key='rebind'))
        self.assertFalse(self.choose({'engine': ['auto']}, key='ocsort'))

    def test_auto_keeps_python_when_cpp_cannot_take_over(self):
        self.assertFalse(self.choose({'engine': ['auto']}, tracker=FakeCameraTracker(False)))
        self.assertFalse(self.choose({'engine': ['auto']}, gb=gimbal(control=False)))   # clicks would be blocked
        self.assertFalse(self.choose({'engine': ['auto']}, gb=gimbal(auto=7)))            # someone is being followed
        self.assertFalse(self.choose({'engine': ['auto']}, replay=True))


if __name__ == '__main__':
    unittest.main()
