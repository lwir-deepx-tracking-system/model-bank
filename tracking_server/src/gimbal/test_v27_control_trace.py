"""Build a controller-only fixture and validate the opt-in JSONL trace."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


GIMBAL = Path(__file__).resolve().parent
FIXTURE = r'''
#include <iostream>
#include "control/auto_tracking_controller.hpp"

int main()
{
    AutoControlConfig config;
    config.deadband_x = config.deadband_y = 0.06;
    AutoTrackingController controller(config);
    Track target;
    target.track_id = 7;
    target.x = 380;
    target.y = 210;
    target.width = 40;
    target.height = 80;
    target.confidence = 0.9f;
    const auto step = [&](std::uint64_t frame, int ms, const TrackLead& lead) {
        FrameMetadata meta;
        meta.frame_id = frame;
        meta.captured_at = std::chrono::steady_clock::time_point(std::chrono::milliseconds(ms));
        const auto demand = controller.calculate(target, 640, 480, meta, 0, lead);
        std::cout << demand.x << ',' << demand.y << ',' << demand.speed << '\n';
    };
    step(1, 100, {7, 32, 20, 42});
    step(1, 100, {7, -32, 20, 42});  // cached frame
    step(2, 90, {7, 32, 20, 42});    // out-of-order time
    step(3, 200, {8, 32, 20, 42});   // wrong track
    target.x = 332;
    step(4, 250, {7, -60, 0, 42});   // direction guard declines the forecast
}
'''


class ControlTraceTest(unittest.TestCase):
    def test_trace_matches_controller_and_skips_rejected_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / 'fixture.cpp'
            executable = fixture.with_suffix('.exe' if os.name == 'nt' else '')
            fixture.write_text(FIXTURE, encoding='utf-8')
            subprocess.run([
                os.environ.get('CXX', 'g++'), '-std=c++17',
                '-Iv27/include', '-Iteam/include', '-Isim/stub',
                str(fixture), 'v27/src/control/auto_tracking_controller.cpp',
                'team/src/control/pid_controller.cpp', '-o', str(executable),
            ], cwd=GIMBAL, check=True)
            env = os.environ.copy()
            env.pop('LWIR_CONTROL_TRACE', None)
            disabled = subprocess.run([executable], env=env, check=True, capture_output=True, text=True)
            env['LWIR_CONTROL_TRACE'] = '1'
            enabled = subprocess.run([executable], env=env, check=True, capture_output=True, text=True)
            env['LWIR_CONTROL_TRACE'] = '10'
            nonmatching = subprocess.run([executable], env=env, check=True, capture_output=True, text=True)

        self.assertEqual(disabled.stderr, '')
        self.assertEqual(nonmatching.stderr, '')
        self.assertEqual(enabled.stdout, disabled.stdout)
        trace = [json.loads(line) for line in enabled.stderr.splitlines()]
        self.assertEqual([row['frame_id'] for row in trace], [1, 3, 4])
        self.assertEqual([row['captured_ns'] for row in trace], [100000000, 200000000, 250000000])
        self.assertTrue(all(row['event'] == 'control_trace' and row['track_id'] == 7 for row in trace))

        first, wrong_id, wrong_direction = trace
        self.assertEqual(first['error_unit'], 'normalized_half_frame')
        self.assertEqual(first['lead_control_epoch'], 42)
        self.assertTrue(first['lead_valid'])
        self.assertAlmostEqual(first['measured_error']['x'], .25)
        self.assertAlmostEqual(first['predicted_error']['x'], .35)
        self.assertAlmostEqual(first['measured_error']['y'], 10 / 240)
        self.assertAlmostEqual(first['predicted_error']['y'], 30 / 240)
        self.assertEqual(first['deadband_hold'], {'x': False, 'y': True})
        self.assertEqual(first['forecast_used'], {'x': True, 'y': False})
        self.assertEqual(first['pid_demand']['y'], 0)
        self.assertEqual(first['demand']['y'], 0)

        self.assertFalse(wrong_id['lead_valid'])
        self.assertEqual(wrong_id['predicted_error'], wrong_id['measured_error'])
        self.assertTrue(wrong_direction['lead_valid'])
        self.assertLess(wrong_direction['predicted_error']['x'], 0)
        self.assertFalse(wrong_direction['forecast_used']['x'])
        self.assertGreater(wrong_direction['pid_demand']['x'], 0)
        self.assertEqual(wrong_direction['demand']['x'], 1)


if __name__ == '__main__':
    unittest.main()
