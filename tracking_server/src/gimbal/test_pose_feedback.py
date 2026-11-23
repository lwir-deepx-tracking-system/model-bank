"""Actual device adapter on a pseudo terminal: no board or motors."""
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import tempfile
import time
import tty
import unittest


FIXTURE = r'''
#include "control/gimbal_device.hpp"
#include <iostream>
#include <iomanip>
int main(int argc, char** argv) {
    GimbalConfig config; config.device = argv[1]; config.speed = 10;
    auto device = GimbalDeviceFactory::create(config);
    device->stop(); device->enable_pose_feedback();
    std::cout << "READY" << std::endl;
    std::string command;
    while (std::cin >> command) {
        if (command == "move") { device->move(1, 0); std::cout << "OK" << std::endl; }
        else {
            const auto pose = device->read_pose();
            if (!pose) std::cout << "NONE" << std::endl;
            else std::cout << std::setprecision(12) << pose->pan << ' ' << pose->tilt << ' '
                << pose->received_at.time_since_epoch().count() << std::endl;
        }
    }
}
'''


class PoseFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = str(Path(cls.temporary.name) / 'pose')
        source = Path(cls.temporary.name) / 'pose.cpp'
        source.write_text(FIXTURE)
        folder = Path(__file__).resolve().parent
        subprocess.run(['g++', '-std=c++17', '-Iteam/include', str(source),
                        'team/src/control/gimbal_device.cpp', '-o', cls.binary], cwd=folder, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.master, self.slave = pty.openpty()
        tty.setraw(self.master)
        self.process = subprocess.Popen([self.binary, os.ttyname(self.slave)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.assertEqual(self.process.stdout.readline().strip(), 'READY')
        received = b''
        while received.count(b'\n') < 3:
            self.assertTrue(select.select([self.master], [], [], 2)[0])
            received += os.read(self.master, 4096)
        self.assertEqual([json.loads(line) for line in received.splitlines()],
                         [{'T': 135}, {'T': 142, 'cmd': 100}, {'T': 131, 'cmd': 1}])

    def tearDown(self):
        self.process.stdin.close()
        self.process.wait(timeout=2)
        self.process.stdout.close()
        os.close(self.master); os.close(self.slave)

    def poll(self):
        self.process.stdin.write('pose\n'); self.process.stdin.flush()
        self.assertTrue(select.select([self.process.stdout], [], [], 2)[0])
        return self.process.stdout.readline().strip()

    def send(self, data):
        os.write(self.master, data)

    def test_no_feedback_returns_immediately(self):
        start = time.monotonic()
        self.assertEqual(self.poll(), 'NONE')
        self.assertLess(time.monotonic() - start, .2)

    def test_actual_zero_angles_and_unchanged_sample_age(self):
        self.send(b'{"T":1001,"pan":0,"tilt":0,"v":12.4}\n')
        sample = self.poll().split()
        self.assertEqual(sample[:2], ['0', '0'])
        time.sleep(.02)
        self.assertEqual(self.poll().split(), sample) # no invented fresh timestamp

    def test_split_packets_and_latest_valid_sample(self):
        self.send(b'{"T":1001,"pan":-180,')
        self.assertEqual(self.poll(), 'NONE')
        self.send(b'"tilt":-90}\n{"T":1001,"pan":180,"tilt":90}\n')
        self.assertEqual(self.poll().split()[:2], ['180', '90'])

    def test_reject_commands_missing_fields_and_malformed_json(self):
        for packet in [b'{"T":133,"X":0,"Y":0}', b'{"T":1001,"pan":0}',
                       b'{"T":1001,"pan":null,"tilt":0}', b'{"T":1001,"pan":NaN,"tilt":0}',
                       b'{"T":1001,"pan":1e999,"tilt":0}', b'{"T":1001,"pan":361,"tilt":0}',
                       b'{"T":1001,"T":133,"pan":0,"tilt":0}',
                       b'{"T":1001,"pan":0,"pan":1,"tilt":0}',
                       b'{"T":1001,"pan":00,"tilt":0}',
                       b'{"T":1001,"nested":{"pan":0,"tilt":0}}',
                       b'{"T":1001,"pan":0,"tilt":0}junk']:
            self.send(packet + b'\n')
            self.assertEqual(self.poll(), 'NONE', packet)

    def test_oversized_line_resync_and_control_write(self):
        # Drain in bounded chunks, then recover on the next line boundary.
        self.send(b'x' * 5000 + b'\n{"T":1001,"pan":25,"tilt":-12.5}\n')
        self.poll()
        self.assertEqual(self.poll().split()[:2], ['25', '-12.5'])
        self.process.stdin.write('move\n'); self.process.stdin.flush()
        self.assertEqual(self.process.stdout.readline().strip(), 'OK')
        self.assertTrue(select.select([self.master], [], [], 2)[0])
        self.assertEqual(json.loads(os.read(self.master, 4096)),
                         {'T': 141, 'X': 1, 'Y': 0, 'SPD': 10})
