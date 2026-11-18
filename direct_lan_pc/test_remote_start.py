"""PC-only checks for starting the Orange Pi from the desktop launcher."""
import sys
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import remote_start


class RemoteStartTests(unittest.TestCase):
    def test_ready_requires_our_direct_protocol(self):
        with patch.object(remote_start, 'Link') as link_type:
            link_type.return_value.request.return_value = ({'status': 200}, b'{"t":123}')
            self.assertTrue(remote_start.ready('192.0.2.2', 8765))
            link_type.return_value.request.assert_called_with(
                {'op': 'route', 'method': 'GET', 'path': '/time'}, timeout=0.8)
            link_type.return_value.request.return_value = ({'status': 200}, b'<html>')
            self.assertFalse(remote_start.ready('192.0.2.2', 8765))

    def test_running_mode_does_not_ssh_or_restart(self):
        with patch.object(remote_start, 'ready', return_value=True), patch.object(remote_start, 'connect') as connect:
            self.assertFalse(remote_start.start('192.0.2.2', 8765, 'user'))
            connect.assert_not_called()

    def test_ambiguous_board_copy_is_refused(self):
        paths = ('/home/user/a/tracking_server/scripts/run_direct_lan.sh\n'
                 '/home/user/b/tracking_server/scripts/run_direct_lan.sh')
        with patch.object(remote_start, 'remote_command', return_value=(0, paths, '')):
            with self.assertRaisesRegex(remote_start.StartupError, '여러 개'):
                remote_start.find_root(Mock())

    def test_explicit_root_is_quoted_and_checked(self):
        with patch.object(remote_start, 'remote_command', return_value=(0, '', '')) as command:
            self.assertEqual(remote_start.find_root(Mock(), '/home/user/model bank'), '/home/user/model bank')
            self.assertIn("'/home/user/model bank/tracking_server/scripts/run_direct_lan.sh'",
                          command.call_args.args[1])

    def test_board_launcher_detaches_without_ssh_file_descriptors(self):
        with patch.object(remote_start, 'remote_command', side_effect=[
                (0, 'created', ''), (0, '1234', '')]) as command:
            self.assertEqual(remote_start.launch(Mock(), '/home/user/model bank'), (1234, True))
            launch_command = command.call_args_list[1].args[1]
            self.assertIn('start_new_session=True', launch_command)
            self.assertIn('close_fds=True', launch_command)
            self.assertIn('run_direct_lan.sh', launch_command)

    def test_detached_command_starts_a_local_standin(self):
        with tempfile.TemporaryDirectory(prefix='modelbank start ') as directory:
            root = Path(directory)
            script = root / 'tracking_server' / 'scripts' / 'run_direct_lan.sh'
            script.parent.mkdir(parents=True)
            marker = root / 'started'
            script.write_text('#!/bin/bash\nprintf started > "' + str(marker) + '"\nsleep 0.2\n')
            def execute(_client, command, timeout=12):
                if command.startswith('if test -e'):
                    return 0, 'existing', ''
                result = subprocess.run(command, shell=True, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        timeout=timeout)
                return result.returncode, result.stdout.strip(), result.stderr.strip()
            with patch.object(remote_start, 'remote_command', side_effect=execute):
                pid, created = remote_start.launch(Mock(), str(root))
            self.assertGreater(pid, 0)
            self.assertFalse(created)
            for _ in range(20):
                if marker.exists():
                    break
                time.sleep(0.02)
            self.assertEqual(marker.read_text(), 'started')

    def test_failed_start_restores_only_marker_it_created(self):
        client = Mock()
        with patch.object(remote_start, 'ready', return_value=False), \
             patch.object(remote_start, 'connect', return_value=client), \
             patch.object(remote_start, 'find_root', return_value='/home/user/model-bank'), \
             patch.object(remote_start, 'launch', return_value=(1234, True)), \
             patch.object(remote_start, 'launch_failure', return_value='NPU busy'), \
             patch.object(remote_start, 'restore_marker') as restore, \
             patch.object(remote_start.time, 'sleep'):
            with self.assertRaisesRegex(remote_start.StartupError, 'NPU busy'):
                remote_start.start('192.0.2.2', 8765, 'user')
            restore.assert_called_once_with(client)
            client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
