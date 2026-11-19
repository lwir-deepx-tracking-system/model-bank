"""Exercise the board session lifecycle with an isolated fake server tree."""
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracking_server.scripts import direct_lan_session as session


class SessionTests(unittest.TestCase):
    def test_pc_window_close_calls_session_restore_after_gimbal_stop(self):
        try:
            from direct_lan_pc.app import App
        except ImportError as exc:
            self.skipTest('PC Tk/Pillow image support is unavailable here: ' + str(exc))
        app = object.__new__(App)
        app.stop = threading.Event()
        app.controls = Mock()
        app.workers = Mock()
        app.link = Mock()
        app.root = Mock()
        app.on_close = Mock()
        app.close()
        app.link.post.assert_called_once_with('/gimbal/stop')
        app.on_close.assert_called_once_with()
        app.root.destroy.assert_not_called()

    def test_restore_uses_original_folder_and_preserves_prior_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            scripts = base / 'tracking_server' / 'scripts'
            scripts.mkdir(parents=True)
            (scripts / 'run_real_gui.sh').write_text('#!/bin/bash\n')
            pause = base / 'existing.pause'
            pause.touch()
            with patch.object(session, 'PAUSE', pause), \
                 patch.object(session, 'direct_running', return_value=False), \
                 patch.object(session, 'web_running', return_value=False), \
                 patch.object(session, 'web_healthy', return_value=True), \
                 patch.object(session, 'wait_npu_idle') as wait_npu, \
                 patch.object(session.subprocess, 'Popen') as launch:
                session.restore_web(str(base), marker_created=False)
                wait_npu.assert_called_once_with()
                self.assertEqual(Path(launch.call_args.kwargs['cwd']), base / 'tracking_server')
                self.assertTrue(pause.exists())

    def test_direct_session_stops_owned_child_and_restores_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'modelbank'
            server = root / 'tracking_server'
            scripts = server / 'scripts'
            scripts.mkdir(parents=True)
            (server / 'start_tracking_server.py').write_text('import time\ntime.sleep(30)\n')
            launcher = scripts / 'run_real_gui.sh'
            launcher.write_text('''#!/bin/bash
python3 start_tracking_server.py --direct-lan-bind 127.0.0.1 &
child=$!
trap 'kill "$child" 2>/dev/null; wait "$child" 2>/dev/null' TERM
wait "$child"
''')
            pause = base / 'keepalive.pause'
            pause.touch()
            proc = subprocess.Popen(['bash', str(launcher)], cwd=server,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                with patch.object(session, 'STATE', base / 'session.json'), \
                     patch.object(session, 'PAUSE', pause):
                    for _ in range(100):
                        if session.launcher(proc.pid, str(root)) and any(
                                session.child(int(item.name), 'direct') and
                                session.process(int(item.name))['parent'] == proc.pid
                                for item in Path('/proc').iterdir() if item.name.isdigit()):
                            break
                        time.sleep(.02)
                    else:
                        self.fail('fake direct server did not start')
                    session.record(str(root), proc.pid, '', True)
                    self.assertTrue(session.STATE.exists())
                    session.finish()
                    self.assertFalse(session.STATE.exists())
                    self.assertFalse(pause.exists())
                    proc.wait(timeout=5)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
