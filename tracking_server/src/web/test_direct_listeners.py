"""The page handler's /netinfo and /pc/ routes (the real do_GET, no sockets). Mock HTTP lifecycle only."""
import http.server
import io
import json
import unittest
import zipfile
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from ..common.state import APP
from . import direct_listeners
from . import page_handler as page


class DirectRoutes(unittest.TestCase):
    def setUp(self):
        previous = vars(APP).copy()
        self.addCleanup(lambda: (vars(APP).clear(), vars(APP).update(previous)))
        vars(APP).clear()
        APP.args = NS(live_bind='127.0.0.1', live_lan_bind=None, live_port=8090)
        APP.LIVE = NS(stopped=False)
        APP.lv_http = NS(ThreadingHTTPServer=Mock(), BaseHTTPRequestHandler=http.server.BaseHTTPRequestHandler)

    def get(self, path, client):
        handler_class = page.make_lv_Handler()
        handler = handler_class.__new__(handler_class)
        handler.path = path
        handler.client_address = (client, 40000)
        handler.wfile = io.BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.send_error = Mock()
        handler.do_GET()
        return handler

    def test_netinfo_for_a_direct_client(self):
        state = dict(mode='lan-only', internet=False, tailscale='absent', updated=0)
        with patch.object(direct_listeners, 'read_state', return_value=state):
            handler = self.get('/netinfo', '10.42.0.20')
        info = json.loads(handler.wfile.getvalue())
        self.assertEqual((info['mode'], info['client_local']), ('lan-only', True))
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_any_call('Cache-Control', 'no-store')

    def test_no_download_for_a_tailscale_client(self):
        handler = self.get('/pc/lwir_connect.zip', '100.64.9.9')
        handler.send_error.assert_called_once_with(404)
        self.assertEqual(handler.wfile.getvalue(), b'')
        info = json.loads(self.get('/netinfo', '100.64.9.9').wfile.getvalue())
        self.assertFalse(info['client_local'])
        self.assertFalse(info['pc_download'])

    @unittest.skipUnless(direct_listeners.pc_files_present(), 'pc_connect files not in this checkout')
    def test_download_for_a_direct_client(self):
        handler = self.get('/pc/lwir_connect.zip', '10.42.0.20')
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_any_call('Content-Disposition', 'attachment; filename="lwir_connect.zip"')
        names = zipfile.ZipFile(io.BytesIO(handler.wfile.getvalue())).namelist()
        self.assertEqual(sorted(names), ['lwir_connect.bat', 'lwir_connect.ps1', 'lwir_connect.sh'])
        script = self.get('/pc/lwir_connect.ps1', '192.168.50.1')
        self.assertIn(b'param(', script.wfile.getvalue())

    def test_unknown_files_are_refused(self):
        for path in ('/pc/', '/pc/other.txt', '/pc/check_find_pi.py', '/pc/lwir_connect.zip.bak'):
            handler = self.get(path, '10.42.0.20')
            handler.send_error.assert_called_once_with(404)


if __name__ == '__main__':
    unittest.main()
