"""Mock HTTP lifecycle only: no sockets, camera, NPU, serial or device access."""
import http.server
import io
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

from ..common.state import APP
from . import page_handler as page


class LanListenerTests(unittest.TestCase):
    def setUp(self):
        previous = vars(APP).copy()
        self.addCleanup(lambda: (vars(APP).clear(), vars(APP).update(previous)))
        vars(APP).clear()
        self.server = Mock()
        self.server.socket.fileno.return_value = 10
        self.thread = Mock()
        self.thread.is_alive.return_value = True
        self.primary = Mock()
        self.primary.socket.fileno.return_value = 11
        self.primary_thread = Mock()
        self.primary_thread.is_alive.return_value = True
        APP.args = NS(live_bind='127.0.0.1', live_lan_bind='192.168.44.2', live_port=8090)
        APP.LIVE = NS(stopped=False)
        APP.lv_http = NS(ThreadingHTTPServer=Mock(return_value=self.server),
                         BaseHTTPRequestHandler=http.server.BaseHTTPRequestHandler)
        APP.lv_th = NS(Thread=Mock(return_value=self.thread))
        APP.lv_srv, APP.lv_srv_thread = self.primary, self.primary_thread
        assigned_patch = patch.object(page, '_lan_address_assigned', return_value=True)
        self.assigned = assigned_patch.start()
        self.addCleanup(assigned_patch.stop)

    def test_explicit_ranges_and_link_local_boundaries(self):
        for address in ('10.2.3.4','172.16.0.1','172.31.255.254','192.168.44.2',
                        '169.254.1.0','169.254.254.255'):
            self.assertEqual(page.lan_ipv4(address),address)
        for address in ('127.0.0.1','0.0.0.0','8.8.8.8','100.64.1.2','172.15.1.2',
                        '172.32.1.2','169.254.0.255','169.254.255.0','192.0.2.1',
                        '224.0.0.1','::1','example.invalid','192.168.001.2'):
            with self.subTest(address=address), self.assertRaises(ValueError):
                page.lan_ipv4(address)

    def test_opt_out_retains_primary_without_secondary(self):
        APP.args.live_lan_bind = None
        page.start_lan_listener(object())
        APP.lv_http.ThreadingHTTPServer.assert_not_called()
        APP.lv_th.Thread.assert_not_called()
        self.assertIs(APP.lv_srv,self.primary)
        self.assertEqual(page.network_state(),{'lan_url':None})

    def test_extra_listener_uses_same_port_handler_and_state(self):
        handler=object()
        page.start_lan_listener(handler)
        APP.lv_http.ThreadingHTTPServer.assert_called_once_with(('192.168.44.2',8090),handler)
        APP.lv_th.Thread.assert_called_once_with(target=self.server.serve_forever,daemon=True)
        self.thread.start.assert_called_once_with()
        self.assertTrue(self.server.daemon_threads)
        self.assertIs(APP.lv_srv,self.primary)
        self.assertIs(APP.lv_srv_lan,self.server)
        self.assertEqual(page.network_state(),{'lan_url':'http://192.168.44.2:8090/'})

    def test_equal_or_wildcard_primary_does_not_duplicate_listener(self):
        for bind in ('192.168.44.2','0.0.0.0',''):
            with self.subTest(bind=bind):
                APP.args.live_bind=bind
                page.start_lan_listener(object())
                self.assertEqual(page.network_state(),{'lan_url':'http://192.168.44.2:8090/'})
                self.assertIsNone(APP.lv_srv_lan)
        APP.lv_http.ThreadingHTTPServer.assert_not_called()
        APP.lv_th.Thread.assert_not_called()
        page.close_lan_listener()
        self.primary.shutdown.assert_not_called()
        self.primary.server_close.assert_not_called()

    def test_failed_bind_is_optional_and_never_advertised(self):
        APP.lv_http.ThreadingHTTPServer.side_effect=OSError('address unavailable')
        page.start_lan_listener(object())
        self.assertEqual(page.network_state(),{'lan_url':None})
        self.assertIsNone(APP.lv_srv_lan)
        self.primary.shutdown.assert_not_called()

    def test_unassigned_reused_primary_is_not_advertised(self):
        APP.args.live_bind='0.0.0.0'
        self.assigned.return_value=False
        page.start_lan_listener(object())
        self.assertEqual(page.network_state(),{'lan_url':None})
        APP.lv_http.ThreadingHTTPServer.assert_not_called()
        self.primary.shutdown.assert_not_called()

    def test_failed_thread_start_closes_socket(self):
        self.thread.start.side_effect=RuntimeError('cannot start thread')
        page.start_lan_listener(object())
        self.server.server_close.assert_called_once_with()
        self.assertEqual(page.network_state(),{'lan_url':None})

    def test_only_live_listener_is_advertised_and_closed_once(self):
        page.start_lan_listener(object())
        self.thread.is_alive.return_value=False
        self.assertEqual(page.network_state(),{'lan_url':None})
        self.thread.is_alive.return_value=True
        self.server.socket.fileno.return_value=-1
        self.assertEqual(page.network_state(),{'lan_url':None})
        self.server.socket.fileno.return_value=10
        APP.LIVE.stopped=True
        self.assertEqual(page.network_state(),{'lan_url':None})
        page.close_lan_listener();page.close_lan_listener()
        self.server.shutdown.assert_called_once_with()
        self.server.server_close.assert_called_once_with()
        self.primary.shutdown.assert_not_called()

    def test_network_get_uses_shared_state_without_mutation(self):
        page.start_lan_listener(object())
        handler_class=page.make_lv_Handler()
        handler=handler_class.__new__(handler_class)
        handler.path='/network?refresh=1'
        handler.wfile=io.BytesIO()
        handler.send_response=Mock();handler.send_header=Mock();handler.end_headers=Mock()
        handler.do_GET()
        self.assertEqual(json.loads(handler.wfile.getvalue()),{'lan_url':'http://192.168.44.2:8090/'})
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_any_call('Cache-Control','no-store')
        self.server.shutdown.assert_not_called()

    def test_live_close_closes_extra_primary_and_loopback(self):
        page.start_lan_listener(object())
        APP.LIVE.camera_bytetrack=Mock()
        APP.LIVE.jcond=MagicMock()
        APP.LIVE.cam_lock=MagicMock()
        APP.LIVE.proc=Mock()
        APP.meas_stop=Mock()
        APP.lv_srv_local=Mock()
        APP.GB=None
        with patch.object(page.replay_eval,'finish'):
            page.live_close()
        self.assertEqual(page.network_state(),{'lan_url':None})
        self.server.shutdown.assert_called_once_with()
        self.server.server_close.assert_called_once_with()
        self.primary.shutdown.assert_called_once_with()
        APP.lv_srv_local.shutdown.assert_called_once_with()


if __name__=='__main__':
    unittest.main()
