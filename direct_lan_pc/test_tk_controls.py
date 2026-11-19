"""Exercise the desktop controls without changing a real Orange Pi."""
import sys
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import App


class TkControlTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest('Tk display unavailable: ' + str(exc))
        self.root.withdraw()
        self.link = Mock()
        self.link.post.return_value = {'text': 'ok'}
        self.link.get.return_value = {}
        with patch('app.threading.Thread.start'):
            self.app = App(self.root, self.link)
        self.app.controls.submit = lambda fn: fn()
        self.app.workers.submit = lambda fn: fn()

    def tearDown(self):
        self.app.stop.set()
        self.app.controls.shutdown(wait=False, cancel_futures=True)
        self.app.workers.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def buttons(self):
        found = {}
        def visit(widget):
            for child in widget.winfo_children():
                if isinstance(child, (tk.Button,)) or child.winfo_class() == 'TButton':
                    found[child.cget('text')] = child
                visit(child)
        visit(self.root)
        return found

    def test_all_action_buttons_send_expected_route(self):
        self.app.replay = {'state': {'on': True, 'fps': 25, 'pos': 100, 'total': 1000}}
        self.app.replay_var.set('recording')
        self.app.seek_var.set('123')
        self.app.measure_n.set('1')
        self.app.measure_stop_button.configure(state='normal')
        expected = {
            '원점 복귀 (Space)': '/gimbal/home', '따라가기 정지': '/gimbal/stop',
            '카메라로': '/replay/stop', '예측 제어 시간 적용': '/rules/set',
            '카메라 초기화': '/camera/reset', '추적 프로그램 재시작': '/system/restart',
            'Orange Pi 재부팅': '/system/reboot?confirm=REBOOT',
            '카메라 C++ ByteTrack': '/tracker/camera', 'Python ByteTrack': '/rules/set',
            '값 적용': '/rules/set', '재생 시작': '/replay/start',
            '카메라로 돌아가기': '/replay/stop', '일시정지': '/replay/pause',
            '계속': '/replay/play', '−10초': '/replay/seek', '+10초': '/replay/seek',
            '+1장': '/replay/step', '처음부터 다시 채점': '/replay/eval/reset',
            '위치 이동': '/replay/seek', '속도 적용': '/replay/speed',
            '측정 시작': '/measure/start', '측정 끝': '/measure/stop',
            '이름·메모 수정': '/measure/note',
        }
        buttons = self.buttons()
        self.assertEqual(set(expected) - set(buttons), set())
        with patch('app.messagebox.askyesno', return_value=True):
            for label, route in expected.items():
                with self.subTest(button=label):
                    self.link.post.reset_mock()
                    buttons[label].invoke()
                    self.assertTrue(self.link.post.called, label)
                    self.assertEqual(self.link.post.call_args.args[0].split('?')[0], route.split('?')[0])
        for label, route in [('현재 진행', '/replay/eval'),
                             ('이 위치까지', '/replay/eval?pos=122'),
                             ('최종 결과', '/replay/eval?view=final'),
                             ('장치 상태', '/healthz'), ('짐벌·사람', '/gimbal/people'),
                             ('자동 모델 기록', '/rules/auto_det_log'), ('현재 모델', '/model')]:
            with self.subTest(button=label):
                self.link.get.reset_mock()
                buttons[label].invoke()
                self.link.get.assert_called_once_with(route)

    def test_bad_state_does_not_stop_periodic_updates(self):
        self.app.events.put(('state', {}))
        self.app.pump()
        self.assertIn('화면 갱신 오류', self.app.message.get())
        self.assertFalse(self.app.stop.is_set())

    def test_switches_and_selection_controls(self):
        switches = {}
        def visit(widget):
            for child in widget.winfo_children():
                if child.winfo_class() == 'TCheckbutton':
                    switches[child.cget('text')] = child
                visit(child)
        visit(self.root)
        expected_keys = {'사진 먼저': 'picture_first', '예측 상자 표시': 'pred_show',
                         '예측 제어': 'lead', 'KeepID': 'relabel',
                         'KeepID 찾는 범위 확장': 'relink_pred',
                         '사람 크기에 따른 모델 자동 선택': 'auto_det',
                         '두 검출 모델 상주': 'resident_det',
                         '놓친 상자를 보일 때와 동일하게': 'lost_same',
                         '온도 Filter': 'warm_on', '사람 후보 표시': 'prop_show',
                         '피부 온도 표시': 'temp_show', '흔들림 보정': 'stab'}
        self.assertFalse(set(expected_keys) - set(switches))
        for label, key in expected_keys.items():
            with self.subTest(switch=label):
                self.link.post.reset_mock()
                switches[label].invoke()
                self.assertTrue(self.link.post.called)
                self.assertIn(key + '=1', self.link.post.call_args.args[0])
        self.link.post.reset_mock()
        switches['정답 상자 그리기'].invoke()
        self.assertIn('/replay/eval/draw?on=1', self.link.post.call_args.args[0])
        self.app.model_ids = [12]
        self.app.model_box['values'] = ['demo']
        self.app.select_model('demo')
        self.assertIn('/model/set?id=12', self.link.post.call_args.args[0])
        self.app.reid_keys = ['test_reid']
        self.app.reid_box['values'] = ['ReID']
        self.app.select_reid('ReID')
        self.assertIn('reid=test_reid', self.link.post.call_args.args[0])
        self.app.predictor_keys = ['test_predictor']
        self.app.predictor_box['values'] = ['Predictor']
        self.app.select_predictor('Predictor')
        self.assertIn('predictor=test_predictor', self.link.post.call_args.args[0])

    def test_video_click_uses_displayed_token(self):
        self.app.frame_seq = 7
        self.app.frame_token = '7:2'
        self.app.image_box = (10, 20, 100, 100)
        self.link.click.return_value = {'answer': 'ok'}
        self.app.on_video_click(Mock(x=60, y=70))
        self.link.click.assert_called_once_with('7:2', .5, .5)

    def test_invalid_replay_position_is_reported_without_callback_error(self):
        self.app.seek_var.set('invalid')
        self.buttons()['이 위치까지'].invoke()
        self.assertEqual(self.app.message.get(), '장 번호를 숫자로 입력하세요')
        self.link.get.assert_not_called()

    def test_full_event_queue_does_not_block_control_result(self):
        for index in range(self.app.events.maxsize):
            self.app.events.put_nowait(('state', index))
        self.app.command('/gimbal/stop')
        self.link.post.assert_called_once_with('/gimbal/stop', None)
        self.assertEqual(self.app.events.qsize(), self.app.events.maxsize)

    def test_recording_bit_depth_label_keeps_original_replay_name(self):
        self.app.replay_ids = ['측정/day/original']
        self.app.replay_names = ['16비트 원본 · 측정/day/original']
        self.app.replay_var.set(self.app.replay_names[0])
        self.app.start_replay()
        route = self.link.post.call_args.args[0]
        self.assertIn('seq=%EC%B8%A1%EC%A0%95%2Fday%2Foriginal', route)


if __name__ == '__main__':
    unittest.main()
