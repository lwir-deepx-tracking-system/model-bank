import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import page_settings


class StartupDetectorTests(unittest.TestCase):
    def fixture(self):
        app = SimpleNamespace(
            args=SimpleNamespace(det='w480', npu_depth=1), det_names=['NEWMRW480'],
            MODEL={'req': None, 'catalog': [], 'history': []},
            MODEL_TEXT={0: 'ready', 3: 'failed'}, RULES={'pending': None},
            SEL={}, eng={}, sel_save=Mock(),
        )
        self.addCleanup(patch.stopall)
        patch.object(page_settings, 'APP', app).start()
        return app

    def test_saved_bank_uses_guarded_registration_before_opening(self):
        app = self.fixture()
        key = 'bank-loaded'
        det = 'bank:' + key
        model = {'det': det, 'key': key, 'id': 2}
        app.bank_refresh = Mock(side_effect=lambda **kwargs: app.MODEL['catalog'].append(model))
        app.model_request = Mock(side_effect=lambda _: app.MODEL.update(req=dict(model, t=0)))
        app.Engine = Mock()
        with patch.dict(page_settings.CORE, {key: 0}), patch.object(page_settings, 'swap_refused', return_value=False):
            with patch.object(page_settings, 'det_changed') as changed:
                page_settings.queue_startup_detector(det)
                page_settings.open_startup_detector()
        app.bank_refresh.assert_called_once_with(force=True)
        app.Engine.assert_called_once_with(key, 0, 1)
        changed.assert_called_once_with(det, key)
        self.assertEqual(app.eng, {key: app.Engine.return_value})
        self.assertIsNone(app.MODEL['req'])
        self.assertEqual(app.MODEL['last']['result'], 0)

    def test_failed_saved_detector_leaves_default_and_removes_request(self):
        app = self.fixture()
        key = 'bad'
        model = {'det': 'bank:' + key, 'key': key, 'id': 2}
        app.MODEL.update(req=dict(model, t=0), catalog=[model])
        app.RULES['pending'] = {'tracker': 'botsort', 'predictor': 'gru_kpt'}
        app.SEL['predictor'] = 'gru_kpt'
        app.Engine = Mock(side_effect=RuntimeError('load refused'))
        with patch.dict(page_settings.CORE, {key: 0}), patch.object(page_settings, 'swap_refused', return_value=False):
            page_settings.open_startup_detector()
        self.assertEqual(app.args.det, 'w480')
        self.assertEqual(app.det_names, ['NEWMRW480'])
        self.assertFalse(app.eng)
        self.assertEqual(app.RULES['pending'], {'tracker': 'botsort'})
        self.assertIsNone(app.MODEL['req'])
        self.assertEqual(app.MODEL['last']['result'], 3)
