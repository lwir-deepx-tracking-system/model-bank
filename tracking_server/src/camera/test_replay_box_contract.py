"""Mismatched or unreviewed GT must not produce presentation scores."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from . import replay_box_contract as contract


class BoxContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lab = dict(ok=True, why=None, frames=[1], gt={1: [(7, 100., 100., 140., 200., 0)]},
                        bad_lines=0, label_sha256='mot', review_sha256='review', review={})
        self.model = dict(source='server_detector', model_keys=('yolo26m_pose_lwir650_20261119_bodybox_kpt_640x480',))
        self.doc = dict(schema=contract.SCHEMA, recording='synthetic/test', human_reviewed=True,
                        source_mot_sha256='mot', source_review_sha256='review',
                        source_box_definition=contract.UPPER, box_definition=contract.UPPER)

    def resolve(self):
        return contract.resolve(self.root, 'synthetic/test', self.lab, self.model, strict=True)

    def write(self):
        (self.root / contract.SIDECAR).write_text(json.dumps(self.doc))

    def test_missing_review_blocks_scores_but_preserves_replay_rows(self):
        state, gt = self.resolve()
        self.assertFalse(state['score_allowed'])
        self.assertEqual(gt, self.lab['gt'])

    def test_verified_matching_definition_preserves_exact_boxes(self):
        self.write()
        state, gt = self.resolve()
        self.assertTrue(state['score_allowed'])
        self.assertEqual(gt, self.lab['gt'])
        self.assertEqual(state['reviewed_box_contract_sha256'], hashlib.sha256((self.root/contract.SIDECAR).read_bytes()).hexdigest())

    def test_changed_labels_review_or_box_definition_block_scores(self):
        for field, value in [('source_mot_sha256', 'changed'), ('source_review_sha256', 'changed'),
                             ('box_definition', contract.FULL), ('human_reviewed', False)]:
            with self.subTest(field=field):
                old = self.doc[field]; self.doc[field] = value; self.write()
                self.assertFalse(self.resolve()[0]['score_allowed'])
                self.doc[field] = old

    def test_unknown_detector_does_not_borrow_active_model_definition(self):
        self.write()
        self.model['model_keys'] = ('unknown',)
        self.assertFalse(self.resolve()[0]['score_allowed'])
        self.assertIsNone(contract.prediction_definition(contract.stored_model({})))
