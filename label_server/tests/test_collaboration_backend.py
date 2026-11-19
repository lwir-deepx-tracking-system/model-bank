"""Real isolated HTTP requests; no production server, labels or camera access."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from label_server.src import collaboration, state, file_ops
from label_server.tests import test_body_parts_backend as fixture


A = 'a' * 32
B = 'b' * 32


class CollaborationHttpTest(unittest.TestCase):
    setUp = fixture.LabelPartsHttpTest.setUp
    request = fixture.LabelPartsHttpTest.request
    snapshot = fixture.LabelPartsHttpTest.snapshot
    save = fixture.LabelPartsHttpTest.save

    def claim(self, owner, first, last, user='worker'):
        return self.request('/api/claim' + fixture.QUERY,
                            dict(owner=owner, user=user, first=first, last=last))

    def entry(self, snapshot, frame, **changes):
        before = collaboration.frame_state(snapshot, frame)
        after = copy.deepcopy(before)
        after.update(changes)
        return dict(frame=frame, before=before, after=after)

    def patch_frames(self, owner, *entries, **extra):
        return self.request('/api/save' + fixture.QUERY,
                            dict(editor_schema=3, owner=owner, frame_patch=list(entries), **extra))

    def test_dueling_claims_and_renewal_release_expiry(self):
        with ThreadPoolExecutor(2) as workers:
            results = list(workers.map(lambda owner: self.claim(owner, 1, 2), (A, B)))
        self.assertEqual(sorted(code for code, _ in results), [200, 409])
        winner = A if results[0][0] == 200 else B
        loser = B if winner == A else A
        public = self.request('/api/claims' + fixture.QUERY)[1]
        self.assertEqual(public['lease_seconds'], 90)
        self.assertEqual(len(public['claims']), 1)
        self.assertNotIn('owner', public['claims'][0])
        self.assertEqual(self.claim(winner, 1, 1)[0], 200)
        self.assertEqual(self.claim(loser, 2, 2)[0], 200)
        mine = self.request('/api/claims' + fixture.QUERY + '&owner=' + winner)[1]['claims']
        self.assertEqual([row['mine'] for row in mine], [True, False])
        self.assertEqual(self.claim(winner, 1, 2)[0], 409)
        self.assertEqual(self.server.frame_claims['example'][winner]['last'], 1)
        self.assertEqual(self.request('/api/release' + fixture.QUERY, {'owner': loser})[0], 200)
        self.assertEqual(self.claim(winner, 1, 2)[0], 200)
        with state.LOCK:
            self.server.frame_claims['example'][winner]['deadline'] = 0
        self.assertEqual(self.request('/api/claims' + fixture.QUERY)[1]['claims'], [])
        self.assertEqual(self.claim(loser, 1, 2)[0], 200)

    def test_disjoint_stale_patches_merge_and_keep_raw_untouched_mot(self):
        before = self.snapshot()
        self.assertTrue(before['frame_claims'])
        self.assertEqual(self.claim(A, 1, 1, 'first')[0], 200)
        self.assertEqual(self.claim(B, 2, 2, 'second')[0], 200)
        first = self.entry(before, 1, mot=['1,1,5.000,6,10,11,1.0000,0,-1,-1'],
                           reviewed=True, parts=fixture.parts_for((1, 1))['annotations'])
        second = self.entry(before, 2, reviewed=True, parts=fixture.parts_for((2, 2))['annotations'])
        with ThreadPoolExecutor(2) as workers:
            results = list(workers.map(lambda job: self.patch_frames(*job), ((A, first), (B, second))))
        self.assertEqual([code for code, _ in results], [200, 200])
        merged = self.snapshot()
        self.assertEqual(merged['review']['reviewed'], [1, 2])
        self.assertEqual(merged['mot'].splitlines(), first['after']['mot'] + second['after']['mot'])
        self.assertEqual({r['id'] for r in merged['parts']['annotations']}, {1, 2})
        for code, receipt in results:
            self.assertTrue(receipt['ok'])
            self.assertEqual(receipt['boxes'], 2)
            self.assertEqual(receipt['version'], receipt['snapshot']['version'])
            self.assertTrue(receipt['snapshot']['frame_claims'])
        # Retry from the old per-frame snapshot must not revert the winner.
        status, error = self.patch_frames(A, first)
        self.assertEqual((status, error['frames']), (409, [1]))
        self.assertEqual(self.snapshot(), merged)

    def test_atomic_multi_frame_conflict_and_claim_ownership(self):
        before = self.snapshot()
        one, two = self.entry(before, 1, reviewed=True), self.entry(before, 2, reviewed=True)
        self.assertEqual(self.patch_frames(A, one)[0], 409)
        self.assertEqual(self.claim(A, 1, 1)[0], 200)
        status, error = self.patch_frames(A, one, two)
        self.assertEqual((status, error['frames']), (409, [2]))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.patch_frames(B, one)[0], 409)
        self.assertEqual(self.claim(A, 1, 2)[0], 200)
        stale_two = copy.deepcopy(two)
        stale_two['before']['reviewed'] = True
        status, error = self.patch_frames(A, one, stale_two)
        self.assertEqual((status, error['frames']), (409, [2]))
        self.assertEqual(self.snapshot(), before)

    def test_legacy_save_purge_and_sequence_changes_blocked_until_release(self):
        before = self.snapshot()
        self.assertEqual(self.claim(A, 1, 1)[0], 200)
        self.assertEqual(self.save(self.original, {}, base=before['version'],
                                   parts=fixture.parts_for((1, 1)), force=True, editor_schema=3)[0], 409)
        self.assertEqual(self.request('/api/purge' + fixture.QUERY, {'base': before['version']})[0], 409)
        self.assertEqual(self.request('/api/seq_rename' + fixture.QUERY, {'name': 'other'})[0], 409)
        self.assertEqual(self.request('/api/seq_trash' + fixture.QUERY, {})[0], 409)
        self.assertEqual(self.snapshot(), before)
        self.request('/api/release' + fixture.QUERY, {'owner': A})
        self.assertEqual(self.save(self.original, {}, base=before['version'], parts=fixture.parts_for((1, 1)))[0], 200)

    def test_parts_order_normalization_and_intentional_removal(self):
        before = self.snapshot()
        parts = fixture.parts_for((1, 1), (2, 2))
        self.assertEqual(self.save(self.original, {}, base=before['version'], parts=parts)[0], 200)
        before = self.snapshot()
        self.claim(A, 1, 1)
        entry = self.entry(before, 1)
        entry['before']['parts'][0]['needs_review'] = False
        entry['before']['parts'][0]['parts'] = dict(reversed(list(entry['before']['parts'][0]['parts'].items())))
        entry['after']['parts'] = []
        self.assertEqual(self.patch_frames(A, entry)[0], 200)
        self.assertEqual(self.snapshot()['parts']['annotations'],
                         [dict(parts['annotations'][1], needs_review=False)])

    def test_invalid_payloads_and_purged_frames_do_not_write(self):
        before = self.snapshot()
        self.assertEqual(self.claim('short', 1, 2)[0], 400)
        for first, last in ((0, 2), (1, 3), (2, 1), (True, 2)):
            self.assertEqual(self.claim(A, first, last)[0], 400)
        self.claim(A, 1, 2)
        one = self.entry(before, 1)
        self.assertEqual(self.patch_frames(A, one, one)[0], 400)
        bad = copy.deepcopy(one)
        bad['after']['mot'] = [fixture.MOT_TWO.strip()]
        self.assertEqual(self.patch_frames(A, bad)[0], 400)
        bad = copy.deepcopy(one)
        bad['after']['parts'] = fixture.parts_for((2, 2))['annotations']
        self.assertEqual(self.patch_frames(A, bad)[0], 400)
        self.assertEqual(self.snapshot(), before)
        with state.LOCK:
            self.server.frame_claims['example'][A]['deadline'] = 0
        self.assertEqual(self.patch_frames(A, one)[0], 409)
        self.claim(A, 1, 2)
        (self.sequence / state.REVIEW_FILE).write_text(json.dumps({'purged': [1], 'removed': [1]}))
        purged = self.snapshot()
        status, result = self.patch_frames(A, self.entry(purged, 1, removed=False))
        self.assertEqual((status, result['frames']), (409, [1]))
        self.assertEqual(self.snapshot(), purged)

    def test_multiple_ignore_rows_with_shared_sentinel_id_are_preserved(self):
        before = self.snapshot()
        ignored = ['1,-1,1,2,3,4,1,1,-1,-1', '1,-1,2,3,4,5,1,1,-1,-1']
        original = before['mot'].replace(fixture.MOT_ONE, fixture.MOT_ONE + '\n'.join(ignored) + '\n')
        self.assertEqual(self.save(original, {}, base=before['version'], parts=fixture.parts_for((1, 1)))[0], 200)
        before = self.snapshot()
        self.claim(A, 1, 1)
        self.assertEqual(self.patch_frames(A, self.entry(before, 1, reviewed=True))[0], 200)
        self.assertEqual(self.snapshot()['mot'], original)

    def test_patch_transaction_failure_rolls_back_all_files(self):
        before = self.snapshot()
        self.claim(A, 1, 2)
        one = self.entry(before, 1, reviewed=True)
        original_replace = file_ops.os.replace
        def fail_parts(source, destination):
            if str(source).endswith(state.PARTS_FILE):
                raise OSError('injected patch failure')
            return original_replace(source, destination)
        with patch.object(file_ops.os, 'replace', side_effect=fail_parts):
            self.assertEqual(self.patch_frames(A, one)[0], 500)
        self.assertEqual(self.snapshot(), before)


if __name__ == '__main__':
    unittest.main()
