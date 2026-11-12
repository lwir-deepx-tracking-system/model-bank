import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from label_server.src import file_ops, handler, sequences, state, transactions
from label_server.src.body_parts import POINT_PARTS, empty_parts, validate_parts


MOT_ONE = '1,1,5,6,10,11,1,0,-1,-1\n'
MOT_TWO = '2,2,50,60,100,110,1,0,-1,-1\n'
QUERY = '?seq=example&user=tester'


def parts_for(*rows):
    parts = empty_parts()
    for frame, person_id in rows:
        parts['annotations'].append({'frame': frame, 'id': person_id, 'parts': {
            'head': {'state': 'visible', 'xywh': [10, 12, 20, 16]},
            'torso': {'state': 'occluded', 'xywh': None},
            'phone': {'state': 'out_of_frame', 'xywh': None},
            'left_shoulder': {'state': 'visible', 'xy': [639.99, 479.99]},
            'right_hip': {'state': 'unknown', 'xy': None},
        }})
    return parts


class PartsValidationTest(unittest.TestCase):
    def test_states_and_geometry(self):
        good = parts_for((1, 1))
        valid_parts = good['annotations'][0]['parts']
        for name in POINT_PARTS:
            valid_parts.setdefault(name, {'state': 'occluded', 'xy': [0, 0]})
        valid_parts['head']['xywh'] = [639, 479, 1, 1]
        normalized = validate_parts(good, [MOT_ONE.strip()], 2)
        self.assertFalse(normalized['annotations'][0]['needs_review'])
        for name, invalid in (
            ('left_shoulder', {'state': 'visible', 'xy': [640, 0]}),
            ('left_shoulder', {'state': 'visible', 'xy': [0, 480]}),
            ('left_shoulder', {'state': 'visible', 'xy': [False, 0]}),
            ('left_shoulder', {'state': 'visible', 'xy': [float('nan'), 0]}),
            ('head', {'state': 'visible', 'xywh': [639, 479, 2, 1]}),
            ('head', {'state': 'occluded', 'xywh': [0, 0, 10, float('inf')]}),
            ('head', {'state': 'visible', 'xywh': [1, 1, 0, 10]}),
            ('head', {'state': 'visible', 'xywh': None}),
            ('head', {'state': 'visible', 'xy': [1, 1]}),
            ('head', {'state': 'unknown', 'xywh': [1, 1, 10, 10]}),
            ('head', {'state': 'out_of_frame', 'xywh': [1, 1, 10, 10]}),
            ('head', {'state': 'occluded'}),
            ('hip', {'state': 'unknown', 'xy': None}),
        ):
            with self.subTest(name=name, invalid=invalid):
                bad = copy.deepcopy(good)
                bad['annotations'][0]['parts'][name] = invalid
                with self.assertRaises(ValueError):
                    validate_parts(bad, [MOT_ONE.strip()], 2)

    def test_exactly_one_person_parent_and_annotation_key(self):
        good = parts_for((1, 1))
        row = MOT_ONE.strip()
        ignore = row.replace(',1,0,-1,-1', ',1,1,-1,-1')
        for lines in ([], [ignore], [row, row], [row, ignore]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                validate_parts(good, lines, 2)
        with self.assertRaises(ValueError):
            validate_parts(parts_for((1, 1), (1, 1)), [row], 2)
        for key, value in (('frame', True), ('frame', 3), ('id', 0), ('id', 1.0), ('needs_review', 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bad = copy.deepcopy(good)
                bad['annotations'][0][key] = value
                validate_parts(bad, [row], 2)


class LabelPartsHttpTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sequence = self.root / 'example'
        (self.sequence / 'images').mkdir(parents=True)
        self.original = MOT_ONE + MOT_TWO
        (self.sequence / state.SOURCE_MOT).write_bytes(self.original.encode('utf-8'))
        for frame in (1, 2):
            (self.sequence / 'images' / f'{frame:06d}.png').write_bytes(bytes([frame]))
        self.backup = self.root / 'backups'

        scopes = (patch.dict(state.SEQS, {'example': {'dir': str(self.sequence), 'frames': 2}}, clear=True),
                  patch.dict(state.RESCAN, {'t': time.time(), 'only': ''}, clear=True),
                  patch.dict(state.PRESENCE, {}, clear=True), patch.object(file_ops, 'HERE', str(self.root)),
                  patch.object(state, 'HERE', str(self.root)))
        for scope in scopes:
            scope.start()
            self.addCleanup(scope.stop)

        self.server = handler.Srv(('127.0.0.1', 0), handler.H)
        self.server.root, self.server.token, self.server.backup = str(self.root), '', str(self.backup)
        self.addCleanup(self.server.server_close)
        thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)

    def request(self, path, body=None):
        conn = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            if body is None:
                conn.request('GET', path)
            else:
                conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            payload = response.read()
            if response.getheader('Content-Type', '').startswith('application/json'):
                payload = json.loads(payload)
            return response.status, payload
        finally:
            conn.close()

    def snapshot(self):
        code, data = self.request('/api/annotations' + QUERY)
        self.assertEqual(code, 200)
        return data

    def save(self, mot, review, **extra):
        return self.request('/api/save' + QUERY, {'mot': mot, 'review': review, **extra})

    def assert_saved(self, **review):
        code, data = self.save(self.original, review, base=self.snapshot()['version'], parts=parts_for((1, 1), (2, 2)))
        self.assertEqual(code, 200, data)
        return self.snapshot()

    def test_snapshot_legacy_save_and_empty_edited_mot(self):
        initial = self.snapshot()
        self.assertEqual(initial['mot'], self.original)
        self.assertEqual(initial['parts'], empty_parts())
        self.assertEqual((initial['parts_supported'], initial['editor_schema'], initial['frames']), (True, 2, 2))
        code, meta = self.request('/api/meta' + QUERY)
        self.assertEqual(code, 200)
        self.assertEqual(meta['version'], initial['version'])
        self.assertEqual(self.request('/data/example/' + state.PARTS_FILE)[0], 404)
        code, result = self.save('', {'reviewed': [1, 2]})
        self.assertEqual(code, 200)
        loaded = self.snapshot()
        self.assertEqual(loaded['mot'], '')
        self.assertEqual(loaded['review']['completed_by'], 'tester')
        self.assertEqual(loaded['version'], result['version'])
        self.assertNotEqual(loaded['version'], initial['version'])
        self.assertFalse((self.sequence / state.PARTS_FILE).exists())
        self.assertEqual((self.sequence / state.SOURCE_MOT).read_text(encoding='utf-8'), self.original)

    def test_sidecar_preconditions_and_validation_before_backups(self):
        initial = self.snapshot()
        parts = parts_for((1, 1))
        self.assertEqual(self.save(MOT_ONE, {}, parts=parts)[0], 428)
        self.assertEqual(self.save(MOT_TWO, {}, base=initial['version'], parts=parts)[0], 400)
        self.assertEqual(self.save(MOT_ONE, {'reviewed': {}}, base=initial['version'], parts=parts)[0], 400)
        self.assertEqual(self.save(MOT_ONE, {}, base=initial['version'], parts=None)[0], 400)
        self.assertFalse(self.backup.exists())
        saved = self.assert_saved()
        self.assertEqual(self.request('/data/example/' + state.PARTS_FILE), (200, saved['parts']))
        self.assertEqual(self.save(MOT_ONE, {}, base=saved['version'], force=True)[0], 428)
        self.assertEqual(self.save(MOT_ONE, {}, base=initial['version'], force=True, parts=parts)[0], 409)
        self.assertEqual(self.snapshot(), saved)

    def test_version_covers_same_size_review_and_parts_changes(self):
        saved = self.assert_saved()
        for name, old, new in ((state.REVIEW_FILE, b'tester', b'editor'), (state.PARTS_FILE, b'"visible"', b'"unknown"')):
            file = self.sequence / name
            stat = file.stat()
            file.write_bytes(file.read_bytes().replace(old, new, 1))
            os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            result = self.snapshot()
            self.assertNotEqual(result['version'], saved['version'])
            saved = result

    def test_purge_keeps_unselected_rows_and_backs_up_three_files(self):
        saved = self.assert_saved(removed=[1])
        previous = {name: (self.sequence / name).read_bytes() for name in state.SAVE_NAMES}
        code, result = self.request('/api/purge' + QUERY, {'base': saved['version']})
        self.assertEqual(code, 200)
        self.assertEqual(result['moved'], 1)
        loaded = self.snapshot()
        self.assertEqual(loaded['mot'], MOT_TWO)
        self.assertEqual(loaded['review']['purged'], [1])
        self.assertEqual(loaded['parts']['annotations'], saved['parts']['annotations'][1:])
        self.assertEqual(loaded['version'], result['version'])
        self.assertEqual((self.sequence / state.SOURCE_MOT).read_text(encoding='utf-8'), self.original)
        self.assertFalse((self.sequence / 'images' / '000001.png').exists())
        self.assertEqual(next(self.root.glob('trash/example/*/000001.png')).read_bytes(), b'\x01')
        for name, prior in previous.items():
            stored = list((self.backup / 'example').glob(f'*before_purge*{name}'))
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].read_bytes(), prior)
        self.assertEqual(self.request('/api/purge' + QUERY, {'base': loaded['version']})[1]['moved'], 0)

    def test_save_and_purge_roll_back_after_third_replace_fails(self):
        before = self.assert_saved(removed=[1])
        real_replace = os.replace

        def fail_last(source, destination):
            if Path(source).name == state.PARTS_FILE:
                raise OSError('injected sidecar replace failure')
            return real_replace(source, destination)

        with patch.object(file_ops.os, 'replace', side_effect=fail_last):
            code, _ = self.save(MOT_TWO, {}, base=before['version'], parts=parts_for((2, 2)))
            self.assertEqual(code, 500)
            self.assertEqual(self.snapshot(), before)
            code, _ = self.request('/api/purge' + QUERY, {'base': before['version']})
            self.assertEqual(code, 500)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual((self.sequence / 'images' / '000001.png').read_bytes(), b'\x01')
        self.assertFalse(list(self.root.glob('trash/example/*/000001.png')))

    def test_failed_first_save_removes_uncommitted_files(self):
        before = self.snapshot()
        real_replace = os.replace

        def fail_last(source, destination):
            if Path(source).name == state.PARTS_FILE:
                raise OSError('injected first sidecar replace failure')
            return real_replace(source, destination)

        with patch.object(file_ops.os, 'replace', side_effect=fail_last):
            code, _ = self.save(MOT_ONE, {}, base=before['version'], parts=parts_for((1, 1)))
        self.assertEqual(code, 500)
        self.assertEqual(self.snapshot(), before)
        self.assertTrue(all(not (self.sequence / name).exists() for name in state.SAVE_NAMES))

    def test_snapshot_waits_until_all_writes_commit(self):
        before = self.assert_saved()
        replaced_mot = threading.Event()
        continue_write = threading.Event()
        real_replace = os.replace

        def pause_after_mot(source, destination):
            real_replace(source, destination)
            if Path(source).name == state.EDITED_MOT:
                replaced_mot.set()
                if not continue_write.wait(3):
                    raise OSError('test did not release paused write')

        with patch.object(file_ops.os, 'replace', side_effect=pause_after_mot), ThreadPoolExecutor(2) as workers:
            write = workers.submit(self.save, MOT_TWO, {}, base=before['version'], parts=parts_for((2, 2)))
            try:
                self.assertTrue(replaced_mot.wait(3))
                read = workers.submit(self.snapshot)
                with self.assertRaises(TimeoutError):
                    read.result(timeout=0.1)
            finally:
                continue_write.set()
            self.assertEqual(write.result(timeout=3)[0], 200)
            after = read.result(timeout=3)
        self.assertEqual(after['mot'], MOT_TWO)
        self.assertEqual(after['parts']['annotations'][0]['id'], 2)

    def crash_during_transaction(self, phase):
        code = """
import os, sys
from pathlib import Path
from uuid import uuid4
from label_server.src import file_ops, state

root, sequence, backup, phase = sys.argv[1:]
state.HERE = root
replace = os.replace
def crash_after_first_label(source, destination):
    replace(source, destination)
    if Path(source).name == state.EDITED_MOT:
        os._exit(76)

if phase == 'after_last_replace':
    file_ops.finish_transaction = lambda directory: os._exit(76)
else:
    file_ops.os.replace = crash_after_first_label
images = ['000001.png', '000002.png'] if phase == 'purge' else []
operation = file_ops.transaction_stamp() + '_' + uuid4().hex
updates = {state.EDITED_MOT: b'', state.REVIEW_FILE: b'{}', state.PARTS_FILE: b'{}'}
file_ops.commit_files(sequence, 'example', backup, 'crash_fixture', updates, operation, images)
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.root), str(self.sequence),
                                 str(self.backup / 'example'), phase], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 76, result.stderr.decode(errors='replace'))
        self.assertTrue((self.sequence / transactions.JOURNAL).is_file())
        self.assertEqual((self.sequence / state.EDITED_MOT).read_bytes(), b'')

    def test_recovery_after_process_exits_at_first_or_last_replace(self):
        before = self.assert_saved(removed=[1])
        journal = self.sequence / transactions.JOURNAL
        self.assertFalse(journal.exists())
        for phase in ('after_first_replace', 'after_last_replace'):
            with self.subTest(phase=phase):
                self.crash_during_transaction(phase)
                self.assertEqual(self.request('/data/example/' + state.EDITED_MOT),
                                 (200, self.original.encode('utf-8')))
                self.assertEqual(self.snapshot(), before)
                self.assertFalse(journal.exists())
        status, _ = self.save(MOT_TWO, {}, base=before['version'], parts=parts_for((2, 2)))
        self.assertEqual(status, 200)
        self.assertFalse(journal.exists())
        self.assertEqual(self.snapshot()['mot'], MOT_TWO)

    def test_startup_scan_recovers_purge_when_all_images_were_moved(self):
        before = self.assert_saved(removed=[1, 2])
        self.crash_during_transaction('purge')
        self.assertEqual(list((self.sequence / 'images').glob('*.png')), [])
        scanned = sequences.scan(str(self.root))
        self.assertEqual(scanned['example']['frames'], 2)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.sequence / transactions.JOURNAL).exists())
        self.assertEqual((self.sequence / 'images' / '000001.png').read_bytes(), b'\x01')
        self.assertEqual((self.sequence / 'images' / '000002.png').read_bytes(), b'\x02')
        self.assertFalse(list(self.root.glob('trash/example/*/*.png')))

    def test_invalid_journal_blocks_reads_and_writes_without_touching_files(self):
        before = self.assert_saved()
        originals = {name: (self.sequence / name).read_bytes() for name in state.SAVE_NAMES}
        journal = self.sequence / transactions.JOURNAL
        payload = {'schema_version': 1, 'files': dict.fromkeys(state.SAVE_NAMES), 'purge': None}
        payload['files']['../unrelated.txt'] = None
        content = json.dumps(payload).encode('utf-8')
        journal.write_bytes(content)
        for path in ('/api/list', '/api/meta' + QUERY, '/api/annotations' + QUERY,
                     '/data/example/' + state.EDITED_MOT, '/data/example/RESULT.json', '/data/example/images/000001.png'):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 500)
        self.assertEqual(self.save(MOT_TWO, {}, base=before['version'], parts=parts_for((2, 2)))[0], 500)
        self.assertEqual(self.request('/api/purge' + QUERY, {'base': before['version']})[0], 500)
        self.assertEqual(journal.read_bytes(), content)
        for name, original in originals.items():
            self.assertEqual((self.sequence / name).read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
