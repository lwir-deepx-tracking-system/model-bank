"""Replay trackers (src/track/replay_backends.py): the stored-result backend and the two older ones.

What is checked here:
  - a stored result played through the web replay path (start_request -> Eval -> on_frame) gives exactly the numbers of
    scoring that file directly with gt_score.Score and replay_metrics.compute;
  - a stored result that does not cover the recording is refused (never listed as usable, never started);
  - 'python' and 'cpp_bytetrack' behave as before;
  - without --replay-tracks-root nothing new is listed or accepted.
Run from the tracking_server folder:  python3 -m unittest src.camera.test_replay_backend
"""
import collections
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from ..common.state import APP
from ..track import gt_score, replay_backends, replay_metrics
from . import replay_eval, replay_link

PNG_HEAD = (b'\x89PNG\r\n\x1a\n' + (13).to_bytes(4, 'big') + b'IHDR' + (640).to_bytes(4, 'big') + (480).to_bytes(4, 'big')
            + bytes([8, 0, 0, 0, 0]) + b'\x00\x00\x00\x00')   # the first 33 bytes of an 8-bit 640x480 PNG: all the recording list reads
FRAMES = 8
# labels: frame,id,x,y,w,h,1,class,-1,-1  (class 0 = person, 1 = ignore region)
LABELS = ''.join(
    '%d,10,%d,100,40,120,1,0,-1,-1\n' % (f, 100 + 4 * f) for f in range(1, FRAMES + 1)) + ''.join(
    '%d,11,%d,104.1,36,90,1,0,-1,-1\n' % (f, 400 - 3 * f) for f in range(3, FRAMES + 1)) + ''.join(
    '%d,90,500,300,100,100,1,1,-1,-1\n' % f for f in range(1, FRAMES + 1)) + ''.join(
    '%d,12,0,200,12,60,1,0,-1,-1\n' % f for f in (2, 3))   # a sliver at the left edge: an ignore region in the new scoring
# stored result: frame,id,x,y,w,h,score,-1,-1,-1  (ids from 0, picture pixels, like run_motip_seq.py writes)
STORED = (
    '1,0,105.250,101.487,40.312,119.921,0.94227,-1,-1,-1\n'
    '2,0,109.500,101.699,40.172,119.746,0.94400,-1,-1,-1\n'
    '2,3,300.000,20.500,30.000,60.000,0.61000,-1,-1,-1\n'      # a false box
    '3,0,113.537,102.101,39.898,118.287,0.94244,-1,-1,-1\n'
    '3,1,392.907,105.757,35.668,89.343,0.84894,-1,-1,-1\n'
    '3,5,2.000,205.000,10.000,50.000,0.40000,-1,-1,-1\n'       # on the edge sliver: ignored in the new scoring, a found person in the old
    # frame 4: nothing found
    '5,2,121.155,100.445,39.997,120.804,0.95450,-1,-1,-1\n'    # person 10 comes back with another id
    '5,1,386.289,104.818,35.548,90.279,0.93722,-1,-1,-1\n'
    '6,2,125.897,99.915,40.491,120.306,0.94884,-1,-1,-1\n'
    '6,1,383.522,120.100,35.437,74.368,0.95494,-1,-1,-1\n'     # y 120.1: y + 16 - 16 is not exact in floating point
    '6,4,520.000,320.000,50.000,50.000,0.50000,-1,-1,-1\n'     # inside the ignore region
    '7,2,129.182,100.349,40.509,119.713,0.94929,-1,-1,-1\n'
    '8,2,133.194,100.074,40.607,120.019,0.94833,-1,-1,-1\n')


def make_recording(root, name, frames=FRAMES, labels=LABELS):
    folder = Path(root, *name.split('/'))
    (folder / 'images').mkdir(parents=True)
    for number in range(1, frames + 1):
        (folder / 'images' / ('%06d.png' % number)).write_bytes(PNG_HEAD)
    if labels:
        (folder / 'tracks_edited_mot.txt').write_text(labels, encoding='utf-8')
        (folder / 'review_state.json').write_text(json.dumps(dict(reviewed=list(range(1, frames + 1)), removed=[])), encoding='utf-8')
    return folder / 'images'


def write_stored(tracks_root, name, key, text=STORED, frames=FRAMES, **changes):
    folder = Path(tracks_root, *name.split('/'))
    folder.mkdir(parents=True, exist_ok=True)
    raw = text.encode('utf-8')
    (folder / (key + '.mot.txt')).write_bytes(raw)
    meta = dict(label='MOTIP', what='test fixture', status='complete_range', error=None, frames_processed=frames, first_processed=1,
                last_processed=frames, covers_whole_sequence=True, sequence_total=frames, png_first=1, png_last=frames,
                source_wh=[640, 480], rows=len(text.splitlines()), mot_sha256=hashlib.sha256(raw).hexdigest(),
                settings=dict(DET_THRESH=0.3), timing_ms=dict(seconds_per_frame_mean=2.0))
    meta.update(changes)
    meta = {field: value for field, value in meta.items() if value != 'DROP'}
    (folder / (key + '.json')).write_text(json.dumps(meta), encoding='utf-8')
    return folder / (key + '.mot.txt'), folder / (key + '.json')


def read_mot(text):
    """An independent reader of the result file: {frame: [(id, x1, y1, x2, y2)]}."""
    rows = collections.defaultdict(list)
    for line in text.splitlines():
        f = line.split(',')
        x, y, w, h = float(f[2]), float(f[3]), float(f[4]), float(f[5])
        rows[int(f[0])].append((int(f[1]), x, y, x + w, y + h))
    return rows


def score_directly(rec_dir, name, pred, frames=FRAMES):
    """The numbers of one first pass, straight from the scoring classes (no replay, no canvas, no stabilization)."""
    lab = replay_eval.load_labels(str(rec_dir), name)
    new, old = gt_score.Score(True), gt_score.Score(False)
    new_h, old_h = gt_score.Score(True, False), gt_score.Score(False, False)
    inputs = []
    for png in range(1, frames + 1):
        gt, out, kept = lab['gt'].get(png, ()), pred.get(png, []), []
        new.frame(gt, out, 1, kept=kept)
        old.frame(gt, out, 1)
        new_h.frame(gt, out, 1)
        old_h.frame(gt, out, 1)
        people, _ = gt_score.split_gt(gt, True)
        inputs.append(([(int(gid), *box) for gid, box in people], [out[j] for j in kept], None))
    return dict(new=dict(new.result(), **replay_metrics.compute(inputs)), old=old.result(), new_h=new_h.result(), old_h=old_h.result())


class Fixture(unittest.TestCase):
    """A replay root with two recordings and (optionally) a stored-results root; the server state is faked."""
    tracks_option = True

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root, self.tracks, self.out = Path(tmp.name, 'rec'), Path(tmp.name, 'tracks'), Path(tmp.name, 'out')
        for folder in (self.root, self.tracks, self.out):
            folder.mkdir()
        self.images = make_recording(self.root, 'opi_sequences/seq_a')
        make_recording(self.root, 'opi_sequences/seq_b')
        self.live = NS(replay=None, stopped=False, cam_hold=False, replay_busy=False, proc=None,
                       sys_lock=threading.Lock(), cam_lock=threading.Lock())
        self.thread = Mock()
        args = NS(replay_root=str(self.root), replay_tracks_root=str(self.tracks) if self.tracks_option else '',
                  measure_dir='', live_wait=0, live_cam='camera', replay_bytetrack_library='')
        for item in (patch.dict(vars(APP), LIVE=self.live, args=args, lv_th=NS(Thread=self.thread), out=self.out,
                                LIVE_W=640, LIVE_H=480),
                     patch.dict(replay_link._C, t=-1e9, items=[], root=None),
                     patch.dict(replay_link._T, t=-1e9, key=None, items=None, found={}, sigs={}),
                     patch.object(replay_link, 'available_bytetrack_library', return_value=None),
                     patch.object(replay_link, 'set_preview_producer'),
                     patch.object(replay_link, '_gimbal_stop'),
                     patch.multiple(replay_eval, EV=None, LAST=None, FINAL=collections.OrderedDict())):
            item.start()
            self.addCleanup(item.stop)

    def start(self, seq, backend=None):
        query = {'seq': [seq]}
        if backend is not None:
            query['backend'] = [backend]
        return replay_link.start_request(query)

    def target(self):
        return self.thread.call_args.kwargs['args'][0]

    def play(self, ev, rows_of, detections_of=lambda png: None, tracker_detections=None, sc=None, frames=FRAMES):
        """Feed one first pass the way the frame loop does. -> [(gt, overlay rows)] per frame."""
        replay_eval.EV = ev
        seen = []
        for slot in range(frames):
            png, t = slot + 1, float(slot + 1)
            ev.tag(t, png, slot)
            seen.append(replay_eval.on_frame(t, rows_of(png), sc, 'image%d' % png, detections_of(png), tracker_detections))
        if ev.metric_worker is not None:
            ev.metric_worker.join(20)
        return seen


class StoredResultReplayTest(Fixture):
    def test_web_path_gives_the_numbers_of_direct_scoring(self):
        mot, _ = write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')
        self.assertEqual(self.start('opi_sequences/seq_a', 'pre:motip')[0], 200)
        target = self.target()
        self.assertEqual((target['backend'], target['tracks']['key'], target['tracks']['label']),
                         ('pre:motip', 'motip', 'MOTIP (저장된 결과)'))
        ev = replay_eval.prepare(target)
        self.assertIsNotNone(ev)
        # The server's own tracker rows and detector candidates are decoys: none of them may reach the numbers.
        decoy_rows = lambda png: [[10.0, 30.0, 60.0, 130.0, 77, 0.9, 0, 0]]
        decoy_detections = lambda png: [[100.0 + 4 * png, 116.0, 140.0 + 4 * png, 236.0, 0.9, 0]]
        shift = (3.0, -2.0)   # picture stabilization is on: stored boxes are real picture coordinates and must not be shifted for scoring
        seen = self.play(ev, decoy_rows, decoy_detections, tracker_detections=[[1, 2, 3, 4, 0.5, 0]], sc=shift)
        got = ev.summary()
        want = score_directly(self.images.parent, 'opi_sequences/seq_a', read_mot(STORED))
        self.assertEqual((got['phase'], got['counts']['scored'], got['metric_status'], got['native_frames']), ('complete', FRAMES, 'ready', FRAMES))
        self.assertEqual(got['new'], want['new'])
        self.assertEqual(got['old'], want['old'])
        self.assertEqual(got['per_frame_hungarian']['new'], want['new_h'])
        self.assertEqual(got['per_frame_hungarian']['old'], want['old_h'])
        # the fixture really exercises the rules (otherwise equal numbers would prove little)
        self.assertEqual((got['new']['idsw'], got['new']['false'], got['new']['ignored_out'], got['new']['missed']), (1, 1, 2, 4))
        self.assertGreater(got['old']['false'], got['new']['false'])
        self.assertIsNotNone(got['new']['hota'])
        # detector-candidate metrics are blank, and the summary says why
        self.assertEqual((got['new']['ap50'], got['new']['map'], got['detector_metrics']), (None, None, False))
        self.assertEqual((got['backend'], got['backend_label'], got['backend_error']), ('pre:motip', 'MOTIP (저장된 결과)', None))
        self.assertEqual(got['backend_meta']['mot_sha256'], hashlib.sha256(mot.read_bytes()).hexdigest())
        # the picture gets the stored boxes (canvas rows moved onto the stabilized picture), not the server tracker's rows
        gt6, overlay6 = seen[5]
        self.assertEqual(len(gt6), 3)
        self.assertEqual([row[4] for row in overlay6], [2, 1, 4])
        self.assertEqual(overlay6[0][:4], [125.897 + 3.0, 99.915 + 16.0 - 2.0, 125.897 + 40.491 + 3.0, 99.915 + 120.306 + 16.0 - 2.0])
        self.assertEqual(seen[3][1], [])   # frame 4: the file has no box, so the picture has none
        # the result file names the backend and carries its description
        name = ev.write('ended')
        saved = json.loads((self.out / name).read_text(encoding='utf-8'))
        self.assertEqual((saved['backend'], saved['backend_label'], saved['detector_metrics']), ('pre:motip', 'MOTIP (저장된 결과)', False))
        self.assertEqual((saved['backend_meta']['key'], saved['backend_meta']['frames'], saved['backend_meta']['rows'],
                          saved['backend_meta']['offline_run']['settings']), ('motip', FRAMES, 13, dict(DET_THRESH=0.3)))
        self.assertEqual(saved['new'], json.loads(json.dumps(want['new'])))
        self.assertIn('stored result file', saved['metric_scope'])

    def test_without_stabilization_the_picture_rows_are_the_stored_rows(self):
        mot, meta = write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')
        backend = replay_backends.PrecomputedTracks('motip', str(mot), str(meta), range(1, FRAMES + 1), 'opi_sequences/seq_a')
        self.assertEqual((backend.id, backend.label, backend.own_detector, backend.slow), ('pre:motip', 'MOTIP (저장된 결과)', True, False))
        rows = backend.update((5, 1, 4), None)
        self.assertEqual(rows, [[121.155, 100.445 + 16.0, 121.155 + 39.997, 100.445 + 120.804 + 16.0, 2, 0.9545, 0, -1],
                                [386.289, 104.818 + 16.0, 386.289 + 35.548, 104.818 + 90.279 + 16.0, 1, 0.93722, 0, -1]])
        self.assertEqual(backend.people(), [(2, 121.155, 100.445, 121.155 + 39.997, 100.445 + 120.804, 0.9545),
                                            (1, 386.289, 104.818, 386.289 + 35.548, 104.818 + 90.279, 0.93722)])
        self.assertEqual((backend.update((4, 1, 3), None), backend.people()), ([], []))
        backend.reset()   # stored rows have no state: a new pass or a seek shows the same boxes again
        self.assertEqual(backend.update((5, 2, 4), None), rows)
        with self.assertRaises(ValueError):
            backend.update((99, 1, 0), None)   # a PNG the result was not made for is an error, not "no boxes"
        backend.close()

    def test_a_png_outside_the_result_stops_scoring_instead_of_showing_nothing(self):
        write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')
        self.start('opi_sequences/seq_a', 'pre:motip')
        ev = replay_eval.prepare(self.target())
        replay_eval.EV = ev
        ev.tag(1.0, 1, 0)
        replay_eval.on_frame(1.0, [], None, 'image1', None, None)
        ev.tag(2.0, 500, 1)   # the folder changed under the replay
        _, overlay = replay_eval.on_frame(2.0, [], None, 'image500', None, None)
        state = ev.summary()
        self.assertEqual(overlay, [])
        self.assertEqual(state['phase'], 'incomplete')
        self.assertIn('PNG 500', state['backend_error'])

    def test_pause_repeat_and_new_pass_keep_showing_the_stored_rows(self):
        write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')
        self.start('opi_sequences/seq_a', 'pre:motip')
        ev = replay_eval.prepare(self.target())
        replay_eval.EV = ev
        ev.tag(1.0, 3, 2)
        first = replay_eval.on_frame(1.0, [], None, 'a', None, None)[1]
        ev.tag(2.0, 3, 2, 6)   # the same frame repeated while paused
        self.assertEqual(replay_eval.on_frame(2.0, [], None, 'b', None, None)[1], first)
        ev.tag(3.0, (1 << 20) | 3, 2)   # second pass
        self.assertEqual(replay_eval.on_frame(3.0, [], None, 'c', None, None)[1], first)
        self.assertEqual([row[4] for row in first], [0, 1, 5])
        # a frame whose PNG number is unknown gets no boxes; the server tracker's rows are not drawn in their place
        self.assertEqual(replay_eval.on_frame(4.0, [[1.0, 17.0, 9.0, 40.0, 3, 0.9, 0, 0]], None, 'd', None, None), (None, []))
        self.assertEqual(ev.summary()['counts']['untagged'], 1)


class StoredResultRefusalTest(Fixture):
    CASES = (
        ('still running', dict(status='running', frames_processed=3, last_processed=3, covers_whole_sequence=False), '끝까지 만든 결과가 아님'),
        ('stopped early', dict(status='stopped_early'), '끝까지 만든 결과가 아님'),
        ('no status', dict(status='DROP'), '끝까지 만든 결과가 아님'),
        ('failed', dict(error='frame 3 is mode RGB'), '오류'),
        ('a range, not the recording', dict(covers_whole_sequence=False), '녹화 전체'),
        ('fewer frames', dict(frames_processed=FRAMES - 1), '처리한 장 수가 녹화와 다름 (결과 7, 녹화 8)'),
        ('no frame count', dict(frames_processed='DROP'), 'frames_processed'),
        ('other first PNG', dict(first_processed=2), '첫 PNG 번호가 녹화와 다름'),
        ('other last PNG', dict(last_processed=FRAMES + 1), '마지막 PNG 번호가 녹화와 다름'),
        ('made for a longer recording', dict(sequence_total=FRAMES + 5), '전체 장 수가 녹화와 다름 (sequence_total 13, 녹화 8)'),
        ('other picture size', dict(source_wh=[1280, 960]), '640x480'),
        ('result file changed after the description', dict(mot_sha256='0' * 64), 'sha256'),
        ('no checksum', dict(mot_sha256='DROP'), 'mot_sha256'),
        ('row count differs', dict(rows=99), '줄 수'),
    )
    BAD_ROWS = (
        ('text instead of numbers', '1,0,abc,1,10,10,0.9,-1,-1,-1\n', '줄을 읽지 못함'),
        ('too few fields', '1,0,5,5,10,10\n', '줄을 읽지 못함'),
        ('not a number', '1,0,nan,5,10,10,0.9,-1,-1,-1\n', '줄을 읽지 못함'),
        ('empty box', '1,0,5,5,0,10,0.9,-1,-1,-1\n', '줄을 읽지 못함'),
        ('negative id', '1,-1,5,5,10,10,0.9,-1,-1,-1\n', '줄을 읽지 못함'),
        ('a PNG the recording does not have', '%d,0,5,5,10,10,0.9,-1,-1,-1\n' % (FRAMES + 1), '녹화에 없는 PNG'),
        ('one id twice in a frame', '1,0,5,5,10,10,0.9,-1,-1,-1\n1,0,50,50,10,10,0.9,-1,-1,-1\n', '두 번'),
    )

    def refused(self, key, why):
        """Refused everywhere: the loader, the list of the page, and the start request."""
        folder = self.tracks / 'opi_sequences' / 'seq_a'
        with self.assertRaises(replay_backends.Refused) as caught:
            replay_backends.PrecomputedTracks(key, str(folder / (key + '.mot.txt')), str(folder / (key + '.json')),
                                              replay_backends.png_numbers(str(self.images)))
        self.assertIn(why, str(caught.exception))
        usable, bad = replay_backends.scan_stored(str(folder), str(self.images))
        self.assertNotIn(key, [item['key'] for item in usable])
        self.assertIn(key, [item['key'] for item in bad])
        code, text = self.start('opi_sequences/seq_a', 'pre:' + key)
        self.assertEqual(code, 409)
        self.assertIn('저장된 결과를 쓸 수 없습니다', text)
        self.assertIn(why, text)
        self.assertFalse(self.live.cam_hold)
        self.thread.assert_not_called()

    def test_description_that_does_not_cover_the_recording(self):
        for number, (case, changes, why) in enumerate(self.CASES):
            with self.subTest(case):
                key = 'case%d' % number
                write_stored(self.tracks, 'opi_sequences/seq_a', key, **changes)
                self.refused(key, why)

    def test_unreadable_rows(self):
        for number, (case, text, why) in enumerate(self.BAD_ROWS):
            with self.subTest(case):
                key = 'rows%d' % number
                write_stored(self.tracks, 'opi_sequences/seq_a', key, text=STORED + text)
                self.refused(key, why)

    def test_recording_that_changed_after_the_result_was_made(self):
        write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')
        (self.images / ('%06d.png' % (FRAMES + 1))).write_bytes(PNG_HEAD)
        self.refused('motip', '처리한 장 수')

    def test_description_missing_or_not_json(self):
        mot, meta = write_stored(self.tracks, 'opi_sequences/seq_a', 'broken')
        meta.write_text('{"status": "complete_range"', encoding='utf-8')
        self.refused('broken', 'JSON')
        meta.unlink()
        usable, bad = replay_backends.scan_stored(str(mot.parent), str(self.images))
        self.assertEqual((usable, [item['key'] for item in bad]), ([], ['broken']))
        self.assertEqual(self.start('opi_sequences/seq_a', 'pre:broken'), (409, replay_link.NO_STORED))

    def test_partial_result_is_listed_as_unusable_and_never_as_a_tracker(self):
        write_stored(self.tracks, 'opi_sequences/seq_b', 'half', status='running', frames_processed=3, last_processed=3,
                     covers_whole_sequence=False)
        info = replay_link.get_info({})
        self.assertEqual([b['id'] for b in info['backends']], ['python'])
        item = next(x for x in info['list'] if x['name'] == 'opi_sequences/seq_b')
        self.assertEqual(item['tracks'], [])
        self.assertIn('half', item['tracks_note'])
        self.assertIn('끝까지 만든 결과가 아님', item['tracks_note'])

    def test_switch_does_not_fall_back_to_the_server_tracker(self):
        target = dict(name='opi_sequences/seq_a', backend='pre:motip', tracks=dict(key='motip', mot='gone', meta='gone', label='x'))
        self.live.cam_hold = self.live.replay_busy = True
        with patch.object(replay_link, '_end_old') as end_old, patch.object(replay_link, '_msg') as message:
            replay_link._switch(target)
        end_old.assert_not_called()   # the camera program was left alone
        self.assertIn('저장된 결과를 열지 못해', message.call_args.args[0])
        self.assertEqual((self.live.replay, self.live.cam_hold, self.live.replay_busy), (None, False, False))


class StoredResultOddFilesTest(Fixture):
    """Files a person may really put there: made for another recording, edited by hand, oddly named, broken."""
    NAME = 'opi_sequences/seq_a'

    def scan(self, name=NAME):
        return replay_backends.scan_stored(str(Path(self.tracks, *name.split('/'))), str(Path(self.root, *name.split('/'), 'images')), name)

    def test_result_made_from_another_recording_is_refused(self):
        # seq_a and seq_b have the same PNG numbers, so every count in the description fits both
        write_stored(self.tracks, self.NAME, 'motip', frames_dir='/pc/data/opi_sequences/seq_b/images')
        usable, bad = self.scan()
        self.assertEqual((usable, [item['key'] for item in bad]), ([], ['motip']))
        self.assertEqual(bad[0]['why'], '다른 녹화로 만든 결과 (결과 seq_b, 녹화 opi_sequences/seq_a)')
        self.assertEqual(self.start(self.NAME, 'pre:motip'), (409, '저장된 결과를 쓸 수 없습니다: ' + bad[0]['why']))
        self.thread.assert_not_called()
        # made from this recording (a Windows path of the PC that made it): usable, and the result file says where it came from
        write_stored(self.tracks, self.NAME, 'motip', frames_dir='C:\\data\\opi_sequences\\seq_a\\images')
        self.assertEqual([item['key'] for item in self.scan()[0]], ['motip'])
        self.assertEqual(self.start(self.NAME, 'pre:motip')[0], 200)
        ev = replay_eval.prepare(self.target())
        self.assertEqual((ev.backend_meta['source_recording'], ev.backend_meta['recording']), ('seq_a', self.NAME))
        # a description that does not say (no frames_dir, or not a .../<recording>/images path) is not checked
        for changes in ({}, dict(frames_dir='/pc/somewhere/frames'), dict(frames_dir=None)):
            mot, meta = write_stored(self.tracks, self.NAME, 'motip', **changes)
            item = replay_backends.PrecomputedTracks('motip', str(mot), str(meta), range(1, FRAMES + 1), self.NAME)
            self.assertEqual(item.meta()['source_recording'], '')

    def test_description_saved_with_a_bom_is_read(self):
        _, meta = write_stored(self.tracks, self.NAME, 'motip')
        meta.write_bytes(b'\xef\xbb\xbf' + meta.read_bytes())   # what Windows PowerShell 5.1 writes after a hand edit
        self.assertEqual(self.scan(), ([dict(id='pre:motip', key='motip', label='MOTIP (저장된 결과)')], []))
        self.assertEqual(self.start(self.NAME, 'pre:motip')[0], 200)

    def test_file_names_that_cannot_be_keys_are_reported_not_skipped(self):
        for key in ('motip', 'motip v2', '_motip', '._motip'):
            write_stored(self.tracks, self.NAME, key)
        usable, bad = self.scan()
        self.assertEqual([item['key'] for item in usable], ['motip'])
        self.assertEqual([(item['key'], item['why']) for item in bad], [('_motip', replay_backends.BAD_KEY), ('motip v2', replay_backends.BAD_KEY)])
        self.assertIn('쓸 수 없는 글자', replay_backends.BAD_KEY)
        item = next(x for x in replay_link.get_info({})['list'] if x['name'] == self.NAME)
        self.assertEqual(item['tracks'], [dict(id='pre:motip', label='MOTIP (저장된 결과)')])
        self.assertEqual(item['tracks_note'], '저장된 결과 _motip 못 씀: %s (외 1개)' % replay_backends.BAD_KEY)
        self.assertEqual(self.start(self.NAME, 'pre:_motip')[0], 400)   # such a name can never be started
        # only an oddly named file in the folder: the reason is still shown
        write_stored(self.tracks, 'opi_sequences/seq_b', 'my result')
        self.assertEqual(self.scan('opi_sequences/seq_b'), ([], [dict(key='my result', why=replay_backends.BAD_KEY)]))

    def test_one_broken_description_does_not_take_the_whole_list_away(self):
        write_stored(self.tracks, self.NAME, 'motip')
        _, deep = write_stored(self.tracks, 'opi_sequences/seq_b', 'deep')
        deep.write_text('[' * 100000, encoding='utf-8')   # json.loads gives RecursionError, not ValueError
        write_stored(self.tracks, 'opi_sequences/seq_b', 'mark', status='\ud800run\x00ning')   # a lone surrogate echoed in the reason
        write_stored(self.tracks, 'opi_sequences/seq_b', 'total', sequence_total='\udfff9')
        info = replay_link.get_info({})
        body = json.dumps(info, ensure_ascii=False).encode('utf-8')   # what GET /replay sends; a lone surrogate would raise here
        self.assertIn('pre:motip', body.decode('utf-8'))
        by_name = {item['name']: item for item in info['list']}
        self.assertEqual(by_name[self.NAME]['tracks'], [dict(id='pre:motip', label='MOTIP (저장된 결과)')])
        self.assertEqual(by_name['opi_sequences/seq_b']['tracks'], [])
        self.assertEqual(by_name['opi_sequences/seq_b']['tracks_note'], '저장된 결과 deep 못 씀: 설명 파일이 올바른 JSON 이 아님 (외 2개)')
        self.assertEqual(self.scan('opi_sequences/seq_b')[1][1:],
                         [dict(key='mark', why='끝까지 만든 결과가 아님 (status running)'),
                          dict(key='total', why='전체 장 수가 녹화와 다름 (sequence_total 9, 녹화 8)')])
        for key, why in (('deep', 'JSON'), ('mark', 'status running'), ('total', 'sequence_total 9')):
            code, text = self.start('opi_sequences/seq_b', 'pre:' + key)
            self.assertEqual(code, 409)
            self.assertIn(why, text)
            json.dumps(dict(ok=False, text=text), ensure_ascii=False).encode('utf-8')
        self.thread.assert_not_called()
        # a failure nobody thought of stays with that one file
        with patch.object(replay_backends, 'PrecomputedTracks', side_effect=KeyError('surprise')):
            self.assertEqual(self.scan(), ([], [dict(key='motip', why='읽지 못함 (KeyError)')]))
            self.assertEqual(self.start(self.NAME, 'pre:motip'), (409, '저장된 결과를 쓸 수 없습니다 (KeyError)'))

    def test_value_that_cannot_be_written_as_utf8_is_not_copied_to_the_result_file(self):
        mot, meta = write_stored(self.tracks, self.NAME, 'motip', what='made on \ud800 PC', label='MO\ud800TIP')
        item = replay_backends.PrecomputedTracks('motip', str(mot), str(meta), range(1, FRAMES + 1), self.NAME)
        self.assertEqual(item.label, 'MOTIP (저장된 결과)')
        self.assertNotIn('what', item.meta()['offline_run'])
        self.assertEqual(item.meta()['offline_run']['settings'], dict(DET_THRESH=0.3))
        json.dumps(item.meta(), ensure_ascii=False).encode('utf-8')

    def test_file_that_cannot_be_read_or_is_too_big(self):
        mot, meta = write_stored(self.tracks, self.NAME, 'motip')
        with self.assertRaises(replay_backends.Refused) as caught:
            replay_backends.PrecomputedTracks('motip', str(mot), str(meta.with_name('gone.json')), range(1, FRAMES + 1))
        self.assertEqual(str(caught.exception), '설명 파일을 읽지 못함 (FileNotFoundError)')
        with patch.object(replay_backends, 'MOT_MAX_BYTES', 10), self.assertRaises(replay_backends.Refused) as caught:
            replay_backends.PrecomputedTracks('motip', str(mot), str(meta), range(1, FRAMES + 1))
        self.assertEqual(str(caught.exception), '결과 파일이 너무 큼')


class StoredResultListCacheTest(Fixture):
    """The page asks GET /replay all the time (also during live camera tracking): unchanged files must not be read again."""

    def minute_later(self, query=None):
        replay_link._C['t'] = replay_link._T['t'] = -1e9   # both 60 s memories have run out: the recording list is a new list
        return {item['name']: item for item in replay_link.get_info(query or {})['list']}

    def test_unchanged_files_are_read_once_and_changed_ones_again(self):
        a, b = 'opi_sequences/seq_a', 'opi_sequences/seq_b'
        mot, meta = write_stored(self.tracks, a, 'motip')
        write_stored(self.tracks, b, 'half', status='running')
        with patch.object(replay_backends, 'scan_stored', wraps=replay_backends.scan_stored) as scan, \
                patch.object(replay_backends, 'PrecomputedTracks', wraps=replay_backends.PrecomputedTracks) as loaded:
            first = self.minute_later()
            self.assertEqual((scan.call_count, loaded.call_count), (2, 2))
            before = replay_link.listing()
            for _ in range(3):
                self.assertEqual(self.minute_later(), first)
            self.assertIsNot(replay_link.listing(), before)   # the recordings were really listed anew each time
            self.assertEqual((scan.call_count, loaded.call_count), (2, 2))   # ... but no stored result was read again
            self.assertEqual(first[a]['tracks'], [dict(id='pre:motip', label='MOTIP (저장된 결과)')])
            self.assertIn('끝까지 만든 결과가 아님', first[b]['tracks_note'])
            # a changed description: only that recording is read again
            write_stored(self.tracks, a, 'motip', label='MOTIP v2')
            self.assertEqual(self.minute_later()[a]['tracks'], [dict(id='pre:motip', label='MOTIP v2 (저장된 결과)')])
            self.assertEqual([call.args[2] for call in scan.call_args_list[2:]], [a])
            # a new result next to it
            write_stored(self.tracks, a, 'second', label='DROP')
            self.assertEqual([t['id'] for t in self.minute_later()[a]['tracks']], ['pre:motip', 'pre:second'])
            self.assertEqual([call.args[2] for call in scan.call_args_list[3:]], [a])
            # the recording got one more PNG: its results no longer cover it
            (self.images / ('%06d.png' % (FRAMES + 1))).write_bytes(PNG_HEAD)
            now = self.minute_later()
            self.assertEqual(now[a]['tracks'], [])
            self.assertIn('처리한 장 수가 녹화와 다름', now[a]['tracks_note'])
            self.assertEqual([call.args[2] for call in scan.call_args_list[4:]], [a])
            (self.images / ('%06d.png' % (FRAMES + 1))).unlink()
            self.assertEqual(len(self.minute_later()[a]['tracks']), 2)
            # results taken away
            for path in list(mot.parent.iterdir()):
                path.unlink()
            self.assertNotIn('tracks', self.minute_later()[a])
            count = scan.call_count
            self.assertNotIn('tracks', self.minute_later()[a])
            self.assertEqual(scan.call_count, count)   # the empty folder is remembered too
            # refresh=1 (opening the replay section) reads everything again
            self.assertIn('tracks_note', self.minute_later({'refresh': ['1']})[b])
            self.assertEqual([call.args[2] for call in scan.call_args_list[count:]], [a, b])

    def test_same_size_and_time_is_not_noticed_in_the_list_but_is_refused_at_start(self):
        name = 'opi_sequences/seq_a'
        mot, _ = write_stored(self.tracks, name, 'motip')
        self.assertEqual(len(self.minute_later()[name]['tracks']), 1)
        stat = mot.stat()
        mot.write_bytes(mot.read_bytes().replace(b'105.250', b'205.250'))   # same length
        os.utime(mot, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual((mot.stat().st_size, mot.stat().st_mtime_ns), (stat.st_size, stat.st_mtime_ns))
        self.assertEqual(len(self.minute_later()[name]['tracks']), 1)   # the known limit of the memory: still listed ...
        code, text = self.start(name, 'pre:motip')
        self.assertEqual(code, 409)   # ... but never played: starting always reads and checks the files again
        self.assertIn('sha256', text)
        self.assertEqual(self.minute_later({'refresh': ['1']})[name]['tracks'], [])


class StoredResultRouteTest(Fixture):
    def test_info_lists_stored_results_per_recording(self):
        write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')
        write_stored(self.tracks, 'opi_sequences/seq_a', 'other', label='DROP')
        info = replay_link.get_info({})
        self.assertEqual(info['backends'], [dict(id='python', label='현재 추적기'),
                                            dict(id='pre:motip', label='MOTIP (저장된 결과)', stored=True),
                                            dict(id='pre:other', label='other (저장된 결과)', stored=True)])
        by_name = {item['name']: item for item in info['list']}
        self.assertEqual(by_name['opi_sequences/seq_a']['tracks'], [dict(id='pre:motip', label='MOTIP (저장된 결과)'),
                                                                   dict(id='pre:other', label='other (저장된 결과)')])
        self.assertNotIn('tracks', by_name['opi_sequences/seq_b'])
        self.assertNotIn('tracks_note', by_name['opi_sequences/seq_a'])
        self.assertNotIn('tracks', replay_link.listing()[0])   # the cached list of recordings itself is not changed

    def test_recording_without_that_result_gets_409(self):
        write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')
        self.assertEqual(self.start('opi_sequences/seq_b', 'pre:motip'), (409, '이 녹화에는 저장된 결과가 없습니다'))
        self.assertEqual(self.start('opi_sequences/seq_a', 'pre:nothing'), (409, '이 녹화에는 저장된 결과가 없습니다'))
        self.assertEqual(self.start('opi_sequences/seq_a', 'pre:../seq_a/motip')[0], 400)
        self.assertFalse(self.live.cam_hold)
        self.thread.assert_not_called()

    def test_restart_and_state_keep_the_stored_backend(self):
        write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')
        self.start('opi_sequences/seq_a', 'pre:motip')
        self.live.replay = dict(self.target(), fin0=0)
        self.live.cam_hold = self.live.replay_busy = False
        self.live.frames_in = 0
        self.assertEqual(replay_link.top_text(self.live.replay), 'REPLAY opi_sequences/seq_a | STORED motip')
        state = replay_link.state()
        self.assertEqual((state['backend'], state['backend_label']), ('pre:motip', 'MOTIP (저장된 결과)'))
        with patch.object(replay_link, 'start_request', return_value=(200, 'ok')) as start:
            replay_link.restart_request()
        self.assertEqual(start.call_args.args[0]['backend'], ['pre:motip'])


class OptionUnsetTest(Fixture):
    tracks_option = False

    def test_nothing_new_is_listed_or_accepted(self):
        write_stored(self.tracks, 'opi_sequences/seq_a', 'motip')   # the files exist, the option is not given
        info = replay_link.get_info({})
        self.assertEqual(info['backends'], [dict(id='python', label='현재 추적기')])
        self.assertIs(info['list'], replay_link.listing())
        self.assertFalse(any('tracks' in item or 'tracks_note' in item for item in info['list']))
        self.assertEqual(self.start('opi_sequences/seq_a', 'pre:motip'), (400, '알 수 없는 재생 추적기'))
        self.assertEqual(self.start('opi_sequences/seq_a', 'nonsense'), (400, '알 수 없는 재생 추적기'))
        self.thread.assert_not_called()

    def test_python_target_and_names_are_as_before(self):
        self.assertEqual(self.start('opi_sequences/seq_a'), (200, '재생 시작: opi_sequences/seq_a'))
        target = self.target()
        self.assertEqual(set(target), {'name', 'dir', 'fps', 'loop', 'bits', 'frames', 'fin0', 'paused', 'backend',
                                       'tracker_library', 'preview', 'pacing'})
        self.assertEqual((target['backend'], target['tracker_library']), ('python', None))
        self.assertEqual(replay_link.top_text(target), 'REPLAY opi_sequences/seq_a')
        self.live.replay = dict(target, fin0=0)
        self.live.frames_in = 0
        self.assertNotIn('backend_label', replay_link.state())

    def test_missing_folder_is_the_same_as_no_option(self):
        with patch.object(APP.args, 'replay_tracks_root', str(self.tracks / 'not_there')):
            self.assertIsNone(replay_link.tracks_root())
            self.assertEqual(self.start('opi_sequences/seq_a', 'pre:motip')[0], 400)


class OlderBackendsUnchangedTest(Fixture):
    tracks_option = False

    def target(self, **more):
        return dict(name='opi_sequences/seq_a', fps=25, loop=False, dir=str(self.images), frames=FRAMES, **more)

    def test_python_rows_are_scored_with_canvas_offset_and_stabilization(self):
        ev = replay_eval.Eval(self.target())
        self.assertIsNone(ev.native_tracker)
        shift = (3.0, -2.0)
        pred = read_mot(STORED)
        # the server's tracker rows: canvas coordinates of the stabilized picture
        rows_of = lambda png: [[b[1] + shift[0], b[2] + 16.0 + shift[1], b[3] + shift[0], b[4] + 16.0 + shift[1], b[0], 0.9, 0, 0]
                               for b in pred.get(png, [])] + [[5.0, 5.0, 9.0, 9.0, 8, 0.9, 2, 0]]   # a non-person row is ignored
        candidates = lambda png: [[100.0 + 4 * png + shift[0], 116.0 + shift[1], 140.0 + 4 * png + shift[0], 236.0 + shift[1], 0.9, 0]]
        seen = self.play(ev, rows_of, candidates, sc=shift)
        self.assertTrue(all(overlay is None for _, overlay in seen))   # None = the page draws the server's rows
        got = ev.summary()
        # the same people after the same conversion the scorer makes (minus 16, then back to the real picture)
        back = lambda png: [(p[0], *p[1:5]) for p in replay_eval.stab_people_real(
            [(int(r[4]), float(r[0]), float(r[1]) - 16.0, float(r[2]), float(r[3]) - 16.0, float(r[5])) for r in rows_of(png) if int(r[6]) == 0],
            shift, 640.0, 480.0)]
        want = score_directly(self.images.parent, 'opi_sequences/seq_a', {png: back(png) for png in range(1, FRAMES + 1)})
        for field in ('ap50', 'map'):
            self.assertIsNotNone(got['new'].pop(field))   # the detector candidates still give AP for the server's own tracker
            want['new'].pop(field)
        self.assertEqual(got['new'], want['new'])
        self.assertEqual(got['old'], want['old'])
        self.assertEqual((got['backend'], got['phase'], got['native_frames']), ('python', 'complete', 0))
        for field in ('backend_label', 'backend_meta', 'detector_metrics'):
            self.assertNotIn(field, got)
        saved = json.loads((self.out / ev.write('ended')).read_text(encoding='utf-8'))
        self.assertIn('AP50 and mAP50:95 use person detections', saved['metric_scope'])
        self.assertNotIn('backend_meta', saved)

    def test_cpp_bytetrack_gets_the_same_calls_as_before(self):
        calls = []

        class FakeNative:
            def __init__(self, library):
                calls.append(('open', library))

            def reset(self):
                calls.append(('reset',))

            def update(self, rows):
                calls.append(('update', rows))
                if rows == 'fail':
                    raise RuntimeError('C++ ByteTrack step failed (-1)')
                if rows == 'bug':
                    raise KeyError('not a tracker error')
                return [[100.0, 116.0, 140.0, 236.0, 7, 0.9, 0, 0]]

            def close(self):
                calls.append(('close',))

        with patch.object(replay_backends, 'NativeByteTrack', FakeNative):
            ev = replay_eval.Eval(self.target(backend='cpp_bytetrack', tracker_library='lib.so'))
            self.assertIsInstance(ev.native_tracker, replay_backends.CppByteTrackBackend)
            self.assertEqual((ev.own_boxes, ev.backend_meta), (False, {}))
            replay_eval.EV = ev
            detections = object()
            ev.tag(1.0, 1, 0)
            gt, overlay = replay_eval.on_frame(1.0, [[0, 0, 1, 1, 99, 0.9, 0, 0]], (3.0, -2.0), 'one', [], detections)
            self.assertEqual(overlay, [[100.0, 116.0, 140.0, 236.0, 7, 0.9, 0, 0]])   # C++ rows as they are (not moved), scored instead of the server's
            self.assertEqual(calls, [('open', 'lib.so'), ('update', detections)])
            ev.tag(2.0, 1, 0, 6)   # repeat of the frame just tracked: last rows, no update
            self.assertEqual(replay_eval.on_frame(2.0, [], None, 'rep', [], detections)[1], overlay)
            ev.tag(3.0, (1 << 20) | 1, 0)   # a new pass resets first
            replay_eval.on_frame(3.0, [], None, 'two', [], detections)
            self.assertEqual(calls[2:], [('reset',), ('update', detections)])
            state = ev.summary()
            self.assertEqual((state['backend'], state['backend_error'], state['native_frames']), ('cpp_bytetrack', None, 2))
            self.assertEqual((state['new']['found'], state['new']['false']), (1, 0))   # the C++ row was scored (the server's own row would be a false box)
            for field in ('backend_label', 'backend_meta', 'detector_metrics'):
                self.assertNotIn(field, state)
            ev.tag(4.0, (1 << 20) | 2, 1)
            with self.assertRaises(KeyError):   # only OSError / RuntimeError / ValueError are tracker failures, as before
                replay_eval.on_frame(4.0, [], None, 'bug', [], 'bug')
            ev.tag(5.0, (1 << 20) | 3, 2)
            self.assertEqual(replay_eval.on_frame(5.0, [], None, 'fail', [], 'fail')[1], [])
            self.assertEqual(calls[-2:], [('update', 'fail'), ('close',)])
            self.assertEqual(ev.summary()['backend_error'], 'C++ ByteTrack step failed (-1)')
            ev.tag(6.0, (1 << 20) | 4, 3)
            self.assertEqual(replay_eval.on_frame(6.0, [], None, 'after', [], detections)[1], [])   # stays off after a failure
            self.assertEqual(calls[-1], ('close',))

    def test_factory_knows_only_real_backends(self):
        with self.assertRaises(ValueError):
            replay_backends.make(dict(backend='pre:motip', dir=str(self.images)))   # no stored files were resolved for it
        with self.assertRaises(ValueError):
            replay_backends.make(dict(backend='something'))
        self.assertEqual([replay_backends.stored_key(text) for text in ('pre:motip', 'pre:a.b-c_1', 'pre:', 'pre:../x', 'pre:a/b', 'python', None)],
                         ['motip', 'a.b-c_1', None, None, None, None, None])


if __name__ == '__main__':
    unittest.main()
