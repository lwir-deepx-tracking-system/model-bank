"""HOTA / AP of a replayed recording: same numbers as before 11-15, computed off the frame loop's interpreter lock.

11-15 on the board: after a recording ended, the frame loop ran at 6 instead of 25 frames a second and the picture came
0.9 s late for about 20 s (stall log: 'GIL_STALL ... replay-metrics ... replay_metrics.py _ap_for_iou'). Two changes:
  - replay_metrics._ap_* does the work shared by the ten IoU thresholds once (checked here against the code it replaced,
    kept below as the reference: every number must be equal, not almost equal),
  - replay_metrics_proc.compute runs it in a child process (same numbers; the old in-thread way when the child fails).

Run from the tracking_server folder: python3 -m unittest src.track.test_replay_metrics_proc
"""
import os
import random
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from . import replay_metrics, replay_metrics_proc


def reference_ap_for_iou(frames, threshold):
    """The AP of one IoU threshold exactly as replay_metrics computed it up to 11-15 (one IoU call per candidate)."""
    total_gt = sum(len(gt) for gt, _, _ in frames)
    if total_gt == 0:
        return None
    scored = []
    for frame_index, (gt, _, det) in enumerate(frames):
        if not len(det):
            continue
        order = np.argsort(-det[:, 0], kind="stable")[:100]
        for local_rank, di in enumerate(order):
            scored.append((float(det[di, 0]), frame_index, local_rank, det[di, 1:5]))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched = [np.zeros(len(gt), dtype=bool) for gt, _, _ in frames]
    tp = np.zeros(len(scored), dtype=np.float64)
    fp = np.zeros(len(scored), dtype=np.float64)
    for i, (_, fi, _, box) in enumerate(scored):
        gt = frames[fi][0]
        if len(gt):
            ious = replay_metrics._iou(box[None, :], gt[:, 1:5])[0]
            eligible = np.flatnonzero((ious >= threshold - np.finfo(float).eps) & ~matched[fi])
            if len(eligible):
                best = eligible[len(eligible) - 1 - np.argmax(ious[eligible[::-1]])]
                matched[fi][best] = True
                tp[i] = 1.0
                continue
        fp[i] = 1.0
    if not len(scored):
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, replay_metrics._RECALL_SAMPLES, side="left")
    sampled = np.zeros_like(replay_metrics._RECALL_SAMPLES)
    valid = indices < len(precision_envelope)
    sampled[valid] = precision_envelope[indices[valid]]
    return float(sampled.mean())


def reference_ap_metrics(frames):
    if any(det is None for _, _, det in frames):
        return None, None
    ap50 = reference_ap_for_iou(frames, 0.50)
    aps = [reference_ap_for_iou(frames, float(t)) for t in replay_metrics._AP_IOUS]
    if ap50 is None or any(v is None for v in aps):
        return None, None
    return 100.0 * ap50, 100.0 * float(np.mean(aps))


def recording(seed, *, frames=None, people=None, clutter=None, ties=False):
    """A made-up recording in the form replay_eval hands over: [(gt rows, track rows, candidate rows or None)].
    People drift, the tracker follows most of them (one number changes half way), the detector adds low-score clutter
    as the pose models do (100-300 boxes a frame on the board). ties=True: rounded scores, exact copies of GT boxes
    and frames without candidates, for the equal-score and equal-IoU rules."""
    r = random.Random(seed)
    frames = r.randint(1, 30) if frames is None else frames
    people = r.randint(0, 6) if people is None else people
    clutter = r.choice([0, 3, 30, 130]) if clutter is None else clutter
    place = [[r.uniform(50, 500), r.uniform(50, 350), r.uniform(20, 60), r.uniform(60, 160)] for _ in range(people)]
    out = []
    for f in range(frames):
        gt, tracks, dets = [], [], []
        for i, p in enumerate(place):
            p[0] += r.uniform(-3, 3); p[1] += r.uniform(-2, 2)
            if r.random() < 0.9:
                gt.append((i + 1, p[0], p[1], p[0] + p[2], p[1] + p[3]))
            if r.random() < 0.85:
                tracks.append((100 + i if f < frames // 2 or i else 200, p[0] + r.uniform(-4, 4), p[1] + r.uniform(-4, 4),
                               p[0] + p[2] + r.uniform(-4, 4), p[1] + p[3] + r.uniform(-4, 4)))
            for _ in range(r.randint(0, 3)):
                score = round(r.uniform(0.05, 0.95), 2) if ties else r.uniform(0.05, 0.95)
                dets.append((score, p[0] + r.uniform(-8, 8), p[1] + r.uniform(-8, 8),
                             p[0] + p[2] + r.uniform(-8, 8), p[1] + p[3] + r.uniform(-8, 8)))
            if ties and gt and r.random() < 0.3:
                dets += [(0.5, *gt[-1][1:]), (0.5, *gt[-1][1:])]
        for _ in range(clutter):
            x, y = r.uniform(0, 600), r.uniform(0, 440)
            score = round(r.uniform(0.001, 0.2), 3) if ties else r.uniform(0.001, 0.2)
            dets.append((score, x, y, x + r.uniform(5, 40), y + r.uniform(10, 80)))
        r.shuffle(dets)
        out.append((gt, tracks, [] if ties and f % 7 == 6 else dets))
    return out


def arrays(frames):
    return [(replay_metrics._rows(gt, 5, 'gt'), replay_metrics._rows(tr, 5, 'tracks'),
             None if det is None else replay_metrics._rows(det, 5, 'detections')) for gt, tr, det in frames]


class SameApAsBefore(unittest.TestCase):
    def test_every_number_equals_the_replaced_code(self):
        for seed in range(30):
            frames = recording(seed, ties=seed % 2 == 0)
            if seed % 7 == 3:
                frames = [(gt, tr, None) for gt, tr, _ in frames]      # a stored result: no candidates
            if seed % 11 == 5:
                frames = [([], tr, det) for _, tr, det in frames]      # nobody labelled
            with self.subTest(seed=seed):
                self.assertEqual(replay_metrics._ap_metrics(arrays(frames)), reference_ap_metrics(arrays(frames)))

    def test_each_threshold_alone(self):
        frames = arrays(recording(4, frames=25, people=5, clutter=130, ties=True))
        prepared = replay_metrics._ap_prepare(frames)
        for threshold in [0.50] + [float(t) for t in replay_metrics._AP_IOUS]:
            self.assertEqual(replay_metrics._ap_for_iou(prepared, threshold), reference_ap_for_iou(frames, threshold))

    def test_equal_iou_takes_the_later_ground_truth_like_coco(self):
        same = [(1, 0, 0, 10, 10), (2, 0, 0, 10, 10), (3, 0, 0, 10, 20)]
        frames = arrays([(same, [], [(0.9, 0, 0, 10, 10), (0.8, 0, 0, 10, 10), (0.7, 0, 0, 10, 10)])])
        self.assertEqual(replay_metrics._ap_metrics(frames), reference_ap_metrics(frames))

    def test_more_than_a_hundred_candidates_in_a_frame(self):
        frames = arrays(recording(9, frames=3, people=4, clutter=260))
        self.assertTrue(all(len(det) > 100 for _, _, det in frames))
        self.assertEqual(replay_metrics._ap_metrics(frames), reference_ap_metrics(frames))

    def test_board_sized_recording_is_fast(self):
        # 370 frames with 190 clutter boxes took 36 s on a PC (about 20 s on the board, in the server) before 11-15.
        frames = recording(1, frames=120, people=3, clutter=190)
        started = time.perf_counter()
        replay_metrics.compute(frames)
        self.assertLess(time.perf_counter() - started, 6.0)


class ChildProcess(unittest.TestCase):
    def setUp(self):
        self.before = dict(replay_metrics_proc.STATS)
        patcher = patch.dict(os.environ)
        patcher.start(); self.addCleanup(patcher.stop)
        os.environ.pop('LWIR_REPLAY_METRICS_PROC', None)

    def count(self, key):
        return replay_metrics_proc.STATS[key] - self.before[key]

    def test_child_gives_the_same_numbers(self):
        for seed in (2, 3, 5, 12):
            frames = recording(seed, ties=seed % 2 == 0)
            if seed == 3:
                frames = [(gt, tr, None) for gt, tr, _ in frames]
            with self.subTest(seed=seed):
                self.assertEqual(replay_metrics_proc.compute(frames), replay_metrics.compute(frames))
        self.assertEqual((self.count('child'), self.count('inline'), self.count('failed')), (4, 0, 0))

    def test_numpy_numbers_and_tuples_go_through(self):
        frames = [([(np.int64(1), np.float32(0), 0.0, 10.0, 10.0)], ((7, 0.0, 0.0, 10.0, np.float64(10)),),
                   np.array([[0.9, 0, 0, 10, 10]]))]
        self.assertEqual(replay_metrics_proc.compute(frames), replay_metrics.compute(frames))
        self.assertEqual(self.count('child'), 1)

    def test_the_work_is_done_outside_this_process(self):
        frames = recording(1, frames=40, people=3, clutter=150)
        seen = []
        real = subprocess.run

        def watch(command, **kwargs):
            seen.append((command, threading.current_thread()))
            return real(command, **kwargs)

        with patch.object(replay_metrics_proc.subprocess, 'run', side_effect=watch), \
             patch.object(replay_metrics_proc, '_inline', side_effect=AssertionError('computed in the server')):
            result = replay_metrics_proc.compute(frames)
        self.assertEqual(result, replay_metrics.compute(frames))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0][1:], ['-B', os.path.abspath(replay_metrics_proc.__file__)])

    def test_switch_off_computes_in_the_caller(self):
        os.environ['LWIR_REPLAY_METRICS_PROC'] = '0'
        frames = recording(6)
        with patch.object(replay_metrics_proc.subprocess, 'run', side_effect=AssertionError('child started')):
            self.assertEqual(replay_metrics_proc.compute(frames), replay_metrics.compute(frames))
        self.assertEqual((self.count('child'), self.count('inline')), (0, 1))

    def test_a_child_that_cannot_start_falls_back(self):
        frames = recording(8)
        with patch.object(replay_metrics_proc.sys, 'executable', os.path.join(os.path.dirname(__file__), 'no_such_python')), \
             patch('builtins.print') as said:
            self.assertEqual(replay_metrics_proc.compute(frames), replay_metrics.compute(frames))
        self.assertEqual((self.count('child'), self.count('inline'), self.count('failed')), (0, 1, 1))
        self.assertIn('child process failed', said.call_args.args[0])

    def test_a_dead_or_slow_child_falls_back(self):
        frames = recording(10)
        dead = subprocess.CompletedProcess([], 1, b'', b'Killed')
        garbage = subprocess.CompletedProcess([], 0, b'not a pickle', b'')
        slow = subprocess.TimeoutExpired('python', 300)
        for answer in (dead, garbage, slow):
            with self.subTest(answer=type(answer).__name__), patch('builtins.print'), \
                 patch.object(replay_metrics_proc.subprocess, 'run',
                              side_effect=[answer] if isinstance(answer, Exception) else None, return_value=answer):
                self.assertEqual(replay_metrics_proc.compute(frames), replay_metrics.compute(frames))
        self.assertEqual((self.count('child'), self.count('failed')), (0, 3))

    def test_bad_input_raises_as_before(self):
        with self.assertRaises(ValueError):
            replay_metrics_proc.compute([([(1, 0, 0, 10)], [], [])])      # a GT row of four numbers
        with self.assertRaises(ValueError), patch('builtins.print'):
            replay_metrics_proc.compute([([], [])])                       # not (gt, tracks, detections)

    def test_replay_scoring_uses_it(self):
        from ..camera import replay_eval
        self.assertIs(replay_eval.replay_metrics_proc, replay_metrics_proc)


if __name__ == "__main__":
    unittest.main()
