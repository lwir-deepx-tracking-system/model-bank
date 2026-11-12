"""Synthetic regressions for the offline v27 analysis."""
import unittest

import analyze_v27 as analysis


def frame(t, *, x=330, y=260, vis=1, lost=0):
    return dict(name='walk_v27_0', phase='run', t=str(t), ns=str(1_000_000_000 + round(t * 1e9)),
                ximg=str(x), yimg=str(y), vis=str(vis), lost=str(lost), sx='0', sy='0', resp='1')


def block(rows, skip=0):
    return analysis.block_values('walk_v27_0', rows, [], skip)


class BlockMetricsTests(unittest.TestCase):
    def test_loss_uses_elapsed_time_and_preserves_both_axes(self):
        result = block([
            frame(0), frame(.1, x=''), frame(.8, y=''), frame(1),
            frame(1.05, x=350, y=280, lost=1), frame(1.2, x='', y=''),
        ], skip=.5)

        self.assertEqual((result['n_run'], result['n_observed'], result['n_lost']), (6, 3, 4))
        self.assertEqual((result['n_scored'], result['n'], result['n_lost_scored']), (4, 2, 3))
        self.assertAlmostEqual(result['attempt_time_s'], 1.2)
        self.assertAlmostEqual(result['lost_time_s'], 1.05)
        self.assertAlmostEqual(result['lost_time_ratio'], .875)
        self.assertAlmostEqual(result['longest_lost_s'], .9)
        self.assertEqual(result['C_lost'], .75)
        self.assertEqual((result['median_abs_dx'], result['p95_abs_dx']), (20, 29))
        self.assertEqual((result['median_abs_dy'], result['p95_abs_dy']), (30, 39))
        self.assertIsNotNone(result['invalid_reason'])

    def test_final_sample_contributes_no_time_without_next_timestamp(self):
        result = block([frame(2), frame(2.3, vis=0)])
        self.assertEqual(result['C_lost'], .5)
        self.assertEqual(result['attempt_time_s'], .3)
        self.assertEqual(result['lost_time_ratio'], 0)
        self.assertEqual(result['longest_lost_s'], 0)

    def test_entirely_missing_observations_still_have_measured_loss(self):
        result = block([frame(i / 20, x='', y='') for i in range(21)])
        self.assertEqual(result['n_observed'], 0)
        self.assertIsNone(result['median_abs_dx'])
        self.assertIsNone(result['B_shake'])
        self.assertEqual(result['C_lost'], 1)
        self.assertEqual(result['lost_time_ratio'], 1)
        self.assertEqual(result['longest_lost_s'], 1)

    def test_missing_duplicate_or_reversed_timestamps_cannot_fake_a_score(self):
        for times in ([], [0], [0, 0], [.5, .25]):
            with self.subTest(times=times):
                result = block([frame(t) for t in times])
                self.assertIsNone(result['lost_time_ratio'])
                self.assertIsNone(result['longest_lost_s'])
                self.assertIsNotNone(result['invalid_reason'])
        second = frame(1)
        second['ns'] = ''
        self.assertIsNone(block([frame(0), second])['attempt_time_s'])

    def test_shake_does_not_bridge_missed_intervals(self):
        result = block([frame(i / 20, vis=i % 2) for i in range(40)])
        self.assertEqual(result['n'], 20)
        self.assertEqual(result['n_shake'], 0)
        self.assertIsNone(result['B_shake'])
        self.assertIsNotNone(result['invalid_reason'])

    def test_complete_observed_run_keeps_legacy_a_b_c_keys(self):
        result = block([frame(i / 20) for i in range(61)], skip=.5)
        self.assertIsNone(result['invalid_reason'])
        self.assertEqual(result['n'], 51)
        self.assertEqual(result['n_shake'], 51)
        self.assertEqual((result['A_mean_abs_dx'], result['B_shake'], result['C_lost']), (10, 0, 0))


class StageDecisionTests(unittest.TestCase):
    @staticmethod
    def stage_row(shake, loss_time=0, **overrides):
        row = dict(status='ok', walk='walk', n=150, A_mean_abs_dx=4, B_shake=shake,
                   C_lost=0, lost_time_ratio=loss_time,
                   starts_per_min=0, stops_per_min=0, reversals_per_min=0)
        row.update(overrides)
        return row

    def test_stage_two_rejects_candidate_that_loses_more_time(self):
        candidates = [('A', 'current', {}), ('B', 'candidate', {})]
        per = {'A': [self.stage_row(2, round=i) for i in range(2)],
               'B': [self.stage_row(1, .03, round=i) for i in range(2)]}
        chosen, _, rows = analysis.choose(candidates, per)
        self.assertEqual(chosen, 'A')
        self.assertIn('놓친 시간 비율', rows['B']['out'])
        for row in per['B']:
            row['lost_time_ratio'] = 0
        self.assertEqual(analysis.choose(candidates, per)[0], 'B')

    def test_stage_three_cannot_call_more_lost_time_an_improvement(self):
        a = self.stage_row(2)
        self.assertEqual(analysis.verdict([(a, self.stage_row(1))] * 2, 2)['verdict'], '덜 흔들림')
        self.assertEqual(analysis.verdict([(a, self.stage_row(1, .06))] * 2, 2)['verdict'], '나빠짐')

    def test_stage_two_cannot_discard_invalid_attempt_and_choose_candidate(self):
        candidates = [('A', 'current', {}), ('B', 'candidate', {})]
        per = {'A': [self.stage_row(2, round=i) for i in range(2)],
               'B': [self.stage_row(1, round=i) for i in range(2)]}
        per['B'].append(self.stage_row(1, status='invalid', round=2, lost_time_ratio=None))
        chosen, _, rows = analysis.choose(candidates, per)
        self.assertEqual(chosen, 'A')
        self.assertIn('판정할 수 없는 시도', rows['B']['out'])

    def test_stage_three_cannot_drop_invalid_pair_to_claim_success(self):
        a, b = self.stage_row(2), self.stage_row(1)
        result = analysis.verdict([(a, b), (a, b), (a, dict(b, status='invalid', lost_time_ratio=None))], 2)
        self.assertNotIn('B_shake', result)
        self.assertIn('판정할 수 없음', result['verdict'])


if __name__ == '__main__':
    unittest.main()
