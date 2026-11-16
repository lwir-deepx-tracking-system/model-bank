# GRU box refiner v2: result (2026-11-15, offline only)

Verdict by the pre-registered rules (PREREG_GRU_V2.json, written before training): **v2 does NOT pass.** It passes every rule on TMOT (thermal) and P1, P3, P4, P5 on DanceTrack, but fails P2 on DanceTrack: its aim-point jitter is not lower than lookup + One Euro (0.039 vs 0.038), and a One Euro tuned with the same val-A rule has smaller height steps on DanceTrack (p99 0.036 vs 0.047). val-B was already seen three times before this (round 1, verification, size-mode workflow); it was used once here.

## What changed against round 1
- Base box = lookup-converted detection (train-split table, bilinear), so a new track starts at the converted box.
- Output = previous output moved toward the converted box by a learned per-component gain, plus a small correction (a learned adaptive filter). Round 1 corrected the raw box from a zero state.
- Loss: accuracy (first 5 rows x3) + jitter of the output itself (camera-compensated) + hinge on one-frame height / width steps + lag against the converted detection along the motion. Round 1's jitter term was relative to the (jerky) target.
- Inputs built by one batched engine; the per-person Kalman is replaced by a fixed-gain velocity filter. New inputs: lookup correction, start flag, rows since start, previous output.
- Training data: random track starts, every piece a fresh state, frame-rate change (half rate 15 %, synthetic double rate 30 %).
- Selection on val-A by smoothness (height step p99, per-track largest step, aim jitter, first-5-rows step) with aim-error, lag and start-step gates. 8 runs; chosen gate / lam_j 4 / lam_l 4 (epoch 29). Size filter chosen 5 Hz, beta 0. Tuned baseline conv_euro_t = centre 1 Hz beta 3, size 0.4 Hz beta 0.

## val-B, near + far streams (target heights; [90 % per-sequence bootstrap]); full tables: results/tables_valB.md

TMOT (6 sequences, 5241 rows)

| method | aim err | aim jitter | height step p99 | per-track largest height step | first-5-rows step (<100 px) | lag vs detection > 0.1 h/frame |
|---|---|---|---|---|---|---|
| raw | 0.212 | 0.077 | 0.255 | 0.183 | 0.061 | 0 |
| tracker row | 0.210 [0.125, 0.280] | 0.044 | 0.172 [0.149, 0.255] | 0.123 | 0.047 | -0.017 |
| One Euro | 0.210 | 0.053 | 0.091 | 0.052 | 0.020 | -0.030 |
| conv_euro | 0.070 [0.064, 0.083] | 0.066 [0.052, 0.078] | 0.099 [0.061, 0.167] | 0.040 [0.036, 0.060] | 0.015 | -0.026 |
| conv_euro tuned | 0.070 | 0.057 | 0.069 [0.042, 0.116] | 0.032 | 0.012 | -0.036 |
| GRU v1 | 0.063 [0.054, 0.078] | 0.037 | 0.193 [0.114, 0.210] | 0.172 | 0.081 | -0.065 |
| GRU v2 | 0.072 [0.065, 0.087] | 0.038 [0.030, 0.046] | 0.045 [0.029, 0.055] | 0.020 [0.018, 0.031] | 0.009 | -0.002 |
| GRU v2 + size filter | 0.072 [0.065, 0.087] | 0.038 [0.030, 0.045] | 0.044 [0.029, 0.054] | 0.019 [0.017, 0.026] | 0.008 | -0.002 |

DanceTrack (8 sequences, 87768 rows)

| method | aim err | aim jitter | height step p99 | per-track largest height step | first-5-rows step (<100 px) | lag vs detection > 0.1 h/frame |
|---|---|---|---|---|---|---|
| raw | 0.100 | 0.063 | 0.191 | 0.185 | 0.057 | 0 |
| tracker row | 0.101 [0.093, 0.110] | 0.040 | 0.132 [0.118, 0.146] | 0.129 | 0.044 | -0.018 |
| One Euro | 0.101 | 0.038 | 0.060 | 0.047 | 0.012 | -0.023 |
| conv_euro | 0.100 [0.093, 0.109] | 0.038 [0.033, 0.043] | 0.059 [0.051, 0.067] | 0.046 [0.041, 0.053] | 0.012 | -0.023 |
| conv_euro tuned | 0.102 | 0.034 | 0.036 [0.030, 0.040] | 0.028 | 0.008 | -0.031 |
| GRU v1 | 0.089 [0.081, 0.098] | 0.033 | 0.077 [0.067, 0.085] | 0.064 | 0.028 | -0.035 |
| GRU v2 | 0.094 [0.087, 0.103] | 0.039 [0.034, 0.045] | 0.051 [0.043, 0.057] | 0.037 [0.032, 0.044] | 0.015 | -0.006 |
| GRU v2 + size filter | 0.094 [0.087, 0.103] | 0.039 [0.034, 0.044] | 0.047 [0.040, 0.054] | 0.032 [0.029, 0.039] | 0.012 | -0.006 |

- Start of track (TMOT, 85 new small-person tracks): height / target at rows 0, 1, 4 = 1.03 / 1.02 / 1.02 (v1 1.40 / 1.20 / 1.04); row-1 steps > 0.15: 0 % (v1 79 %).
- Lag, TMOT >= 100 px, > 0.1 h/frame, against the raw detection (n 639): v2 -0.005, tracker -0.023, v1 -0.099.
- Size lag (height behind when the target height changes fast): v2 -0.060 TMOT / -0.065 DanceTrack, with size filter -0.062 / -0.070; conv_euro -0.015 / -0.059; v1 -0.042 / -0.043. v2's height is the slowest to follow fast size changes. Not part of the pass rules.
- Accuracy against v1: aim error +0.009 (TMOT), +0.005 (DanceTrack). v1's extra accuracy came with lag behind the detection.

## Device-rate recordings (3 recordings, 23-24 fps, no GT, 5895 steps, 42 tracks; reference = tracker row height); results/tables_device.md

| method | height step p99 | per-track largest height step | aim step p99 | aim jitter | lag 0.03-0.1 (n 361) | first-5-rows step (n 177) | height / tracker height |
|---|---|---|---|---|---|---|---|
| tracker row | 0.071 | 0.094 | 0.121 | 0.0191 | -0.004 | 0.017 | 1.00 |
| One Euro | 0.035 | 0.031 | 0.120 | 0.0186 | -0.011 | 0.003 | 1.00 |
| conv_euro | 0.027 | 0.026 | 0.124 | 0.0207 | -0.008 | 0.006 | 0.64 |
| conv_euro tuned | 0.017 | 0.018 | 0.124 | 0.0191 | -0.011 | 0.004 | 0.64 |
| GRU v1 | 0.039 | 0.050 | 0.121 | 0.0182 | -0.022 | 0.026 | 0.75 |
| GRU v2 | 0.018 | 0.020 | 0.130 | 0.0208 | +0.001 | 0.011 | 0.72 |
| GRU v2 + size filter | 0.016 | 0.016 | 0.130 | 0.0208 | +0.001 | 0.009 | 0.72 |

The > 0.1 bin has only 82 steps (v2 +0.037, others about 0): too few to read.

## Cost (PC, one thread, numpy, lookup + inputs + GRU + size filter in one batch)
1 person 0.52 ms, 10 people 0.64 ms, 20 people 0.72 ms, 30 people 1.13 ms per frame. numpy = torch within 1.7e-4 px (12,380 rows).

## Doubts
1. Fails P2 on DanceTrack; a tuned lookup + One Euro is smoother in height there, at the price of more lag (-0.031 vs -0.006).
2. At device rate the aim-point jitter and aim step p99 of v2 are slightly worse than the tracker row (0.0208 vs 0.0191; 0.130 vs 0.121). The gain is in height steps only. For the gimbal aim this is no improvement.
3. On the device recordings (close people) every converted method draws a box 64-75 % of the tracker box height. The lookup's large-box rows come from DanceTrack and the single fixed target ratio; whether this box is right for close people on our camera was not checked by eye.
4. v2 follows fast size changes slowest (size lag above).
5. The frame-rate doubling is synthetic (interpolated frames with resampled noise). The device test has 3 recordings, few people, one site.
6. The lag loss ties the output to the detection; on TMOT the detection is ahead of the GT target, so centre error against GT is a little higher than v1 by design.
7. The fixed-gain velocity filter replaced the Kalman without a separate ablation. The 'res' runs (no feedback) scored worse on val-A (best J 0.74 with a gate violation vs 0.68).
8. Reset rule differs slightly from round 1 (camera-moved jump test, missed > 30); v1 numbers use its own rule and reproduce round 1.
9. val-A P3 (aim) failed in the code check on val-A for TMOT but passed on val-B; TMOT intervals are wide (5-6 sequences).
10. Not done: server code, device run, KeepID hand-over, by-eye check of drawn boxes.

## Addendum 2026-11-15 night: by-eye check (doubt 3), lookup fix, server port

**By eye (eyecheck/before/sheet_01..06.jpg, 24 frames, near and far detector, 3 device-rate recordings):** the output was wrong for close people. The green box was the detector box shrunk to about 0.7 in width and height around a point a little above its centre: it started below the top of the head (forehead / hair line), ended at the chest and was narrower than the shoulders. The tracker row (white) there already is the body box. Numbers on all rows: for detector boxes >= 240 px the hip keypoints sat at 0.95 of the detector box height but at 1.27-1.30 of the v2 box (below its bottom edge); v2 height / tracker height 0.71-0.72.

**Cause (train split only, results/lutcheck_train.json):** the two lookup rows 'detector box 170-230 px' and '>= 230 px' (h ratio 0.86 and 0.64) were fitted on 1,659 and 300 DanceTrack pairs in one aspect cell (no thermal pair that large) and copied to every aspect column. In those pairs the target (a fixed share of the ground-truth box) is too small: the detectors' own hip keypoints are at 0.95-0.97 of the detector box, as in every other row, but at 1.11 / 1.50 of the target. In the row below (130-170 px, 7,470 pairs) target / detector = 1.00 for both detectors and both datasets.

**Fix (g2_lutfix.py, data/lut_train_fix.pkl):** rows >= 170 px take the 130-170 px row (identity within 0.8 %). The weights are the SAME selected run (g2_gate_j4_l4, epoch 29), not retrained and not re-selected; size filter and tuned baseline unchanged. Re-exported (new sha256 b03ea8ad...; numpy = torch 1.4e-4 px). Everything from before the fix is kept in before_lutfix/.

**After (eyecheck/after/):** the green box starts at the top of the head and is as wide as the detector box within a few px; its bottom edge is still about 10 % of the height above the tracker row's. v2 height / tracker height 0.89 (>= 240 px), 0.95 (160-240 px); hips at 1.06-1.10 of the v2 box. The remaining 11 % comes from the GRU's own correction, which learned the same wrong large-box targets; removing it needs a retrain with those train pairs dropped or re-targeted (not done: it would be a new selection).

**val-B again (the 2nd look for v2, 5th for this line; no re-selection):** val-B has almost no boxes >= 170 px, so nothing moved: TMOT v2 + size aim 0.072, aim jitter 0.038, dh p99 0.044, per-track largest 0.019, first-5 0.008, lag -0.002 (all as before); DanceTrack 0.094 / 0.039 / 0.047 / 0.032 / 0.012 / -0.006 (as before). conv_euro TMOT dh p99 0.099 -> 0.112, tuned 0.069 -> 0.074. Verdict unchanged: all rules pass on TMOT, P2 fails on DanceTrack.

**Device-rate recordings again (results/tables_device.md; before in before_lutfix/results/):**

| method | height step p99 | per-track largest | aim step p99 | aim jitter | lag 0.03-0.1 | first-5-rows step | height / tracker |
|---|---|---|---|---|---|---|---|
| tracker row | 0.071 | 0.094 | 0.121 | 0.0191 | -0.004 | 0.017 | 1.00 |
| One Euro | 0.035 | 0.031 | 0.120 | 0.0186 | -0.011 | 0.003 | 1.00 |
| conv_euro (before -> after) | 0.027 -> 0.035 | 0.026 -> 0.031 | 0.124 -> 0.121 | 0.0207 -> 0.0186 | -0.008 -> -0.011 | 0.006 -> 0.003 | 0.64 -> 1.00 |
| conv_euro tuned | 0.017 -> 0.022 | 0.018 -> 0.023 | 0.124 -> 0.119 | 0.0191 -> 0.0179 | -0.011 -> -0.015 | 0.004 -> 0.003 | 0.64 -> 1.00 |
| GRU v2 | 0.018 -> 0.032 | 0.020 -> 0.031 | 0.130 -> 0.127 | 0.0208 -> 0.0209 | +0.001 -> -0.003 | 0.011 -> 0.016 | 0.72 -> 0.89 |
| GRU v2 + size filter | 0.016 -> 0.027 | 0.016 -> 0.024 | 0.130 -> 0.127 | 0.0208 -> 0.0209 | +0.001 -> -0.003 | 0.009 -> 0.012 | 0.72 -> 0.89 |

Part of the small height steps before the fix was only the smaller box (steps are divided by the tracker height). After the fix v2 + size filter is still smoother in height than the tracker row (0.027 vs 0.071) and One Euro (0.035), but a tuned One Euro is smoother (0.022). Aim jitter and aim step stay slightly worse than the tracker row. The > 0.1 bin (82 steps): v2 +0.049, too few to read.

**Server port:** tracking_server/src/track/box_refine.py mode `gru2` (G2Model, G2Engine, Refiner). results/server_parity.json: the server mode equals g2_lib on all 159,814 rows of val-B (41 streams) and the 3 device recordings (9 streams): 0 px with the hand-over off. With the server's id hand-over on (g2_lib has none) 238 hand-overs change 26,880 rows (up to 32 px on one row): the hand-over keeps a warm state where the offline numbers start a new one; its effect on the numbers was not scored. Cost of Refiner.step on the PC, one thread (results/server_cost.json): 1 / 10 / 20 / 30 people 0.93 / 1.30 / 1.43 / 1.62 ms per frame.

New doubts: (a) the 11 % shrink for close people is still there; (b) the lookup fix is justified on the train split and by eye, but the GRU never saw 'large box + identity conversion' inputs in training; (c) the hand-over path is unscored; (d) still nothing measured on the device.
