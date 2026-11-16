# GRU box refiner v2: what the server must compute

Status (2026-11-15 night): ported as mode `gru2` in tracking_server/src/track/box_refine.py (G2Model, G2Engine; default off). The lookup table in the npz is the FIXED one (rows >= 170 px = the 130-170 px row, REPORT.md addendum); sha256 b03ea8ad.... Server = this reference on 159,814 rows (results/server_parity.json). The server adds the id hand-over of the other modes (not in g2_lib). Reference implementation: `scripts/g2_lib.py` (`Engine.step`, `model_step_np`, `conv_delta`, `GRUNp`). Port it as is; the offline numbers were made with exactly this code, all people of a frame in one batch.

Weights: `export/box_refine_gru_v2_tmotdance_20261115.npz` (sha256 in the `.json` next to it). The npz holds everything needed:

| key | meaning |
| --- | --- |
| `cell.weight_ih` (144x78), `cell.weight_hh` (144x48), `cell.bias_ih`, `cell.bias_hh` | GRU cell, PyTorch gate order r, z, n |
| `head.weight` (8x48), `head.bias` (8) | output: 4 gate logits (cx, cy, log w, log h), then 4 corrections |
| `arch` | `gate` (the chosen one) |
| `lut_tab` (11x8x4), `lut_lh_cen` (11), `lut_ar_cen` (8) | lookup conversion, read with bilinear interpolation |
| `kc` 0.4, `ks` log 2.2 | correction range |
| `vb` 0.35 | velocity filter gain |
| `size_euro` (mc_s, beta_s, dcut) | size filter after the GRU |
| `lost_max` 30, `jump_h` 1.5, `clip` 5 | reset rule, input clip |

Coordinates: the server's frame canvas (640 wide, picture rows 16..496), boxes as (cx, cy, w, h). This model needs keypoints, so only for the two pose detectors (near, far / general).

## Input per frame

- For every published person id: the detection box `z` matched to the track this frame (round-1 rule: the matched detection when its IoU with the tracker row is at least 0.3, else the tracker row), with w and h clamped to at least 1, and its 17 keypoints (x, y, visibility 0..1; `None` -> 51 zeros).
- `H`: this frame's camera warp (2x3, previous frame -> this frame), or none.
- `t`: frame time in seconds (only the size filter uses it).

## State per id (arrays over all live ids)

`zp` previous detection box, `op` previous GRU output box, `v` velocity (px per frame), `cam` this frame's camera shift of `zp`, `miss` frames since last seen, `k` rows since the state started, `h` GRU state (48), and for the size filter `sx` (log w, log h), `sdx`, `lsc` (camera log scale summed since last seen), `tl` (time last seen).

## Every frame, in this order

1. All live ids (seen or not): `miss += 1`. Drop ids with `miss > 31` (more than 30 frames missed).
2. If there is a warp `H = [R | tr]`, scale `s = sqrt(|det R|)`, for all live ids: `zp` centre -> `R c + tr`, size x `s`; `cam` = new `zp` centre - old; `op` the same way; `v` -> `R v`; `lsc += log s`. No warp: `cam = 0`.
3. For the N ids seen this frame (one batch):
   - `start` = id has no state, or `|z centre - zp centre| > 1.5 x zp height` (zp already camera-moved). On start: `v = 0`, `gap = 0`, previous box = `z`, GRU state 0, `cam = 0`, `k = 0`. Otherwise `gap = miss - 1`, `k += 1`.
   - velocity filter: `inn = (z_c - (zp_c + v (gap+1))) / h`; `v += 0.35 x ((z_c - zp_c) / (gap+1) - v)`.
   - lookup: `d = bilinear(lut_tab; log h, w / h)` (clamped to the centre range), converted box `C = (cx + d0 h, cy + d1 h, w e^d2, h e^d3)`.
   - 74 inputs: `[0:2] v / h` (after the update), `[2:4] inn`, `[4:6] (z_c - zp_c) / h`, `[6:8] log(z_wh / zp_wh)`, `[8] log(h / 100)`, `[9] log(w / h)`, `[10:14]` picture edges / h (left, right, top, bottom), `[14:65]` keypoints ((x - cx) / h x vis, (y - cy) / h x vis, vis) with vis = visibility >= 0.5, `[65] min(gap, 30) / 15`, `[66:68] cam / h`, `[68:72] d`, `[72] start`, `[73] min(k, 10) / 10`. Clip to +-5.
   - 4 more inputs from the previous output (zeros on start): `(op_c - C_c) / C_h`, `log(op_wh / C_wh)`, clipped to +-5. Total 78.
   - GRU step from the id's state (zeros on start) -> `o` (8).
   - In (cx, cy, log w, log h): `base = op + sigmoid(o[0:4]) x (C - op)`; on start `base = C`. Output centre = `base_c + 0.4 x C_h x tanh(o[4:6])`, log size = `base_s + log 2.2 x tanh(o[6:8])`.
   - Store: `op` = output, GRU state, `zp = z`, `v`, `miss = 0`, `k`.
4. Size filter (One Euro on log w and log h only; the centre stays the GRU centre): on start `sx` = output log size, `sdx = 0`. Otherwise `xp = sx + lsc`, `dt = t - tl` (at least 1 ms), `d = (ls - xp) / dt`, `sdx = a(dcut, dt) d + (1 - a) sdx`, `cut = mc_s + beta_s |sdx|`, `sx = a(cut, dt) ls + (1 - a) xp`, with `a(c, dt) = 1 / (1 + 1 / (2 pi c dt))`. Then `lsc = 0`, `tl = t`. Drawn / gimbal box = (GRU centre, exp(sx)). The GRU's own feedback `op` stays the unfiltered output.

## Reset rules (one rule for everything)

- More than 30 frames missed, or the 1.5-height jump above: the GRU state, the velocity, the previous output and the size filter all restart together at the converted box.
- Tracker reset / detector model reload / camera restart: drop all states.
- Detector switch near <-> far: keep the state (the training data had such switches). No blend (`box_blend = 0`).
- Not built or tested here: handing a state over to a new id (KeepID). Round-1 judge item 2 still applies.

## Checks to add when porting

- Parity: `results/parity.json` (torch vs this engine: 1.7e-4 px on 12,380 rows). A server port should reproduce `Engine.step` outputs on saved rows to about 1e-9.
- Cost on the PC: `results/cost.json`. Measure on the device with 20-30 people before turning it on.
- Not measured on the device or on the live camera. See `REPORT.md` for the limits.
