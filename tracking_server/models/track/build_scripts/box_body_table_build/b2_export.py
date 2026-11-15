"""body_v2/scripts/b2_export.py - the body mode's table file: data/table_v2.npz (train split, b2_table.py fit) + the size filter chosen on val-A
(results/valA_grid.json) -> <out dir>/box_body_table_<date>.npz + .json (how it was made). Usage: python b2_export.py <out dir> <date>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

B2 = Path(__file__).resolve().parent.parent


def main():
    out, date = Path(sys.argv[1]), sys.argv[2]; z = np.load(B2 / 'data' / 'table_v2.npz', allow_pickle=False); va = json.loads((B2 / 'results' / 'valA_grid.json').read_text(encoding='utf-8'))
    tv = json.loads((B2 / 'results' / 'table_v2.json').read_text(encoding='utf-8')); ch = va['choice']; assert ch['feasible'], ch
    euro = np.array([ch['mc_s'], ch['beta_s'], 1.0]); name = 'box_body_table_%s' % date
    np.savez(out / (name + '.npz'), tab=z['tab'], lh_cen=z['lh_cen'], ar_cen=z['ar_cen'], person=z['person'], euro=euro)
    sha = hashlib.sha256((out / (name + '.npz')).read_bytes()).hexdigest(); kinds = tv['kind']; R = tv['rules']; Pn = tv['person']
    cnt = {k: sum(row.count(k) for row in kinds) for k in ('body', 'fit', 'fill')}
    info = {
        'status': '상체 네모로 통일(box_refine=body) 2판 변환표 + 사람 전체 상자 고정 비율 + 크기 필터 · 장치 미확인',
        'made': 'pos_refine_20261115/body_v2 (PREREG_BODY_V2.json, written before any val-A / val-B / device number of this version): b2_table.py fit (train split only), b2_eval.py valA (filter), b2_export.py',
        'keys': {'tab': '[11, 8, 4] (dx / h, dy / h, log w ratio, log h ratio) at the cell centres, bilinear between them', 'lh_cen': 'log detector-box height (px) of the 11 rows', 'ar_cen': 'w / h of the 8 columns',
                 'person': '(top offset, height, width) / person box height: the whole-person -> body box constant for detectors that give whole-person boxes', 'euro': 'size filter (min cutoff Hz, beta per log unit per s, derivative cutoff Hz)'},
        'table_rule': 'train split, our two pose detectors, corrected head-to-hip targets, 402,155 pairs. Body cell (>= %d pairs with both hips seen, median hip line in %s of the detector box) = identity; a fitted cell is kept only with >= %d pairs (N: per-pair spread 0.078 log, design effect 16 -> standard error ~2 %%), >= %d sequences, a sequence-bootstrap 90 %% interval of the height ratio narrower than x %.2f and no contradiction with the hips; every other cell takes the nearest kept or body cell of its row.' % (R['min_hip_pairs'], R['body_hip_range'], R['N'], R['min_sequences'], R['max_boot90_ratio']),
        'cells': cnt, 'changes_from_first_table': 'every box of the body-box shape (aspect 0.55-0.80) and every box from 100 px up is identity (the first table: 0.94 height and +0.03 h to the right above 230 px, 0.60 / 0.61 at 65-100 px x aspect 0.65-0.8); whole-person cells under 100 px keep fitted 0.48-0.53 height',
        'person_rule': 'train pairs whose detection is a body box by its own hips against the dataset whole-person box (>= 60 px): median top %+.4f, height %.4f, width %.4f x person HEIGHT (log MAD 0.052; x person width 0.22), n %d; TMOT %.3f, DanceTrack %.3f height. Project BOX_ONLY 0.531, stage 3a 0.50-0.55' % (Pn['top'], Pn['height'], Pn['width_over_height'], Pn['n'], Pn['by_dataset']['TMOT']['height'], Pn['by_dataset']['DT']['height']),
        'family': "server: detector key with 'bodybox' (our pose detectors) -> table; any other detector -> person constant",
        'euro_rule': 'val-A grid min cutoff {0.2..8} Hz x beta {0..10}: minimise height-step p99 + per-track largest step (relative to the unfiltered converted row) subject to size lag in fast size changes >= tracker - 0.02, first-second under-size (< 0.9 of the converted height) <= 5 %%, fast-growth median >= 0.90, |log h| <= unfiltered + 0.005. Chosen %.1f Hz, beta %.1f (J %.3f; the first version 0.4 Hz beta 0: J %.3f but fails the lag and under-size limits)' % (ch['mc_s'], ch['beta_s'], ch['J'], va['pushed_filter_point']['J']),
        'sha256': sha}
    (out / (name + '.json')).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8'); print('EXPORT', name, sha, euro.tolist(), flush=True)


if __name__ == '__main__':
    main()
