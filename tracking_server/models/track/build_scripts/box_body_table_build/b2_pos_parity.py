"""body_v2/scripts/b2_pos_parity.py - position memory (GRU / Mamba keypoint files) numpy server == torch, after the box_refine trim
(scripts/parity_both.pos_parity, the 11-15 check, on the current worktree; the box refiner part of parity_both is not run: gru was removed).
Usage: python b2_pos_parity.py <out.json>"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import parity_both as PB   # noqa: E402

wt = Path(PB.P.PATHS['worktree']); res = {}
for arm in ('gru_kpt', 'mamba_kpt'):
    f = wt / 'models' / 'track' / ('pos_%s_tmotdance_20261115.npz' % arm)
    res['pos_' + arm] = dict(PB.pos_parity(str(f), arm, ROOT), file=f.name); print('PARITY pos', arm, res['pos_' + arm], flush=True)
res['ok'] = all(v['max_abs_px'] < 1e-3 for v in res.values() if isinstance(v, dict)); res['worktree_head'] = __import__('subprocess').run(['git', '-C', str(wt), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print('POS_PARITY_OK' if res['ok'] else 'POS_PARITY_FAIL', flush=True)
