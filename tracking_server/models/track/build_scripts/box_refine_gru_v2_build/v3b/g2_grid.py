"""gru_v2/scripts/g2_grid.py - the 8 pre-registered GRU v2 runs (arch x lam_j x lam_l), 4 at a time x 2 threads on the PC CPU. Logs runs/<stem>.log."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'runs'
GRID = [(a, j, l) for a in ('gate', 'res') for j in (1, 4) for l in (1, 4)]
extra = sys.argv[1:]
t0 = time.time(); pend = list(GRID); act = []; bad = 0
while pend or act:
    while pend and len(act) < 4:
        a, j, l = pend.pop(0); stem = 'g3_%s_j%g_l%g' % (a, j, l); lf = open(RUNS / (stem + '.log'), 'w')
        act.append((stem, subprocess.Popen([sys.executable, str(HERE / 'g2_train.py'), a, str(j), str(l)] + extra, stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE)), lf))
    time.sleep(5)
    for x in list(act):
        if x[1].poll() is not None:
            act.remove(x); x[2].close(); last = (RUNS / (x[0] + '.log')).read_text(errors='replace').strip().splitlines()[-1]
            bad += x[1].returncode != 0; print('RUN %s exit %d (%.0f s) %s' % (x[0], x[1].returncode, time.time() - t0, last), flush=True)
print('GRID_DONE bad %d' % bad, flush=True); sys.exit(1 if bad else 0)
