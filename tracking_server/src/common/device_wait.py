"""장치 관리: 같은 장치에서 다른 측정 프로그램이 돌고 있으면 끝나기를 기다린다."""
import os
import time
import re


# ------------------------------------------------------------------------------------------------------------ M1 guard
def other_m1_evals(root='/proc', pid=None):
    """Other running M1 latency scripts (any process whose command line contains m1_<x>eval<y>.py), excluding this process and its
    ancestors. [] where /proc does not exist (PC dry run)."""
    if not os.path.isdir(root): return []
    pat = re.compile(rb'm1_[A-Za-z0-9_]*eval[A-Za-z0-9_]*\.py'); anc = set(); p = pid or os.getpid()
    while p and p > 1 and p not in anc:
        anc.add(p)
        try: p = int(open(os.path.join(root, str(p), 'stat')).read().rsplit(')', 1)[1].split()[1])
        except Exception: break
    found = []
    for d in os.listdir(root):
        if d.isdigit() and int(d) not in anc:
            try: c = open(os.path.join(root, d, 'cmdline'), 'rb').read()
            except Exception: continue
            if pat.search(c): found.append((int(d), c.replace(b'\0', b' ').decode(errors='replace').strip()[:200]))
    return sorted(found)


def wait_m1_free(max_s=1800, every=15):
    t = time.monotonic()
    while True:
        o = other_m1_evals()
        if not o: return time.monotonic() - t
        if time.monotonic() - t > max_s: raise SystemExit('OTHER_M1_EVAL_STILL_RUNNING %s' % o)
        print('WAIT_OTHER_M1_EVAL', o, flush=True); time.sleep(every)
