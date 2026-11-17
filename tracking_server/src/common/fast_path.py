"""Switches for the same-result speedups of the frame loop (2026-11-15). Off by default (the old way); each gives exactly the old result.

  embed  appearance crops (BoT-SORT, KeepID) go to the NPU together, up to 4 at a time, also while KeepID's refresh crops of the
         last frame are still on their own slots (before: at most 2 at a time, and one by one on every frame that followed a
         refresh, which is every other frame). Same crops, same model, same NPU core: the features are the same bytes.
  kpts   the anchor of every person box kept by NMS (for its keypoints) is looked up for all boxes at once among the anchors
         that can match, instead of one pass over all 6,300 anchors per box (about 100 boxes a frame at the low score limit).

On at start: export LWIR_FAST_EMBED=1 and export LWIR_FAST_KPTS=1 in scripts/live.env (embed also needs LWIR_REID_BIRTH_PIPELINE=1
and --rl-async, as before). While running (not kept over a restart):
POST /rules/set?fast_embed=0|1, fast_kpts=0|1, fast_kpts_check=0|1. With fast_kpts_check on, both keypoint lookups run on every
frame and STAT counts the frames where they differ (for a check on a recording; it costs the time of both).
"""
import os

FAST = {'embed': os.environ.get('LWIR_FAST_EMBED') == '1', 'kpts': os.environ.get('LWIR_FAST_KPTS') == '1', 'kpts_check': False}
STAT = {'kpts_checked': 0, 'kpts_differ': 0, 'kpts_loop_rows': 0, 'embed_batches': 0, 'embed_beside_refresh': 0}
EMBED_MAX = 4   # crops on the NPU at a time (BoT-SORT asks for at most 4 a frame)


def set_from(query):
    """query = parse_qs dict of /rules/set -> the names changed"""
    changed = []
    for name in ('embed', 'kpts', 'kpts_check'):
        key = 'fast_' + name
        if key in query:
            FAST[name] = query[key][0] in ('1', 'true', 'on'); changed.append(key)
    if 'fast_kpts_check' in query and FAST['kpts_check']: STAT['kpts_checked'] = STAT['kpts_differ'] = 0
    return changed


def state():
    return dict(fast_embed=FAST['embed'], fast_kpts=FAST['kpts'], fast_kpts_check=FAST['kpts_check'], fast_stat=dict(STAT))
