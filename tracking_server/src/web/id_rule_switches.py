"""Two /rules switches of the 2026-11-20 ID-rules audit. Both start at the behaviour before the switch and are not kept
over a restart (start values: LWIR_NESTED_DUP, LWIR_TRACK_THR in scripts/live.env).

  POST /rules/set?nested_dup=named|kpt|off          which detectors drop the weak full-height duplicate of an upper-body
                                                    box (src/detect/pose_duplicate.py). named = only the three detectors
                                                    of 11-18 / 11-19 (default), kpt = every detector with keypoint heads.
  POST /rules/set?track_thr=engine_default|py|cpp   tracker score limits of BoT-SORT / OC-SORT / SORT: each engine's own
                                                    (default), or one set on both engines (src/track/threshold_set.py).
GET /rules shows nested_dup, nested_dup_active, nested_dup_dropped, track_thr, track_thr_sets, track_thr_now.

install(APP) puts both in front of the existing /rules parser: these two keys are checked first and changed only when the
rest of the query was taken too (the rule of the other switches in page_settings.rules_request_2).
"""
from ..detect import pose_duplicate
from ..detect.model_scan import KPT_DETS
from ..track import threshold_set

KEYS = ('nested_dup', 'track_thr')


def _tracker(app):
    """(running tracker, its page row) or (None, None)"""
    values = vars(app)
    rules = values.get('RULES') or {}
    return values.get('tracker'), rules.get('tracker')


def state(app):
    det = getattr(vars(app).get('args'), 'det', None)   # vars(): a name APP does not hold yet must not raise here
    tracker, kind = _tracker(app)
    return dict(pose_duplicate.state(det, KPT_DETS), **threshold_set.state(tracker, kind))


def request(app, query, rest_request):
    """query: parse_qs dict. -> (ok, text). rest_request(query without our keys) is the parser that was there before."""
    mine = {key: str(query[key][0]).strip().lower() for key in KEYS if key in query}
    if not mine:
        return rest_request(query)
    if 'nested_dup' in mine and mine['nested_dup'] not in pose_duplicate.MODES:
        return False, '겹친 상자 지우기 값은 named, kpt, off 중 하나'
    if 'track_thr' in mine and mine['track_thr'] not in threshold_set.KEYS:
        return False, '추적기 기준값 세트는 engine_default, py, cpp 중 하나'
    rest = {key: value for key, value in query.items() if key not in mine}
    ok, text = rest_request(rest) if rest else (True, '적용됨')
    if not ok:
        return ok, text
    if 'nested_dup' in mine:
        pose_duplicate.set_mode(mine['nested_dup'])
    if 'track_thr' in mine:
        threshold_set.set_key(mine['track_thr'], *_tracker(app))
    print('RULES set %s' % ' '.join('%s=%s' % item for item in sorted(mine.items())), flush=True)
    return ok, text


def install(app):
    """Wrap app.rules_request / app.rules_state (set by app.py just before this call)."""
    rest_request, rest_state = app.rules_request, app.rules_state
    def state_with_switches():
        out = rest_state()
        try:
            out = dict(out, **state(app))
        except Exception as error:   # the page lives on GET /rules: these extra keys must never take it down
            out = dict(out, id_rule_switches_error=repr(error)[:120])
        return out

    app.rules_request = lambda query: request(app, query, rest_request)
    app.rules_state = state_with_switches
