"""One selectable set of tracker score limits for the Python and the C++ engine (2026-11-20, ID-rules audit).

Since 11-18 the two engines of the same tracker do not use the same numbers, so choosing the engine on the page also
changes the rules:

  BoT-SORT                          new track   first match   unconfirmed match
    Python (botsort.py)                0.50        0.9           0.6
    C++ (native_tracker_core.hpp)      0.25        0.8           0.7
  OC-SORT, SORT   new track 0.50 in Python (the ByteTrack base class), 0.25 in C++. Their first match has its own limit
                  and the unconfirmed match is 0.7 in both engines, so only `new` differs.
  ByteTrack rows (confirm3, rebind) have one engine in this factory (Python: 0.50 / 0.9 / 0.7) and are left alone.

new = lowest detection score that may start a track; match = highest cost (1 - IoU x score, or the appearance cost) of a
pair in the first association; unconfirmed = the same limit for tracks that are not confirmed yet.

The switch (POST /rules/set?track_thr=...; start value LWIR_TRACK_THR; not kept over a restart):
  engine_default  (default) every engine keeps its own numbers = the behaviour before the switch
  py              the Python numbers on both engines
  cpp             the C++ numbers on both engines
It is applied to every tracker the page factory makes and, when changed, to the running tracker at once (tracks are kept).
With engine_default a tracker is not touched at all (nothing is read from it or written on it) unless a set was put on
that same tracker before; then its own numbers, remembered at that moment, are put back.

The C++ `unconfirmed` limit is a settable field only in extensions built from the 2026-11-20 source or later
(bash scripts/build_native_trackers.sh). An older build keeps 0.7 whatever is chosen; state() then says so in `note`.
"""
import os

from . import native_backend

KEYS = ('engine_default', 'py', 'cpp')
SETS = {'py': dict(new=0.50, match=0.9, unconfirmed=0.6),
        'cpp': dict(new=0.25, match=0.8, unconfirmed=0.7)}
FIELDS = {'botsort': ('new', 'match', 'unconfirmed'), 'ocsort': ('new',), 'sort': ('new',)}   # what a set changes, per page row
CPP_UNCONFIRMED = 0.7   # the fixed value of a C++ build without the settable field
REBUILD = 'C++ unconfirmed 0.7 고정: 이 빌드는 값을 받지 못함 (scripts/build_native_trackers.sh 로 다시 빌드한 뒤 서버 재시작)'
OWN = '_thr_engine_default'   # attribute on a tracker that got a set: the numbers it was made with


def _start_key():
    key = os.environ.get('LWIR_TRACK_THR', '').strip().lower()
    return key if key in KEYS else 'engine_default'


CHOICE = {'key': _start_key()}


def _inner(tracker):
    """the tracker object itself (native_backend wraps an extension object that takes no new attributes)"""
    try:
        return object.__getattribute__(tracker, '_wrapped')
    except AttributeError:
        return tracker


def _is_cpp(tracker):
    return native_backend.backend_of(tracker) == 'C++'


def _known(tracker, name):
    """A C++ tracker takes ANY attribute name (it has a __dict__), so a name its build does not know would be stored and
    never read by the tracker. For C++ only a field of the extension type counts."""
    return not _is_cpp(tracker) or hasattr(type(_inner(tracker)), name)


def _read(tracker, name):
    if not _known(tracker, name):
        return CPP_UNCONFIRMED if name == 'unconfirmed' else None
    value = getattr(tracker, name, None)
    return float(value) if isinstance(value, (int, float)) else None


def apply(tracker, kind):
    """Put the chosen numbers on `tracker`, made for the page row `kind` ('botsort', 'ocsort', 'sort', ...). -> the tracker."""
    fields = FIELDS.get(kind, ())
    if tracker is None or not fields:
        return tracker
    own = getattr(tracker, OWN, None)
    if CHOICE['key'] == 'engine_default':
        if own is None:
            return tracker   # never changed
        values = own
    else:
        if own is None:
            own = {name: _read(tracker, name) for name in fields}
            try:
                setattr(tracker, OWN, own)
            except Exception:   # an object that takes no attributes keeps the set until the next tracker is made
                pass
        values = SETS[CHOICE['key']]
    for name in fields:
        value = values.get(name)
        if value is not None and _known(tracker, name) and _read(tracker, name) != value:
            setattr(tracker, name, value)
    return tracker


def factory(make):
    """Wrap the page's tracker factory make(kind) so every tracker it returns carries the chosen set."""
    def make_with_thresholds(kind):
        return apply(make(kind), kind)
    return make_with_thresholds


def set_key(key, tracker=None, kind=None):
    """-> True when taken; the running tracker (if given, with its page row) gets the numbers at once."""
    if key not in KEYS:
        return False
    CHOICE['key'] = key
    apply(tracker, kind)
    return True


def state(tracker=None, kind=None):
    """For GET /rules: the choice, the two sets, and the numbers the running tracker really has now."""
    now = None
    if tracker is not None:
        fields = FIELDS.get(kind, ())
        note = ''
        if not fields:
            note = '이 추적기 줄은 엔진이 하나라 기준값 세트를 쓰지 않음'
        elif CHOICE['key'] != 'engine_default':
            missed = [name for name in fields if _read(tracker, name) != SETS[CHOICE['key']][name]]
            if missed == ['unconfirmed'] and _is_cpp(tracker):
                note = REBUILD
            elif missed:
                note = '적용되지 않은 값: ' + ', '.join(missed)
        shown = ('new',) if kind in ('ocsort', 'sort') else ('new', 'match', 'unconfirmed')
        now = dict(kind=kind, backend=native_backend.backend_of(tracker), set_fields=list(fields), note=note,
                   **{name: _read(tracker, name) for name in shown})
    return dict(track_thr=CHOICE['key'], track_thr_sets={key: dict(values) for key, values in SETS.items()}, track_thr_now=now)
