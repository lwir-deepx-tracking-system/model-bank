"""Frame-range leases and per-frame compare-and-swap. Call with state.LOCK held.

Owner tokens are bearer secrets, kept only in this server process. A restart
expires all leases; clients must reclaim before writing. Public lists omit tokens.
"""
from copy import deepcopy
import json
import re
import time

from .body_parts import empty_parts, validate_parts
from .state import MOT_LINE

LEASE_SECONDS = 90


class Conflict(Exception):
    def __init__(self, message, frames=()):
        super().__init__(message)
        self.frames = sorted(set(frames))


def owner_token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{32,128}', value):
        raise ValueError('owner must be a random 32-128 digit hex token')
    return value.lower()


def live_claims(store, sequence):
    now = time.monotonic()
    active = store.setdefault(sequence, {})
    for owner in list(active):
        if active[owner]['deadline'] <= now:
            del active[owner]
    return active


def public_claims(store, sequence, owner=None):
    result = []
    for key, claim in live_claims(store, sequence).items():
        row = {k: claim[k] for k in ('user', 'first', 'last', 'expires_at')}
        row['lease_seconds'] = LEASE_SECONDS
        if owner is not None:
            row['mine'] = key == owner
        result.append(row)
    return sorted(result, key=lambda row: (row['first'], row['last']))


def claim_range(store, sequence, owner, user, first, last, frames):
    owner = owner_token(owner)
    if (not isinstance(user, str) or not user.strip() or len(user) > 40
            or any(ord(ch) < 32 for ch in user)):
        raise ValueError('user must be a nonempty name of at most 40 characters')
    if type(first) is not int or type(last) is not int or not 1 <= first <= last <= frames:
        raise ValueError('claim range must be inclusive original frame numbers')
    active = live_claims(store, sequence)
    collisions = []
    for key, claim in active.items():
        if key != owner:
            collisions.extend(range(max(first, claim['first']), min(last, claim['last']) + 1))
    if collisions:
        raise Conflict('frame range is already claimed', collisions)
    record = dict(user=user.strip(), first=first, last=last,
                  deadline=time.monotonic() + LEASE_SECONDS,
                  expires_at=time.time() + LEASE_SECONDS)
    active[owner] = record
    return dict(owner=owner, user=record['user'], first=first, last=last,
                expires_at=record['expires_at'], lease_seconds=LEASE_SECONDS)


def release_range(store, sequence, owner):
    live_claims(store, sequence).pop(owner_token(owner), None)


def frame_state(snapshot, frame):
    return dict(mot=[line for line in snapshot['mot'].replace('\r', '').split('\n')
                     if line.strip() and int(line.split(',', 1)[0]) == frame],
                parts=[row for row in snapshot['parts']['annotations'] if row['frame'] == frame],
                reviewed=frame in {int(v) for v in snapshot['review'].get('reviewed', [])},
                removed=frame in {int(v) for v in snapshot['review'].get('removed', [])})


def _validated_state(value, frame, frames):
    if not isinstance(value, dict) or set(value) != {'mot', 'parts', 'reviewed', 'removed'}:
        raise ValueError('before/after must contain mot, parts, reviewed and removed')
    if type(value['reviewed']) is not bool or type(value['removed']) is not bool:
        raise ValueError('reviewed and removed must be booleans')
    if not isinstance(value['mot'], list) or not isinstance(value['parts'], list):
        raise ValueError('mot and parts must be lists')
    ids = set()
    for line in value['mot']:
        if not isinstance(line, str) or not MOT_LINE.fullmatch(line):
            raise ValueError('patch MOT must contain raw individual MOT lines')
        columns = line.split(',')
        if int(columns[0]) != frame:
            raise ValueError('patch MOT frame mismatch')
        if int(columns[7]) == 0:
            if int(columns[1]) in ids:
                raise ValueError('duplicate person id')
            ids.add(int(columns[1]))
    parts = dict(empty_parts(), annotations=value['parts'])
    normalized = validate_parts(parts, value['mot'], frames)['annotations']
    if any(row['frame'] != frame for row in normalized):
        raise ValueError('patch part frame mismatch')
    return dict(value, parts=normalized)


def _canonical(value):
    return (tuple(sorted(value['mot'])),
            json.dumps(sorted(value['parts'], key=lambda row: (row['frame'], row['id'])),
                       sort_keys=True, allow_nan=False, separators=(',', ':')),
            value['reviewed'], value['removed'])


def merge_patch(snapshot, edits, claim):
    """Validate the whole patch before producing a merged, independent snapshot."""
    if not isinstance(edits, list) or not edits or len(edits) > snapshot['frames']:
        raise ValueError('frame_patch must be a nonempty list bounded by frame count')
    replacements = {}
    conflicts = []
    purged = {int(v) for v in snapshot['review'].get('purged', [])}
    for item in edits:
        if not isinstance(item, dict) or set(item) != {'frame', 'before', 'after'}:
            raise ValueError('patch entry must contain frame, before and after')
        frame = item['frame']
        if type(frame) is not int or not 1 <= frame <= snapshot['frames'] or frame in replacements:
            raise ValueError('invalid or repeated patch frame')
        before = _validated_state(item['before'], frame, snapshot['frames'])
        after = _validated_state(item['after'], frame, snapshot['frames'])
        replacements[frame] = after
        if not claim['first'] <= frame <= claim['last'] or frame in purged:
            conflicts.append(frame)
            continue
        current = _validated_state(frame_state(snapshot, frame), frame, snapshot['frames'])
        if _canonical(before) != _canonical(current):
            conflicts.append(frame)
    if conflicts:
        raise Conflict('frame changed or is outside the live claim', conflicts)
    merged = deepcopy(snapshot)
    output = []
    inserted = set()
    for line in snapshot['mot'].replace('\r', '').split('\n'):
        if not line.strip():
            continue
        frame = int(line.split(',', 1)[0])
        if frame not in replacements:
            output.append(line)  # Keep unchanged MOT lines byte-for-byte, without reformatting.
        elif frame not in inserted:
            output.extend(replacements[frame]['mot'])
            inserted.add(frame)
    for frame in sorted(set(replacements) - inserted):
        output.extend(replacements[frame]['mot'])
    merged['mot'] = '\n'.join(output) + ('\n' if output else '')
    kept_parts = [row for row in snapshot['parts']['annotations'] if row['frame'] not in replacements]
    kept_parts.extend(row for frame in sorted(replacements) for row in replacements[frame]['parts'])
    merged['parts'] = dict(empty_parts(), annotations=kept_parts)
    # Validate existing and changed associations together before touching disk.
    validate_parts(merged['parts'], output, snapshot['frames'])
    for key in ('reviewed', 'removed'):
        numbers = {int(v) for v in snapshot['review'].get(key, [])}
        for frame, after in replacements.items():
            (numbers.add if after[key] else numbers.discard)(frame)
        merged['review'][key] = sorted(numbers)
    return merged
