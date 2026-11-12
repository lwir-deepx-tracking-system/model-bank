"""Validate the body-part sidecar against the person MOT saved with it."""

from collections import Counter
import math


IMAGE_SIZE = (640, 480)
BOX_PARTS = frozenset(('head', 'torso', 'phone'))
POINT_PARTS = frozenset(
    side + '_' + joint
    for side in ('left', 'right')
    for joint in ('shoulder', 'hip', 'elbow', 'wrist')
)
STATES = ('visible', 'occluded', 'out_of_frame', 'unknown')


def empty_parts():
    return {'schema_version': 1, 'image_size': list(IMAGE_SIZE), 'annotations': []}


def validate_parts(data, mot_lines, frames):
    if (not isinstance(data, dict) or data.keys() != empty_parts().keys()
            or type(data['schema_version']) is not int or data['schema_version'] != 1
            or not isinstance(data['image_size'], list)
            or any(type(value) is not int for value in data['image_size'])
            or data['image_size'] != list(IMAGE_SIZE)
            or not isinstance(data['annotations'], list)):
        raise ValueError('person_parts.json: schema_version, image_size 또는 annotations 형식이 맞지 않습니다.')

    parent_counts = Counter()
    person_keys = set()
    for line in mot_lines:
        fields = line.split(',')
        key = (int(fields[0]), int(fields[1]))
        parent_counts[key] += 1
        if int(fields[7]) == 0:
            person_keys.add(key)

    normalized = empty_parts()
    seen = set()
    for row in data['annotations']:
        if (not isinstance(row, dict)
                or row.keys() - {'frame', 'id', 'parts', 'needs_review'}
                or type(row.get('frame')) is not int or not 1 <= row['frame'] <= frames
                or type(row.get('id')) is not int or row['id'] <= 0
                or type(row.get('needs_review', False)) is not bool
                or not isinstance(row.get('parts'), dict)):
            raise ValueError('person_parts.json: frame, id, needs_review 또는 parts 형식이 맞지 않습니다.')
        key = (row['frame'], row['id'])
        label = f'{key[0]}번 프레임, 사람 {key[1]}'
        if key in seen:
            raise ValueError(f'{label}: 신체부위 기록이 중복됐습니다.')
        if parent_counts[key] != 1 or key not in person_keys:
            raise ValueError(f'{label}: 같은 프레임에 이 번호의 사람 상자가 하나 있어야 합니다.')
        seen.add(key)
        for part, item in row['parts'].items():
            _validate_part(part, item, label)
        normalized['annotations'].append({
            'frame': key[0], 'id': key[1], 'needs_review': row.get('needs_review', False), 'parts': row['parts'],
        })
    return normalized


def _validate_part(part, item, label):
    if part not in BOX_PARTS and part not in POINT_PARTS:
        raise ValueError(f'{label}: 지원하지 않는 신체부위입니다 ({part}).')
    is_box = part in BOX_PARTS
    field = 'xywh' if is_box else 'xy'
    if not isinstance(item, dict) or item.keys() != {'state', field} or item['state'] not in STATES:
        raise ValueError(f'{label}, {part}: 상태 또는 좌표 형식이 맞지 않습니다.')

    state, geometry = item['state'], item[field]
    if geometry is None and state != 'visible':
        return
    if state not in ('visible', 'occluded') or not _valid_geometry(geometry, is_box):
        raise ValueError(f'{label}, {part}: 상태에 맞는 영상 범위 내 좌표를 입력해 주세요.')


def _valid_geometry(geometry, is_box):
    if not isinstance(geometry, list) or len(geometry) != (4 if is_box else 2):
        return False
    width, height = IMAGE_SIZE
    if any(type(value) not in (int, float) or not 0 <= value <= width or not math.isfinite(value)
           for value in geometry):
        return False
    x, y = geometry[:2]
    if is_box:
        w, h = geometry[2:]
        return w > 0 and h > 0 and x + w <= width and y + h <= height
    return x < width and y < height
