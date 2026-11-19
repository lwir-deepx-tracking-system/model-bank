"""Optional native frame geometry. Calls use frame-owned, contiguous arrays."""
from collections import deque
import ctypes as c
import os
from pathlib import Path
from threading import Lock

import numpy as np

lib = None
if os.environ.get('LWIR_CPP_FRAME_MATH', '1') == '1':
    try:
        lib = c.CDLL(str(Path(__file__).with_name('liblwir_frame_math.so')))
    except OSError:
        pass


def bind(name, args, result=c.c_int):
    fn = getattr(lib, 'lwir_' + name, None)
    if fn is not None:
        fn.argtypes, fn.restype = args, result
    return fn


f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')

_keepid = bind('keepid_geometry', [f64, i64, i32, c.c_int, f64, c.c_int, c.c_double, f64, u8])


def keepid_geometry(rows, detections, near_heights):
    # Small frames cost less to handle directly. P from the live tracker
    # contains Python floats, whose arithmetic defines the frozen features.
    if _keepid is None or len(rows) * (len(rows) + len(detections)) < 32:
        return None
    if any(type(box) not in (tuple, list) or len(box) != 4 or
           any(type(v) not in (float, int) for v in box)
           for box in detections.values()):
        return None
    order = {key: i for i, key in enumerate(detections)}
    try:
        track_ids = np.array([row[4] for row in rows], np.int64)
    except OverflowError:
        return None
    indices = np.array([order.get(row[7], -1) for row in rows], np.int32)
    boxes = np.array([row[:4] for row in rows], np.float64)
    det_boxes = np.array(list(detections.values()), np.float64).reshape(-1, 4)
    geometry = np.empty((len(rows), 4), np.float64)
    fallback = np.empty(len(rows), np.uint8)
    if not _keepid(boxes, track_ids, indices, len(rows), det_boxes, len(det_boxes),
                   near_heights, geometry, fallback):
        return None
    return [None if old else (float(g[0]), int(g[1]), int(g[2]), int(g[3]))
            for g, old in zip(geometry, fallback)]


_position = bind('position_seen', [f64, f64, f64, u8, c.c_int, f32, f32, c.c_int,
                                   c.c_int, c.c_int, f64, f64])
_float32_with_float64_scalar = np.result_type(np.empty(0, np.float32), np.float64(0)) == np.float64
_no_boxes = np.empty((0, 4), np.float32)
_no_poses = np.empty((0, 17, 3), np.float32)


def position_available():
    return _position is not None and _float32_with_float64_scalar


def position_seen(states, innovations, previous, pose_result, posetrack, edges):
    boxes, poses = ((pose_result['boxes'], pose_result['kpts']) if pose_result else
                     (_no_boxes, _no_poses))
    count = len(states)
    if (boxes.dtype != np.float32 or poses.dtype != np.float32 or
            boxes.shape != (len(boxes), 4) or poses.shape != (len(boxes), 17, 3) or
            states.dtype != np.float64 or states.shape != (count, 6)):
        return None
    innovations = np.ascontiguousarray(innovations, dtype=np.float64)
    has_previous = np.array([z is not None for z in previous], np.uint8)
    prior = np.zeros((count, 4), np.float64)
    for i, z in enumerate(previous):
        if z is not None:
            prior[i] = z
    z = np.empty((count, 4), np.float64)
    features = np.empty((count, 68), np.float64)
    if innovations.shape != (count, 4) or not _position(
            np.ascontiguousarray(states), innovations, prior, has_previous, count,
            np.ascontiguousarray(boxes), np.ascontiguousarray(poses), len(boxes),
            posetrack, edges, z, features):
        return None
    logs = features[:, 8:12]
    np.log(logs, out=logs)
    np.clip(features, -5, 5, out=features)
    return z, features


_nested = bind('nested_people', [f32, f32, c.c_int, u8])


def nested_people(boxes, poses):
    if (_nested is None or boxes.dtype != np.float32 or poses.dtype != np.float32
            or boxes.shape != (len(boxes), 5) or poses.shape != (len(boxes), 17, 3)):
        return None
    keep = np.empty(len(boxes), np.bool_)
    if _nested(np.ascontiguousarray(boxes), np.ascontiguousarray(poses), len(boxes), keep.view(np.uint8)):
        return keep
    return None


_merge = bind('merge_heads', [c.POINTER(c.c_void_p), i32, i32, c.c_int, c.POINTER(c.c_void_p)])
_anchors = bind('pose_anchors', [f32, c.c_int, f32, c.c_int, i64])
_head_planes = np.array([4, 16, 51], np.int32)


def merge_heads(values, slots):
    if _merge is None or len(values) != len(slots) or len(slots) not in (6, 9):
        return None
    raw = tuple(np.asarray(value) for value in values)
    if any(a.dtype != np.float32 or not a.flags.c_contiguous or a.size != s.size
           for a, s in zip(raw, slots)):
        return None
    spatial = np.array([slot.size // 4 for slot in slots[:3]], np.int32)
    planes = _head_planes[:len(slots) // 3]
    if any(a.size != int(planes[i // 3]) * int(spatial[i % 3]) for i, a in enumerate(raw)):
        return None
    total = int(spatial.sum())
    result = [np.empty((1, group, total), np.float32) for group in planes]
    src = (c.c_void_p * len(raw))(*(a.ctypes.data for a in raw))
    dst = (c.c_void_p * len(result))(*(a.ctypes.data for a in result))
    return result if _merge(src, spatial, planes, len(result), dst) else None


def pose_anchors(decoded, rows):
    if (_anchors is None or decoded.ndim != 2 or decoded.shape[0] < 5
            or decoded.dtype != np.float32 or rows.dtype != np.float32
            or not decoded.flags.c_contiguous or rows.ndim != 2 or rows.shape[1] != 6):
        return None
    picked = np.empty(len(rows), np.int64)
    if _anchors(decoded, decoded.shape[1], np.ascontiguousarray(rows), len(rows), picked):
        return picked
    return None


_body = bind('body_deltas', [f64, f64, c.c_int, f64, c.c_int, f64, c.c_int, f64])
_motion = bind('motion_chain', [f64, c.c_int, c.c_int, c.c_int, c.c_double, c.c_double, f64])


def body_deltas(table, heights, ratios, log_height, aspect):
    if _body is None or len(heights) < 2 or len(ratios) < 2:
        return None
    inputs = np.column_stack((log_height, aspect))
    if inputs.dtype != np.float64:
        return None
    out = np.empty((len(inputs), 4), np.float64)
    if _body(np.ascontiguousarray(table), np.ascontiguousarray(heights), len(heights),
             np.ascontiguousarray(ratios), len(ratios), inputs, len(inputs), out):
        return out
    return None


class MotionHistory:
    """Keep the normal frame history and a contiguous view for native queries."""
    def __init__(self, size):
        self.history = deque(maxlen=size)
        self.buffer = np.empty((size, 8), np.float64)
        self.cursor = 0
        self.lock = Lock()

    def __iter__(self):
        with self.lock:
            return iter(tuple(self.history))

    def __len__(self):
        return len(self.history)

    def append(self, entry):
        start, end, warp = entry
        with self.lock:
            self.history.append(entry)
            row = self.buffer[self.cursor]
            row[:2] = start, end
            try:
                matrix = np.asarray(warp, dtype=np.float64)
                row[2:] = matrix.ravel() if matrix.shape == (2, 3) else np.nan
            except (TypeError, ValueError):
                row[2:] = np.nan
            self.cursor = (self.cursor + 1) % len(self.buffer)

    def compose(self, start, end):
        size = len(self.buffer)
        out = np.empty((2, 3), np.float64)
        with self.lock:
            count = len(self.history)
            if _motion(self.buffer, size, (self.cursor - count) % size, count, start, end, out):
                return out
        return None


def motion_history(size):
    return MotionHistory(size) if _motion is not None else deque(maxlen=size)
