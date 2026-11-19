"""Per-frame result records kept by the frame loop (APP.preds, APP.tracks_out, APP.tracks_base), stored compactly.

2026-11-13: the server keeps the last 3000-6000 frames of these records in memory and writes them to files only when the run
ends. Kept as nested Python lists (one list per detection and per track row) they grew to about a million objects that
Python's garbage collector has to walk; on the board a full (generation 2) collection then took 32 -> 68 -> 97 -> 129 ms of
CPU while holding the GIL, and each time the camera reader could not take frames (camera_events.log: GC_SLOW, GIL_STALL and
cam_pipe INPUT_GAP lines at the same moments). Numbers in a NumPy array are not objects the collector walks, so the records
now keep arrays and are turned into exactly the old lists only when a file is written. The files are unchanged.
"""
import numpy as np

TRACK_KEY = 'tracks_xyxy_id_conf_class_detectionindex'
HISTORY_LIMIT = 6000
HISTORY_TRIM = 3000
RELEASE_PER_FRAME = 16


class LiveHistoryTrim:
    """Keep the same visible window while spreading memory release across frames."""

    def __init__(self):
        self._retired = []

    def trim(self, histories):
        for records in histories:
            # The slice keeps the objects alive when the live history drops them.
            self._retired.append(records[:HISTORY_TRIM])
            del records[:HISTORY_TRIM]

    def release(self):
        if not self._retired:
            return
        for chunk in self._retired:
            del chunk[-RELEASE_PER_FRAME:]
        if not any(self._retired):
            self._retired.clear()


def pred_arrays(pc, *, dy=0.0):
    """Pack detections [N, >=6] into a PRED record with the optional y shift."""
    boxes = pc[:, :4]
    if dy:
        boxes = boxes.copy()
        boxes[:, 1] += dy
        boxes[:, 3] += dy
        np.round(boxes, 2, out=boxes)
    else:
        boxes = np.round(boxes, 2)
    return boxes, np.round(pc[:, 4], 5), pc[:, 5].astype(int)


def track_array(rows, dy):
    """rows = track rows (x1, y1, x2, y2, id, score, class, detection index); y1 and y2 are shifted by -dy, nothing is rounded here."""
    out = np.empty((len(rows), 8), np.float64)
    if not len(rows):
        return out
    if isinstance(rows, np.ndarray) or all(len(row) == 8 for row in rows):
        packed = np.asarray(rows, dtype=np.float64)
        if packed.ndim == 2 and packed.shape[1] >= 8:
            out[:] = packed[:, :8]
            out[:, 1] -= dy
            out[:, 3] -= dy
            return out
    for k, r in enumerate(rows):
        out[k, :4] = [float(v) - (dy if j in (1, 3) else 0.0) for j, v in enumerate(r[:4])]
        out[k, 4:] = [float(r[4]), float(r[5]), float(r[6]), float(r[7])]
    return out


def plain(record):
    """A record as it was written before 11-13 (lists and Python numbers): the same json.dumps text."""
    out = dict(record)
    for key in ('xyxy', 'conf', 'cls'):
        if isinstance(out.get(key), np.ndarray): out[key] = out[key].tolist()
    t = out.get(TRACK_KEY)
    if isinstance(t, np.ndarray):
        out[TRACK_KEY] = [[round(float(a[0]), 2), round(float(a[1]), 2), round(float(a[2]), 2), round(float(a[3]), 2),
                           int(a[4]), round(float(a[5]), 5), int(a[6]), int(a[7])] for a in t]
    return out
