"""Model Bank row format bridge for the team's offline C++ tracker presets."""
import ctypes as C
from pathlib import Path

import numpy as np


class _Det(C.Structure):
    _fields_ = [(name, C.c_float) for name in ("x1", "y1", "x2", "y2", "score")] + [("cls", C.c_int)]


class _Track(C.Structure):
    _fields_ = [(name, C.c_float) for name in ("x1", "y1", "x2", "y2", "score")] + [
        (name, C.c_int) for name in ("id", "cls", "source_index")]


class TeamTracker:
    """Explicit offline selection; no implicit Python fallback or server lifecycle emulation."""

    def __init__(self, library, kind):
        code = {"bytetrack": 0, "ocsort": 1}[kind]
        lib = C.CDLL(str(Path(library).resolve(strict=True)))
        lib.lwirtk_abi.argtypes, lib.lwirtk_abi.restype = [], C.c_int
        if lib.lwirtk_abi() != 1:
            raise RuntimeError("Unsupported team tracker ABI")
        lib.lwirtk_create.argtypes, lib.lwirtk_create.restype = [C.c_int], C.c_void_p
        lib.lwirtk_close.argtypes, lib.lwirtk_close.restype = [C.c_void_p], None
        lib.lwirtk_reset.argtypes, lib.lwirtk_reset.restype = [C.c_void_p], C.c_int
        lib.lwirtk_step.argtypes = [C.c_void_p, C.POINTER(_Det), C.c_int, C.POINTER(_Track), C.c_int]
        lib.lwirtk_step.restype = C.c_int
        self._lib = lib
        self._handle = lib.lwirtk_create(code)
        if not self._handle:
            raise RuntimeError(f"Could not create team {kind}")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("Team tracker is closed")

    def close(self):
        if self._handle:
            self._lib.lwirtk_close(self._handle)
            self._handle = None

    def reset(self):
        self._check_open()
        if self._lib.lwirtk_reset(self._handle):
            raise RuntimeError("Could not reset team tracker")

    def update(self, rows):
        self._check_open()
        rows = np.asarray(rows, dtype=np.float32)
        if rows.shape == (0,):
            rows = rows.reshape(0, 6)
        if (rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all()
                or np.any(rows[:, 4:] < 0) or np.any(rows[:, 4] > 1)
                or np.any(rows[:, 5] > 2147483520) or np.any(rows[:, 5] != np.trunc(rows[:, 5]))):
            raise ValueError("Expected finite [N,6] xyxy, score in [0,1], and nonnegative integer class IDs")
        n = len(rows)
        inputs = (_Det * n)(*(_Det(*row[:5], int(row[5])) for row in rows))
        output = (_Track * n)()
        size = self._lib.lwirtk_step(self._handle, inputs, n, output, n)
        if size < 0:
            self.close()  # after a native failure the frame may have partially advanced
            raise RuntimeError(f"Team tracker step failed ({size})")
        return [[r.x1, r.y1, r.x2, r.y2, r.id, r.score, r.cls, r.source_index]
                for r in output[:size]]
