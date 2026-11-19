"""Validated fixed-buffer views for the optional native dual-tile backend."""

import ctypes as c
import math
import os
from pathlib import Path

import numpy as np

from ..common.shared_values import HERE


def parallel_wait_requested():
    try:
        with (HERE / 'scripts' / 'split_wait_mode').open('rb') as setting:
            mode = setting.read(16).strip()
        if mode in (b'sequential', b'parallel'):
            return mode == b'parallel'
    except OSError:
        pass
    return os.environ.get('LWIR_SPLIT_PARALLEL_WAIT') == '1'


class TensorInfo(c.Structure):
    _fields_ = (("name", c.c_char * 256), ("type", c.c_int), ("elem_size", c.c_uint32),
                ("ndim", c.c_int), ("shape", c.c_int64 * 8), ("size", c.c_uint64))


class Timing(c.Structure):
    _fields_ = tuple((field, c.c_uint64 * 2) for field in
                     ("submit_start", "submit_done", "wait_start", "wait_done", "wait_cpu")) + (
                         ("npu_us", c.c_uint32 * 2), ("latency_us", c.c_int * 2), ("job_id", c.c_int * 2))


class CoreMetrics:
    def __init__(self, pair, core):
        self.pair, self.core = pair, core

    def get_npu_inference_time(self):
        return self.pair.timing.npu_us[self.core]

    def get_latency(self):
        return self.pair.timing.latency_us[self.core]


class NativeTiles:
    def __init__(self, library, model, input_shape, shapes):
        self.lib = c.CDLL(str(library))
        ptr, size, u64ptr = c.c_void_p, c.c_size_t, c.POINTER(c.c_uint64)
        for name, args, result in (
            ("open", (c.c_char_p, u64ptr, u64ptr), ptr),
            ("metadata", (ptr, c.c_int, c.c_int, c.POINTER(TensorInfo), c.POINTER(c.c_int)), c.c_int),
            ("submit", (ptr, ptr, ptr, size, c.POINTER(Timing)), c.c_int),
            ("wait", (ptr, ptr, ptr, size, c.POINTER(Timing)), c.c_int),
            ("error", (), c.c_char_p), ("close", (ptr,), None),
        ):
            function = getattr(self.lib, "lwir_split_" + name)
            function.argtypes, function.restype = args, result
        input_size, output_size = c.c_uint64(), c.c_uint64()
        self.handle = self.lib.lwir_split_open(str(model).encode(), c.byref(input_size), c.byref(output_size))
        if not self.handle:
            raise RuntimeError(self.error())
        self.timing = Timing()
        self.parallel_wait = False
        self.input_size, self.output_size = input_size.value, output_size.value
        try:
            self.layout = self.layouts(input_shape, shapes)
            self.buffers = [np.empty(self.output_size, np.uint8) for _ in range(2)]
            self.views = [self.tensor_views(buffer) for buffer in self.buffers]
            self.engines = [CoreMetrics(self, core) for core in (0, 1)]
            if parallel_wait_requested():
                set_mode = getattr(self.lib, 'lwir_split_set_parallel_wait', None)
                if set_mode is not None:
                    set_mode.argtypes, set_mode.restype = (ptr, c.c_int), c.c_int
                    self.parallel_wait = set_mode(self.handle, 1) == 0
        except BaseException:
            self.close()
            raise

    def error(self):
        value = self.lib.lwir_split_error()
        return value.decode("utf-8", errors="replace") if value else "DX-RT native split failed"

    def metadata(self, core, output):
        count = c.c_int()
        if self.lib.lwir_split_metadata(self.handle, core, output, None, c.byref(count)):
            raise RuntimeError(self.error())
        capacity = count.value
        if not 1 <= capacity <= 64:
            raise ValueError("Unexpected native tensor metadata count")
        data = (TensorInfo * capacity)()
        if self.lib.lwir_split_metadata(self.handle, core, output, data, c.byref(count)):
            raise RuntimeError(self.error())
        if count.value != capacity or any(not 1 <= item.ndim <= 8 for item in data):
            raise ValueError("Native tensor dimensions or count changed")
        return [(item.name.decode(), item.type, item.elem_size, tuple(item.shape[:item.ndim]), item.size)
                for item in data]

    def layouts(self, input_shape, shapes):
        first = None
        for core in (0, 1):
            inputs = self.metadata(core, 0)
            outputs = self.metadata(core, 1)
            if (len(inputs) != 1 or inputs[0][1:] != (2, 1, input_shape, self.input_size)
                    or self.input_size != math.prod(input_shape)):
                raise ValueError("Native tile input layout differs")
            if len(outputs) != len(shapes) or {entry[0] for entry in outputs} != shapes.keys():
                raise ValueError("Native tile output names differ")
            for name, dtype, elem_size, shape, bytes_ in outputs:
                if dtype != 1 or elem_size != 4 or shape != shapes[name] or bytes_ != math.prod(shape) * 4:
                    raise ValueError(f"Native output layout differs for {name}")
            if first is not None and outputs != first:
                raise ValueError("Native tile outputs disagree on layout")
            first = outputs
        if sum(item[4] for item in first) != self.output_size:
            raise ValueError("Native tile output byte count differs")
        return first

    def tensor_views(self, buffer):
        offset = 0
        result = {}
        for name, _, _, shape, bytes_ in self.layout:
            result[name] = np.frombuffer(buffer, dtype=np.float32, count=bytes_ // 4, offset=offset).reshape(shape)
            offset += bytes_
        return result

    def submit(self, inputs):
        if self.lib.lwir_split_submit(self.handle, inputs[0].ctypes.data, inputs[1].ctypes.data,
                                      self.input_size, c.byref(self.timing)):
            raise RuntimeError(self.error())
        return list(self.timing.job_id)

    def wait(self, jobs, names, *, owned=False):
        # An eager feeder can collect the next frame before the tracker has
        # decoded this one. Each handed-off frame owns its SDK destinations.
        buffers = [np.empty_like(buf) for buf in self.buffers] if owned else self.buffers
        self.wait_into(jobs, [buffer.ctypes.data for buffer in buffers])
        views = [self.tensor_views(buf) for buf in buffers] if owned else self.views
        return [[tile[name] for name in names] for tile in views]

    def wait_into(self, jobs, destinations):
        if list(self.timing.job_id) != jobs:
            raise RuntimeError("Native split job identity changed")
        if self.lib.lwir_split_wait(self.handle, *destinations, self.output_size, c.byref(self.timing)):
            raise RuntimeError(self.error())

    def close(self):
        if self.handle:
            self.lib.lwir_split_close(self.handle)
            self.handle = None
