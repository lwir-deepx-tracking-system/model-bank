"""Reader of the camera's temperature ring (11-14, research feature, display / log only; nothing here runs unless it is switched on).

cam_pipe v2.11, started with LWIR_CAM_TEMP=N (1..300), calls the camera SDK's CalcTemp after every N-th frame it passes on and copies the
whole temperature map into a ring of 4 slots in shared memory. The layout is written at the top of cam_pipe.cpp (C++ side, with
static_asserts); this module is the server's ONE Python description of it (constants + reader). src/camera/test_cam_temp_ring.py (cam_pipe's
own layout test) and src/camera/test_temp_ring.py write rings byte by byte from that note and read them back.

Layout (native byte order = little endian on the board):
  file /dev/shm/lwir_temp_ring_<uid>
  file header, 64 bytes at 0, RING_HDR '<4s7IQ2If2I4x': magic 'LWT1', version 1, ring_id, nslots (4), stride, payload (= width*height*2),
    width, height, pid, every (N), state (1 = maps are written, 2 = stopped for this run, 3 = cam_pipe ended normally), emissivity
    (0 = SetEmissivity not called), max_ms, maps (maps written so far; map k is in slot (k - 1) % nslots; stored after that slot is complete)
  slot i at 64 + i * stride: slot header, 64 bytes, SLOT_HDR '<Q4sHHQIHHfIIIHH12x': frame (cam_pipe frame number of the picture = the
    LWF1 stamp header's seq; 0 while the slot is written), magic 'LWTS', version 1, flags (bit 0: the read before CalcTemp had the AGC on),
    t_mono_ns (CLOCK_MONOTONIC ns when that frame's RecvImage returned; the clock of Python's time.monotonic on Linux), calc_us, width,
    height, emissivity, map_seq (k), payload, ring_id, t_offset (5000), t_div (100); then height rows of width uint16: T_C = (v - t_offset) / t_div
  Reading (as in the cam_pipe note, the same rule as the picture ring camera_extra.Ring): k = maps; slot = (k - 1) % nslots; f1 = the slot's
  frame (0 = being written: take map k - 1); copy the slot header and the map; f2 = the slot's frame again; use the copy only if f1 == f2, the
  copied header's frame == f1 and it is map k of this ring ('gone' otherwise: overwritten before the first look).

Start: --cam-temp N (app.py) or the environment variable LWIR_CAM_TEMP passes LWIR_CAM_TEMP=N on to cam_pipe (apply_cli below). With neither,
nothing is passed and cam_pipe runs exactly as before."""
import collections
import mmap
import os
import stat
import struct

import numpy as np

ENV = 'LWIR_CAM_TEMP'
SHM_DIR = '/dev/shm'
NAME = 'lwir_temp_ring_%d'
RING_MAGIC, SLOT_MAGIC, VERSION = b'LWT1', b'LWTS', 1
RING_HDR = struct.Struct('<4s7IQ2If2I4x')        # magic, version, ring_id, nslots, stride, payload, width, height, pid, every, state, emissivity, max_ms, maps
SLOT_HDR = struct.Struct('<Q4sHHQIHHfIIIHH12x')   # frame, magic, version, flags, t_mono_ns, calc_us, width, height, emissivity, map_seq, payload, ring_id, t_offset, t_div
RING_FIELDS = ('magic', 'version', 'ring_id', 'nslots', 'stride', 'payload', 'width', 'height', 'pid', 'every', 'state', 'emissivity', 'max_ms', 'maps')
SLOT_FIELDS = ('frame', 'magic', 'version', 'flags', 't_mono_ns', 'calc_us', 'width', 'height', 'emissivity', 'map_seq', 'payload', 'ring_id', 't_offset', 't_div')
HDR_N = 64                                       # file header and slot header
MAPS_OFF = 56                                    # u32 maps in the file header
STATE_ON, STATE_STOPPED, STATE_ENDED = 1, 2, 3
FLAG_AGC_READ = 1
TEMP_OFFSET, TEMP_DIV = 5000.0, 100.0            # what cam_pipe v2.11 writes into every slot (T_C = (v - 5000) / 100)
assert RING_HDR.size == HDR_N and SLOT_HDR.size == HDR_N
_Q8 = struct.Struct('<Q')

# frame = cam_pipe frame number, t_ns = CLOCK_MONOTONIC ns of that frame, values = uint16 [height, width] (a private copy), state / every = from the file header
TempMap = collections.namedtuple('TempMap', 'frame map_seq ring_id t_ns width height emissivity calc_us flags t_offset t_div state every values')


def ring_path():
    return os.path.join(SHM_DIR, NAME % (os.getuid() if hasattr(os, 'getuid') else 0))


def stride(w, h):
    """bytes of one slot: header + map, rounded up to 64 (as cam_pipe makes it)"""
    return (HDR_N + int(w) * int(h) * 2 + 63) // 64 * 64


def to_celsius(v, t_offset=TEMP_OFFSET, t_div=TEMP_DIV):
    """map values (any shape, uint16) -> float32 degrees C"""
    return (np.asarray(v, np.float32) - np.float32(t_offset)) / np.float32(t_div)


def from_celsius(c, t_offset=TEMP_OFFSET, t_div=TEMP_DIV):
    """degrees C -> the uint16 values cam_pipe stores (tests, synthetic maps)"""
    return np.clip(np.round(np.asarray(c, np.float64) * t_div + t_offset), 0, 65535).astype(np.uint16)


def apply_cli(n):
    """11-14, called once at the live start before cam_pipe is started: --cam-temp N > 0 puts LWIR_CAM_TEMP=N in this process's
    environment, so every start of cam_pipe inherits it (lv_spawn, camera_extra.spawn). 0 (the default) changes nothing: a LWIR_CAM_TEMP
    given in the environment is passed on as it is, and without one nothing is passed. Returns the value cam_pipe will get ('' = none)."""
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        os.environ[ENV] = str(n)
    v = os.environ.get(ENV, '')
    if v:
        print('CAM_TEMP cam_pipe gets %s=%s (%s): temperature maps in %s' % (ENV, v, '--cam-temp' if n > 0 else 'environment', ring_path()), flush=True)
    return v


def ring_info(buf):
    """the file header as a dict, or None when buf is not a whole version-1 temperature ring"""
    if len(buf) < HDR_N:
        return None
    h = dict(zip(RING_FIELDS, RING_HDR.unpack_from(buf, 0)))
    if h['magic'] != RING_MAGIC or h['version'] != VERSION:
        return None
    if not 1 <= h['nslots'] <= 64 or h['stride'] < HDR_N + h['payload'] or h['payload'] != h['width'] * h['height'] * 2 or len(buf) < HDR_N + h['nslots'] * h['stride']:
        return None
    return h


class TempRing:
    """Read-only access to the ring. newest() maps the file, takes the newest complete map and unmaps it again (a few tens of
    microseconds; a cam_pipe that starts again replaces the file, so nothing is kept open). The map is copied only when the newest map is
    another one than last time (ring id, map number); otherwise the map returned last time is returned again."""
    def __init__(self, path=None):
        self.path = path or ring_path()
        self.last = None
        self.info = None   # the file header of the last read (state, every, maps, ...)
        self.stats = dict(reads=0, maps=0, same=0, missing=0, empty=0, writing=0, torn=0, gone=0, bad=0, last_error='')
        self._between = None   # test hook: called after a copy, before the frame number is read again

    def newest(self):
        """-> TempMap of the newest complete map, or None (no ring, no map yet, newest two being written / overwritten)"""
        self.stats['reads'] += 1
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
        except FileNotFoundError:
            self.stats['missing'] += 1; self.info = None
            return None
        except OSError as e:
            self.stats['bad'] += 1; self.stats['last_error'] = 'open: %s' % e
            return None
        mm = None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or (hasattr(os, 'getuid') and st.st_uid != os.getuid()):
                self.stats['bad'] += 1; self.stats['last_error'] = 'not a regular file of this user'
                return None
            if st.st_size < HDR_N:
                self.stats['empty'] += 1
                return None
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            return self._pick(mm)
        except (OSError, ValueError, struct.error) as e:
            self.stats['bad'] += 1; self.stats['last_error'] = '%s: %s' % (type(e).__name__, str(e)[:120])
            return None
        finally:
            if mm is not None:
                mm.close()
            os.close(fd)

    def _pick(self, mm):
        info = ring_info(mm)
        self.info = info
        if info is None:
            self.stats['bad'] += 1; self.stats['last_error'] = 'not a version-1 temperature ring (magic %r)' % bytes(mm[:4])
            return None
        k = info['maps']
        for kk in (k, k - 1):
            if kk < 1:
                self.stats['empty'] += 1
                return None
            r = self._slot(mm, info, kk)
            if isinstance(r, TempMap):
                return r
            self.stats[r] += 1
        return None

    def _slot(self, mm, info, k):
        off = HDR_N + ((k - 1) % info['nslots']) * info['stride']
        f1 = _Q8.unpack_from(mm, off)[0]
        if f1 == 0:
            return 'writing'
        last = self.last
        same = last is not None and last.ring_id == info['ring_id'] and last.map_seq == k
        head = mm[off:off + HDR_N]
        vals = None if same else mm[off + HDR_N:off + HDR_N + info['payload']]
        if self._between is not None: self._between()
        f2 = _Q8.unpack_from(mm, off)[0]
        s = dict(zip(SLOT_FIELDS, SLOT_HDR.unpack(head)))
        if f2 != f1 or s['frame'] != f1:
            return 'torn'
        if (s['magic'] != SLOT_MAGIC or s['version'] != VERSION or s['map_seq'] != k or s['ring_id'] != info['ring_id']
                or s['width'] != info['width'] or s['height'] != info['height'] or s['payload'] != info['payload'] or s['t_div'] == 0):
            return 'gone'
        if same and last.frame == f1 and last.t_ns == s['t_mono_ns']:
            self.stats['same'] += 1
            if (last.state, last.every) != (info['state'], info['every']):
                self.last = last._replace(state=int(info['state']), every=int(info['every']))
            return self.last
        if vals is None:   # the cached map number now holds another frame (cannot happen with one writer): copy it next time
            return 'torn'
        self.last = TempMap(f1, k, info['ring_id'], s['t_mono_ns'], s['width'], s['height'], float(s['emissivity']), int(s['calc_us']), int(s['flags']),
                            int(s['t_offset']), int(s['t_div']), int(info['state']), int(info['every']),
                            np.frombuffer(vals, '<u2').reshape(s['height'], s['width']))
        self.stats['maps'] += 1
        return self.last
