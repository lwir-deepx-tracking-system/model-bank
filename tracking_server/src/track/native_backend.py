"""Optional runtime selection of same-name native tracker backends.

Native extension ABI v1:
  module: ``_native_<kind>`` in this directory's ignored ``_native/`` folder
  ``API_VERSION = 1``
  ``create_tracker(mode: str, kwargs: dict) -> tracker``
  optional ``SUPPORTED_MODES`` iterable of mode strings

No compilation occurs at runtime. A missing, incompatible, import-failing, or
mode-incompatible extension falls back to the supplied zero-argument Python
factory. Loaded extensions stay pinned to their file identity for the life of
the process. Use ``backend_status(kind)`` for the next tracker and
``backend_of(active_tracker)`` to label the actual instance.

``LWIR_NATIVE_TRACKERS=0`` in the environment keeps every tracker on Python
even when extensions are built.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
import importlib.util
import logging
import os
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType
from typing import Callable


_LOG = logging.getLogger(__name__)
_SOURCE_DIR = Path(__file__).resolve().parent
_SHARED_SOURCES = ("native_tracker_core.hpp",)   # included by sort.cpp, ocsort.cpp and botsort.cpp
_DISABLED = "disabled by LWIR_NATIVE_TRACKERS=0"
_NATIVE_DIR = _SOURCE_DIR / "_native"
_KIND_MODULE = {
    "bytetrack": "bytetrack",
    "confirm3": "bytetrack",
    "rebind": "bytetrack",
    "botsort": "botsort",
    "ocsort": "ocsort",
    "sort": "sort",
}
_WARNED: set[tuple[str, str]] = set()
_IMPORT_LOCK = RLock()
_RESTART_REQUIRED = "C++ extension changed after loading; server restart required"
_REQUIRED_TRACKER_ATTRIBUTES = (
    "update", "tracked", "lost", "frame_id", "next_id", "n_init", "kf",
    "new_this_frame", "lost_this_frame", "rebinds_motion", "rebinds_app",
    "rebinds_fallback",
)


@dataclass(frozen=True)
class _ExtensionImport:
    path: Path
    fingerprint: tuple[int, int, int, int, int]
    module: ModuleType | None
    error: str | None


_IMPORTS: dict[str, _ExtensionImport] = {}


def _module_stem(kind: str) -> str | None:
    return _KIND_MODULE.get(str(kind).strip().lower())


def _native_mode(kind: str, mode: str) -> str:
    # confirm3 is the same ByteTrack engine with distinct n_init kwargs.
    return "bytetrack" if str(kind).strip().lower() == "confirm3" else mode


def _fingerprint(stat_result) -> tuple[int, int, int, int, int]:
    return (stat_result.st_dev, stat_result.st_ino, stat_result.st_size,
            stat_result.st_mtime_ns, stat_result.st_ctime_ns)


def _load_module(kind: str):
    stem = _module_stem(kind)
    if stem is None:
        return None, f"unknown tracker kind: {kind}"
    if os.environ.get("LWIR_NATIVE_TRACKERS", "1").strip() == "0":
        return None, _DISABLED
    module_name = f"_native_{stem}"
    source = _SOURCE_DIR / f"{stem}.cpp"
    with _IMPORT_LOCK:
        return _load_extension(module_name, source)


def _load_extension(module_name: str, source: Path):
    previous = _IMPORTS.get(module_name)
    try:
        if not source.is_file():
            return None, f"{source.name} not present"
        binary = next(
            (candidate for suffix in importlib.machinery.EXTENSION_SUFFIXES
             if (candidate := _NATIVE_DIR / f"{module_name}{suffix}").is_file()),
            None,
        )
        if binary is None:
            return None, "matching extension not built"
        resolved_binary = binary.resolve()
        if resolved_binary.parent != _NATIVE_DIR.resolve():
            return None, "extension is outside _native"
        binary_stat = resolved_binary.stat()
        fingerprint = _fingerprint(binary_stat)
        if previous and (previous.path != resolved_binary or previous.fingerprint != fingerprint):
            return None, _RESTART_REQUIRED
        newest_source = max(path.stat().st_mtime_ns for path in
                            (source, *(source.parent / name for name in _SHARED_SOURCES)) if path.is_file())
        if binary_stat.st_mtime_ns < newest_source:
            return None, "C++ source is newer than the extension; rebuild needed"
    except Exception as exc:
        return None, f"{module_name} unavailable ({type(exc).__name__})"
    if previous:
        return previous.module, previous.error
    if module_name in sys.modules:
        # A module loaded outside this loader has no known on-disk identity.
        return None, _RESTART_REQUIRED
    module = None
    reason = None
    try:
        spec = importlib.util.spec_from_file_location(module_name, resolved_binary)
        if spec is None or spec.loader is None:
            return None, f"{module_name} has no extension loader"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if _fingerprint(resolved_binary.stat()) != fingerprint:
            reason = _RESTART_REQUIRED
        else:
            reason = _validate_module(module, module_name)
    except FileNotFoundError:
        reason = _RESTART_REQUIRED
    except Exception as exc:
        reason = f"{module_name} unavailable ({type(exc).__name__})"
    if reason is not None:
        if module is not None and sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        module = None
    # CPython can keep an extension mapped even when initialization fails.
    _IMPORTS[module_name] = _ExtensionImport(resolved_binary, fingerprint, module, reason)
    return module, reason


def _validate_module(module, module_name: str) -> str | None:
    if getattr(module, "API_VERSION", None) != 1:
        return f"{module_name} API_VERSION is not 1"
    if not callable(getattr(module, "create_tracker", None)):
        return f"{module_name} has no create_tracker(mode, kwargs)"
    return None


def _supports(module, kind: str, mode: str) -> bool:
    supported = getattr(module, 'SUPPORTED_MODES', None)
    if supported is None:
        return mode == _module_stem(kind)
    return isinstance(supported, (tuple, list, set, frozenset)) and mode in supported


def backend_status(kind: str, mode: str | None = None) -> dict:
    """Describe the available backend for the next tracker instance."""
    module, reason = _load_module(kind)
    if module is not None:
        native_mode = _native_mode(kind, mode or _default_mode(kind))
        if not _supports(module, kind, native_mode):
            module = None
            reason = f"mode {native_mode!r} not supported"
    return dict(backend="C++" if module is not None else "Python",
                restart_required=reason == _RESTART_REQUIRED, reason=reason)


def backend_label(kind: str, mode: str | None = None) -> str:
    """Return the available backend for a kind/mode: ``C++`` or ``Python``."""
    return backend_status(kind, mode)["backend"]


def _default_mode(kind: str) -> str:
    kind = str(kind).strip().lower()
    if kind == "confirm3":
        return "bytetrack"
    if kind in _KIND_MODULE:
        return kind
    return ""


class _LabeledTracker:
    """Transparent attribute proxy for extension objects without writable attrs."""

    __slots__ = ("_wrapped", "_backend_label")

    def __init__(self, wrapped, label: str):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_backend_label", label)

    @property
    def backend_label(self) -> str:
        return object.__getattribute__(self, "_backend_label")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        if name in _LabeledTracker.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)


def _tag(instance, label: str):
    try:
        setattr(instance, "backend_label", label)
        if getattr(instance, "backend_label", None) == label:
            return instance
    except Exception:
        pass
    wrapped = _LabeledTracker(instance, label)
    return wrapped


def backend_of(instance) -> str:
    """Return the actual backend attached to an active object, else ``Python``."""
    if isinstance(instance, _LabeledTracker):
        return instance.backend_label
    label = getattr(instance, "backend_label", None)
    if label in ("C++", "Python"):
        return label
    return "Python"


def _validate_tracker(instance) -> str | None:
    if not callable(getattr(instance, 'update', None)):
        return 'tracker update is not callable'
    missing = [name for name in _REQUIRED_TRACKER_ATTRIBUTES if not hasattr(instance, name)]
    return "missing tracker fields: " + ", ".join(missing) if missing else None


def _warn_once(kind: str, reason: str) -> None:
    key = (str(kind), reason)
    if key not in _WARNED:
        _WARNED.add(key)
        _LOG.warning("tracker %s: using Python backend (%s)", kind, reason)


def create(kind: str, mode: str, kwargs: dict, fallback: Callable[[], object], native: bool = True):
    """Create the same-name native tracker when valid, otherwise use fallback.

    ``fallback`` is always called with zero arguments. ``kwargs`` is copied
    before it reaches the native extension so the caller's mapping is untouched.
    ``native=False`` is the operator's explicit Python choice: no extension is tried.
    """
    if not native:
        instance = fallback()
        if instance is None:
            raise RuntimeError("Python fallback factory returned None")
        return _tag(instance, "Python")
    module, reason = _load_module(kind)
    native_mode = _native_mode(kind, mode)
    if module is not None:
        if not _supports(module, kind, native_mode):
            reason = f"mode {native_mode!r} not supported"
            module = None
    if module is not None:
        try:
            instance = module.create_tracker(native_mode, dict(kwargs or {}))
            if instance is None:
                raise RuntimeError("create_tracker returned None")
            invalid = _validate_tracker(instance)
            if invalid:
                reason = invalid
                module = None
            else:
                return _tag(instance, "C++")
        except Exception as exc:
            reason = f"native creation failed ({type(exc).__name__})"
    _warn_once(kind, reason or "native backend unavailable")
    instance = fallback()
    if instance is None:
        raise RuntimeError("Python fallback factory returned None")
    return _tag(instance, "Python")
