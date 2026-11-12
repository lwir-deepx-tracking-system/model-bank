"""Optional runtime selection of same-name native tracker backends.

Native extension ABI v1:
  module: ``_native_<kind>`` in this directory's ignored ``_native/`` folder
  ``API_VERSION = 1``
  ``create_tracker(mode: str, kwargs: dict) -> tracker``
  optional ``SUPPORTED_MODES`` iterable of mode strings

No compilation occurs at runtime. A missing, incompatible, import-failing, or
mode-incompatible extension falls back to the supplied zero-argument Python
factory. Use ``backend_of(active_tracker)`` to label the actual instance.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
from pathlib import Path
import sys
from typing import Callable


_LOG = logging.getLogger(__name__)
_NATIVE_DIR = Path(__file__).resolve().parent / "_native"
_KIND_MODULE = {
    "bytetrack": "bytetrack",
    "confirm3": "bytetrack",
    "rebind": "bytetrack",
    "botsort": "botsort",
    "ocsort": "ocsort",
    "sort": "sort",
}
_WARNED: set[tuple[str, str]] = set()
_REQUIRED_TRACKER_ATTRIBUTES = (
    "update", "tracked", "lost", "frame_id", "next_id", "n_init", "kf",
    "new_this_frame", "lost_this_frame", "rebinds_motion", "rebinds_app",
    "rebinds_fallback",
)


def _module_stem(kind: str) -> str | None:
    return _KIND_MODULE.get(str(kind).strip().lower())


def _native_mode(kind: str, mode: str) -> str:
    # confirm3 is the same ByteTrack engine with distinct n_init kwargs.
    return "bytetrack" if str(kind).strip().lower() == "confirm3" else mode


def _load_module(kind: str):
    stem = _module_stem(kind)
    if stem is None:
        return None, f"unknown tracker kind: {kind}"
    module_name = f"_native_{stem}"
    source = Path(__file__).resolve().parent / f"{stem}.cpp"
    if not source.is_file():
        return None, f"{source.name} not present"
    binary = next(
        (candidate for suffix in importlib.machinery.EXTENSION_SUFFIXES
         if (candidate := _NATIVE_DIR / f"{module_name}{suffix}").is_file()),
        None,
    )
    if binary is None:
        return None, f"matching extension not built"
    try:
        if binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
            return None, 'C++ source is newer than the extension; rebuild needed'
        resolved_binary = binary.resolve()
        if resolved_binary.parent != _NATIVE_DIR.resolve():
            return None, 'extension is outside _native'
        cached = sys.modules.get(module_name)
        if cached is not None and Path(getattr(cached, "__file__", "")).resolve() == resolved_binary:
            module = cached
        else:
            spec = importlib.util.spec_from_file_location(module_name, resolved_binary)
            if spec is None or spec.loader is None:
                return None, f"{module_name} has no extension loader"
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
    except Exception as exc:
        return None, f"{module_name} unavailable ({type(exc).__name__})"
    if getattr(module, "API_VERSION", None) != 1:
        return None, f"{module_name} API_VERSION is not 1"
    if not callable(getattr(module, "create_tracker", None)):
        return None, f"{module_name} has no create_tracker(mode, kwargs)"
    return module, None


def _supports(module, kind: str, mode: str) -> bool:
    supported = getattr(module, 'SUPPORTED_MODES', None)
    if supported is None:
        return mode == _module_stem(kind)
    return isinstance(supported, (tuple, list, set, frozenset)) and mode in supported


def backend_label(kind: str, mode: str | None = None) -> str:
    """Return the available backend for a kind/mode: ``C++`` or ``Python``."""
    module, _ = _load_module(kind)
    if module is not None:
        native_mode = _native_mode(kind, mode or _default_mode(kind))
        if not _supports(module, kind, native_mode):
            module = None
    return "C++" if module is not None else "Python"


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


def create(kind: str, mode: str, kwargs: dict, fallback: Callable[[], object]):
    """Create the same-name native tracker when valid, otherwise use fallback.

    ``fallback`` is always called with zero arguments. ``kwargs`` is copied
    before it reaches the native extension so the caller's mapping is untouched.
    """
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
