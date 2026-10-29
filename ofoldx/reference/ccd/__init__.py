r"""Build, publish and retrieve versioned Chemical Component Dictionary artifacts."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...chemistry.definitions import load_library
    from .build import build_artifact
    from .hub import download, upload

__all__ = ["build_artifact", "download", "load_library", "upload"]

_EXPORTS = {
    "build_artifact": ".build",
    "download": ".hub",
    "load_library": "...chemistry.definitions",
    "upload": ".hub",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
