"""Public managed Python API for MojoVec.

The implementation is split into private modules so the public package remains
small, stable, and easy for IDEs and documentation tools to inspect.
"""

from __future__ import annotations

from ._collection import Collection, load, recover
from ._runtime import _native_backend
from ._types import (
    DEFAULT_MMAP_THRESHOLD_BYTES,
    WAL_ASYNC,
    WAL_SYNC,
    Metadata,
    QueryResult,
    Where,
)

__version__ = "0.7.0"


def native_backend() -> str:
    """Return the native CPU backend selected for this Python process.

    Returns:
        "avx2" or "avx512" for Linux fat wheels and "native" for
        platform-specific builds such as Apple Silicon.
    """
    return _native_backend


# Re-exported objects should appear to users and documentation tools as members
# of the public package rather than implementation modules.
Collection.__module__ = __name__
load.__module__ = __name__
recover.__module__ = __name__

__all__ = [
    "Collection",
    "DEFAULT_MMAP_THRESHOLD_BYTES",
    "Metadata",
    "QueryResult",
    "WAL_ASYNC",
    "WAL_SYNC",
    "Where",
    "__version__",
    "load",
    "native_backend",
    "recover",
]
