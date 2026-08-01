"""Native extension loading and Mojo runtime compatibility helpers."""

from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path

from ._dispatch import load_native


def _runtime_library_directories() -> list[Path]:
    """Return likely Mojo runtime locations without recursively scanning disk."""
    directories: list[Path] = []
    configured = os.environ.get("MOJO_RUNTIME_LIB_DIR")
    if configured:
        directories.append(Path(configured).expanduser())

    directories.append(Path.home() / ".pixi" / "envs" / "mojo" / "lib")
    directories.append(Path(sys.prefix) / "lib")

    try:
        for packages in site.getsitepackages():
            directories.append(Path(packages) / "modular" / "lib")
    except AttributeError:
        pass

    unique: list[Path] = []
    for directory in directories:
        if directory not in unique:
            unique.append(directory)
    return unique


def _preload_mojo_runtime() -> bool:
    """Load a separately installed Mojo runtime for unbundled source builds."""
    if sys.platform == "darwin":
        suffix = ".dylib"
    elif sys.platform.startswith("linux"):
        suffix = ".so"
    else:
        return False

    library_names = (
        "libMSupportGlobals",
        "libAsyncRTRuntimeGlobals",
        "libAsyncRTMojoBindings",
        "libKGENCompilerRTShared",
    )
    for directory in _runtime_library_directories():
        compiler_runtime = directory / f"libKGENCompilerRTShared{suffix}"
        if not compiler_runtime.is_file():
            continue
        try:
            for name in library_names:
                library = directory / f"{name}{suffix}"
                if library.is_file():
                    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return True
    return False


try:
    _native, _native_backend = load_native()
except ImportError as native_import_error:
    runtime_names = ("libKGENCompilerRTShared", "libAsyncRTMojoBindings")
    if not any(name in str(native_import_error) for name in runtime_names):
        raise
    if not _preload_mojo_runtime():
        raise
    _native, _native_backend = load_native()
