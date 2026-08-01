"""Select a CPU-compatible Mojo native extension before loading native code."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from types import ModuleType


_X86_64_MACHINES = frozenset({"amd64", "x86_64"})

# Linux /proc/cpuinfo spellings for the x86-64-v3 ISA level. The compiler may
# use any instruction in this level, so checking AVX2 alone is not sufficient.
_X86_64_V3_FLAGS = frozenset(
    {
        "avx",
        "avx2",
        "bmi1",
        "bmi2",
        "f16c",
        "fma",
        "movbe",
        "popcnt",
        "sse4_1",
        "sse4_2",
        "xsave",
    }
)
_X86_64_V4_FLAGS = _X86_64_V3_FLAGS | frozenset(
    {
        "avx512bw",
        "avx512cd",
        "avx512dq",
        "avx512f",
        "avx512vl",
    }
)
_BACKEND_ENVIRONMENT_VARIABLE = "MOJOVEC_FORCE_BACKEND"


def _parse_linux_cpu_flags(cpuinfo: str) -> frozenset[str]:
    """Return the intersection of flags advertised for every logical CPU."""
    per_cpu: list[frozenset[str]] = []
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"flags", "features"}:
            per_cpu.append(frozenset(value.lower().split()))
    if not per_cpu:
        return frozenset()
    flags = per_cpu[0]
    for cpu_flags in per_cpu[1:]:
        flags &= cpu_flags
    return flags


def _linux_cpu_flags() -> frozenset[str]:
    try:
        return _parse_linux_cpu_flags(
            Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        )
    except OSError:
        return frozenset()


def _select_linux_x86_backend(
    flags: frozenset[str], forced: str | None = None
) -> str:
    if forced is not None:
        forced = forced.strip().lower()
        if forced not in {"avx2", "avx512"}:
            raise ImportError(
                f"{_BACKEND_ENVIRONMENT_VARIABLE} must be 'avx2' or 'avx512'"
            )

    if forced == "avx512" or (forced is None and _X86_64_V4_FLAGS <= flags):
        missing = _X86_64_V4_FLAGS - flags
        if missing:
            raise ImportError(
                "AVX-512 backend requested, but the CPU is missing: "
                + ", ".join(sorted(missing))
            )
        return "avx512"

    missing = _X86_64_V3_FLAGS - flags
    if missing:
        raise ImportError(
            "MojoVec Linux wheels require an x86-64-v3 CPU (AVX2). Missing: "
            + ", ".join(sorted(missing))
        )
    return "avx2"


def load_native() -> tuple[ModuleType, str]:
    """Import the fastest native extension safe for the current CPU."""
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in _X86_64_MACHINES:
        backend = _select_linux_x86_backend(
            _linux_cpu_flags(), os.environ.get(_BACKEND_ENVIRONMENT_VARIABLE)
        )
        module = importlib.import_module(f"{__package__}._{backend}._native")
        return module, backend

    return importlib.import_module(f"{__package__}._native"), "native"
