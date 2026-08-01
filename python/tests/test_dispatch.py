import pytest

from mojovec._dispatch import (
    _X86_64_V3_FLAGS,
    _X86_64_V4_FLAGS,
    _parse_linux_cpu_flags,
    _select_linux_x86_backend,
)


def test_parse_linux_cpu_flags_uses_features_common_to_every_cpu():
    flags = _parse_linux_cpu_flags(
        """
processor : 0
flags : avx avx2 fma avx512f
processor : 1
flags : avx avx2 fma
"""
    )
    assert flags == frozenset({"avx", "avx2", "fma"})


def test_dispatch_prefers_avx512():
    assert _select_linux_x86_backend(_X86_64_V4_FLAGS) == "avx512"


def test_dispatch_falls_back_to_avx2():
    assert _select_linux_x86_backend(_X86_64_V3_FLAGS) == "avx2"
    assert (
        _select_linux_x86_backend(_X86_64_V4_FLAGS, forced="avx2") == "avx2"
    )


def test_dispatch_rejects_unsupported_cpu():
    with pytest.raises(ImportError, match="require an x86-64-v3 CPU"):
        _select_linux_x86_backend(frozenset({"sse2"}))


def test_dispatch_rejects_unsafe_forced_avx512():
    with pytest.raises(ImportError, match="AVX-512 backend requested"):
        _select_linux_x86_backend(_X86_64_V3_FLAGS, forced="avx512")


def test_dispatch_rejects_unknown_override():
    with pytest.raises(ImportError, match="MOJOVEC_FORCE_BACKEND"):
        _select_linux_x86_backend(_X86_64_V4_FLAGS, forced="fastest")
