"""
Defines fundamental types and compile-time constants for metrics and quantizers.
"""

from std.os import getenv, setenv

comptime MetricType = Int
comptime METRIC_L2 = 0
comptime METRIC_INNER_PRODUCT = 1

comptime QuantizerType = Int
comptime QT_8bit = 0
comptime QT_fp16 = 1

comptime StorageKind = Int
comptime STORAGE_FLAT = 0
comptime STORAGE_SQ8 = 1

comptime DispatchMode = Int
comptime MODE_AUTO: DispatchMode = 0
comptime MODE_SERIAL: DispatchMode = 1
comptime MODE_RUNTIME: DispatchMode = 2
comptime MODE_PTHREAD: DispatchMode = 3


def set_pq_dispatch_mode(mode: Int):
    """Sets the execution dispatch mode for Product Quantization training.

    Modes:
        MODE_AUTO (0): Default behavior (native pthreads).
        MODE_SERIAL (1): Sequential execution in the caller thread.
        MODE_RUNTIME (2): Parallel execution via max.algorithm.parallelize.
        MODE_PTHREAD (3): Native POSIX pthreads.
    """
    _ = setenv("MOJOVEC_PQ_DISPATCH_MODE", String(mode), True)


def get_pq_dispatch_mode() -> Int:
    """Returns the current dispatch mode for Product Quantization training."""
    var val = getenv("MOJOVEC_PQ_DISPATCH_MODE")
    if val == "1" or val == "serial" or val == "SERIAL":
        return MODE_SERIAL
    elif val == "2" or val == "runtime" or val == "RUNTIME":
        return MODE_RUNTIME
    elif val == "3" or val == "pthread" or val == "PTHREAD":
        return MODE_PTHREAD
    return MODE_AUTO


def set_hnsw_dispatch_mode(mode: Int):
    """Sets the execution dispatch mode for HNSW index search.

    Modes:
        MODE_AUTO (0): Default behavior (heuristic selection).
        MODE_SERIAL (1): Sequential execution in the caller thread.
        MODE_RUNTIME (2): Parallel execution via max.algorithm.parallelize.
        MODE_PTHREAD (3): Native POSIX pthreads.
    """
    _ = setenv("MOJOVEC_HNSW_DISPATCH_MODE", String(mode), True)


def get_hnsw_dispatch_mode() -> Int:
    """Returns the current dispatch mode for HNSW index search."""
    var val = getenv("MOJOVEC_HNSW_DISPATCH_MODE")
    if val == "1" or val == "serial" or val == "SERIAL":
        return MODE_SERIAL
    elif val == "2" or val == "runtime" or val == "RUNTIME":
        return MODE_RUNTIME
    elif val == "3" or val == "pthread" or val == "PTHREAD":
        return MODE_PTHREAD
    return MODE_AUTO
