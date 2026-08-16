"""Collection metric and sparse payload serialization helpers."""

from std.io.file import FileHandle
from std.collections import InlineArray
from std.collections.span import Span

from mojovec.api.metadata import (
    METADATA_BOOL,
    METADATA_FLOAT,
    METADATA_INT,
    METADATA_STRING,
    Metadata,
)
from mojovec.core.types import (
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    MetricType,
)


comptime METRIC_COSINE = 2


def _parse_metric(value: String) raises -> MetricType:
    if value == "l2":
        return METRIC_L2
    if value == "cosine":
        return METRIC_COSINE
    if value == "ip":
        return METRIC_INNER_PRODUCT
    raise Error('metric must be "l2", "cosine", or "ip".')


def _metric_name(metric: MetricType) -> String:
    if metric == METRIC_COSINE:
        return "cosine"
    if metric == METRIC_INNER_PRODUCT:
        return "ip"
    return "l2"


def _index_metric(metric: MetricType) -> MetricType:
    # Cosine similarity is inner product over unit-normalized vectors.
    if metric == METRIC_COSINE:
        return METRIC_INNER_PRODUCT
    return metric


def _write_string(mut file: FileHandle, value: String) raises:
    from mojovec.io.serialization import write_int

    write_int(file, value.byte_length())
    file.write_bytes(value.as_bytes())


def _read_string(mut file: FileHandle) raises -> String:
    from mojovec.io.serialization import check_size_limit, read_int

    var size = read_int(file)
    check_size_limit(size, 1_048_576)
    var data = file.read_bytes(size)
    if len(data) != size:
        raise Error("Unexpected end of serialized String.")
    return String(from_utf8=data)


def _write_float64(mut file: FileHandle, value: Float64) raises:
    var storage = InlineArray[Float64, 1](uninitialized=True)
    storage[0] = value
    file.write_bytes(
        Span[UInt8](
            unsafe_ptr=storage.unsafe_ptr().unsafe_bitcast[UInt8](), length=8
        )
    )


def _read_float64(mut file: FileHandle) raises -> Float64:
    var data = file.read_bytes(8)
    if len(data) != 8:
        raise Error("Unexpected end of serialized Float64.")
    var value = data.unsafe_ptr().unsafe_bitcast[Float64]()[unsafe_offset=0]
    _ = len(data)
    return value


def _write_metadata(mut file: FileHandle, metadata: Metadata) raises:
    from mojovec.io.serialization import write_bool, write_int

    write_int(file, metadata.count())
    for index in range(metadata.count()):
        _write_string(file, metadata._key_at(index))
        var value = metadata._value_at(index)
        write_int(file, value.kind())
        if value.kind() == METADATA_STRING:
            _write_string(file, value.as_string())
        elif value.kind() == METADATA_INT:
            write_int(file, value.as_int())
        elif value.kind() == METADATA_FLOAT:
            _write_float64(file, value.as_float())
        elif value.kind() == METADATA_BOOL:
            write_bool(file, value.as_bool())
        else:
            raise Error("Unsupported metadata value kind.")


def _read_metadata(mut file: FileHandle) raises -> Metadata:
    from mojovec.io.serialization import (
        check_size_limit,
        read_bool,
        read_int,
    )

    var field_count = read_int(file)
    check_size_limit(field_count, 65_536)
    var metadata = Metadata()
    for _ in range(field_count):
        var key = _read_string(file)
        if metadata.contains(key):
            raise Error("Duplicate metadata keys are not allowed.")
        var kind = read_int(file)
        if kind == METADATA_STRING:
            metadata.set(key, _read_string(file))
        elif kind == METADATA_INT:
            metadata.set(key, read_int(file))
        elif kind == METADATA_FLOAT:
            metadata.set(key, _read_float64(file))
        elif kind == METADATA_BOOL:
            metadata.set(key, read_bool(file))
        else:
            raise Error("Invalid metadata value kind.")
    return metadata^
