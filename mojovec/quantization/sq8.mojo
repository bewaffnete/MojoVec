"""Metric-specialized scalar-quantization codecs and distance dispatch."""

import std.math as math

from mojovec.utils.distances import (
    sq8_dot_product_simd,
    sq8_l2_from_dot,
    sq8_signed_dot_product_simd,
)


@always_inline
def _encode_sq8_vector[SYMMETRIC: Bool](
    source: Pointer[Float32, _],
    destination: Pointer[UInt8, MutUntrackedOrigin],
    dimension: Int,
    minimum: Float32,
    inverse_scale: Float32,
) -> UInt32:
    """Encodes one vector with the metric-selected SQ8 codec."""
    var norm: UInt32 = 0
    comptime if SYMMETRIC:
        var signed_destination = destination.unsafe_bitcast[Int8]()
        for component in range(dimension):
            var scaled = math.round(source[unsafe_offset=component] * inverse_scale)
            var code = Int8(math.clamp(scaled, -127, 127))
            signed_destination[unsafe_offset=component] = code
            var wide_code = Int32(code)
            norm += UInt32(wide_code * wide_code)
    else:
        for component in range(dimension):
            var scaled = math.round(
                (source[unsafe_offset=component] - minimum) * inverse_scale
            )
            var code = UInt8(math.clamp(scaled, 0, 255))
            destination[unsafe_offset=component] = code
            norm += UInt32(code) * UInt32(code)
    return norm


def _encode_sq8_batch[SYMMETRIC: Bool](
    source: Pointer[Float32, _],
    destination: Pointer[UInt8, MutUntrackedOrigin],
    norms: Pointer[UInt32, MutUntrackedOrigin],
    vector_count: Int,
    dimension: Int,
    minimum: Float32,
    inverse_scale: Float32,
):
    """Encodes a contiguous vector batch without a metric branch per value."""
    for vector_index in range(vector_count):
        norms[unsafe_offset=vector_index] = _encode_sq8_vector[SYMMETRIC](
            source.unsafe_offset(vector_index * dimension),
            destination.unsafe_offset(vector_index * dimension),
            dimension,
            minimum,
            inverse_scale,
        )


@always_inline
def _sq8_code_distance[INNER_PRODUCT: Bool](
    query: Pointer[UInt8, _],
    database: Pointer[UInt8, _],
    query_norm: UInt32,
    database_norm: UInt32,
    dimension: Int,
    scale_squared: Float32,
) -> Float32:
    """Runs the metric-specialized byte kernel used by Flat and HNSW."""
    comptime if INNER_PRODUCT:
        var dot = sq8_signed_dot_product_simd(
            query.unsafe_bitcast[Int8](), database.unsafe_bitcast[Int8](), dimension
        )
        return -Float32(dot) * scale_squared
    else:
        var dot = sq8_dot_product_simd(query, database, dimension)
        var distance = sq8_l2_from_dot(query_norm, database_norm, dot)
        return Float32(distance) * scale_squared
