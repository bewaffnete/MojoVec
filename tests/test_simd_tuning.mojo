"""Correctness gate for width/unroll tuning and the legacy short-vector guard."""

from std.memory.alloc import unsafe_alloc
from std.testing import assert_almost_equal, TestSuite
from mojovec.utils.distances import (
    l2_distance_simd,
    inner_product_simd,
    l2_distance_simd_batch4,
    inner_product_simd_batch4,
    l2_distance_short4,
    inner_product_short4,
    l2_distance_tuned,
    inner_product_tuned,
    l2_distance_tuned_batch4,
    inner_product_tuned_batch4,
)


def check_variant[
    width: Int, unroll: Int, guarded: Bool
](d: Int, offset: Int) raises:
    var storage = unsafe_alloc[Float32](4 * d + offset + 1)
    var query_storage = unsafe_alloc[Float32](d + offset + 1)
    var x = storage.unsafe_offset(offset)
    var y = query_storage.unsafe_offset(offset)
    for i in range(4 * d):
        x[unsafe_offset=i] = Float32((i * 17 + 3) % 127 - 63) / 64.0
    for i in range(d):
        y[unsafe_offset=i] = Float32((i * 29 + 5) % 127 - 63) / 64.0
    var batch_l2 = l2_distance_simd_batch4[width](
        x,
        x.unsafe_offset(d),
        x.unsafe_offset(2 * d),
        x.unsafe_offset(3 * d),
        y,
        d,
    )
    var batch_ip = inner_product_simd_batch4[width](
        x,
        x.unsafe_offset(d),
        x.unsafe_offset(2 * d),
        x.unsafe_offset(3 * d),
        y,
        d,
    )
    var tuned_l2 = l2_distance_tuned_batch4(
        x,
        x.unsafe_offset(d),
        x.unsafe_offset(2 * d),
        x.unsafe_offset(3 * d),
        y,
        d,
    )
    var tuned_ip = inner_product_tuned_batch4(
        x,
        x.unsafe_offset(d),
        x.unsafe_offset(2 * d),
        x.unsafe_offset(3 * d),
        y,
        d,
    )
    for row in range(4):
        var expected_l2: Float64 = 0.0
        var expected_ip: Float64 = 0.0
        for i in range(d):
            var a = Float64(x[unsafe_offset=row * d + i])
            var b = Float64(y[unsafe_offset=i])
            expected_l2 += (a - b) * (a - b)
            expected_ip += a * b
        var row_ptr = x.unsafe_offset(row * d)
        assert_almost_equal(
            Float64(l2_distance_simd[width, unroll, guarded](row_ptr, y, d)),
            expected_l2,
            atol=0.001,
        )
        assert_almost_equal(
            Float64(inner_product_simd[width, unroll, guarded](row_ptr, y, d)),
            expected_ip,
            atol=0.001,
        )
        assert_almost_equal(Float64(batch_l2[row]), expected_l2, atol=0.001)
        assert_almost_equal(Float64(batch_ip[row]), expected_ip, atol=0.001)
        assert_almost_equal(
            Float64(l2_distance_tuned(row_ptr, y, d)), expected_l2, atol=0.001
        )
        assert_almost_equal(
            Float64(inner_product_tuned(row_ptr, y, d)), expected_ip, atol=0.001
        )
        assert_almost_equal(Float64(tuned_l2[row]), expected_l2, atol=0.001)
        assert_almost_equal(Float64(tuned_ip[row]), expected_ip, atol=0.001)
        if d <= 4:
            assert_almost_equal(
                Float64(l2_distance_short4(row_ptr, y, d)),
                expected_l2,
                atol=0.000001,
            )
            assert_almost_equal(
                Float64(inner_product_short4(row_ptr, y, d)),
                expected_ip,
                atol=0.000001,
            )
    storage.unsafe_free()
    query_storage.unsafe_free()


def test_all_fp32_widths_and_unroll_factors() raises:
    comptime for power in range(5):
        comptime width = 4 << power
        comptime for unroll_power in range(4):
            for d in [
                0,
                1,
                2,
                3,
                4,
                7,
                8,
                15,
                16,
                17,
                31,
                32,
                33,
                63,
                64,
                65,
                127,
                128,
                129,
                384,
                768,
                1536,
            ]:
                for offset in [0, 1]:
                    check_variant[width, 1 << unroll_power, True](d, offset)


def test_short_vector_workaround_can_be_bypassed() raises:
    # Experimental path only: production keeps the old guard until Linux
    # correctness AND timing have been checked for both x86 wheel targets.
    comptime for power in range(5):
        comptime width = 4 << power
        for d in [0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63]:
            if d < width:
                check_variant[width, 4, False](d, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
