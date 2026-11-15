from std.testing import (
    assert_true,
    assert_almost_equal,
    assert_equal,
    TestSuite,
)
from std.memory.alloc import unsafe_alloc
from std.random.philox import Random

from mojovec.utils.distances import (
    inner_product_simd,
    l2_distance_simd,
    sq8_dot_product_simd,
    sq8_signed_dot_product_simd,
    sq8_signed_dot_product_simd_batch4,
)


def l2_scalar(
    x: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    d: Int,
) -> Float32:
    var res: Float32 = 0.0
    for i in range(d):
        var diff = x[unsafe_offset=i] - y[unsafe_offset=i]
        res += diff * diff
    return res


def ip_scalar(
    x: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    d: Int,
) -> Float32:
    var res: Float32 = 0.0
    for i in range(d):
        res += x[unsafe_offset=i] * y[unsafe_offset=i]
    return res


def check_distance(d: Int) raises:
    var x = unsafe_alloc[Float32](d)
    var y = unsafe_alloc[Float32](d)
    var generator = Random(seed=UInt64(d))

    for i in range(d):
        var values = generator.step_uniform()
        x[unsafe_offset=i] = values[0] * 20.0 - 10.0
        y[unsafe_offset=i] = values[1] * 20.0 - 10.0

    # Check L2
    var expected_l2 = l2_scalar(x, y, d)
    var actual_l2 = l2_distance_simd[4](x, y, d)
    assert_almost_equal(actual_l2, expected_l2, atol=1e-4)

    # Check IP
    var expected_ip = ip_scalar(x, y, d)
    var actual_ip = inner_product_simd[4](x, y, d)
    assert_almost_equal(actual_ip, expected_ip, atol=1e-4)

    x.unsafe_free()
    y.unsafe_free()


def test_distances_odd_dimensions() raises:
    check_distance(1)
    check_distance(3)
    check_distance(7)
    check_distance(13)
    check_distance(15)
    check_distance(27)
    check_distance(31)
    check_distance(33)
    check_distance(127)


def test_distances_exact_simd_multiples() raises:
    check_distance(4)
    check_distance(8)
    check_distance(16)
    check_distance(32)
    check_distance(64)


def test_distance_accumulators_are_zero_for_sub_simd_vectors() raises:
    # PQ uses width-eight kernels for subvectors that can contain only four
    # components. This exercises the tail-only path that exposed undefined
    # default SIMD accumulator values in Linux optimized wheels.
    var x = unsafe_alloc[Float32](4)
    var y = unsafe_alloc[Float32](4)
    for index in range(4):
        x[unsafe_offset=index] = Float32(index + 1)
        y[unsafe_offset=index] = Float32(index) * 0.25
    assert_almost_equal(
        l2_distance_simd[8](x, y, 4), l2_scalar(x, y, 4), atol=1e-6
    )
    assert_almost_equal(
        inner_product_simd[8](x, y, 4), ip_scalar(x, y, 4), atol=1e-6
    )
    x.unsafe_free()
    y.unsafe_free()


def test_unsigned_sq8_dot_matches_scalar() raises:
    for dimension in [
        0,
        1,
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
        257,
    ]:
        var x = unsafe_alloc[UInt8](dimension + 1)
        var y = unsafe_alloc[UInt8](dimension + 1)
        for pattern in range(3):
            var expected: UInt32 = 0
            for index in range(dimension):
                var x_value = UInt8(255)
                var y_value = UInt8(255)
                if pattern == 1:
                    x_value = UInt8((index * 37) % 256)
                    y_value = UInt8((index * 53 + 11) % 256)
                elif pattern == 2:
                    x_value = UInt8(0)
                x[unsafe_offset=index] = x_value
                y[unsafe_offset=index] = y_value
                expected += UInt32(x_value) * UInt32(y_value)
            assert_equal(sq8_dot_product_simd(x, y, dimension), expected)
        x.unsafe_free()
        y.unsafe_free()


def test_signed_sq8_dot_handles_vector_tails() raises:
    for dimension in [
        0,
        1,
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
        257,
    ]:
        var x = unsafe_alloc[Int8](dimension + 1)
        var y = unsafe_alloc[Int8](dimension + 1)
        var expected: Int32 = 0
        for index in range(dimension):
            var x_value = Int8((index * 37) % 256 - 128)
            var y_value = Int8((index * 53 + 11) % 256 - 128)
            x[unsafe_offset=index] = x_value
            y[unsafe_offset=index] = y_value
            expected += Int32(x_value) * Int32(y_value)
        assert_equal(sq8_signed_dot_product_simd(x, y, dimension), expected)
        x.unsafe_free()
        y.unsafe_free()


def test_signed_sq8_batch4_matches_individual_kernels() raises:
    for dimension in [
        0,
        1,
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
        257,
    ]:
        var query = unsafe_alloc[Int8](dimension + 1)
        var x0 = unsafe_alloc[Int8](dimension + 1)
        var x1 = unsafe_alloc[Int8](dimension + 1)
        var x2 = unsafe_alloc[Int8](dimension + 1)
        var x3 = unsafe_alloc[Int8](dimension + 1)
        var expected0: Int32 = 0
        var expected1: Int32 = 0
        var expected2: Int32 = 0
        var expected3: Int32 = 0
        for index in range(dimension):
            var q = Int8((index * 11) % 256 - 128)
            query[unsafe_offset=index] = q
            x0[unsafe_offset=index] = Int8((index * 17 + 1) % 256 - 128)
            x1[unsafe_offset=index] = Int8((index * 29 + 2) % 256 - 128)
            x2[unsafe_offset=index] = Int8(-128)
            x3[unsafe_offset=index] = Int8(127)
            expected0 += Int32(x0[unsafe_offset=index]) * Int32(q)
            expected1 += Int32(x1[unsafe_offset=index]) * Int32(q)
            expected2 += Int32(x2[unsafe_offset=index]) * Int32(q)
            expected3 += Int32(x3[unsafe_offset=index]) * Int32(q)
        var batched = sq8_signed_dot_product_simd_batch4(
            x0, x1, x2, x3, query, dimension
        )
        # Use independent scalar references: comparing two SIMD kernels alone
        # can hide a shared bug in their target selection or arithmetic.
        assert_equal(batched[0], expected0)
        assert_equal(batched[1], expected1)
        assert_equal(batched[2], expected2)
        assert_equal(batched[3], expected3)
        assert_equal(
            batched[0], sq8_signed_dot_product_simd(x0, query, dimension)
        )
        assert_equal(
            batched[1], sq8_signed_dot_product_simd(x1, query, dimension)
        )
        assert_equal(
            batched[2], sq8_signed_dot_product_simd(x2, query, dimension)
        )
        assert_equal(
            batched[3], sq8_signed_dot_product_simd(x3, query, dimension)
        )
        query.unsafe_free()
        x0.unsafe_free()
        x1.unsafe_free()
        x2.unsafe_free()
        x3.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
