from std.testing import assert_true, assert_almost_equal, assert_equal, TestSuite
from std.memory import alloc
from std.random.philox import Random

from mojovec.utils.distances import (
    inner_product_simd,
    l2_distance_simd,
    sq8_signed_dot_product_simd,
    sq8_signed_dot_product_simd_batch4,
)

def l2_scalar(x: UnsafePointer[Float32, MutUntrackedOrigin], y: UnsafePointer[Float32, MutUntrackedOrigin], d: Int) -> Float32:
    var res: Float32 = 0.0
    for i in range(d):
        var diff = x[i] - y[i]
        res += diff * diff
    return res

def ip_scalar(x: UnsafePointer[Float32, MutUntrackedOrigin], y: UnsafePointer[Float32, MutUntrackedOrigin], d: Int) -> Float32:
    var res: Float32 = 0.0
    for i in range(d):
        res += x[i] * y[i]
    return res

def check_distance(d: Int) raises:
    var x = alloc[Float32](d)
    var y = alloc[Float32](d)
    var generator = Random(seed=UInt64(d))
    
    for i in range(d):
        var values = generator.step_uniform()
        x[i] = values[0] * 20.0 - 10.0
        y[i] = values[1] * 20.0 - 10.0
        
    # Check L2
    var expected_l2 = l2_scalar(x, y, d)
    var actual_l2 = l2_distance_simd[4](x, y, d)
    assert_almost_equal(actual_l2, expected_l2, atol=1e-4)
    
    # Check IP
    var expected_ip = ip_scalar(x, y, d)
    var actual_ip = inner_product_simd[4](x, y, d)
    assert_almost_equal(actual_ip, expected_ip, atol=1e-4)
    
    x.free()
    y.free()

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
    var x = alloc[Float32](4)
    var y = alloc[Float32](4)
    for index in range(4):
        x[index] = Float32(index + 1)
        y[index] = Float32(index) * 0.25
    assert_almost_equal(
        l2_distance_simd[8](x, y, 4), l2_scalar(x, y, 4), atol=1e-6
    )
    assert_almost_equal(
        inner_product_simd[8](x, y, 4), ip_scalar(x, y, 4), atol=1e-6
    )
    x.free()
    y.free()


def test_signed_sq8_dot_handles_vector_tails() raises:
    for dimension in [1, 7, 16, 63, 64, 127, 128, 129]:
        var x = alloc[Int8](dimension)
        var y = alloc[Int8](dimension)
        var expected: Int32 = 0
        for index in range(dimension):
            var x_value = Int8((index * 37) % 255 - 127)
            var y_value = Int8((index * 53 + 11) % 255 - 127)
            x[index] = x_value
            y[index] = y_value
            expected += Int32(x_value) * Int32(y_value)
        assert_equal(
            sq8_signed_dot_product_simd(x, y, dimension), expected
        )
        x.free()
        y.free()


def test_signed_sq8_batch4_matches_individual_kernels() raises:
    var dimension = 129
    var query = alloc[Int8](dimension)
    var x0 = alloc[Int8](dimension)
    var x1 = alloc[Int8](dimension)
    var x2 = alloc[Int8](dimension)
    var x3 = alloc[Int8](dimension)
    for index in range(dimension):
        query[index] = Int8((index * 11) % 255 - 127)
        x0[index] = Int8((index * 17 + 1) % 255 - 127)
        x1[index] = Int8((index * 29 + 2) % 255 - 127)
        x2[index] = Int8((index * 41 + 3) % 255 - 127)
        x3[index] = Int8((index * 59 + 4) % 255 - 127)
    var batched = sq8_signed_dot_product_simd_batch4(
        x0, x1, x2, x3, query, dimension
    )
    assert_equal(batched[0], sq8_signed_dot_product_simd(x0, query, dimension))
    assert_equal(batched[1], sq8_signed_dot_product_simd(x1, query, dimension))
    assert_equal(batched[2], sq8_signed_dot_product_simd(x2, query, dimension))
    assert_equal(batched[3], sq8_signed_dot_product_simd(x3, query, dimension))
    query.free()
    x0.free()
    x1.free()
    x2.free()
    x3.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
