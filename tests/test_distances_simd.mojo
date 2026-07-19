from std.testing import assert_true, assert_almost_equal, TestSuite
from std.memory import alloc
from std.random import random_float64

from mojovec.utils.distances import l2_distance_simd, inner_product_simd

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
    
    for i in range(d):
        x[i] = Float32(random_float64(-10.0, 10.0))
        y[i] = Float32(random_float64(-10.0, 10.0))
        
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

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
