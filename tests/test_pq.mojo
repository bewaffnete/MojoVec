from std.memory.span import Span
from std.memory import alloc
from std.random import random_float64
from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite
from mojovec.quantization.pq import ProductQuantizer
from mojovec.utils.distances import l2_distance_simd

def test_pq_encoding_error() raises:
    var d = 16
    var n = 1000
    var M = 4
    var ksub = 256
    
    var data = alloc[Float32](n * d)
    for i in range(n * d):
        data[i] = Float32(random_float64(-1.0, 1.0))
        
    var pq = ProductQuantizer(d, M, ksub)
    
    pq.train(n, data)
    assert_true(pq.is_trained, "Should be trained")
    
    var codes = alloc[UInt8](n * M)
    pq.compute_codes(n, data, codes)
    
    var decoded = alloc[Float32](n * d)
    pq.decode(n, codes, decoded)
    
    # Check reconstruction error
    var total_error: Float32 = 0.0
    for i in range(n):
        var ptr_orig = data + i * d
        var ptr_dec = decoded + i * d
        total_error += l2_distance_simd[4](ptr_orig, ptr_dec, d)
        
    var mse = total_error / Float32(n)
    # Theoretically MSE for uniform data PQ should be within reasonable bounds
    assert_true(mse < 1.0, "PQ Reconstruction Error (MSE) is too high")
    
    data.free()
    codes.free()
    decoded.free()

def test_pq_symmetric_distances() raises:
    var d = 16
    var n = 100
    var M = 4
    var ksub = 256
    
    var data = alloc[Float32](n * d)
    for i in range(n * d):
        data[i] = Float32(random_float64(-1.0, 1.0))
        
    var pq = ProductQuantizer(d, M, ksub)
    pq.train(n, data)
    
    var codes = alloc[UInt8](n * M)
    pq.compute_codes(n, data, codes)
    
    var decoded = alloc[Float32](n * d)
    pq.decode(n, codes, decoded)
    
    var query = data + 0
    var dis_table = alloc[Float32](M * ksub)
    pq.compute_distance_table(query, dis_table)
    
    # Compute ADC distance to vector 0
    var approx_dist: Float32 = 0.0
    var codes_0 = codes + 0
    for m in range(M):
        var k = Int(codes_0[m])
        approx_dist += dis_table[m * ksub + k]
        
    # Asymmetric distance (ADC) from vector 0 to its encoded representation
    # must perfectly equal the L2 distance between the original vector 0 and the decoded vector 0.
    var exact_decoded_dist = l2_distance_simd[4](query, decoded, d)
    assert_almost_equal(approx_dist, exact_decoded_dist, atol=1e-5)
    
    data.free()
    codes.free()
    decoded.free()
    dis_table.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
