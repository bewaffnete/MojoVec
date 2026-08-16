from std.collections.span import Span
from std.memory.alloc import unsafe_alloc
from std.random.philox import Random
from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite
from mojovec.quantization.pq import ProductQuantizer
from mojovec.utils.distances import l2_distance_simd

def test_pq_encoding_error() raises:
    var d = 16
    var n = 1000
    var M = 4
    var ksub = 256
    var generator = Random(seed=UInt64(181))
    
    var data = unsafe_alloc[Float32](n * d)
    var values = generator.step_uniform()
    for i in range(n * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        data[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0
        
    var pq = ProductQuantizer(d, M, ksub)
    
    var data_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=data, length=n * d
    )
    pq.train(data_span)
    assert_true(pq.is_trained, "Should be trained")
    
    var codes = unsafe_alloc[UInt8](n * M)
    var codes_span = Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=codes, length=n * M
    )
    pq.compute_codes(data_span, codes_span)
    
    var decoded = unsafe_alloc[Float32](n * d)
    var decoded_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=decoded, length=n * d
    )
    pq.decode(codes_span, decoded_span)
    
    # Check reconstruction error
    var total_error: Float32 = 0.0
    for i in range(n):
        var ptr_orig = data.unsafe_offset(i * d)
        var ptr_dec = decoded.unsafe_offset(i * d)
        total_error += l2_distance_simd[4](ptr_orig, ptr_dec, d)
        
    var mse = total_error / Float32(n)
    # Theoretically MSE for uniform data PQ should be within reasonable bounds
    assert_true(mse < 1.0, "PQ Reconstruction Error (MSE) is too high")
    
    data.unsafe_free()
    codes.unsafe_free()
    decoded.unsafe_free()

def test_pq_symmetric_distances() raises:
    var d = 16
    var n = 100
    var M = 4
    var ksub = 256
    var generator = Random(seed=UInt64(191))
    
    var data = unsafe_alloc[Float32](n * d)
    var values = generator.step_uniform()
    for i in range(n * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        data[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0
        
    var pq = ProductQuantizer(d, M, ksub)
    var data_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=data, length=n * d
    )
    pq.train(data_span)
    
    var codes = unsafe_alloc[UInt8](n * M)
    var codes_span = Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=codes, length=n * M
    )
    pq.compute_codes(data_span, codes_span)
    
    var decoded = unsafe_alloc[Float32](n * d)
    var decoded_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=decoded, length=n * d
    )
    pq.decode(codes_span, decoded_span)
    
    var query = data
    var dis_table = unsafe_alloc[Float32](M * ksub)
    var query_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=query, length=d
    )
    var table_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=dis_table, length=M * ksub
    )
    pq.compute_distance_table(query_span, table_span)
    
    # Compute ADC distance to vector 0
    var approx_dist: Float32 = 0.0
    var codes_0 = codes
    for m in range(M):
        var k = Int(codes_0[unsafe_offset=m])
        approx_dist += dis_table[unsafe_offset=m * ksub + k]
        
    # Asymmetric distance (ADC) from vector 0 to its encoded representation
    # must perfectly equal the L2 distance between the original vector 0 and the decoded vector 0.
    var exact_decoded_dist = l2_distance_simd[4](query, decoded, d)
    assert_almost_equal(approx_dist, exact_decoded_dist, atol=1e-5)
    
    data.unsafe_free()
    codes.unsafe_free()
    decoded.unsafe_free()
    dis_table.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
