from std.collections.span import Span
from std.testing import assert_equal, assert_true, assert_almost_equal, TestSuite
from mojovec.core.types import METRIC_L2, METRIC_INNER_PRODUCT, QT_8bit, QT_fp16
from mojovec.index.index_scalar_quantizer import IndexScalarQuantizer
from std.memory.alloc import unsafe_alloc

def test_sq8_bounds() raises:
    var d = 4
    var n = 3
    var index = IndexScalarQuantizer(d, QT_8bit, METRIC_L2)
    
    var data = unsafe_alloc[Float32](n * d)
    # vmin = [0, -10, 50, 0]
    data[unsafe_offset=0] = 0.0; data[unsafe_offset=1] = -10.0; data[unsafe_offset=2] = 50.0; data[unsafe_offset=3] = 0.0
    # mid = [50, 0, 100, 5]
    data[unsafe_offset=4] = 50.0; data[unsafe_offset=5] = 0.0; data[unsafe_offset=6] = 100.0; data[unsafe_offset=7] = 5.0
    # vmax = [100, 10, 150, 10]
    data[unsafe_offset=8] = 100.0; data[unsafe_offset=9] = 10.0; data[unsafe_offset=10] = 150.0; data[unsafe_offset=11] = 10.0
    
    index.train(Span[Float32](unsafe_ptr=data, length=n * d))
    
    # Verify trained vmin and vdiff
    assert_almost_equal(index.sq.vmin_at(0), 0.0, atol=1e-5)
    assert_almost_equal(index.sq.vmin_at(1), -10.0, atol=1e-5)
    assert_almost_equal(index.sq.vmin_at(2), 50.0, atol=1e-5)
    assert_almost_equal(index.sq.vmin_at(3), 0.0, atol=1e-5)
    
    # vdiff = vmax - vmin
    assert_almost_equal(index.sq.vdiff_at(0), 100.0, atol=1e-5)
    assert_almost_equal(index.sq.vdiff_at(1), 20.0, atol=1e-5)
    assert_almost_equal(index.sq.vdiff_at(2), 100.0, atol=1e-5)
    assert_almost_equal(index.sq.vdiff_at(3), 10.0, atol=1e-5)
    
    # Add exactly the vmin and vmax to see their encoded bytes
    index.add(Span[Float32, MutUntrackedOrigin](unsafe_ptr=data, length=n * d))
    
    # Store codes securely
    var codes_copy = unsafe_alloc[UInt8](n * d)
    for i in range(n * d):
        codes_copy[unsafe_offset=i] = index.codes[unsafe_offset=i]
        
    # Vector 0
    assert_equal(Int(codes_copy[unsafe_offset=0]), 0)
    
    # Vector 2
    assert_equal(Int(codes_copy[unsafe_offset=8]), 255)
    
    # Vector 1
    var code_val_0 = Int(codes_copy[unsafe_offset=4])
    var code_val_1 = Int(codes_copy[unsafe_offset=5])
    var code_val_2 = Int(codes_copy[unsafe_offset=6])
    var code_val_3 = Int(codes_copy[unsafe_offset=7])
    print("Vector 1 codes:", code_val_0, code_val_1, code_val_2, code_val_3)
    
    assert_true(abs(code_val_0 - 128) <= 1, "Expected ~128")
    assert_true(abs(code_val_1 - 128) <= 1, "Expected ~128")
    assert_true(abs(code_val_2 - 128) <= 1, "Expected ~128")
    assert_true(abs(code_val_3 - 128) <= 1, "Expected ~128")
    
    codes_copy.unsafe_free()
    data.unsafe_free()

def test_sq16_fp16_conversion() raises:
    # QT_fp16 doesn't require training, it just converts bits directly
    var d = 2
    var n = 1
    var index = IndexScalarQuantizer(d, QT_fp16, METRIC_L2)
    
    var data = unsafe_alloc[Float32](n * d)
    data[unsafe_offset=0] = 1.0; data[unsafe_offset=1] = -1.0
    
    index.add(Span[Float32, MutUntrackedOrigin](unsafe_ptr=data, length=n * d))
    
    # Verify decoding matches closely (FP16 should exactly represent 1.0 and -1.0)
    var decoded = unsafe_alloc[Float32](d)
    index.sq.decode(index.codes, decoded)
    
    assert_almost_equal(decoded[unsafe_offset=0], 1.0, atol=1e-5)
    assert_almost_equal(decoded[unsafe_offset=1], -1.0, atol=1e-5)
    
    data.unsafe_free()
    decoded.unsafe_free()


def test_sq_distance_computer_shared_calibration_lifetime() raises:
    """Repeated query contexts share calibration without sharing ownership bugs."""
    var index = IndexScalarQuantizer(2, QT_8bit, METRIC_L2)
    var data: List[Float32] = [0.0, 0.0, 1.0, 1.0]
    index.train(Span(data))
    index.add(Span(data))

    var query: List[Float32] = [0.0, 0.0]
    for _ in range(1024):
        var computer = index.get_distance_computer(query.unsafe_ptr())
        assert_almost_equal(computer.distance(0), 0.0, atol=1e-5)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
