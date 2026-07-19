from std.memory.span import Span
from std.testing import assert_equal, assert_true, assert_almost_equal, TestSuite
from mojovec.core.types import METRIC_L2, METRIC_INNER_PRODUCT, QT_8bit, QT_fp16
from mojovec.index.index_scalar_quantizer import IndexScalarQuantizer
from std.memory import alloc

def test_sq8_bounds() raises:
    var d = 4
    var n = 3
    var index = IndexScalarQuantizer(d, QT_8bit, METRIC_L2)
    
    var data = alloc[Float32](n * d)
    # vmin = [0, -10, 50, 0]
    data[0] = 0.0; data[1] = -10.0; data[2] = 50.0; data[3] = 0.0
    # mid = [50, 0, 100, 5]
    data[4] = 50.0; data[5] = 0.0; data[6] = 100.0; data[7] = 5.0
    # vmax = [100, 10, 150, 10]
    data[8] = 100.0; data[9] = 10.0; data[10] = 150.0; data[11] = 10.0
    
    index.train(n, data)
    
    # Verify trained vmin and vdiff
    assert_almost_equal(index.sq.vmin[0], 0.0, atol=1e-5)
    assert_almost_equal(index.sq.vmin[1], -10.0, atol=1e-5)
    assert_almost_equal(index.sq.vmin[2], 50.0, atol=1e-5)
    assert_almost_equal(index.sq.vmin[3], 0.0, atol=1e-5)
    
    # vdiff = vmax - vmin
    assert_almost_equal(index.sq.vdiff[0], 100.0, atol=1e-5)
    assert_almost_equal(index.sq.vdiff[1], 20.0, atol=1e-5)
    assert_almost_equal(index.sq.vdiff[2], 100.0, atol=1e-5)
    assert_almost_equal(index.sq.vdiff[3], 10.0, atol=1e-5)
    
    # Add exactly the vmin and vmax to see their encoded bytes
    index.add(Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d))
    
    # Store codes securely
    var codes_copy = alloc[UInt8](n * d)
    for i in range(n * d):
        codes_copy[i] = index.codes[i]
        
    # Vector 0
    assert_equal(Int(codes_copy[0]), 0)
    
    # Vector 2
    assert_equal(Int(codes_copy[8]), 255)
    
    # Vector 1
    var code_val_0 = Int(codes_copy[4])
    var code_val_1 = Int(codes_copy[5])
    var code_val_2 = Int(codes_copy[6])
    var code_val_3 = Int(codes_copy[7])
    print("Vector 1 codes:", code_val_0, code_val_1, code_val_2, code_val_3)
    
    assert_true(abs(code_val_0 - 128) <= 1, "Expected ~128")
    assert_true(abs(code_val_1 - 128) <= 1, "Expected ~128")
    assert_true(abs(code_val_2 - 128) <= 1, "Expected ~128")
    assert_true(abs(code_val_3 - 128) <= 1, "Expected ~128")
    
    codes_copy.free()
    data.free()

def test_sq16_fp16_conversion() raises:
    # QT_fp16 doesn't require training, it just converts bits directly
    var d = 2
    var n = 1
    var index = IndexScalarQuantizer(d, QT_fp16, METRIC_L2)
    
    var data = alloc[Float32](n * d)
    data[0] = 1.0; data[1] = -1.0
    
    index.add(Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d))
    
    # Verify decoding matches closely (FP16 should exactly represent 1.0 and -1.0)
    var decoded = alloc[Float32](d)
    index.sq.decode(index.codes, decoded)
    
    assert_almost_equal(decoded[0], 1.0, atol=1e-5)
    assert_almost_equal(decoded[1], -1.0, atol=1e-5)
    
    data.free()
    decoded.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
