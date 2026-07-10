from std.testing import assert_equal, assert_true, TestSuite
from mojovec.core.types import METRIC_L2, METRIC_INNER_PRODUCT, QT_8bit, QT_fp16
from mojovec.index.index_scalar_quantizer import IndexScalarQuantizer

def test_index_scalar_quantizer_sq8_l2() raises:
    var index = IndexScalarQuantizer(4, QT_8bit, METRIC_L2)

    var data = alloc[Float32](12)
    for i in range(4): data[i] = 0.0
    for i in range(4): data[4 + i] = 100.0
    for i in range(4): data[8 + i] = 200.0

    # SQ8 requires explicit training (convention: add() is a no-op until trained).
    index.train(3, data)
    index.add(3, data)
    assert_equal(index.ntotal, 3)
    
    var query = alloc[Float32](4)
    for i in range(4): query[i] = 110.0
        
    var distances = alloc[Float32](2)
    var labels = alloc[Int](2)
    
    index.search(1, query, 2, distances, labels)
    
    assert_true((labels[0] == 1 and labels[1] == 2) or (labels[0] == 2 and labels[1] == 1))
    
    query.free()
    distances.free()
    labels.free()
    data.free()

def test_index_scalar_quantizer_fp16_ip() raises:
    var index = IndexScalarQuantizer(4, QT_fp16, METRIC_INNER_PRODUCT)
    
    var data = alloc[Float32](12)
    for i in range(4):
        data[i] = 1.0
        data[4 + i] = 2.0
        data[8 + i] = 3.0
        
    index.add(3, data)
    
    var query = alloc[Float32](4)
    for i in range(4): query[i] = 1.0
        
    var distances = alloc[Float32](2)
    var labels = alloc[Int](2)
    
    index.search(1, query, 2, distances, labels)
    
    var has_12 = False
    var has_8 = False
    if abs(distances[0] - 12.0) < 0.1 or abs(distances[1] - 12.0) < 0.1:
        has_12 = True
    if abs(distances[0] - 8.0) < 0.1 or abs(distances[1] - 8.0) < 0.1:
        has_8 = True
        
    assert_true(has_12, "FP16 Should have IP approx 12.0")
    assert_true(has_8, "FP16 Should have IP approx 8.0")
    assert_true((labels[0] == 1 and labels[1] == 2) or (labels[0] == 2 and labels[1] == 1))
    
    query.free()
    distances.free()
    labels.free()
    data.free()

def test_index_scalar_quantizer_sq8_negative_bounds() raises:
    var index = IndexScalarQuantizer(2, QT_8bit, METRIC_L2)
    var data = alloc[Float32](6)
    data[0] = -100.0; data[1] = -100.0
    data[2] = 0.0; data[3] = 0.0
    data[4] = 100.0; data[5] = 100.0
    index.train(3, data)
    index.add(3, data)
    
    var query = alloc[Float32](2)
    query[0] = -10.0; query[1] = -10.0
    
    var distances = alloc[Float32](1)
    var labels = alloc[Int](1)
    index.search(1, query, 1, distances, labels)
    
    assert_equal(labels[0], 1)
    assert_true(abs(distances[0] - 200.0) < 25.0, "Distance should be approx 200")
    
    query.free()
    distances.free()
    labels.free()
    data.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
