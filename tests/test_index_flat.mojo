from std.memory.span import Span
from std.testing import assert_equal, assert_true, TestSuite
from mojovec.index.index_flat import IndexFlat
from mojovec.core.types import METRIC_L2, METRIC_INNER_PRODUCT

def test_index_flat_l2() raises:
    var index = IndexFlat(4, METRIC_L2)
    var data = alloc[Float32](12)
    for i in range(4):
        data[i] = 0.0
        data[4 + i] = 1.0
        data[8 + i] = 2.0
    index.add(Span[Float32, MutUntrackedOrigin](ptr=data, length=3 * 4))
    
    var query = alloc[Float32](4)
    for i in range(4):
        query[i] = 1.1
        
    var distances = alloc[Float32](2)
    var labels = alloc[Int](2)
    
    labels[0] = -1
    labels[1] = -1
    
    var span_dist_1 = Span[Float32, MutUntrackedOrigin](ptr=distances, length=1 * 2)
    var span_labels_1 = Span[Int, MutUntrackedOrigin](ptr=labels, length=1 * 2)
    index.search(Span[Float32, MutUntrackedOrigin](ptr=query, length=1 * 4), 2, span_dist_1, span_labels_1)
    
    assert_true(labels[0] == 1 or labels[1] == 1)
    assert_true(labels[0] == 2 or labels[1] == 2)
    
    var has_0_04 = False
    var has_3_24 = False
    if abs(distances[0] - 0.04) < 0.001 or abs(distances[1] - 0.04) < 0.001:
        has_0_04 = True
    if abs(distances[0] - 3.24) < 0.001 or abs(distances[1] - 3.24) < 0.001:
        has_3_24 = True
        
    assert_true(has_0_04, "Should have distance approx 0.04")
    assert_true(has_3_24, "Should have distance approx 3.24")
    
    query.free()
    distances.free()
    labels.free()
    data.free()


def test_index_flat_inner_product() raises:
    var index = IndexFlat(4, METRIC_INNER_PRODUCT)
    var data = alloc[Float32](12)
    for i in range(4):
        data[i] = 1.0
        data[4 + i] = 2.0
        data[8 + i] = 3.0
    index.add(Span[Float32, MutUntrackedOrigin](ptr=data, length=3 * 4))
    
    var query = alloc[Float32](4)
    for i in range(4):
        query[i] = 1.0
        
    var distances = alloc[Float32](2)
    var labels = alloc[Int](2)
    var span_dist_2 = Span[Float32, MutUntrackedOrigin](ptr=distances, length=1 * 2)
    var span_labels_2 = Span[Int, MutUntrackedOrigin](ptr=labels, length=1 * 2)
    index.search(Span[Float32, MutUntrackedOrigin](ptr=query, length=1 * 4), 2, span_dist_2, span_labels_2)
    
    var has_12 = False
    var has_8 = False
    if abs(distances[0] - 12.0) < 0.001 or abs(distances[1] - 12.0) < 0.001:
        has_12 = True
    if abs(distances[0] - 8.0) < 0.001 or abs(distances[1] - 8.0) < 0.001:
        has_8 = True
        
    assert_true(has_12, "Should have IP 12.0")
    assert_true(has_8, "Should have IP 8.0")
    assert_true((labels[0] == 1 and labels[1] == 2) or (labels[0] == 2 and labels[1] == 1))
    
    query.free()
    distances.free()
    labels.free()
    data.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
