from std.collections.span import Span
from std.memory.alloc import unsafe_alloc
from std.testing import TestSuite, assert_equal, assert_true
from mojovec.index.index_hnsw import IndexHNSW
from mojovec.index.index_flat import IndexFlat
from mojovec.core.types import METRIC_L2

def test_hnsw_structure_construction() raises:
    var d = 2
    var nb = 10
    var xb = unsafe_alloc[Float32](nb * d)
    for i in range(nb * d):
        xb[unsafe_offset=i] = Float32(i) # Deterministic points
        
    var storage = IndexFlat(d, METRIC_L2)
    var hnsw = IndexHNSW[IndexFlat](storage^, d, METRIC_L2, M=2)
    
    hnsw.add(Span[Float32, MutUntrackedOrigin](unsafe_ptr=xb, length=nb * d))
    
    # Verify graph structure and nodes
    assert_equal(hnsw.hnsw.ntotal, 10, "HNSW ntotal should match added points")
    assert_true(hnsw.hnsw.M == 2, "M parameter should be 2")

    xb.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
