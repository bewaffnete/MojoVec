from std.memory.span import Span
from std.memory import alloc
from std.random import random_float64
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_ivf_pq import IndexIVFPQ

from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite

def test_ivf_pq() raises:
    var d = 16
    var n = 1000
    var nlist = 10
    var M = 4
    var nq = 10
    var k = 5
    
    var data = alloc[Float32](n * d)
    for i in range(n * d):
        data[i] = Float32(random_float64(-1.0, 1.0))
        
    var queries = alloc[Float32](nq * d)
    for i in range(nq * d):
        queries[i] = Float32(random_float64(-1.0, 1.0))
        
    pass  # print("Allocating flat_quantizer")
    var flat_quantizer = alloc[IndexFlat](1)
    flat_quantizer.init_pointee_move(IndexFlat(d))
    pass  # print("Allocating IndexIVFPQ")
    var ivf = IndexIVFPQ[IndexFlat](flat_quantizer, d, nlist, M)
    
    pass  # print("Training IVFPQ...")
    ivf.train(n, data)
    assert_true(ivf.is_trained, "Should be trained")
    
    pass  # print("Adding vectors...")
    
    var ids = alloc[Int](n)
    for i in range(n):
        ids[i] = i * 10  # Custom IDs
    ivf.add_with_ids(Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d), ids)
    assert_true(ivf.ntotal == n, "Total should match")
    
    ivf.nprobe = 3
    pass  # print("Searching IVFPQ...")
    var dists = alloc[Float32](nq * k)
    var labels = alloc[Int](nq * k)
    
    var span_dist_1 = Span[Float32, MutUntrackedOrigin](ptr=dists, length=nq * k)
    var span_labels_1 = Span[Int, MutUntrackedOrigin](ptr=labels, length=nq * k)
    ivf.search(Span[Float32, MutUntrackedOrigin](ptr=queries, length=nq * d), k, span_dist_1, span_labels_1)
    

    print("All IndexIVFPQ tests passed!")
    
    # Keep ivf alive
    _ = ivf.ntotal
    
    flat_quantizer.free()
    data.free()
    queries.free()
    ids.free()
    dists.free()
    labels.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
