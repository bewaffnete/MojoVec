from std.memory.span import Span
from std.memory import alloc
from std.random import random_float64
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_ivf_flat import IndexIVFFlat

from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite

def test_ivf_flat() raises:
    var d = 16
    var n = 1000
    var nlist = 10
    var nq = 10
    var k = 5
    
    var data = alloc[Float32](n * d)
    for i in range(n * d):
        data[i] = Float32(random_float64(-1.0, 1.0))
        
    var queries = alloc[Float32](nq * d)
    for i in range(nq * d):
        queries[i] = Float32(random_float64(-1.0, 1.0))
    pass  # print("Data and queries .")
    var flat_quantizer = alloc[IndexFlat](1)
    flat_quantizer.init_pointee_move(IndexFlat(d))
    var ivf = IndexIVFFlat[IndexFlat](flat_quantizer, d, nlist)
    
    pass  # print("Training IVF...")
    ivf.train(n, data)
    assert_true(ivf.is_trained, "Should be trained")
    
    pass  # print("Adding vectors...")
    
    var ids = alloc[Int](n)
    for i in range(n):
        ids[i] = i * 10  # Custom IDs
    ivf.add_with_ids(Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d), ids)
    assert_true(ivf.ntotal == n, "Total should match")
    
    ivf.nprobe = 3
    pass  # print("Searching IVF...")
    var dists = alloc[Float32](nq * k)
    var labels = alloc[Int](nq * k)
    
    var span_dist_1 = Span[Float32, MutUntrackedOrigin](ptr=dists, length=nq * k)
    var span_labels_1 = Span[Int, MutUntrackedOrigin](ptr=labels, length=nq * k)
    ivf.search(Span[Float32, MutUntrackedOrigin](ptr=queries, length=nq * d), k, span_dist_1, span_labels_1)
    

    print("All IndexIVFFlat tests passed!")
    
    # Keep ivf alive
    _ = ivf.ntotal
    
    flat_quantizer.free()
    data.free()
    queries.free()
    ids.free()
    dists.free()
    labels.free()

def test_ivf_flat_exact_match() raises:
    var d = 4
    var data = alloc[Float32](4)
    for i in range(4): data[i] = Float32(i)
    var flat_quantizer = alloc[IndexFlat](1)
    flat_quantizer.init_pointee_move(IndexFlat(d))
    var ivf = IndexIVFFlat[IndexFlat](flat_quantizer, d, 1)
    
    ivf.train(1, data)
    ivf.add(Span[Float32, MutUntrackedOrigin](ptr=data, length=1 * d))
    
    var distances = alloc[Float32](1)
    var labels = alloc[Int](1)
    var span_dist_2 = Span[Float32, MutUntrackedOrigin](ptr=distances, length=1 * 1)
    var span_labels_2 = Span[Int, MutUntrackedOrigin](ptr=labels, length=1 * 1)
    ivf.search(Span[Float32, MutUntrackedOrigin](ptr=data, length=1 * d), 1, span_dist_2, span_labels_2)
    assert_true(labels[0] == 0, "Should match id 0")
    assert_almost_equal(distances[0], 0.0, msg="Distance should be 0")
    
    distances.free()
    labels.free()
    data.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
