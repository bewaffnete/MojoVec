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
    ivf.train(
        Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d)
    )
    assert_true(ivf.is_trained, "Should be trained")
    
    pass  # print("Adding vectors...")
    
    var ids = alloc[Int](n)
    for i in range(n):
        ids[i] = i * 10  # Custom IDs
    ivf.add_with_ids(
        Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d),
        Span[Int, MutUntrackedOrigin](ptr=ids, length=n),
    )
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
    
    ivf.train(Span[Float32, MutUntrackedOrigin](ptr=data, length=d))
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


def test_ivf_flat_sparse_results_have_deterministic_tail() raises:
    comptime d = 4
    comptime k = 5
    var data = alloc[Float32](d)
    for i in range(d):
        data[i] = 0.0
    var ids = alloc[Int](1)
    ids[0] = 42

    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(IndexFlat(d))
    var index = IndexIVFFlat[IndexFlat](quantizer, d, 1)
    var data_span = Span[Float32, MutUntrackedOrigin](
        ptr=data, length=d
    )
    index.train(data_span)
    index.add_with_ids(
        data_span,
        Span[Int, MutUntrackedOrigin](ptr=ids, length=1),
    )

    var distances = alloc[Float32](k)
    var labels = alloc[Int](k)
    for i in range(k):
        distances[i] = -7.0
        labels[i] = -7
    var distance_span = Span[mut=True, Float32, MutUntrackedOrigin](
        ptr=distances, length=k
    )
    var label_span = Span[mut=True, Int, MutUntrackedOrigin](
        ptr=labels, length=k
    )
    index.search(
        data_span,
        k,
        distance_span,
        label_span,
    )

    assert_equal(labels[0], 42)
    assert_true(distances[0] == 0.0)
    for i in range(1, k):
        assert_equal(labels[i], -1)
        assert_true(distances[i] == 1e38)

    index.quantizer.free()
    data.free()
    ids.free()
    distances.free()
    labels.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
