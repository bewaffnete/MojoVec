from std.memory.span import Span
from std.memory import alloc
from std.random import random_float64
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_ivf_pq import IndexIVFPQ
from std.testing import assert_true, assert_equal, TestSuite

def test_ivf_pq_crud() raises:
    # Test internal CRUD operations on IVFPQ (training, adding, and checking Inverted Lists)
    var d = 4
    var n = 100
    var nlist = 4
    var M = 2
    
    var data = alloc[Float32](n * d)
    for i in range(n * d):
        data[i] = Float32(random_float64(-1.0, 1.0))
        
    var flat_quantizer = alloc[IndexFlat](1)
    flat_quantizer.init_pointee_move(IndexFlat(d))
    var ivf = IndexIVFPQ[IndexFlat](flat_quantizer, d, nlist, M)
    
    # Assert not trained
    assert_true(not ivf.is_trained)
    
    # Train
    ivf.train(
        Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d)
    )
    assert_true(ivf.is_trained, "Should be trained")
    
    # Ensure quantizer has the centroids
    assert_equal(ivf.quantizer[0].ntotal, nlist)
    
    # Add vectors
    var ids = alloc[Int](n)
    for i in range(n):
        ids[i] = i * 10
    ivf.add_with_ids(
        Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d),
        Span[Int, MutUntrackedOrigin](ptr=ids, length=n),
    )
    
    # Verify elements are successfully distributed into the inverted lists
    assert_equal(ivf.ntotal, n)
    
    var total_in_lists = 0
    for i in range(nlist):
        var size = ivf.invlists.list_size(i)
        assert_true(size >= 0)
        total_in_lists += size
        
    assert_equal(total_in_lists, n)
    
    # Perform a dummy search to ensure no crashes on precomputed tables
    var queries = alloc[Float32](1 * d)
    for i in range(d): queries[i] = 0.5
        
    var dists = alloc[Float32](1 * 5)
    var labels = alloc[Int](1 * 5)
    
    var span_d = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists, length=5)
    var span_l = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels, length=5)
    ivf.search(Span[Float32, MutUntrackedOrigin](ptr=queries, length=d), 5, span_d, span_l)
               
    # Search should return valid results if lists have elements
    var num_found = 0
    for i in range(5):
        if labels[i] != -1:
            num_found += 1
            
    assert_true(num_found > 0)
    
    _ = ivf.ntotal
    flat_quantizer.free()
    data.free()
    queries.free()
    ids.free()
    dists.free()
    labels.free()


def test_ivf_pq_sparse_results_have_deterministic_tail() raises:
    comptime d = 4
    comptime k = 5
    var data = alloc[Float32](d)
    for i in range(d):
        data[i] = 0.0
    var ids = alloc[Int](1)
    ids[0] = 42

    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(IndexFlat(d))
    var index = IndexIVFPQ[IndexFlat](quantizer, d, 1, 2)
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
