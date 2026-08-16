from std.collections.span import Span
from std.memory.alloc import unsafe_alloc
from std.random.philox import Random
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_ivf_flat import IndexIVFFlat

from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite

def test_ivf_flat() raises:
    var d = 16
    var n = 1000
    var nlist = 10
    var nq = 10
    var k = 5
    var generator = Random(seed=UInt64(101))
    
    var data = unsafe_alloc[Float32](n * d)
    var values = generator.step_uniform()
    for i in range(n * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        data[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0
        
    var queries = unsafe_alloc[Float32](nq * d)
    values = generator.step_uniform()
    for i in range(nq * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        queries[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0
    pass  # print("Data and queries .")
    var ivf = IndexIVFFlat[IndexFlat](IndexFlat(d), d, nlist)
    
    pass  # print("Training IVF...")
    ivf.train(
        Span[Float32, MutUntrackedOrigin](unsafe_ptr=data, length=n * d)
    )
    assert_true(ivf.is_trained, "Should be trained")
    
    pass  # print("Adding vectors...")
    
    var ids = unsafe_alloc[Int](n)
    for i in range(n):
        ids[unsafe_offset=i] = i * 10  # Custom IDs
    ivf.add_with_ids(
        Span[Float32, MutUntrackedOrigin](unsafe_ptr=data, length=n * d),
        Span[Int, MutUntrackedOrigin](unsafe_ptr=ids, length=n),
    )
    assert_true(ivf.ntotal == n, "Total should match")
    
    ivf.nprobe = 3
    pass  # print("Searching IVF...")
    var dists = unsafe_alloc[Float32](nq * k)
    var labels = unsafe_alloc[Int](nq * k)
    
    var span_dist_1 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=dists, length=nq * k)
    var span_labels_1 = Span[Int, MutUntrackedOrigin](unsafe_ptr=labels, length=nq * k)
    ivf.search(Span[Float32, MutUntrackedOrigin](unsafe_ptr=queries, length=nq * d), k, span_dist_1, span_labels_1)
    

    # Keep ivf alive
    _ = ivf.ntotal
    
    data.unsafe_free()
    queries.unsafe_free()
    ids.unsafe_free()
    dists.unsafe_free()
    labels.unsafe_free()

def test_ivf_flat_exact_match() raises:
    var d = 4
    var data = unsafe_alloc[Float32](4)
    for i in range(4): data[unsafe_offset=i] = Float32(i)
    var ivf = IndexIVFFlat[IndexFlat](IndexFlat(d), d, 1)
    
    ivf.train(Span[Float32, MutUntrackedOrigin](unsafe_ptr=data, length=d))
    ivf.add(Span[Float32, MutUntrackedOrigin](unsafe_ptr=data, length=1 * d))
    
    var distances = unsafe_alloc[Float32](1)
    var labels = unsafe_alloc[Int](1)
    var span_dist_2 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=distances, length=1 * 1)
    var span_labels_2 = Span[Int, MutUntrackedOrigin](unsafe_ptr=labels, length=1 * 1)
    ivf.search(Span[Float32, MutUntrackedOrigin](unsafe_ptr=data, length=1 * d), 1, span_dist_2, span_labels_2)
    assert_true(labels[unsafe_offset=0] == 0, "Should match id 0")
    assert_almost_equal(distances[unsafe_offset=0], 0.0, msg="Distance should be 0")
    
    distances.unsafe_free()
    labels.unsafe_free()
    data.unsafe_free()


def test_ivf_flat_sparse_results_have_deterministic_tail() raises:
    comptime d = 4
    comptime k = 5
    var data = unsafe_alloc[Float32](d)
    for i in range(d):
        data[unsafe_offset=i] = 0.0
    var ids = unsafe_alloc[Int](1)
    ids[unsafe_offset=0] = 42

    var index = IndexIVFFlat[IndexFlat](IndexFlat(d), d, 1)
    var data_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=data, length=d
    )
    index.train(data_span)
    index.add_with_ids(
        data_span,
        Span[Int, MutUntrackedOrigin](unsafe_ptr=ids, length=1),
    )

    var distances = unsafe_alloc[Float32](k)
    var labels = unsafe_alloc[Int](k)
    for i in range(k):
        distances[unsafe_offset=i] = -7.0
        labels[unsafe_offset=i] = -7
    var distance_span = Span[mut=True, Float32, MutUntrackedOrigin](
        unsafe_ptr=distances, length=k
    )
    var label_span = Span[mut=True, Int, MutUntrackedOrigin](
        unsafe_ptr=labels, length=k
    )
    index.search(
        data_span,
        k,
        distance_span,
        label_span,
    )

    assert_equal(labels[unsafe_offset=0], 42)
    assert_true(distances[unsafe_offset=0] == 0.0)
    for i in range(1, k):
        assert_equal(labels[unsafe_offset=i], -1)
        assert_true(distances[unsafe_offset=i] == 1e38)

    data.unsafe_free()
    ids.unsafe_free()
    distances.unsafe_free()
    labels.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
