from std.testing import assert_true, assert_equal, TestSuite
from std.memory.alloc import unsafe_alloc
from std.random.philox import Random
from std.collections.span import Span
from std.io.file import FileHandle

from mojovec.core.types import METRIC_L2, QT_8bit
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_flat_sq8 import IndexFlatSQ8
from mojovec.index.index_hnsw import IndexHNSW
from mojovec.index.index_ivf_flat import IndexIVFFlat
from mojovec.io.serialization import (
    write_index_flat, read_index_flat,
    write_index_flat_sq8, read_index_flat_sq8,
    write_index_hnsw, read_index_hnsw,
    write_index_ivf_flat, read_index_ivf_flat
)

comptime d = 16
comptime n = 200
comptime k = 5
comptime nq = 10

struct Dataset:
    var db: Pointer[Float32, MutUntrackedOrigin]
    var queries: Pointer[Float32, MutUntrackedOrigin]

    def __init__(out self):
        var generator = Random(seed=UInt64(73))
        self.db = unsafe_alloc[Float32](n * d)
        var values = generator.step_uniform()
        for i in range(n * d):
            if i != 0 and i % 4 == 0:
                values = generator.step_uniform()
            self.db[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0
            
        self.queries = unsafe_alloc[Float32](nq * d)
        values = generator.step_uniform()
        for i in range(nq * d):
            if i != 0 and i % 4 == 0:
                values = generator.step_uniform()
            self.queries[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0

    def free(self):
        self.db.unsafe_free()
        self.queries.unsafe_free()

def assert_search_results(
    dists1: Pointer[Float32, MutUntrackedOrigin], labels1: Pointer[Int, MutUntrackedOrigin], 
    dists2: Pointer[Float32, MutUntrackedOrigin], labels2: Pointer[Int, MutUntrackedOrigin]
) raises:
    for i in range(nq * k):
        assert_equal(labels1[unsafe_offset=i], labels2[unsafe_offset=i], "Labels mismatch after deserialization")
        assert_true(dists1[unsafe_offset=i] == dists2[unsafe_offset=i], "Distances mismatch after deserialization")

def test_flat_io() raises:
    var ds = Dataset()
    var index = IndexFlat(d, METRIC_L2)
    index.add(Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.db, length=n * d))
    
    var dists1 = unsafe_alloc[Float32](nq * k)
    var labels1 = unsafe_alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](unsafe_ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](unsafe_ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_flat.bin", "w")
    write_index_flat(f_w, index)
    f_w.close()
    
    var f_r = open("test_flat.bin", "r")
    var index2 = read_index_flat(f_r)
    f_r.close()
    
    var dists2 = unsafe_alloc[Float32](nq * k)
    var labels2 = unsafe_alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](unsafe_ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](unsafe_ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.unsafe_free()
    labels1.unsafe_free()
    dists2.unsafe_free()
    labels2.unsafe_free()

def test_flat_sq8_io() raises:
    var ds = Dataset()
    var index = IndexFlatSQ8(d, METRIC_L2)
    index.add(Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.db, length=n * d))
    
    var dists1 = unsafe_alloc[Float32](nq * k)
    var labels1 = unsafe_alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](unsafe_ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](unsafe_ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_flat_sq8.bin", "w")
    write_index_flat_sq8(f_w, index)
    f_w.close()
    
    var f_r = open("test_flat_sq8.bin", "r")
    var index2 = read_index_flat_sq8(f_r)
    f_r.close()
    
    var dists2 = unsafe_alloc[Float32](nq * k)
    var labels2 = unsafe_alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](unsafe_ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](unsafe_ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.unsafe_free()
    labels1.unsafe_free()
    dists2.unsafe_free()
    labels2.unsafe_free()

def test_hnsw_io() raises:
    var ds = Dataset()
    var index = IndexHNSW[IndexFlat](IndexFlat(d, METRIC_L2), d, METRIC_L2, 32)
    index.add(Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.db, length=n * d))
    
    var dists1 = unsafe_alloc[Float32](nq * k)
    var labels1 = unsafe_alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](unsafe_ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](unsafe_ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_hnsw.bin", "w")
    write_index_hnsw(f_w, index)
    f_w.close()
    
    var f_r = open("test_hnsw.bin", "r")
    var index2 = read_index_hnsw(f_r)
    f_r.close()
    
    var dists2 = unsafe_alloc[Float32](nq * k)
    var labels2 = unsafe_alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](unsafe_ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](unsafe_ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.unsafe_free()
    labels1.unsafe_free()
    dists2.unsafe_free()
    labels2.unsafe_free()

def test_ivf_flat_io() raises:
    var ds = Dataset()
    var index = IndexIVFFlat[IndexFlat](
        IndexFlat(d, METRIC_L2), d, 8, METRIC_L2
    )
    index.train(
        Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.db, length=n * d)
    )
    index.add(Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.db, length=n * d))
    index.nprobe = 4
    
    var dists1 = unsafe_alloc[Float32](nq * k)
    var labels1 = unsafe_alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](unsafe_ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](unsafe_ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_ivf_flat.bin", "w")
    write_index_ivf_flat(f_w, index)
    f_w.close()
    
    var f_r = open("test_ivf_flat.bin", "r")
    var index2 = read_index_ivf_flat(f_r)
    f_r.close()
    index2.nprobe = 4
    
    var dists2 = unsafe_alloc[Float32](nq * k)
    var labels2 = unsafe_alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](unsafe_ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](unsafe_ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](unsafe_ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.unsafe_free()
    labels1.unsafe_free()
    dists2.unsafe_free()
    labels2.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
