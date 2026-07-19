from std.testing import assert_true, assert_equal, TestSuite
from std.memory import alloc
from std.random import random_float64
from std.memory.span import Span
from std.io.file import FileHandle

from mojovec.core.types import METRIC_L2, QT_8bit
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_flat_sq8 import IndexFlatSQ8
from mojovec.index.index_hnsw import IndexHNSW
from mojovec.index.index_ivf_flat import IndexIVFFlat
from mojovec.index.index_ivf_pq import IndexIVFPQ
from mojovec.io.serialization import (
    write_index_flat, read_index_flat,
    write_index_flat_sq8, read_index_flat_sq8,
    write_index_hnsw, read_index_hnsw,
    write_index_ivf_flat, read_index_ivf_flat,
    write_index_ivf_pq, read_index_ivf_pq
)

comptime d = 16
comptime n = 200
comptime k = 5
comptime nq = 10

struct Dataset:
    var db: UnsafePointer[Float32, MutUntrackedOrigin]
    var queries: UnsafePointer[Float32, MutUntrackedOrigin]

    def __init__(out self):
        self.db = alloc[Float32](n * d)
        for i in range(n * d): 
            self.db[i] = Float32(random_float64(-1.0, 1.0))
            
        self.queries = alloc[Float32](nq * d)
        for i in range(nq * d): 
            self.queries[i] = Float32(random_float64(-1.0, 1.0))

    def free(self):
        self.db.free()
        self.queries.free()

def assert_search_results(
    dists1: UnsafePointer[Float32, MutUntrackedOrigin], labels1: UnsafePointer[Int, MutUntrackedOrigin], 
    dists2: UnsafePointer[Float32, MutUntrackedOrigin], labels2: UnsafePointer[Int, MutUntrackedOrigin]
) raises:
    for i in range(nq * k):
        assert_equal(labels1[i], labels2[i], "Labels mismatch after deserialization")
        assert_true(dists1[i] == dists2[i], "Distances mismatch after deserialization")

def test_flat_io() raises:
    var ds = Dataset()
    var index = IndexFlat(d, METRIC_L2)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=n * d))
    
    var dists1 = alloc[Float32](nq * k)
    var labels1 = alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_flat.bin", "w")
    write_index_flat(f_w, index)
    f_w.close()
    
    var f_r = open("test_flat.bin", "r")
    var index2 = read_index_flat(f_r)
    f_r.close()
    
    var dists2 = alloc[Float32](nq * k)
    var labels2 = alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.free()
    labels1.free()
    dists2.free()
    labels2.free()

def test_flat_sq8_io() raises:
    var ds = Dataset()
    var index = IndexFlatSQ8(d, METRIC_L2)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=n * d))
    
    var dists1 = alloc[Float32](nq * k)
    var labels1 = alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_flat_sq8.bin", "w")
    write_index_flat_sq8(f_w, index)
    f_w.close()
    
    var f_r = open("test_flat_sq8.bin", "r")
    var index2 = read_index_flat_sq8(f_r)
    f_r.close()
    
    var dists2 = alloc[Float32](nq * k)
    var labels2 = alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.free()
    labels1.free()
    dists2.free()
    labels2.free()

def test_hnsw_io() raises:
    var ds = Dataset()
    var index = IndexHNSW[IndexFlat](IndexFlat(d, METRIC_L2), d, METRIC_L2, 32)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=n * d))
    
    var dists1 = alloc[Float32](nq * k)
    var labels1 = alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_hnsw.bin", "w")
    write_index_hnsw(f_w, index)
    f_w.close()
    
    var f_r = open("test_hnsw.bin", "r")
    var index2 = read_index_hnsw(f_r)
    f_r.close()
    
    var dists2 = alloc[Float32](nq * k)
    var labels2 = alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.free()
    labels1.free()
    dists2.free()
    labels2.free()

def test_ivf_flat_io() raises:
    var ds = Dataset()
    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(IndexFlat(d, METRIC_L2))
    var index = IndexIVFFlat[IndexFlat](quantizer, d, 8, METRIC_L2)
    index.train(n, ds.db)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=n * d))
    index.nprobe = 4
    
    var dists1 = alloc[Float32](nq * k)
    var labels1 = alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_ivf_flat.bin", "w")
    write_index_ivf_flat(f_w, index)
    f_w.close()
    
    var f_r = open("test_ivf_flat.bin", "r")
    var index2 = read_index_ivf_flat(f_r)
    f_r.close()
    index2.nprobe = 4
    
    var dists2 = alloc[Float32](nq * k)
    var labels2 = alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.free()
    labels1.free()
    dists2.free()
    labels2.free()
    index.quantizer.free()
    index2.quantizer.free()

def test_ivf_pq_io() raises:
    var ds = Dataset()
    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(IndexFlat(d, METRIC_L2))
    var index = IndexIVFPQ[IndexFlat](quantizer, d, 8, 4, METRIC_L2)
    index.train(n, ds.db)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=n * d))
    index.nprobe = 4
    
    var dists1 = alloc[Float32](nq * k)
    var labels1 = alloc[Int](nq * k)
    var q_span1 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span1 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists1, length=nq * k)
    var l_span1 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels1, length=nq * k)
    index.search(q_span1, k, d_span1, l_span1)
    
    var f_w = open("test_ivf_pq.bin", "w")
    write_index_ivf_pq(f_w, index)
    f_w.close()
    
    var f_r = open("test_ivf_pq.bin", "r")
    var index2 = read_index_ivf_pq(f_r)
    f_r.close()
    index2.nprobe = 4
    
    var dists2 = alloc[Float32](nq * k)
    var labels2 = alloc[Int](nq * k)
    var q_span2 = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var d_span2 = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists2, length=nq * k)
    var l_span2 = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels2, length=nq * k)
    index2.search(q_span2, k, d_span2, l_span2)
    
    assert_search_results(dists1, labels1, dists2, labels2)
    
    ds.free()
    dists1.free()
    labels1.free()
    dists2.free()
    labels2.free()
    index.quantizer.free()
    index2.quantizer.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
