from std.memory import alloc
from std.random import random_float64
from src.mojovec.index.index_flat import IndexFlat
from src.mojovec.index.index_ivf_flat import IndexIVFFlat
from src.mojovec.index.index_ivf_pq import IndexIVFPQ
from src.mojovec.io.serialization import write_index_flat, read_index_flat
from src.mojovec.io.serialization import write_index_ivf_flat, read_index_ivf_flat
from src.mojovec.io.serialization import write_index_ivf_pq, read_index_ivf_pq
from std.io.file import FileHandle

def assert_true(cond: Bool, msg: String = "Assertion failed") raises:
    if not cond:
        raise Error(msg)

def test_flat() raises:
    var d = 16
    var n = 100
    var k = 5
    var data = alloc[Float32](n * d)
    var queries = alloc[Float32](k * d)
    for i in range(n * d): data[i] = Float32(random_float64(-1.0, 1.0))
    for i in range(k * d): queries[i] = Float32(random_float64(-1.0, 1.0))
    
    var index = IndexFlat(d)
    index.add(n, data)
    
    # Search before
    var dists1 = alloc[Float32](k * k)
    var labels1 = alloc[Int](k * k)
    index.search(k, queries, k, dists1, labels1)
    
    var f_w = open("test_flat.bin", "w")
    write_index_flat(f_w, index)
    f_w.close()
    
    var f_r = open("test_flat.bin", "r")
    var index2 = read_index_flat(f_r)
    f_r.close()
    
    # Search after
    var dists2 = alloc[Float32](k * k)
    var labels2 = alloc[Int](k * k)
    index2.search(k, queries, k, dists2, labels2)
    
    for i in range(k * k):
        assert_true(labels1[i] == labels2[i], "Labels mismatch Flat")
        assert_true(dists1[i] == dists2[i], "Dists mismatch Flat")
        
    print("Flat IO test passed!")
    
    data.free()
    queries.free()
    dists1.free()
    labels1.free()
    dists2.free()
    labels2.free()

def test_ivf_pq() raises:
    print("Starting IVFPQ test")
    var d = 16
    var n = 1000
    var k = 5
    var data = alloc[Float32](n * d)
    var queries = alloc[Float32](k * d)
    for i in range(n * d): data[i] = Float32(random_float64(-1.0, 1.0))
    for i in range(k * d): queries[i] = Float32(random_float64(-1.0, 1.0))
    
    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(IndexFlat(d))
    var index = IndexIVFPQ[IndexFlat](quantizer, d, 10, 4)
    index.train(n, data)
    var ids = alloc[Int](n)
    for i in range(n): ids[i] = i
    index.add_with_ids(n, data, ids)
    index.nprobe = 3
    
    print("Searching before save")
    # Search before
    var dists1 = alloc[Float32](k * k)
    var labels1 = alloc[Int](k * k)
    index.search(k, queries, k, dists1, labels1)
    
    print("Writing index")
    var f_w = open("test_ivfpq.bin", "w")
    write_index_ivf_pq(f_w, index)
    f_w.close()
    
    print("Reading index")
    var f_r = open("test_ivfpq.bin", "r")
    var quantizer_r = alloc[IndexFlat](1)
    quantizer_r.init_pointee_move(IndexFlat(d))
    var index2 = IndexIVFPQ[IndexFlat](quantizer_r, d, 10, 4)
    read_index_ivf_pq(f_r, index2)
    f_r.close()
    index2.nprobe = 3
    
    print("Searching after load")
    # Search after
    var dists2 = alloc[Float32](k * k)
    var labels2 = alloc[Int](k * k)
    index2.search(k, queries, k, dists2, labels2)
    
    for i in range(k * k):
        assert_true(labels1[i] == labels2[i], "Labels mismatch IVFPQ")
        assert_true(dists1[i] == dists2[i], "Dists mismatch IVFPQ")
        
    print("IVFPQ IO test passed!")
    
    data.free()
    queries.free()
    dists1.free()
    labels1.free()
    dists2.free()
    labels2.free()
    # index2's quantizer is allocated in read_index_ivf_pq, we should free it ideally but script ends anyway.

def main() raises:
    test_flat()
    test_ivf_pq()
    print("All IO tests passed!")
