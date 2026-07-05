from std.memory import alloc
from std.random import random_float64
from src.mojovec.index.index_flat import IndexFlat
from src.mojovec.index.index_ivf_flat import IndexIVFFlat

def assert_true(cond: Bool, msg: String = "Assertion failed") raises:
    if not cond:
        raise Error(msg)

def main() raises:
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
        
    var flat_quantizer = alloc[IndexFlat](1)
    flat_quantizer[0] = IndexFlat(d)
    var ivf = IndexIVFFlat[IndexFlat](flat_quantizer, d, nlist)
    
    print("Training IVF...")
    ivf.train(n, data)
    assert_true(ivf.is_trained, "Should be trained")
    
    print("Adding vectors...")
    
    var ids = alloc[Int](n)
    for i in range(n):
        ids[i] = i * 10  # Custom IDs
    ivf.add_with_ids(n, data, ids)
    assert_true(ivf.ntotal == n, "Total should match")
    
    ivf.nprobe = 3
    print("Searching IVF...")
    var dists = alloc[Float32](nq * k)
    var labels = alloc[Int](nq * k)
    
    ivf.search(nq, queries, k, dists, labels)
    
    for i in range(nq):
        print("Query", i, "top-1 ID:", labels[i * k], "dist:", dists[i * k])
        
    print("All IndexIVFFlat tests passed!")
    
    # Keep ivf alive
    _ = ivf.ntotal
    
    flat_quantizer.free()
    data.free()
    queries.free()
    ids.free()
    dists.free()
    labels.free()
