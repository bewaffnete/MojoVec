from mojovec.index.index_hnsw import IndexHNSW
from mojovec.index.index_flat import IndexFlat
from mojovec.core.types import METRIC_L2
from std.random import rand
from std.memory import alloc

from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite

def test_hnsw_recall() raises:
    var d = 16
    var nb = 1000
    var nq = 10
    var k = 5
    var M = 16
    
    # 1. Generate data
    var xb = alloc[Float32](nb * d)
    var xq = alloc[Float32](nq * d)
    rand(xb, nb * d)
    rand(xq, nq * d)
    
    # 2. Build and search Flat Index (Ground Truth)
    var flat = IndexFlat(d, METRIC_L2)
    flat.add(nb, xb)
    
    var gt_dist = alloc[Float32](nq * k)
    var gt_labels = alloc[Int](nq * k)
    flat.search(nq, xq, k, gt_dist, gt_labels)
    
    # 3. Build and search HNSW Index
    var storage = IndexFlat(d, METRIC_L2)
    var hnsw = IndexHNSW[IndexFlat](storage^, d, METRIC_L2, M=M)
    hnsw.hnsw.efConstruction = 40
    hnsw.hnsw.efSearch = 40
    
    hnsw.add(nb, xb)
    
    var hnsw_dist = alloc[Float32](nq * k)
    var hnsw_labels = alloc[Int](nq * k)
    hnsw.search(nq, xq, k, hnsw_dist, hnsw_labels)
    
    # 4. Compare recall
    for i in range(5):
        pass  # print("Query", i)
        pass  # print("GT  : ", end="")
        for j in range(k):
            pass  # print(gt_labels[i * k + j], end=" ")
        pass  # print("\nHNSW: ", end="")
        for j in range(k):
            pass  # print(hnsw_labels[i * k + j], end=" ")
        pass  # print("\n")
        
    var matches = 0
    for i in range(nq):
        for j in range(k):
            var hnsw_lbl = hnsw_labels[i * k + j]
            # Check if this label is in the top-k of ground truth
            for l in range(k):
                if gt_labels[i * k + l] == hnsw_lbl:
                    matches += 1
                    break
                    
    var recall = Float32(matches) / Float32(nq * k)
    pass  # print("HNSW Recall@5:", recall)
    
    assert_true(recall > 0.8, "Recall is too low, HNSW implementation might be flawed.")
    
    xb.free()
    xq.free()
    gt_dist.free()
    gt_labels.free()
    hnsw_dist.free()
    hnsw_labels.free()
    
def test_hnsw_edge_cases() raises:
    var d = 4
    var storage = IndexFlat(d, METRIC_L2)
    var hnsw = IndexHNSW[IndexFlat](storage^, d, METRIC_L2, M=4)
    
    # 1. efSearch < k should not crash but maybe just return whatever it can
    hnsw.hnsw.efSearch = 1
    
    var xb = alloc[Float32](8)
    for i in range(8): xb[i] = Float32(i)
    hnsw.add(2, xb)
    
    var xq = alloc[Float32](4)
    for i in range(4): xq[i] = 1.0
        
    var dist = alloc[Float32](5)
    var labels = alloc[Int](5)
    hnsw.search(1, xq, 5, dist, labels) # k=5 > efSearch=1
    
    # Check that padded labels are -1
    var num_valid = 0
    for i in range(5):
        if labels[i] != -1:
            num_valid += 1
            
    assert_true(num_valid <= 2, "Cannot return more than database size")
    
    xb.free()
    xq.free()
    dist.free()
    labels.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
