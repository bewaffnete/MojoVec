from src.mojovec.index.index_hnsw import IndexHNSW
from src.mojovec.index.index_flat import IndexFlat
from src.mojovec.core.types import METRIC_L2
from std.random import rand
from std.memory import alloc

def assert_true(cond: Bool, msg: String = "Assertion failed") raises:
    if not cond:
        raise Error(msg)

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
        print("Query", i)
        print("GT  : ", end="")
        for j in range(k):
            print(gt_labels[i * k + j], end=" ")
        print("\nHNSW: ", end="")
        for j in range(k):
            print(hnsw_labels[i * k + j], end=" ")
        print("\n")
        
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
    print("HNSW Recall@5:", recall)
    
    assert_true(recall > 0.8, "Recall is too low, HNSW implementation might be flawed.")
    
    xb.free()
    xq.free()
    gt_dist.free()
    gt_labels.free()
    hnsw_dist.free()
    hnsw_labels.free()
    
def main() raises:
    print("Testing HNSW Index...")
    test_hnsw_recall()
    print("All HNSW tests passed!")
