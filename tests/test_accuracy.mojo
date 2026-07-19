from std.memory.span import Span
from std.testing import assert_true, TestSuite
from std.random import random_float64, seed
from std.memory import alloc

from mojovec.core.types import METRIC_L2, QT_8bit
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_ivf_flat import IndexIVFFlat
from mojovec.index.index_ivf_pq import IndexIVFPQ
from mojovec.index.index_scalar_quantizer import IndexScalarQuantizer
from mojovec.index.index_hnsw import IndexHNSW

comptime d = 32
comptime nb = 5000
comptime nq = 100
comptime k = 10

struct Dataset:
    var db: UnsafePointer[Float32, MutUntrackedOrigin]
    var queries: UnsafePointer[Float32, MutUntrackedOrigin]

    def __init__(out self):
        seed(42)
        self.db = alloc[Float32](nb * d)
        for i in range(nb * d): 
            self.db[i] = Float32(random_float64(-1.0, 1.0))
            
        self.queries = alloc[Float32](nq * d)
        for i in range(nq * d): 
            self.queries[i] = Float32(random_float64(-1.0, 1.0))

    def free(self):
        self.db.free()
        self.queries.free()

def get_ground_truth(ds: Dataset) -> UnsafePointer[Int, MutUntrackedOrigin]:
    var index = IndexFlat(d, METRIC_L2)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=nb * d))
    
    var gt_dist = alloc[Float32](nq * k)
    var gt_labels = alloc[Int](nq * k)
    
    var queries_span = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var dist_span = Span[mut=True, Float32, MutUntrackedOrigin](ptr=gt_dist, length=nq * k)
    var labels_span = Span[mut=True, Int, MutUntrackedOrigin](ptr=gt_labels, length=nq * k)
    
    index.search(queries_span, k, dist_span, labels_span)
    gt_dist.free()
    return gt_labels

def compute_recall(gt_labels: UnsafePointer[Int, MutUntrackedOrigin], test_labels: UnsafePointer[Int, MutUntrackedOrigin], nq: Int, k: Int) -> Float32:
    var matches = 0
    for i in range(nq):
        var gt_offset = i * k
        var test_offset = i * k
        for j in range(k):
            var gt_val = gt_labels[gt_offset + j]
            for m in range(k):
                if test_labels[test_offset + m] == gt_val:
                    matches += 1
                    break
    return Float32(matches) / Float32(nq * k)

def test_accuracy_hnsw() raises:
    var ds = Dataset()
    var gt_labels = get_ground_truth(ds)
    
    var index = IndexHNSW[IndexFlat](IndexFlat(d, METRIC_L2), d, METRIC_L2, 32)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=nb * d))
    
    var test_dist = alloc[Float32](nq * k)
    var test_labels = alloc[Int](nq * k)
    
    var queries_span = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var dist_span = Span[mut=True, Float32, MutUntrackedOrigin](ptr=test_dist, length=nq * k)
    var labels_span = Span[mut=True, Int, MutUntrackedOrigin](ptr=test_labels, length=nq * k)
    
    index.search(queries_span, k, dist_span, labels_span)
    
    var recall = compute_recall(gt_labels, test_labels, nq, k)
    print("HNSW Recall@10:", recall)
    assert_true(recall >= 0.80, "HNSW Recall too low")
    
    ds.free()
    gt_labels.free()
    test_dist.free()
    test_labels.free()

def test_accuracy_ivf_flat() raises:
    var ds = Dataset()
    var gt_labels = get_ground_truth(ds)
    
    var nlist = 64
    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(IndexFlat(d, METRIC_L2))
    var index = IndexIVFFlat[IndexFlat](quantizer, d, nlist, METRIC_L2)
    index.train(nb, ds.db)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=nb * d))
    index.nprobe = 16
    
    var test_dist = alloc[Float32](nq * k)
    var test_labels = alloc[Int](nq * k)
        
    var queries_span = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var dist_span = Span[mut=True, Float32, MutUntrackedOrigin](ptr=test_dist, length=nq * k)
    var labels_span = Span[mut=True, Int, MutUntrackedOrigin](ptr=test_labels, length=nq * k)
    
    index.search(queries_span, k, dist_span, labels_span)
    
    var recall = compute_recall(gt_labels, test_labels, nq, k)
    print("IVFFlat Recall@10:", recall)
    assert_true(recall >= 0.75, "IVFFlat Recall too low")
    
    ds.free()
    gt_labels.free()
    test_dist.free()
    test_labels.free()
    index.quantizer.free()

def test_accuracy_ivf_pq() raises:
    var ds = Dataset()
    var gt_labels = get_ground_truth(ds)
    
    var nlist = 64
    var m = 8  # 8 subquantizers
    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(IndexFlat(d, METRIC_L2))
    var index = IndexIVFPQ[IndexFlat](quantizer, d, nlist, m, METRIC_L2)
    index.train(nb, ds.db)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=nb * d))
    index.nprobe = 16
    
    var test_dist = alloc[Float32](nq * k)
    var test_labels = alloc[Int](nq * k)
    
    var queries_span = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var dist_span = Span[mut=True, Float32, MutUntrackedOrigin](ptr=test_dist, length=nq * k)
    var labels_span = Span[mut=True, Int, MutUntrackedOrigin](ptr=test_labels, length=nq * k)
    
    index.search(queries_span, k, dist_span, labels_span)
    
    var recall = compute_recall(gt_labels, test_labels, nq, k)
    print("IVFPQ Recall@10:", recall)
    assert_true(recall >= 0.50, "IVFPQ Recall too low")
    
    ds.free()
    gt_labels.free()
    test_dist.free()
    test_labels.free()
    index.quantizer.free()

def test_accuracy_sq8() raises:
    var ds = Dataset()
    var gt_labels = get_ground_truth(ds)
    
    var index = IndexScalarQuantizer(d, QT_8bit, METRIC_L2)
    index.train(nb, ds.db)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=ds.db, length=nb * d))
    
    var test_dist = alloc[Float32](nq * k)
    var test_labels = alloc[Int](nq * k)
        
    var queries_span = Span[Float32, MutUntrackedOrigin](ptr=ds.queries, length=nq * d)
    var dist_span = Span[mut=True, Float32, MutUntrackedOrigin](ptr=test_dist, length=nq * k)
    var labels_span = Span[mut=True, Int, MutUntrackedOrigin](ptr=test_labels, length=nq * k)
    
    index.search(queries_span, k, dist_span, labels_span)
    
    var recall = compute_recall(gt_labels, test_labels, nq, k)
    print("SQ8 Recall@10:", recall)
    assert_true(recall >= 0.90, "SQ8 Recall too low")
    
    ds.free()
    gt_labels.free()
    test_dist.free()
    test_labels.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
