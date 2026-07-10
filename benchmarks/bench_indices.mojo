from std.memory import alloc
from std.random import random_float64
from std.time import perf_counter_ns
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_ivf_flat import IndexIVFFlat
from mojovec.index.index_ivf_pq import IndexIVFPQ
from mojovec.index.index_hnsw import IndexHNSW
from mojovec.core.types import METRIC_L2

def compute_recall(ground_truth_labels: UnsafePointer[Int, MutUntrackedOrigin], 
                   pred_labels: UnsafePointer[Int, MutUntrackedOrigin], 
                   nq: Int, k: Int) -> Float32:
    var hits = 0
    for i in range(nq):
        var start_gt = i * k
        var start_pred = i * k
        var true_top1 = ground_truth_labels[start_gt]
        var found = False
        for j in range(k):
            if pred_labels[start_pred + j] == true_top1:
                found = True
                break
        if found: hits += 1
    return Float32(hits) / Float32(nq)

def main() raises:
    var d = 128
    var n = 100000
    var nq = 10000
    var k = 10
    
    print("Generating random dataset...")
    var data = alloc[Float32](n * d)
    var queries = alloc[Float32](nq * d)
    
    from std.random import random_si64
    
    var num_clusters = 1024
    var cluster_centers = alloc[Float32](num_clusters * d)
    for i in range(num_clusters * d):
        cluster_centers[i] = Float32(random_float64(-1.0, 1.0))
        
    for i in range(n):
        var c = Int(random_si64(0, Int64(num_clusters - 1)))
        for j in range(d):
            var noise = Float32(random_float64(-0.2, 0.2))
            data[i * d + j] = cluster_centers[c * d + j] + noise
            
    for i in range(nq):
        var c = Int(random_si64(0, Int64(num_clusters - 1)))
        for j in range(d):
            var noise = Float32(random_float64(-0.2, 0.2))
            queries[i * d + j] = cluster_centers[c * d + j] + noise
            
    cluster_centers.free()
        
    print("Dataset generated: N =", n, "D =", d, "Queries =", nq)
    print("--------------------------------------------------")
    
    # 1. Exact Search (IndexFlat)
    print("[1] IndexFlat (Exact Baseline)")
    var flat_index = IndexFlat(d)
    var t0 = perf_counter_ns()
    flat_index.add(n, data)
    var build_flat = Float64(perf_counter_ns() - t0) / 1e9
    print("Build time:", build_flat, "s")
    
    var gt_dists = alloc[Float32](nq * k)
    var gt_labels = alloc[Int](nq * k)
    
    t0 = perf_counter_ns()
    flat_index.search(nq, queries, k, gt_dists, gt_labels)
    var search_flat = Float64(perf_counter_ns() - t0) / 1e9
    var qps_flat = Float64(nq) / search_flat
    print("Search time:", search_flat, "s")
    print("QPS:", Int(qps_flat))
    print("Recall@10: 1.000")
    print("--------------------------------------------------")
    
    # 2. IndexIVFFlat
    var nlist = 1024
    print("[2] IndexIVFFlat (nlist =", nlist, ")")
    var quantizer1 = alloc[IndexFlat](1)
    quantizer1.init_pointee_move(IndexFlat(d))
    var ivf_flat = IndexIVFFlat[IndexFlat](quantizer1, d, nlist, METRIC_L2)
    ivf_flat.train(n, data)
    t0 = perf_counter_ns()
    ivf_flat.add(n, data)
    print("Build time:", Float64(perf_counter_ns() - t0) / 1e9, "s")
    
    # 3. IndexIVFPQ
    print("--------------------------------------------------")
    var M_pq = 64
    print("[3] IndexIVFPQ (nlist =", nlist, "M =", M_pq, ")")
    var quantizer2 = alloc[IndexFlat](1)
    quantizer2.init_pointee_move(IndexFlat(d))
    var ivf_pq = IndexIVFPQ[IndexFlat](quantizer2, d, nlist, M_pq, METRIC_L2)
    ivf_pq.train(n, data)
    t0 = perf_counter_ns()
    ivf_pq.add(n, data)
    print("Build time:", Float64(perf_counter_ns() - t0) / 1e9, "s")
    print("--------------------------------------------------")
    
    var pred_dists = alloc[Float32](nq * k)
    var pred_labels = alloc[Int](nq * k)
    
    print("[2] IndexIVFFlat Search:")
    var nprobe_list = [1, 5, 10, 50]
    for i in range(len(nprobe_list)):
        ivf_flat.nprobe = nprobe_list[i]
        t0 = perf_counter_ns()
        ivf_flat.search(nq, queries, k, pred_dists, pred_labels)
        var search_t = Float64(perf_counter_ns() - t0) / 1e9
        var qps = Float64(nq) / search_t
        var recall = compute_recall(gt_labels, pred_labels, nq, k)
        print("  nprobe=", ivf_flat.nprobe, " | QPS:", Int(qps), "| Recall@10:", recall)
        
    print("--------------------------------------------------")
    print("[3] IndexIVFPQ Search:")
    for i in range(len(nprobe_list)):
        ivf_pq.nprobe = nprobe_list[i]
        t0 = perf_counter_ns()
        ivf_pq.search(nq, queries, k, pred_dists, pred_labels)
        var search_t = Float64(perf_counter_ns() - t0) / 1e9
        var qps = Float64(nq) / search_t
        var recall = compute_recall(gt_labels, pred_labels, nq, k)
        print("  nprobe=", ivf_pq.nprobe, " | QPS:", Int(qps), "| Recall@10:", recall)
    print("--------------------------------------------------")
    
    # 4. IndexHNSW
    var M_hnsw = 32
    print("[4] IndexHNSW (M =", M_hnsw, ")")
    var hnsw = IndexHNSW[IndexFlat](IndexFlat(d), d, METRIC_L2, M_hnsw)
    hnsw.hnsw.efConstruction = 200
    
    t0 = perf_counter_ns()
    hnsw.add(n, data)
    var add_hnsw = Float64(perf_counter_ns() - t0) / 1e9
    print("Build time:", add_hnsw, "s")
    
    var efs = [10, 50, 100, 200]
    for i in range(len(efs)):
        var ef = efs[i]
        hnsw.hnsw.efSearch = ef
        t0 = perf_counter_ns()
        hnsw.search(nq, queries, k, pred_dists, pred_labels)
        var search_t = Float64(perf_counter_ns() - t0) / 1e9
        var qps = Float64(nq) / search_t
        var recall = compute_recall(gt_labels, pred_labels, nq, k)
        print("  efSearch=", ef, " | QPS:", Int(qps), "| Recall@10:", recall)
    print("--------------------------------------------------")
    
    data.free()
    queries.free()
    gt_dists.free()
    gt_labels.free()
    pred_dists.free()
    pred_labels.free()
    
    quantizer1.destroy_pointee()
    quantizer1.free()
    quantizer2.destroy_pointee()
    quantizer2.free()
