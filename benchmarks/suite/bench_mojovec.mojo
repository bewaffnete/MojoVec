from std.time import perf_counter_ns
from std.memory import alloc, memcpy
from src.mojovec.index.index_hnsw import IndexHNSW
from src.mojovec.index.index_flat import IndexFlat
from src.mojovec.core.types import METRIC_L2

def load_bin_data(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var data = f.read_bytes()
    return data^

def main() raises:
    var n = 100_000
    var q = 10_000
    var d = 128
    var k = 10
    var M = 32
    var efConstruction = 200
    
    print("Loading data...")
    var db_data = load_bin_data("benchmarks/suite/db.bin")
    var queries_data = load_bin_data("benchmarks/suite/queries.bin")
    var gt_data = load_bin_data("benchmarks/suite/groundtruth.bin")
    
    var db = alloc[Float32](n * d)
    var queries = alloc[Float32](q * d)
    var gt = alloc[Int32](q * k)
    
    memcpy(dest=db, src=db_data.unsafe_ptr().bitcast[Float32](), count=n * d)
    memcpy(dest=queries, src=queries_data.unsafe_ptr().bitcast[Float32](), count=q * d)
    memcpy(dest=gt, src=gt_data.unsafe_ptr().bitcast[Int32](), count=q * k)
    
    print("--------------------------------------------------")
    print("[MojoVec] IndexHNSW (M=" + String(M) + ", efConstruction=" + String(efConstruction) + ")")
    
    var hnsw = IndexHNSW[IndexFlat](IndexFlat(d), d, METRIC_L2, M)
    hnsw.hnsw.efConstruction = efConstruction
    
    var t0 = perf_counter_ns()
    hnsw.add(n, db)
    var t1 = perf_counter_ns()
    var build_time = Float64(t1 - t0) / 1e9
    print("Build time: " + String(build_time) + " s")
    
    var ef_list = List[Int]()
    ef_list.append(10)
    ef_list.append(50)
    ef_list.append(100)
    ef_list.append(200)
    
    var D_res = alloc[Float32](q * k)
    var I_res = alloc[Int](q * k)
    
    print("Search:")
    for i in range(len(ef_list)):
        var ef = ef_list[i]
        hnsw.hnsw.efSearch = ef
        
        # warmup
        hnsw.search(100, queries, 10, D_res, I_res)
        
        var t_s_0 = perf_counter_ns()
        hnsw.search(q, queries, 10, D_res, I_res)
        var t_s_1 = perf_counter_ns()
        var search_time = Float64(t_s_1 - t_s_0) / 1e9
        
        var qps = Float64(q) / search_time
        
        # Calculate recall
        var recall_sum: Float64 = 0.0
        for qi in range(q):
            var hits = 0
            for j in range(k):
                var res_id = I_res[qi * k + j]
                for g in range(k):
                    if res_id == Int(gt[qi * k + g]):
                        hits += 1
                        break
            recall_sum += Float64(hits) / Float64(k)
        var recall = recall_sum / Float64(q)
        
        print("  efSearch=" + String(ef) + " | QPS: " + String(qps) + " | Recall@10: " + String(recall))
        
    db.free()
    queries.free()
    gt.free()
    D_res.free()
    I_res.free()
