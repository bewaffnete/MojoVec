from std.time import perf_counter_ns
from std.memory import alloc, memcpy
from mojovec import Client
from std.collections import List

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
    
    var db_ptr = db_data.unsafe_ptr().bitcast[Float32]()
    var queries_ptr = queries_data.unsafe_ptr().bitcast[Float32]()
    var gt = gt_data.unsafe_ptr().bitcast[Int32]()
    
    # Convert pointer data to List[Float32] and List[Int] for new API
    var db_list = List[Float32](capacity=n * d)
    var ids_list = List[Int](capacity=n)
    for i in range(n):
        ids_list.append(i)
        for j in range(d):
            db_list.append(db_ptr[i * d + j])
            
    var queries_list = List[Float32](capacity=q * d)
    for i in range(q * d):
        queries_list.append(queries_ptr[i])
    
    print("--------------------------------------------------")
    print("[MojoVec] Collection API (HNSW, M=" + String(M) + ", efConstruction=" + String(efConstruction) + ")")
    
    var client = Client()
    # Note: efConstruction is defaulted in Collection, we can't easily change it
    # without modifying Collection.__init__, but we will benchmark the API's default (which is 200).
    var collection = client.create_collection("bench_hnsw", dimension=d)
    
    var t0 = perf_counter_ns()
    collection.add(ids_list, db_list)
    var t1 = perf_counter_ns()
    var build_time = Float64(t1 - t0) / 1e9
    print("Build time: " + String(build_time) + " s")
    
    var ef_list = List[Int]()
    ef_list.append(10)
    ef_list.append(40)
    ef_list.append(50)
    ef_list.append(100)
    ef_list.append(200)
    
    print("Search:")
    for i in range(len(ef_list)):
        var ef = ef_list[i]
        collection.set_ef_search(ef)
        
        # warmup
        _ = collection.query(queries_list, n_results=10)
        
        var t_s_0 = perf_counter_ns()
        var results = collection.query(queries_list, n_results=10)
        var t_s_1 = perf_counter_ns()
        var search_time = Float64(t_s_1 - t_s_0) / 1e9
        
        var qps = Float64(q) / search_time
        
        # Calculate recall
        var recall_sum: Float64 = 0.0
        for qi in range(q):
            var hits = 0
            for j in range(k):
                var res_id = results.ids[qi][j]
                for g in range(k):
                    if res_id == Int(gt[qi * k + g]):
                        hits += 1
                        break
            recall_sum += Float64(hits) / Float64(k)
        var recall = recall_sum / Float64(q)
        
        print("  efSearch=" + String(ef) + " | QPS: " + String(qps) + " | Recall@10: " + String(recall))
