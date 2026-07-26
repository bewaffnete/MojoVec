from std.time import perf_counter_ns
from std.memory import alloc, memcpy
from std.collections import List
from std.memory.span import Span
from mojovec.index.index_flat import IndexFlat
from mojovec.core.types import METRIC_L2

def load_bin_data(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var data = f.read_bytes()
    return data^

def main() raises:
    var n = 1_000_000
    var q = 10_000
    var d = 128
    var k = 10

    print("Loading data...")
    var db_data = load_bin_data("benchmarks/sift1m/sift_base.fvecs")
    var queries_data = load_bin_data("benchmarks/sift1m/sift_query.fvecs")

    var db_ptr = db_data.unsafe_ptr().bitcast[Float32]()
    var queries_ptr = queries_data.unsafe_ptr().bitcast[Float32]()

    var db_list = List[Float32](capacity=n * d)
    for i in range(n):
        var offset = i * (d + 1) + 1
        for j in range(d):
            db_list.append(db_ptr[offset + j])

    var queries_list = List[Float32](capacity=q * d)
    for i in range(q):
        var offset = i * (d + 1) + 1
        for j in range(d):
            queries_list.append(queries_ptr[offset + j])

    print("--------------------------------------------------")
    print("[MojoVec] Flat Exhaustive Search Benchmark")

    var index = IndexFlat(d, METRIC_L2)

    var t0 = perf_counter_ns()
    var db_span = Span[Float32](ptr=db_list.unsafe_ptr(), length=len(db_list))
    index.add(db_span)
    var t1 = perf_counter_ns()
    print("Add time: " + String(Float64(t1 - t0) / 1e9) + " s")

    var dist_ptr = alloc[Float32](q * k)
    var labels_ptr = alloc[Int](q * k)
    var q_span = Span[Float32](ptr=queries_list.unsafe_ptr(), length=len(queries_list))
    var d_span = Span[mut=True, Float32](ptr=dist_ptr, length=q * k)
    var l_span = Span[mut=True, Int](ptr=labels_ptr, length=q * k)

    # Warmup
    index.search(q_span, k, d_span, l_span)

    # Run
    var total_time: Float64 = 0.0
    var iters = 5
    for _ in range(iters):
        t0 = perf_counter_ns()
        index.search(q_span, k, d_span, l_span)
        t1 = perf_counter_ns()
        total_time += Float64(t1 - t0) / 1e9

    var avg_time = total_time / Float64(iters)
    var qps = Float64(q) / avg_time
    print("Avg Search time: " + String(avg_time) + " s")
    print("QPS: " + String(qps))

    dist_ptr.free()
    labels_ptr.free()
