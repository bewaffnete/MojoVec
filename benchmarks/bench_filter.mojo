from std.memory.span import Span
from std.memory import alloc
from std.random import random_float64
from std.time import perf_counter_ns
from mojovec.index.index_flat import IndexFlat

def bench_filter() raises:
    print("=== Benchmarking Soft Delete / Filter Performance ===")
    var d = 128
    var n = 100000
    var nq = 100
    var k = 10
    
    # 1. Prepare data
    var data = alloc[Float32](n * d)
    for i in range(n * d):
        data[i] = Float32(random_float64(-1.0, 1.0))
        
    var queries = alloc[Float32](nq * d)
    for i in range(nq * d):
        queries[i] = Float32(random_float64(-1.0, 1.0))
        
    # 2. Build index
    var index = IndexFlat(d)
    index.add(Span[Float32, MutUntrackedOrigin](ptr=data, length=n * d))
    
    # 3. Create Filter (delete every even vector)
    var filter_size = (n + 7) // 8
    var filter = alloc[UInt8](filter_size)
    for i in range(filter_size): filter[i] = 0
    for i in range(n):
        if i % 2 == 0:
            # Set bit to 1 indicating deleted
            var byte_idx = i // 8
            var bit_idx = i % 8
            filter[byte_idx] |= UInt8(1 << bit_idx)
            
    # Output spans
    var dists = alloc[Float32](nq * k)
    var labels = alloc[Int](nq * k)
    var span_d = Span[mut=True, Float32, MutUntrackedOrigin](ptr=dists, length=nq * k)
    var span_l = Span[mut=True, Int, MutUntrackedOrigin](ptr=labels, length=nq * k)
    
    var span_filter_empty = Span[UInt8, _](ptr=alloc[UInt8](0), length=0)
    var span_filter_active = Span[UInt8, _](ptr=filter, length=filter_size)
    
    # Warmup
    index.search(Span[Float32, MutUntrackedOrigin](ptr=queries, length=nq * d), k, span_d, span_l, span_filter_empty)
    
    # Benchmark without filter
    var iters = 5
    var start_no_filter = perf_counter_ns()
    for _ in range(iters):
        index.search(Span[Float32, MutUntrackedOrigin](ptr=queries, length=nq * d), k, span_d, span_l, span_filter_empty)
    var end_no_filter = perf_counter_ns()
    var time_no_filter = Float64(end_no_filter - start_no_filter) / 1e6 / Float64(iters)
    
    # Benchmark with filter
    var start_filter = perf_counter_ns()
    for _ in range(iters):
        index.search(Span[Float32, MutUntrackedOrigin](ptr=queries, length=nq * d), k, span_d, span_l, span_filter_active)
    var end_filter = perf_counter_ns()
    var time_filter = Float64(end_filter - start_filter) / 1e6 / Float64(iters)
    
    print("Number of database vectors:", n)
    print("Dimensionality:", d)
    print("Number of queries:", nq)
    print("Search k:", k)
    print("Time WITHOUT filter (ms / query batch):", time_no_filter)
    print("Time WITH filter (ms / query batch):", time_filter)
    
    var overhead = (time_filter - time_no_filter) / time_no_filter * 100.0
    print("Filter Overhead:", overhead, "%")
    
    data.free()
    queries.free()
    dists.free()
    labels.free()
    filter.free()

def main() raises:
    bench_filter()
