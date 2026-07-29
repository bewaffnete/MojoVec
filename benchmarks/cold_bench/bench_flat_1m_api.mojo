from std.collections import List
from std.ffi import external_call
from std.time import perf_counter_ns
from mojovec import Collection
from sift1m_data import (
    first_vectors,
    load_fvecs,
    load_ground_truth,
    recall_at_k,
)


def main() raises:
    comptime dimension = 128
    comptime query_count = 10_000
    comptime k = 10
    comptime loops_per_sample = 5
    comptime samples = 10

    print("Loading saved Flat SIFT1M collection...")
    var collection = Collection.load(
        "benchmarks/cold_bench/mojovec_hnsw_flat_1m.mojovec"
    )
    collection.set_ef_search(96)

    var queries = load_fvecs(
        "benchmarks/data/sift1m/sift_query.fvecs",
        query_count,
        dimension,
    )
    var ground_truth = load_ground_truth(
        "benchmarks/data/sift1m/sift_groundtruth.ivecs",
        query_count,
        k,
    )
    var warmup_queries = first_vectors(queries, 256, dimension)

    # Initialize an owned result without running a search.
    var empty_queries = List[Float32]()
    var results = collection.query(empty_queries, n_results=k)
    var measured_seconds: Float64 = 0.0

    for sample in range(samples):
        print("cooldown before sample " + String(sample + 1) + "...")
        _ = external_call["sleep", UInt](UInt(30))
        _ = collection.query(warmup_queries, n_results=k)

        var started = perf_counter_ns()
        for _ in range(loops_per_sample):
            results = collection.query(queries, n_results=k)
        var elapsed = Float64(perf_counter_ns() - started) / 1e9
        measured_seconds += elapsed

        var qps = Float64(query_count * loops_per_sample) / elapsed
        print("sample " + String(sample + 1) + ": " + String(qps) + " QPS")

    var aggregate_qps = Float64(
        query_count * loops_per_sample * samples
    ) / measured_seconds
    var recall = recall_at_k(results, ground_truth, query_count, k)

    print("aggregate: " + String(aggregate_qps) + " QPS")
    print("Recall@10: " + String(recall))
