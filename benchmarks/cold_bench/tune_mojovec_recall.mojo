from std.collections import List
from std.memory.alloc import unsafe_alloc
from std.collections.span import Span
from std.time import perf_counter_ns
from mojovec import Collection


def load_bytes(path: String) raises -> List[UInt8]:
    var file = open(path, "r")
    var data = file.read_bytes()
    file.close()
    return data^


def run_sweep[
    query_origin: Origin,
    ground_truth_origin: Origin,
](
    path: String,
    name: String,
    queries: Span[Float32, query_origin],
    ground_truth: Pointer[Int32, ground_truth_origin],
) raises:
    comptime query_count = 10_000
    comptime k = 10

    print("\n" + name)
    var collection = Collection.unsafe_load(path)
    var distance_storage = unsafe_alloc[Float32](query_count * k)
    var label_storage = unsafe_alloc[Int](query_count * k)
    var distances = Span[mut=True, Float32](
        unsafe_ptr=distance_storage, length=query_count * k
    )
    var labels = Span[mut=True, Int](
        unsafe_ptr=label_storage, length=query_count * k
    )
    var ef_values = List[Int]()
    ef_values.append(48)
    ef_values.append(64)
    ef_values.append(80)
    ef_values.append(96)
    ef_values.append(112)
    ef_values.append(128)
    ef_values.append(160)
    ef_values.append(192)
    ef_values.append(256)

    for ef_index in range(len(ef_values)):
        var ef = ef_values[ef_index]
        collection.set_ef_search(ef)
        var started = perf_counter_ns()
        collection._query_into(queries, k, labels, distances)
        var elapsed = Float64(perf_counter_ns() - started) / 1e9

        var hits = 0
        for query_id in range(query_count):
            for result_index in range(k):
                var result_id = label_storage[query_id * k + result_index]
                for truth_index in range(k):
                    if result_id == Int(
                        ground_truth[query_id * 101 + 1 + truth_index]
                    ):
                        hits += 1
                        break

        var recall = Float64(hits) / Float64(query_count * k)
        var qps = Float64(query_count) / elapsed
        print(
            "efSearch="
            + String(ef)
            + " | QPS="
            + String(qps)
            + " | Recall@10="
            + String(recall)
        )

    distance_storage.unsafe_free()
    label_storage.unsafe_free()


def main() raises:
    comptime dimension = 128
    comptime query_count = 10_000

    var query_data = load_bytes(
        "benchmarks/data/sift1m/sift_query.fvecs"
    )
    var ground_truth_data = load_bytes(
        "benchmarks/data/sift1m/sift_groundtruth.ivecs"
    )
    var source = query_data.unsafe_ptr().unsafe_bitcast[Float32]()
    var queries = List[Float32](capacity=query_count * dimension)
    for i in range(query_count):
        var offset = i * (dimension + 1) + 1
        for j in range(dimension):
            queries.append(source[offset + j])

    var query_span = Span[Float32](
        unsafe_ptr=queries.unsafe_ptr(), length=len(queries)
    )
    var ground_truth = ground_truth_data.unsafe_ptr().unsafe_bitcast[Int32]()
    run_sweep(
        "benchmarks/cold_bench/mojovec_hnsw_flat_1m.mojovec",
        "MojoVec Flat",
        query_span,
        ground_truth,
    )
    run_sweep(
        "benchmarks/cold_bench/mojovec_hnsw_sq8_1m.mojovec",
        "MojoVec SQ8",
        query_span,
        ground_truth,
    )
