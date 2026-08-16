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


def main() raises:
    comptime dimension = 128
    comptime query_count = 10_000
    comptime output_capacity = 128
    comptime k = 10

    var collection = Collection.unsafe_load(
        "benchmarks/cold_bench/mojovec_hnsw_sq8_1m.mojovec"
    )
    var query_data = load_bytes(
        "benchmarks/data/sift1m/sift_query.fvecs"
    )
    var ground_truth_data = load_bytes(
        "benchmarks/data/sift1m/sift_groundtruth.ivecs"
    )
    var source = query_data.unsafe_ptr().unsafe_bitcast[Float32]()
    var ground_truth = ground_truth_data.unsafe_ptr().unsafe_bitcast[Int32]()
    var queries = List[Float32](capacity=query_count * dimension)
    for i in range(query_count):
        var offset = i * (dimension + 1) + 1
        for j in range(dimension):
            queries.append(source[offset + j])
    var query_span = Span[Float32](
        unsafe_ptr=queries.unsafe_ptr(), length=len(queries)
    )

    var distance_storage = unsafe_alloc[Float32](query_count * output_capacity)
    var label_storage = unsafe_alloc[Int](query_count * output_capacity)
    var ef_values = List[Int]()
    ef_values.append(96)
    var rerank_values = List[Int]()
    rerank_values.append(20)
    rerank_values.append(30)
    rerank_values.append(40)
    rerank_values.append(60)
    rerank_values.append(80)
    rerank_values.append(96)

    for ef_index in range(len(ef_values)):
        var ef = ef_values[ef_index]
        collection.set_ef_search(ef)
        for rerank_index in range(len(rerank_values)):
            var rerank_k = rerank_values[rerank_index]
            if rerank_k > ef:
                continue
            var distances = Span[mut=True, Float32](
                unsafe_ptr=distance_storage, length=query_count * rerank_k
            )
            var labels = Span[mut=True, Int](
                unsafe_ptr=label_storage, length=query_count * rerank_k
            )
            var started = perf_counter_ns()
            collection._query_into(
                query_span, rerank_k, labels, distances
            )
            var elapsed = Float64(perf_counter_ns() - started) / 1e9

            var hits = 0
            for query_id in range(query_count):
                for result_index in range(k):
                    var result_id = label_storage[
                        query_id * rerank_k + result_index
                    ]
                    for truth_index in range(k):
                        if result_id == Int(
                            ground_truth[
                                query_id * 101 + 1 + truth_index
                            ]
                        ):
                            hits += 1
                            break

            var recall = Float64(hits) / Float64(query_count * k)
            var qps = Float64(query_count) / elapsed
            print(
                "efSearch="
                + String(ef)
                + " | rerank="
                + String(rerank_k)
                + " | QPS="
                + String(qps)
                + " | Recall@10="
                + String(recall)
            )

    distance_storage.unsafe_free()
    label_storage.unsafe_free()
