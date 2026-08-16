from std.collections import List
from mojovec import QueryResults


def _read_bytes(path: String) raises -> List[UInt8]:
    """Reads a dataset file; binary decoding stays outside API benchmarks."""
    var file = open(path, "r")
    var data = file.read_bytes()
    file.close()
    return data^


def load_fvecs(
    path: String,
    vector_count: Int,
    dimension: Int,
) raises -> List[Float32]:
    """Loads flattened vectors from the SIFT1M .fvecs format."""
    var data = _read_bytes(path)
    var source = data.unsafe_ptr().unsafe_bitcast[Float32]()
    var vectors = List[Float32](capacity=vector_count * dimension)

    for vector_index in range(vector_count):
        var source_offset = vector_index * (dimension + 1) + 1
        for component in range(dimension):
            vectors.append(source[unsafe_offset=source_offset + component])

    _ = len(data)
    return vectors^


def load_ground_truth(
    path: String,
    query_count: Int,
    k: Int,
) raises -> List[Int]:
    """Loads the first k neighbors per query from SIFT1M .ivecs."""
    comptime stored_neighbors = 100
    var data = _read_bytes(path)
    var source = data.unsafe_ptr().unsafe_bitcast[Int32]()
    var ground_truth = List[Int](capacity=query_count * k)

    for query_index in range(query_count):
        var source_offset = query_index * (stored_neighbors + 1) + 1
        for neighbor in range(k):
            ground_truth.append(Int(source[unsafe_offset=source_offset + neighbor]))

    _ = len(data)
    return ground_truth^


def first_vectors(
    vectors: List[Float32],
    vector_count: Int,
    dimension: Int,
) -> List[Float32]:
    """Copies a small prefix for an untimed API warm-up."""
    var prefix = List[Float32](capacity=vector_count * dimension)
    for index in range(vector_count * dimension):
        prefix.append(vectors[index])
    return prefix^


def recall_at_k(
    results: QueryResults,
    ground_truth: List[Int],
    query_count: Int,
    k: Int,
) -> Float64:
    """Computes set-intersection recall for public QueryResults."""
    var recall_sum: Float64 = 0.0
    for query_index in range(query_count):
        var hits = 0
        for result_index in range(k):
            var result_id = results.ids[query_index][result_index]
            for truth_index in range(k):
                if result_id == ground_truth[query_index * k + truth_index]:
                    hits += 1
                    break
        recall_sum += Float64(hits) / Float64(k)
    return recall_sum / Float64(query_count)
