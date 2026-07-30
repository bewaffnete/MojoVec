"""
Cosine, inner-product, and squared-L2 search with the managed Collection API.

The `metric` argument is independent of vector storage:

- `quantized=False` selects Flat Float32 HNSW;
- `quantized=True` selects SQ8 HNSW with exact Float32 reranking;
- `metric="l2"`, `"cosine"`, or `"ip"` selects neighbor ordering.

All public `QueryResults.distances` values are smaller-is-better:

- l2: squared Euclidean distance;
- cosine: 1 - cosine similarity;
- ip: 1 - inner product.

Cosine input vectors and queries are normalized automatically. MojoVec creates
owned normalized index rows; it never modifies the caller's managed Lists.
Cosine rejects zero and non-finite vectors because their direction is undefined.
"""

from std.collections import List

from mojovec import Client, QueryResults


def print_results(title: String, results: QueryResults):
    print("\n" + title)
    for rank in range(len(results.ids[0])):
        print(
            "  rank",
            rank + 1,
            "| id =",
            results.ids[0][rank],
            "| distance =",
            results.distances[0][rank],
        )


def main() raises:
    var client = Client()

    # ------------------------------------------------------------------
    # Cosine distance: direction matters; vector magnitude does not.
    # ------------------------------------------------------------------
    var cosine = client.create_collection(
        "cosine_demo",
        dimension=2,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=False,
        metric="cosine",
    )
    var cosine_embeddings = [
        Float32(2.0), Float32(0.0),  # ID 10: same direction as query
        Float32(1.0), Float32(1.0),  # ID 20: 45 degrees from query
        Float32(0.0), Float32(3.0),  # ID 30: orthogonal to query
    ]
    cosine.add([10, 20, 30], cosine_embeddings)

    # [100, 0] and [1, 0] produce identical cosine rankings and distances.
    # The original stored input remains [2, 0, 1, 1, 0, 3].
    var cosine_results = cosine.query(
        [Float32(100.0), Float32(0.0)],
        n_results=3,
    )
    print("collection metric:", cosine.metric())
    print_results("Cosine distance (1 - cosine similarity)", cosine_results)

    # ------------------------------------------------------------------
    # Inner product: both direction and magnitude affect the result.
    # ------------------------------------------------------------------
    var inner_product = client.create_collection(
        "ip_demo",
        dimension=2,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=True,
        metric="ip",
    )
    inner_product.add(
        [10, 20, 30],
        [
            Float32(1.0), Float32(0.0),  # dot([1, 0]) = 1
            Float32(1.0), Float32(1.0),  # dot([1, 0]) = 1
            Float32(2.0), Float32(2.0),  # dot([1, 0]) = 2
        ],
    )
    var ip_results = inner_product.query(
        [Float32(1.0), Float32(0.0)],
        n_results=3,
    )
    print_results("Inner-product distance (1 - dot product)", ip_results)

    # IP distances may be negative: a dot product of 2 produces distance -1.
    # This is expected. Smaller distance still means a better match.

    # ------------------------------------------------------------------
    # L2 remains the default and does not normalize vectors.
    # ------------------------------------------------------------------
    var l2 = client.create_collection(
        "l2_demo",
        dimension=2,
        quantized=False,
        metric="l2",
    )
    l2.add(
        [10, 20],
        [
            Float32(1.0), Float32(0.0),
            Float32(2.0), Float32(0.0),
        ],
    )
    var l2_results = l2.query(
        [Float32(1.0), Float32(0.0)],
        n_results=2,
    )
    print_results("Squared L2 distance", l2_results)

    # save/load, mmap loading, compaction, filters, WAL recovery, and hybrid
    # vector ranking preserve the collection's selected metric automatically.
