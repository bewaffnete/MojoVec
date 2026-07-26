"""
Training and querying an IVF-PQ collection with the MojoVec API.

IVF-PQ targets datasets where memory usage matters more than exact distances.
It combines:

- IVF (inverted file): searches only selected coarse clusters.
- PQ (product quantization): stores compact codes instead of full vectors.

Unlike HNSW Flat/SQ8 collections, an IVF-PQ index must be trained before it can
encode vectors. CollectionIVFPQ supports both explicit training and automatic
training on the first add().

This example uses deterministic synthetic data so it produces repeatable output.
"""

from mojovec import Client, QueryResults
from std.collections import List


def make_ids(count: Int, first_id: Int) -> List[Int]:
    """Creates stable application IDs unrelated to internal vector positions."""
    var ids = List[Int](capacity=count)
    for i in range(count):
        ids.append(first_id + i)
    return ids^


def make_clustered_embeddings(
    count: Int, dimension: Int, num_clusters: Int
) -> List[Float32]:
    """
    Creates flattened vectors grouped around deterministic cluster centers.

    Each row has one dominant component. A small deterministic jitter prevents
    every vector in a cluster from being exactly identical.
    """
    var embeddings = List[Float32](capacity=count * dimension)
    for row in range(count):
        var cluster = row % num_clusters
        for column in range(dimension):
            var center: Float32 = 0.0
            if column == cluster % dimension:
                center = 1.0
            var jitter = Float32((row * 17 + column * 13) % 23) / 1000.0
            embeddings.append(center + jitter)
    return embeddings^


def make_queries(dimension: Int) -> List[Float32]:
    """Creates two queries near cluster 0 and cluster 3."""
    var queries = List[Float32](capacity=2 * dimension)
    for column in range(dimension):
        var value: Float32 = 0.0
        if column == 0:
            value = 1.0
        queries.append(value)
    for column in range(dimension):
        var value: Float32 = 0.0
        if column == 3:
            value = 1.0
        queries.append(value)
    return queries^


def print_results(title: String, results: QueryResults):
    print("\n" + title)
    for query_index in range(len(results.ids)):
        print("  Query", query_index)
        for rank in range(len(results.ids[query_index])):
            print(
                "    rank",
                rank + 1,
                "| id =",
                results.ids[query_index][rank],
                "| approximate squared L2 distance =",
                results.distances[query_index][rank],
            )


def main() raises:
    comptime dimension = 8
    comptime num_vectors = 512
    comptime nlist = 8
    comptime pq_subvectors = 2

    # Important parameter constraints:
    #
    # - Every vector must contain exactly `dimension` Float32 values.
    # - `nlist` is the number of coarse IVF clusters. Larger values create
    #   smaller posting lists but require enough representative training data.
    # - `pq_subvectors` is called M in the API. dimension must be divisible by M.
    #   Here, dimension=8 and M=2, so each PQ subvector has four components.
    # - M in IVF-PQ is unrelated to the HNSW graph parameter also named M.
    var ids = make_ids(num_vectors, first_id=50_000)
    var embeddings = make_clustered_embeddings(
        num_vectors, dimension, num_clusters=nlist
    )
    var queries = make_queries(dimension)

    var client = Client()
    var collection = client.create_ivfpq_collection(
        "compressed_documents",
        dimension=dimension,
        nlist=nlist,
        M=pq_subvectors,
    )

    # ---------------------------------------------------------------------
    # Explicit training workflow.
    # ---------------------------------------------------------------------
    # In production, train on a representative sample of the same embedding
    # distribution that will later be indexed. Training data controls both the
    # coarse IVF centroids and the PQ codebooks, so poor samples reduce recall.
    print("Training IVF-PQ on", num_vectors, "representative vectors...")
    collection.train(embeddings)

    # After explicit training, add() only encodes and inserts the rows.
    # Application IDs are stored separately and returned by query().
    print("Adding", num_vectors, "vectors...")
    collection.add(ids, embeddings)

    comptime k = 5
    var results = collection.query(queries, n_results=k)
    print_results("Explicitly trained IVF-PQ results", results)

    # ---------------------------------------------------------------------
    # Automatic training workflow.
    # ---------------------------------------------------------------------
    # If train() is omitted, the first add() trains on that same batch before
    # inserting it. This is convenient for experiments, but explicit training
    # is preferable when the first ingestion batch is small or unrepresentative.
    var auto_collection = client.create_ivfpq_collection(
        "auto_trained_documents",
        dimension=dimension,
        nlist=nlist,
        M=pq_subvectors,
    )

    print("\nAuto-training and adding through the first add() call...")
    auto_collection.add(ids, embeddings)
    var auto_results = auto_collection.query(queries, n_results=k)
    print_results("Automatically trained IVF-PQ results", auto_results)

    # The distances above are approximate. Evaluate IVF-PQ with recall@k on
    # your own dataset; do not compare approximate distances for exact equality.
