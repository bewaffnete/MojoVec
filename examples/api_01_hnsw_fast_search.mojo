"""
Complete HNSW collection lifecycle with the high-level MojoVec API.

This example is the recommended starting point for the high-level MojoVec API.
It demonstrates:

1. How embeddings are represented as one flattened List[Float32].
2. How to create an SQ8-quantized or exact Flat HNSW collection.
3. The difference between add(), upsert(), update(), and delete().
4. How to query several embeddings with the convenient query() API.
5. How ef_search changes the search/recall trade-off.

All collections in this example use squared L2 distance. Smaller distances are
better, and a distance of 0.0 means that the two vectors are identical.

The public API uses managed Lists and QueryResults. User code never allocates
output buffers and never calls free().
"""

from mojovec import Client, QueryResults
from std.collections import List


def append_vector(
    mut values: List[Float32],
    x0: Float32,
    x1: Float32,
    x2: Float32,
    x3: Float32,
):
    """Appends one four-dimensional vector to a flattened embedding list."""
    values.append(x0)
    values.append(x1)
    values.append(x2)
    values.append(x3)


def print_results(title: String, results: QueryResults):
    """Prints every query row returned by Collection.query()."""
    print("\n" + title)
    for query_index in range(len(results.ids)):
        print("  Query", query_index)
        for rank in range(len(results.ids[query_index])):
            print(
                "    rank",
                rank + 1,
                "| id =",
                results.ids[query_index][rank],
                "| squared L2 distance =",
                results.distances[query_index][rank],
            )


def main() raises:
    # ---------------------------------------------------------------------
    # 1. Prepare a small, readable dataset.
    # ---------------------------------------------------------------------
    # MojoVec accepts a flattened row-major list:
    #
    #   [vector_0_dim_0, vector_0_dim_1, ...,
    #    vector_1_dim_0, vector_1_dim_1, ...]
    #
    # Five vectors with dimension=4 therefore require exactly 5 * 4 values.
    comptime dimension = 4

    var ids = List[Int]()
    ids.append(101)
    ids.append(202)
    ids.append(303)
    ids.append(404)
    ids.append(505)

    var embeddings = List[Float32]()
    append_vector(embeddings, 1.0, 0.0, 0.0, 0.0)  # ID 101
    append_vector(embeddings, 0.0, 1.0, 0.0, 0.0)  # ID 202
    append_vector(embeddings, 0.0, 0.0, 1.0, 0.0)  # ID 303
    append_vector(embeddings, 0.0, 0.0, 0.0, 1.0)  # ID 404
    append_vector(embeddings, 0.9, 0.1, 0.0, 0.0)  # ID 505

    # ---------------------------------------------------------------------
    # 2. Create the recommended SQ8 HNSW collection.
    # ---------------------------------------------------------------------
    var client = Client()
    var sq8 = client.create_collection(
        "documents_sq8",
        dimension=dimension,
        M=16,
        ef_construction=100,
        ef_search=32,
        quantized=True,
    )

    # Parameter overview:
    #
    # - dimension:
    #     Number of Float32 components in every vector.
    # - M:
    #     Approximate number of graph connections per node. Larger values
    #     usually improve recall but consume more memory and build time.
    # - ef_construction:
    #     Candidate-list size used while building the graph. Larger values
    #     usually improve graph quality and increase construction time.
    # - ef_search:
    #     Candidate-list size used during queries. It can be changed later.
    # - quantized=True:
    #     Store/search vectors with SQ8 acceleration. Use False for Flat.

    print("Created collection:", sq8.name())
    print("  dimension:", sq8.dimension())
    print("  quantized:", sq8.is_quantized())
    print("  active records before add:", sq8.count())

    # add() only accepts new IDs. It raises if an ID already exists or appears
    # twice in the same batch.
    sq8.add(ids, embeddings)
    print("  active records after add:", sq8.count())
    print("  deleted/internal historical records:", sq8.count_deleted())

    # ---------------------------------------------------------------------
    # 3. Query multiple embeddings in one call.
    # ---------------------------------------------------------------------
    var queries = List[Float32]()
    append_vector(queries, 1.0, 0.0, 0.0, 0.0)
    append_vector(queries, 0.0, 0.0, 0.8, 0.2)

    # With two queries and n_results=3, results.ids and results.distances both
    # contain two rows with three values per row.
    var initial_results = sq8.query(queries, n_results=3)
    print_results("Initial SQ8 results", initial_results)

    # ef_search is a runtime latency/recall knob. It must be positive and no
    # greater than 2048. A useful production rule is ef_search >= n_results.
    sq8.set_ef_search(64)

    # ---------------------------------------------------------------------
    # 4. Modify the collection using Chroma-style operations.
    # ---------------------------------------------------------------------
    # upsert() inserts missing IDs and replaces existing IDs.
    var upsert_ids = List[Int]()
    upsert_ids.append(202)  # Existing: replace its vector.
    upsert_ids.append(606)  # Missing: insert a new record.

    var upsert_embeddings = List[Float32]()
    append_vector(upsert_embeddings, 0.75, 0.25, 0.0, 0.0)  # ID 202
    append_vector(upsert_embeddings, 0.0, 0.1, 0.9, 0.0)  # ID 606
    sq8.upsert(upsert_ids, upsert_embeddings)

    # update() is stricter: every ID must already exist.
    var update_ids = List[Int]()
    update_ids.append(303)
    var update_embeddings = List[Float32]()
    append_vector(update_embeddings, 0.0, 0.0, 0.95, 0.05)
    sq8.update(update_ids, update_embeddings)

    # delete() is a soft delete. Unknown IDs are ignored. Deleted nodes remain
    # in the HNSW graph but are filtered out of returned results.
    var delete_ids = List[Int]()
    delete_ids.append(404)
    delete_ids.append(999_999)  # This ID does not exist; delete is idempotent.
    sq8.delete(delete_ids)

    print("\nAfter upsert, update, and delete:")
    print("  active records:", sq8.count())
    print("  deleted/internal historical records:", sq8.count_deleted())

    var modified_results = sq8.query(queries, n_results=3)
    print_results("SQ8 results after mutations", modified_results)

    # ---------------------------------------------------------------------
    # 5. Create an exact Flat collection through the same API.
    # ---------------------------------------------------------------------
    # quantized=False changes only the vector storage/distance implementation.
    # add(), query(), update(), delete(), save(), and load() remain identical.
    var flat = client.create_collection(
        "documents_flat",
        dimension=dimension,
        M=16,
        ef_construction=100,
        ef_search=64,
        quantized=False,
    )
    flat.add(ids, embeddings)

    print("\nCreated collection:", flat.name())
    print("  quantized:", flat.is_quantized())
    print("  active records:", flat.count())

    var flat_results = flat.query(queries, n_results=3)
    print_results("Exact Flat results", flat_results)
