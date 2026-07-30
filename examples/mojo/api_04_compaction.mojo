"""
Inspecting and compacting a MojoVec HNSW collection.

MojoVec implements update, upsert, and delete through soft deletion:

- an updated vector is appended as a new internal row;
- the previous internal row remains in storage but cannot be returned;
- an explicitly deleted record also remains in storage until compaction.

This keeps normal mutations simple and search-safe, but historical rows consume
memory and graph traversal work. Collection.stats() exposes that accumulation.
Collection.compact() rebuilds immediately when garbage exists, while
compact_if_needed() rebuilds only after a configured deleted-row ratio.

The replacement graph is fully constructed before it is installed. If the
rebuild fails, the original collection remains unchanged.
"""

from std.collections import List

from mojovec import Client, CollectionStats, CompactReport


comptime DIMENSION = 4


def append_vector(
    mut embeddings: List[Float32],
    x0: Float32,
    x1: Float32,
    x2: Float32,
    x3: Float32,
):
    embeddings.append(x0)
    embeddings.append(x1)
    embeddings.append(x2)
    embeddings.append(x3)


def print_stats(title: String, stats: CollectionStats):
    print("\n" + title)
    print("  active records:", stats.active_count)
    print("  deleted historical records:", stats.deleted_count)
    print("  total stored rows:", stats.total_count)
    print("  deleted ratio:", stats.deleted_ratio)
    print("  dimension:", stats.dimension)
    print("  quantized:", stats.quantized)
    print("  M:", stats.M)
    print("  ef_construction:", stats.ef_construction)
    print("  ef_search:", stats.ef_search)


def print_report(report: CompactReport):
    print("\nCompaction report")
    print("  performed:", report.performed)
    print("  reclaimed rows:", report.reclaimed_records)
    print("  elapsed seconds:", report.elapsed_seconds)
    print("  stored rows before:", report.before.total_count)
    print("  stored rows after:", report.after.total_count)


def main() raises:
    var client = Client()
    var collection = client.create_collection(
        "documents",
        dimension=DIMENSION,
        M=16,
        ef_construction=100,
        ef_search=32,
        quantized=True,
    )

    var ids = List[Int]()
    var embeddings = List[Float32]()
    for record in range(8):
        ids.append(1_000 + record)
        append_vector(
            embeddings,
            Float32(record),
            Float32(record + 1),
            Float32(record + 2),
            Float32(record + 3),
        )
    collection.add(ids, embeddings)
    print_stats("After initial add", collection.stats())

    # Replacing three IDs creates three deleted historical rows. The active
    # record count stays at eight while total stored rows grows to eleven.
    var update_ids = List[Int]()
    update_ids.append(1_001)
    update_ids.append(1_003)
    update_ids.append(1_005)
    var updated_embeddings = List[Float32]()
    append_vector(updated_embeddings, 101.0, 102.0, 103.0, 104.0)
    append_vector(updated_embeddings, 103.0, 104.0, 105.0, 106.0)
    append_vector(updated_embeddings, 105.0, 106.0, 107.0, 108.0)
    collection.update(update_ids, updated_embeddings)

    # Deleting one more active ID leaves seven visible records and four
    # deleted rows out of eleven stored rows.
    collection.delete([1_007])
    var before = collection.stats()
    print_stats("Before compaction", before)

    var query = List[Float32]()
    append_vector(query, 101.0, 102.0, 103.0, 104.0)
    var before_result = collection.query(query, n_results=1)
    print("\nNearest ID before compaction:", before_result.ids[0][0])

    # 4 / 11 is above 25%, so this call rebuilds the graph. With a threshold
    # above the current ratio it would return performed=False without work.
    var report = collection.compact_if_needed(deleted_ratio=0.25)
    print_report(report)
    print_stats("After compaction", collection.stats())

    var after_result = collection.query(query, n_results=1)
    print("\nNearest ID after compaction:", after_result.ids[0][0])

    # There is no garbage left, so an immediate second request is a no-op.
    var no_op = collection.compact_if_needed(deleted_ratio=0.25)
    print("Second compaction performed:", no_op.performed)
