"""
Persisting and restoring Flat and SQ8 collections with MojoVec.

Collection.save() persists:

- collection name and vector dimension;
- whether storage is Flat or SQ8;
- application IDs and soft-delete flags;
- stored vectors and the complete HNSW graph.

Collection.load() inspects the file and reconstructs the correct storage type,
so callers do not need to pass quantized=True/False while loading.

Saved files place vector and HNSW arrays in aligned regions. Large files are
memory-mapped automatically; this small example forces mmap with a zero-byte
threshold so that the behavior is visible without generating a large fixture.
The Collection owns and releases the mapping automatically.

save() publishes through a synchronized temporary file and atomic rename.
The completed snapshot includes a checksum that load() validates before
parsing metadata or exposing memory-mapped arrays.
snapshot() additionally returns an independent point-in-time reader, allowing
one writer to continue changing its in-memory collection while existing
readers keep querying the previous mapped file.

The example writes two files under /tmp to avoid polluting the repository.
"""

from mojovec import Client, Collection, QueryResults
from std.collections import List


def append_vector(
    mut values: List[Float32],
    x0: Float32,
    x1: Float32,
    x2: Float32,
    x3: Float32,
):
    values.append(x0)
    values.append(x1)
    values.append(x2)
    values.append(x3)


def make_ids() -> List[Int]:
    var ids = List[Int]()
    ids.append(10_001)
    ids.append(10_002)
    ids.append(10_003)
    ids.append(10_004)
    ids.append(10_005)
    return ids^


def make_embeddings() -> List[Float32]:
    var embeddings = List[Float32]()
    append_vector(embeddings, 1.0, 0.0, 0.0, 0.0)
    append_vector(embeddings, 0.0, 1.0, 0.0, 0.0)
    append_vector(embeddings, 0.0, 0.0, 1.0, 0.0)
    append_vector(embeddings, 0.0, 0.0, 0.0, 1.0)
    append_vector(embeddings, 0.8, 0.2, 0.0, 0.0)
    return embeddings^


def make_query() -> List[Float32]:
    var query = List[Float32]()
    append_vector(query, 0.9, 0.1, 0.0, 0.0)
    return query^


def print_results(title: String, results: QueryResults):
    print(title)
    for rank in range(len(results.ids[0])):
        print(
            "  rank",
            rank + 1,
            "| id =",
            results.ids[0][rank],
            "| squared L2 distance =",
            results.distances[0][rank],
        )


def round_trip(
    label: String,
    mut collection: Collection,
    path: String,
    query: List[Float32],
) raises:
    """Saves, loads, inspects, and queries one collection."""
    print("\n============================================================")
    print(label)
    print("============================================================")
    print("Before save:")
    print("  name:", collection.name())
    print("  dimension:", collection.dimension())
    print("  quantized:", collection.is_quantized())
    print("  active records:", collection.count())
    print("  deleted/internal historical records:", collection.count_deleted())

    var before = collection.query(query, n_results=3)
    print_results("Results before save:", before)

    print("Saving to:", path)
    collection.save(path)

    # load() is a static factory. The file header determines whether MojoVec
    # reconstructs IndexHNSW[IndexFlat] or IndexHNSW[IndexFlatSQ8].
    #
    # Production code can normally omit mmap_threshold_bytes. The default
    # maps files >= 64 MiB and copies smaller files to owned heap memory.
    var loaded = Collection.load(path, mmap_threshold_bytes=0)
    print("Loaded collection:")
    print("  name:", loaded.name())
    print("  dimension:", loaded.dimension())
    print("  quantized:", loaded.is_quantized())
    print("  memory mapped:", loaded.is_memory_mapped())
    print("  active records:", loaded.count())
    print("  deleted/internal historical records:", loaded.count_deleted())

    # ef_search is scalar runtime state, so tuning it does not detach the
    # read-only mapped arrays.
    loaded.set_ef_search(64)
    print("  mapped after set_ef_search:", loaded.is_memory_mapped())
    var after = loaded.query(query, n_results=3)
    print_results("Results after load:", after)

    # add/update/upsert/compaction need writable graph storage. On the first
    # such operation MojoVec copies the mapped arrays to owned memory
    # automatically. No mmap handles, pointers, or free calls reach the API.

    # snapshot() combines atomic save and load. Existing snapshots remain
    # pinned to their complete point-in-time state when this path is published
    # again, making them suitable for concurrent read traffic.
    var snapshot = collection.snapshot(path, mmap_threshold_bytes=0)
    print("Snapshot reader:")
    print("  memory mapped:", snapshot.is_memory_mapped())
    print("  active records:", snapshot.count())
    var snapshot_results = snapshot.query(query, n_results=3)
    print_results("Results from point-in-time snapshot:", snapshot_results)

    # For deterministic data, returned IDs should match before and after the
    # round trip. Floating-point distances should be compared with a tolerance,
    # especially for SQ8 storage.


def populate_and_mutate(
    client: Client,
    name: String,
    quantized: Bool,
    ids: List[Int],
    embeddings: List[Float32],
) raises -> Collection:
    """Builds state that exercises ID and deletion serialization."""
    var collection = client.create_collection(
        name,
        dimension=4,
        M=16,
        ef_construction=100,
        ef_search=32,
        quantized=quantized,
    )
    collection.add(ids, embeddings)

    # update() appends a replacement internally and soft-deletes the previous
    # version. Only the replacement remains visible under ID 10_003.
    var update_ids = List[Int]()
    update_ids.append(10_003)
    var update_embeddings = List[Float32]()
    append_vector(update_embeddings, 0.0, 0.1, 0.9, 0.0)
    collection.update(update_ids, update_embeddings)

    # Explicitly delete another application ID. Both the replacement mapping
    # and all deletion flags must survive serialization.
    var delete_ids = List[Int]()
    delete_ids.append(10_004)
    collection.delete(delete_ids)

    return collection^


def main() raises:
    var client = Client()
    var ids = make_ids()
    var embeddings = make_embeddings()
    var query = make_query()

    var sq8 = populate_and_mutate(
        client,
        "persistent_sq8",
        True,
        ids,
        embeddings,
    )
    round_trip(
        "SQ8 collection round trip",
        sq8,
        "/tmp/mojovec_example_sq8.bin",
        query,
    )

    var flat = populate_and_mutate(
        client,
        "persistent_flat",
        False,
        ids,
        embeddings,
    )
    round_trip(
        "Flat collection round trip",
        flat,
        "/tmp/mojovec_example_flat.bin",
        query,
    )

    print("\nExample files:")
    print("  /tmp/mojovec_example_sq8.bin")
    print("  /tmp/mojovec_example_flat.bin")
