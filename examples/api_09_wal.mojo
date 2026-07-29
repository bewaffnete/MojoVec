"""
Optional, batched write-ahead logging with MojoVec.

WAL is useful when MojoVec itself must recover mutations made after the latest
snapshot. It remains disabled by default, so applications backed by another
source of truth keep the smallest possible ingestion path.

The two durability policies are:

- WAL_ASYNC: append one committed frame per public mutation batch, then let the
  application group several calls behind one flush_wal() durability barrier;
- WAL_SYNC: perform one fsync after every complete add/upsert/update/delete
  call, reducing the loss window at the cost of ingestion throughput.

Queries never read or synchronize the WAL. Collection owns the open file and
all recovery memory; callers do not manage file handles or free buffers.
"""

from std.collections import List
from std.time import perf_counter_ns

from mojovec import (
    Collection,
    Metadata,
    WAL_ASYNC,
    WAL_SYNC,
)


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


def metadata(category: String, year: Int) -> Metadata:
    var value = Metadata()
    value.set("category", category)
    value.set("year", year)
    value.set("published", True)
    return value^


def main() raises:
    # A unique /tmp fixture keeps this example safe to run repeatedly. A real
    # application normally keeps stable snapshot and WAL paths.
    var run_id = perf_counter_ns()
    var snapshot_path = String("/tmp/mojovec_wal_example_", run_id, ".bin")
    var wal_path = String("/tmp/mojovec_wal_example_", run_id, ".wal")

    var collection = Collection(
        dimension=4,
        M=16,
        ef_construction=100,
        ef_search=32,
        quantized=True,
        name="durable_articles",
    )

    # Recovery always begins with a complete atomic snapshot. enable_wal()
    # refuses to overwrite a non-empty log, preventing accidental data loss.
    collection.save(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_ASYNC)
    print("WAL enabled:", collection.wal_enabled())

    # The flattened embedding layout is:
    # [record_0 dimensions..., record_1 dimensions...].
    var embeddings = List[Float32]()
    append_vector(embeddings, 1.0, 0.0, 0.0, 0.0)
    append_vector(embeddings, 0.0, 1.0, 0.0, 0.0)

    var metadatas = List[Metadata]()
    metadatas.append(metadata("mojo", 2026))
    metadatas.append(metadata("systems", 2025))

    # This whole two-record add is one WAL frame and one sequence number.
    collection.add(
        [101, 102],
        embeddings,
        metadatas,
        [
            String("Fast vector search written in Mojo"),
            String("Durable storage and recovery"),
        ],
    )
    print("sequence after batched add:", collection.wal_sequence())

    # Updates/upserts preserve the same public API. Each complete call advances
    # the sequence once, regardless of the number or dimension of its vectors.
    var replacement_metadata = List[Metadata]()
    replacement_metadata.append(metadata("mojo", 2027))
    collection.update(
        [101],
        [Float32(0.9), Float32(0.1), Float32(0.0), Float32(0.0)],
        replacement_metadata,
        [String("Updated Mojo vector-search article")],
    )
    collection.delete([102])
    print("sequence after update and delete:", collection.wal_sequence())

    # WAL_ASYNC deliberately avoids fsync on each mutation. This single call
    # makes every preceding committed frame durable as one group.
    collection.flush_wal()

    # A process restart closes the owned handle automatically. disable_wal()
    # is used only to emulate that boundary inside this one program.
    collection.disable_wal()

    # recover() atomically loads the snapshot, replays only complete frames,
    # validates checksums and sequence continuity, and resumes the same WAL.
    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        durability=WAL_ASYNC,
        memory_mapped=False,
    )
    print("\nRecovered collection:")
    print("  active records:", recovered.count())
    print("  latest sequence:", recovered.wal_sequence())
    print("  document 101:", recovered.get_document(101))
    print(
        "  metadata year:",
        recovered.get_metadata(101).get_int("year"),
    )

    var results = recovered.query(
        [Float32(0.9), Float32(0.1), Float32(0.0), Float32(0.0)],
        n_results=1,
    )
    print("  nearest ID:", results.ids[0][0])

    # checkpoint() first publishes and synchronizes the complete collection,
    # then atomically rotates WAL frames already covered by that snapshot.
    recovered.checkpoint(snapshot_path)
    print("Checkpoint published; WAL remains enabled for later writes.")

    # Choose WAL_SYNC instead when every public mutation call must cross its
    # own fsync durability boundary:
    #
    # collection.enable_wal(wal_path, durability=WAL_SYNC)
    #
    # A non-empty existing WAL must be opened with Collection.recover(), never
    # silently replaced through enable_wal().
    _ = WAL_SYNC
