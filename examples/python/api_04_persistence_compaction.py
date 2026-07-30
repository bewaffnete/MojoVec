"""Atomic persistence, mmap loading, snapshots, and compaction."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import mojovec


def main() -> None:
    with TemporaryDirectory(prefix="mojovec-persistence-") as directory:
        root = Path(directory)
        database_path = root / "articles.mojovec"
        snapshot_path = root / "point-in-time.mojovec"

        collection = mojovec.Collection(
            dimension=4,
            M=16,
            ef_construction=96,
            ef_search=48,
            quantized=False,
            name="persistent_articles",
            metric="cosine",
        )
        collection.add(
            [1, 2, 3],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            metadatas=[
                {"version": 1},
                {"version": 1},
                {"version": 1},
            ],
            documents=["first", "second", "third"],
        )

        # Updates append a replacement internal row and soft-delete the old
        # one. Explicit delete creates another reclaimable row.
        collection.update([1], [[0.9, 0.1, 0.0, 0.0]])
        collection.upsert([2], [[0.1, 0.9, 0.0, 0.0]])
        collection.delete([3])
        print("before save:", collection.stats())

        # save() writes a complete checksummed temporary file, synchronizes it,
        # and atomically publishes it at the destination path.
        collection.save(database_path)

        # Production loads map only sufficiently large files by default. A
        # zero threshold forces mmap for this tiny tutorial fixture.
        mapped = mojovec.load(
            database_path,
            memory_mapped=True,
            mmap_threshold_bytes=0,
        )
        print("mapped:", mapped.is_memory_mapped())
        print("loaded name/metric:", mapped.name(), mapped.metric())
        print("loaded document:", mapped.get_document(1))

        # Read-only queries keep the mapping active.
        result = mapped.query([[1.0, 0.0, 0.0, 0.0]], n_results=2)
        print("mapped query IDs:", result["ids"])
        print("still mapped:", mapped.is_memory_mapped())

        # The first mutation transparently materializes owned writable arrays.
        mapped.add([4], [[0.0, 0.0, 0.0, 1.0]], documents=["fourth"])
        print("mapped after mutation:", mapped.is_memory_mapped())

        # snapshot() publishes and returns an independent point-in-time reader.
        point_in_time = mapped.snapshot(
            snapshot_path,
            memory_mapped=True,
            mmap_threshold_bytes=0,
        )
        mapped.add([5], [[0.5, 0.5, 0.0, 0.0]], documents=["fifth"])
        print("writer count:", mapped.count())
        print("point-in-time count:", point_in_time.count())

        # compact_if_needed() defaults to a 25% deleted-row threshold. Use an
        # explicit zero threshold here to demonstrate a guaranteed rebuild.
        report = mapped.compact_if_needed(deleted_ratio=0.0)
        print("compaction report:", report)
        print("after compaction:", mapped.stats())

        copied = mojovec.Collection.load(
            snapshot_path,
            memory_mapped=False,
        )
        print("forced heap load:", copied.is_memory_mapped())


if __name__ == "__main__":
    main()
