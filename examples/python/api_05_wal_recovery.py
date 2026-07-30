"""Optional write-ahead logging and crash recovery from Python."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import mojovec


def main() -> None:
    with TemporaryDirectory(prefix="mojovec-wal-") as directory:
        root = Path(directory)
        snapshot_path = root / "base.mojovec"
        wal_path = root / "mutations.wal"

        collection = mojovec.Collection(
            dimension=3,
            quantized=True,
            name="durable_articles",
        )

        # WAL recovery always starts from a complete atomic snapshot.
        collection.save(snapshot_path)

        # WAL_ASYNC appends complete checksummed frames without fsync on every
        # mutation. flush_wal() creates an explicit group durability boundary.
        collection.enable_wal(wal_path, durability=mojovec.WAL_ASYNC)
        collection.add(
            ids=[101, 202],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            metadatas=[
                {"category": "mojo", "year": 2026},
                {"category": "systems", "year": 2025},
            ],
            documents=[
                "Fast vector search written in Mojo",
                "Durable storage and recovery",
            ],
        )
        print("sequence after one batched add:", collection.wal_sequence())

        collection.update(
            ids=[101],
            embeddings=[[0.9, 0.1, 0.0]],
            metadatas=[{"category": "mojo", "year": 2027}],
            documents=["Updated Mojo vector-search article"],
        )
        collection.delete([202])
        collection.flush_wal()
        print("durable sequence:", collection.wal_sequence())

        # This emulates process shutdown while keeping the tutorial in one
        # process. Collection normally closes its owned WAL handle on drop.
        collection.disable_wal()

        recovered = mojovec.recover(
            snapshot_path,
            wal_path,
            durability=mojovec.WAL_ASYNC,
            memory_mapped=False,
        )
        print("\nrecovered:")
        print("  active records:", recovered.count())
        print("  sequence:", recovered.wal_sequence())
        print("  metadata:", recovered.get_metadata(101))
        print("  document:", recovered.get_document(101))
        print("  WAL resumed:", recovered.wal_enabled())

        result = recovered.query([[0.9, 0.1, 0.0]], n_results=1)
        print("  nearest ID:", result["ids"][0][0])

        # checkpoint() first publishes the full applied state and then rotates
        # WAL frames already covered by that snapshot.
        recovered.checkpoint(snapshot_path)
        recovered.upsert(
            [303],
            [[0.0, 0.0, 1.0]],
            documents=["Mutation after checkpoint"],
        )
        recovered.flush_wal()
        print("sequence after checkpoint and new write:", recovered.wal_sequence())
        recovered.disable_wal()

        # WAL_SYNC is the alternative when each mutation call must perform its
        # own fsync. It lowers the loss window but costs ingestion throughput.
        print("available durability constants:", mojovec.WAL_ASYNC, mojovec.WAL_SYNC)


if __name__ == "__main__":
    main()
