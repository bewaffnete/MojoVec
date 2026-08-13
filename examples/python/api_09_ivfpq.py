"""Train, tune, query, and persist a compressed IVF-PQ collection."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

import mojovec


DIMENSION = 8
COUNT = 256


def make_vectors() -> list[list[float]]:
    return [
        [
            ((row * 17 + column * 29) % 251) / 251.0
            + (2.0 if column == row % DIMENSION else 0.0)
            for column in range(DIMENSION)
        ]
        for row in range(COUNT)
    ]


def main() -> None:
    ids = list(range(50_000, 50_000 + COUNT))
    embeddings = make_vectors()

    collection = mojovec.IVFPQCollection(
        dimension=DIMENSION,
        nlist=8,
        pq_subvectors=2,
        nprobe=4,
        metric="cosine",
        name="compressed-vectors",
    )

    # Explicit training is preferable when a representative sample is
    # available. It needs at least max(nlist, 256) vectors.
    collection.train(embeddings)
    collection.add(ids, embeddings)
    print(collection)
    print("Stats:", collection.stats())

    result = collection.query([embeddings[0], embeddings[17]], n_results=5)
    print("First query IDs:", result["ids"][0])
    print("First query distances:", result["distances"][0])

    # Higher nprobe usually improves recall while scanning more candidates.
    collection.set_nprobe(8)
    tuned = collection.query(embeddings[0], n_results=5)
    print("Tuned IDs:", tuned["ids"][0])

    with TemporaryDirectory(prefix="mojovec-ivfpq-") as directory:
        path = Path(directory) / "vectors.ivfpq"
        collection.save(path)
        loaded = mojovec.IVFPQCollection.load(path)
        print("Loaded:", loaded)


if __name__ == "__main__":
    main()
