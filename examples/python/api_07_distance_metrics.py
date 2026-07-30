"""Compare squared-L2, cosine, and inner-product collection metrics."""

from __future__ import annotations

import mojovec


def show(metric: str, quantized: bool) -> None:
    collection = mojovec.Collection(
        dimension=2,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=quantized,
        name=f"{metric}_{'sq8' if quantized else 'flat'}",
        metric=metric,
    )
    collection.add(
        [10, 20, 30],
        [
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, 3.0],
        ],
    )
    result = collection.query([[1.0, 0.0]], n_results=3)
    print(f"\nmetric={metric}, storage={collection.storage_kind()}")
    for rank, record_id in enumerate(result["ids"][0]):
        print(
            f"  rank {rank + 1}: id={record_id}, "
            f"distance={result['distances'][0][rank]:.6f}"
        )


def main() -> None:
    # All returned distances are smaller-is-better:
    #
    # l2     -> squared Euclidean distance
    # cosine -> 1 - cosine similarity
    # ip     -> 1 - inner product
    #
    # Cosine normalizes stored vectors and queries automatically. IP distances
    # can be negative when the dot product is greater than one.
    show("l2", quantized=False)
    show("cosine", quantized=False)
    show("ip", quantized=True)


if __name__ == "__main__":
    main()
