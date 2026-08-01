"""Complete managed Collection lifecycle from Python.

Run from the repository root after installing the built wheel:

    python examples/python/api_01_collection_crud.py

Python accepts both nested vectors and the flattened row-major representation
used by Mojo. Nested vectors are usually easier to read, so this tutorial uses
them throughout.
"""

from __future__ import annotations

import mojovec


def print_vector_results(title: str, result: mojovec.QueryResult) -> None:
    """Print one or more vector-query rows."""
    print(f"\n{title}")
    for query_index, ids in enumerate(result["ids"]):
        print(f"  query {query_index}")
        for rank, record_id in enumerate(ids):
            if record_id < 0:
                print(f"    rank {rank + 1}: <no match>")
                continue
            print(
                f"    rank {rank + 1}: id={record_id}, "
                f"distance={result['distances'][query_index][rank]:.6f}"
            )


def main() -> None:
    # quantized=True selects SQ8 storage. Set it to False for Flat Float32
    # storage without changing any CRUD or query calls.
    collection = mojovec.Collection(
        dimension=4,
        M=16,
        ef_construction=96,
        ef_search=32,
        quantized=True,
        name="python_getting_started",
        metric="l2",
    )

    ids = [101, 202, 303, 404]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    # add() accepts only IDs that do not exist yet.
    collection.add(ids=ids, embeddings=embeddings)
    print("created:", collection)
    print("active rows:", collection.count())

    # A batch is inferred from the nested rows. The returned outer lists have
    # one row per query and one item per requested rank.
    initial = collection.query(
        query_embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0],
        ],
        n_results=2,
    )
    print_vector_results("initial search", initial)

    # upsert() replaces existing IDs and inserts missing IDs in one batch.
    collection.upsert(
        ids=[202, 505],
        embeddings=[
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0],
        ],
    )

    # update() is stricter: every ID must already exist.
    collection.update(
        ids=[303],
        embeddings=[[0.0, 0.0, 0.95, 0.05]],
    )

    # delete() is idempotent and ignores unknown IDs. It is a soft deletion;
    # compact() can later remove historical internal rows.
    collection.delete([404, 999_999])

    print("\nafter mutations:")
    print("  active rows:", collection.count())
    print("  historical/deleted rows:", collection.count_deleted())
    print("  stats:", collection.stats())

    collection.set_ef_search(64)
    final = collection.query([1.0, 0.0, 0.0, 0.0], n_results=3)
    print_vector_results("search after mutations", final)

    # Legacy aliases remain available for existing programs.
    collection.upsert_batch([606], [0.0, 0.0, 0.1, 0.9])
    alias_result = collection.query_batch(
        [0.0, 0.0, 0.0, 1.0],
        n_results=1,
    )
    print_vector_results("query_batch compatibility alias", alias_result)


if __name__ == "__main__":
    main()
