"""Batch metadata/documents and Chroma-style typed where filters."""

from __future__ import annotations

import mojovec


def print_payload_results(title: str, result: mojovec.QueryResult) -> None:
    print(f"\n{title}")
    for rank, record_id in enumerate(result["ids"][0]):
        if record_id < 0:
            print(f"  rank {rank + 1}: <no match>")
            continue
        print(
            f"  rank {rank + 1}: id={record_id}, "
            f"distance={result['distances'][0][rank]:.6f}"
        )
        print("    metadata:", result["metadatas"][0][rank])
        print("    document:", result["documents"][0][rank])


def main() -> None:
    collection = mojovec.Collection(
        dimension=3,
        quantized=False,
        name="catalog",
        metric="cosine",
    )

    # Metadata is one scalar dictionary per record. Supported values are str,
    # int, float, and bool. Documents are one string per record.
    collection.add(
        ids=[10, 20, 30, 40],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ],
        metadatas=[
            {
                "category": "guide",
                "year": 2024,
                "score": 0.91,
                "published": True,
            },
            {
                "category": "reference",
                "year": 2026,
                "score": 0.99,
                "published": True,
            },
            {
                "category": "internal",
                "year": 2027,
                "score": 0.75,
                "published": False,
            },
            {
                "category": "guide",
                "year": 2028,
                "score": 0.88,
                "published": True,
            },
        ],
        documents=[
            "Getting started with MojoVec",
            "Complete Python API reference",
            "Private implementation notes",
            "Hybrid search deployment guide",
        ],
    )

    print("metadata for ID 20:", collection.get_metadata(20))
    print("document for ID 20:", collection.get_document(20))

    # Multiple top-level fields are implicitly ANDed. Explicit $and/$or/$not
    # can express nested logic. $in and $nin require homogeneous scalar lists.
    where = {
        "$and": [
            {"year": {"$gte": 2025}},
            {"category": {"$in": ["guide", "reference"]}},
            {"published": True},
            {
                "$or": [
                    {"score": {"$gt": 0.95}},
                    {"year": {"$gte": 2028}},
                ]
            },
        ]
    }
    filtered = collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=4,
        where=where,
    )
    print_payload_results("nested typed filter", filtered)

    # Vector-only updates retain existing metadata and documents.
    collection.update([20], [[0.9, 0.1, 0.0]])
    assert collection.get_document(20) == "Complete Python API reference"

    # Supplying payloads replaces each complete payload object for that row.
    collection.upsert(
        [20],
        [[0.85, 0.15, 0.0]],
        metadatas=[
            {
                "category": "reference",
                "year": 2029,
                "score": 1.0,
                "published": True,
            }
        ],
        documents=["Updated Python API reference"],
    )
    print("\nafter payload replacement:")
    print("  metadata:", collection.get_metadata(20))
    print("  document:", collection.get_document(20))

    # $not negates a complete expression. This differs from $ne for records
    # where the field is absent.
    not_internal = collection.query(
        [[1.0, 0.0, 0.0]],
        n_results=4,
        where={"$not": {"category": {"$eq": "internal"}}},
    )
    print_payload_results("$not internal", not_internal)


if __name__ == "__main__":
    main()
