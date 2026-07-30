"""Native BM25 and hybrid HNSW + BM25 search through one collection."""

from __future__ import annotations

import mojovec


def print_ranked_results(title: str, result: mojovec.QueryResult) -> None:
    print(f"\n{title}")
    print("  distances populated:", bool(result["distances"]))
    print("  scores populated:", bool(result["scores"]))
    for query_index, ids in enumerate(result["ids"]):
        print(f"  query {query_index}")
        for rank, record_id in enumerate(ids):
            if record_id < 0:
                print(f"    rank {rank + 1}: <no match>")
                continue
            score = result["scores"][query_index][rank]
            print(
                f"    rank {rank + 1}: id={record_id}, score={score:.6f}"
            )
            print("      document:", result["documents"][query_index][rank])
            print("      metadata:", result["metadatas"][query_index][rank])


def main() -> None:
    collection = mojovec.Collection(
        dimension=4,
        M=16,
        ef_construction=96,
        ef_search=64,
        quantized=True,
        name="knowledge_base",
    )
    collection.add(
        ids=[101, 202, 303, 404],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        metadatas=[
            {"category": "guide", "published": True},
            {"category": "internals", "published": True},
            {"category": "release", "published": False},
            {"category": "tutorial", "published": True},
        ],
        documents=[
            "Vector search introduction",
            "HNSW graph traversal and vector search",
            "Database release notes",
            "BM25 full text search tutorial",
        ],
    )

    # query_texts selects BM25. Scores are larger-is-better and distances are
    # empty. The analyzer applies Unicode lowercase and word boundaries plus
    # bundled English/Russian stopwords; it deliberately does not stem.
    bm25 = collection.query(
        query_texts=["HNSW vector search", "full text tutorial"],
        n_results=3,
        where={"published": True},
    )
    print_ranked_results("batched BM25", bm25)

    # Passing a string list as the first argument is the overloaded shorthand.
    shorthand = collection.query(["database release"], n_results=2)
    print_ranked_results("BM25 positional shorthand", shorthand)

    # Hybrid search aligns each embedding with the text at the same batch
    # position. RRF combines ranks, never raw distance and BM25 score values.
    hybrid = collection.query_hybrid(
        query_embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        query_texts=[
            "HNSW graph traversal",
            "full text tutorial",
        ],
        n_results=3,
        rrf_k=60,
        candidate_multiplier=4,
        where={"category": {"$nin": ["release"]}},
    )
    print_ranked_results("hybrid RRF", hybrid)

    # Replacing a document deactivates its old postings immediately.
    collection.update(
        ids=[101],
        embeddings=[[0.95, 0.05, 0.0, 0.0]],
        documents=["Exact and quantized nearest-neighbor search in Mojo"],
    )
    updated = collection.query(
        query_texts=["quantized nearest neighbor"],
        n_results=2,
    )
    print_ranked_results("BM25 after document update", updated)


if __name__ == "__main__":
    main()
