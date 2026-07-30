"""
Hybrid HNSW + BM25 search with reciprocal rank fusion (RRF).

The two source scores are intentionally never added directly:

- vector distances are smaller-is-better;
- BM25 relevance scores are larger-is-better;
- RRF converts each source to ranks and combines those ranks.

Every query embedding is paired with the text at the same batch position.
Hybrid results use `scores` and leave `distances` empty.
"""

from std.collections import List

from mojovec import Client, Metadata, QueryResults, Where


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


def article_metadata(category: String, published: Bool) -> Metadata:
    var metadata = Metadata()
    metadata.set("category", category)
    metadata.set("published", published)
    return metadata^


def print_results(title: String, results: QueryResults) raises:
    print("\n", title)
    print("Hybrid leaves distances empty:", len(results.distances) == 0)
    for query_index in range(len(results.ids)):
        print(" query", query_index)
        for rank in range(len(results.ids[query_index])):
            var record_id = results.ids[query_index][rank]
            if record_id < 0:
                print("  rank", rank, "<no match>")
                continue
            print(
                "  rank",
                rank,
                "ID",
                record_id,
                "RRF score",
                results.scores[query_index][rank],
            )
            print("   document:", results.documents[query_index][rank])
            print(
                "   category:",
                results.metadatas[query_index][rank].get_string(
                    "category"
                ),
            )


def main() raises:
    var client = Client()
    var collection = client.create_collection(
        "hybrid_articles",
        dimension=DIMENSION,
        M=16,
        ef_construction=100,
        ef_search=64,
        quantized=True,
    )

    var embeddings = List[Float32]()
    append_vector(embeddings, 1.0, 0.0, 0.0, 0.0)
    append_vector(embeddings, 0.9, 0.1, 0.0, 0.0)
    append_vector(embeddings, 0.0, 1.0, 0.0, 0.0)
    append_vector(embeddings, 0.0, 0.0, 1.0, 0.0)

    var metadatas = List[Metadata]()
    metadatas.append(article_metadata("guide", True))
    metadatas.append(article_metadata("internals", True))
    metadatas.append(article_metadata("release", False))
    metadatas.append(article_metadata("tutorial", True))

    collection.add(
        [101, 202, 303, 404],
        embeddings,
        metadatas,
        [
            String("Vector search introduction"),
            String("HNSW graph traversal and vector search"),
            String("Database release notes"),
            String("BM25 full text search tutorial"),
        ],
    )

    # The embedding and text at position zero form one hybrid query. HNSW and
    # BM25 each retrieve 4 * n_results candidates by default. ID 202 is strong
    # in both lists, so it receives two reciprocal-rank contributions.
    var single = collection.query_hybrid(
        [0.95, 0.05, 0.0, 0.0],
        [String("HNSW vector search")],
        n_results=3,
    )
    print_results("One hybrid query", single)

    # Batching is row-major for embeddings. These two flattened vectors align
    # with the two strings in the same order.
    var batched_embeddings = List[Float32]()
    append_vector(batched_embeddings, 1.0, 0.0, 0.0, 0.0)
    append_vector(batched_embeddings, 0.0, 0.0, 1.0, 0.0)
    var batched = collection.query_hybrid(
        batched_embeddings,
        [
            String("graph traversal"),
            String("full text tutorial"),
        ],
        n_results=2,
        rrf_k=60,
        candidate_multiplier=4,
    )
    print_results("Two aligned hybrid queries", batched)

    # The typed filter is evaluated before both HNSW and BM25 produce their
    # candidate rankings. Corpus-wide BM25 statistics remain unchanged.
    var published = collection.query_hybrid(
        [0.0, 1.0, 0.0, 0.0],
        [String("database release")],
        where=Where.eq("published", True),
        n_results=3,
    )
    print_results("Only published records", published)

    # Stopwords remove every text term here, so RRF receives only the vector
    # ranking and behaves as a rank-based vector fallback.
    var vector_fallback = collection.query_hybrid(
        [1.0, 0.0, 0.0, 0.0],
        [String("the and и на")],
        n_results=2,
    )
    print_results("Vector fallback", vector_fallback)
