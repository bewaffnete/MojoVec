"""
Native BM25 search over collection documents.

BM25 complements vector search:

- vector query input is a flattened List[Float32];
- BM25 query input is a List[String];
- vector results use `distances` where smaller is better;
- BM25 results use `scores` where larger is better.

Documents, metadata, vectors, and both search paths share the same application
IDs and lifecycle. Applications do not create, serialize, or compact a text
index separately.
"""

from std.collections import List

from mojovec import Client, Collection, Metadata, QueryResults, Where


comptime DIMENSION = 4
comptime DATABASE_PATH = "/tmp/mojovec_bm25_example.mojovec"


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


def article_metadata(
    category: String, language: String, published: Bool
) -> Metadata:
    var metadata = Metadata()
    metadata.set("category", category)
    metadata.set("language", language)
    metadata.set("published", published)
    return metadata^


def print_bm25_results(title: String, results: QueryResults) raises:
    print("\n", title)
    print("BM25 leaves distances empty:", len(results.distances) == 0)
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
                "score",
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
        "knowledge_base",
        dimension=DIMENSION,
        M=16,
        ef_construction=100,
        ef_search=32,
        quantized=True,
    )

    var ids = [101, 202, 303, 404]
    var embeddings = List[Float32]()
    append_vector(embeddings, 1.0, 0.0, 0.0, 0.0)
    append_vector(embeddings, 0.0, 1.0, 0.0, 0.0)
    append_vector(embeddings, 0.0, 0.0, 1.0, 0.0)
    append_vector(embeddings, 0.0, 0.0, 0.0, 1.0)

    var metadatas = List[Metadata]()
    metadatas.append(article_metadata("docs", "en", True))
    metadatas.append(article_metadata("internals", "en", True))
    metadatas.append(article_metadata("tutorial", "ru", True))
    metadatas.append(article_metadata("notes", "en", False))

    var documents = [
        String("Fast vector search with Mojo and HNSW"),
        String("HNSW graph layers, links, and candidate traversal"),
        String("«Быстрый» полнотекстовый поиск — BM25 на Mojo"),
        String("Private draft about vector database experiments"),
    ]
    collection.add(ids, embeddings, metadatas, documents)

    # Indexing and querying share one text analyzer: Unicode lowercase,
    # Unicode punctuation/symbol/emoji boundaries, and bundled English/Russian
    # stopwords. There is deliberately no stemming, so exact word forms remain
    # distinct. Apostrophes and underscores inside terms are preserved.
    #
    # Overload resolution selects BM25 because the query batch is List[String].
    # Repeated query terms are deduplicated. Repeated document terms still
    # affect term frequency through the standard BM25 saturation formula.
    var batched = collection.query(
        [
            String("vector search"),
            String("HNSW graph"),
            String("ПОИСК mojo"),
        ],
        n_results=3,
    )
    print_bm25_results("Three text queries in one call", batched)

    # The typed metadata filter is evaluated before candidates enter the
    # result set. BM25 corpus statistics remain global and deterministic.
    var public_results = collection.query(
        [String("vector database")],
        where=Where.eq("published", True),
        n_results=3,
    )
    print_bm25_results("Only published records", public_results)

    # Supplying a new document deactivates old postings for that application
    # ID. Supplying only a vector or metadata would preserve the document.
    collection.update(
        [101],
        [0.9, 0.1, 0.0, 0.0],
        ["Exact and quantized nearest-neighbor search in Mojo"],
    )
    print_bm25_results(
        "Old terms no longer match ID 101",
        collection.query([String("HNSW")], n_results=4),
    )
    print_bm25_results(
        "New terms are indexed immediately",
        collection.query([String("quantized nearest neighbor")], n_results=2),
    )

    # Documents are serialized with the collection. The BM25 postings are
    # rebuilt automatically and are never a second file to keep in sync.
    collection.save(DATABASE_PATH)
    var loaded = Collection.load(DATABASE_PATH)
    print_bm25_results(
        "Search after load",
        loaded.query([String("quantized search")], n_results=2),
    )

    # update() created a historical vector/document version. compact() removes
    # it and rebuilds both HNSW and BM25 from active records.
    var report = loaded.compact()
    print("\nCompaction performed:", report.performed)
    print("Reclaimed historical rows:", report.reclaimed_records)
    print_bm25_results(
        "Search after compaction",
        loaded.query([String("quantized search")], n_results=2),
    )
