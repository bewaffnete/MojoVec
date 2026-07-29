"""
Storing typed metadata and documents, then receiving both from queries.

Metadata is an owned string-to-scalar record. Its supported value types are:

- String
- Int
- Float64
- Bool

Metadata and document batches align with IDs. A vector-only update or upsert
inherits both current snapshots. Passing either payload explicitly replaces
that payload. Both follow records through save/load and compaction.

The next tutorial, api_06_where_filters.mojo, uses the same metadata snapshots
for typed `where` queries backed by automatic bitmap indexes.
"""

from std.collections import List

from mojovec import Client, Collection, Metadata, Where


comptime DIMENSION = 4
comptime DATABASE_PATH = "/tmp/mojovec_metadata_example.mojovec"


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


def print_record(collection: Collection, record_id: Int) raises:
    var metadata = collection.get_metadata(record_id)
    print("ID:", record_id)
    print("  title:", metadata.get_string("title"))
    print("  year:", metadata.get_int("year"))
    print("  score:", metadata.get_float("score"))
    print("  published:", metadata.get_bool("published"))
    print("  document:", collection.get_document(record_id))


def main() raises:
    var client = Client()
    var collection = client.create_collection(
        "articles",
        dimension=DIMENSION,
        M=16,
        ef_construction=100,
        ef_search=32,
        quantized=True,
    )

    # Metadata is constructed explicitly, so Mojo catches type mistakes at the
    # getter boundary rather than silently converting stored values.
    var first = Metadata()
    first.set("title", "Mojo vector search")
    first.set("year", 2026)
    first.set("score", Float64(0.97))
    first.set("published", True)

    var second = Metadata()
    second.set("title", "HNSW internals")
    second.set("year", 2025)
    second.set("score", Float64(0.91))
    second.set("published", False)

    var ids = List[Int]()
    ids.append(101)
    ids.append(202)

    var embeddings = List[Float32]()
    append_vector(embeddings, 1.0, 0.0, 0.0, 0.0)
    append_vector(embeddings, 0.0, 1.0, 0.0, 0.0)

    # There must be exactly one Metadata and one document String for every ID
    # when both payload batches are supplied.
    var metadatas = List[Metadata]()
    metadatas.append(first.copy())
    metadatas.append(second.copy())
    var documents = List[String]()
    documents.append("A practical guide to vector search written in Mojo.")
    documents.append("An explanation of HNSW layers, links, and traversal.")
    collection.add(ids, embeddings, metadatas, documents)

    print("After add")
    print_record(collection, 101)

    # The returned value is an owned copy. Mutating it does not mutate the
    # collection. To replace stored metadata, pass it to update or upsert.
    var detached = collection.get_metadata(101)
    detached.set("title", "local-only change")
    print(
        "\nStored title after changing returned copy:",
        collection.get_metadata(101).get_string("title"),
    )

    # A vector-only update preserves both active payload snapshots.
    collection.update([101], [0.9, 0.1, 0.0, 0.0])
    print(
        "Title after vector-only update:",
        collection.get_metadata(101).get_string("title"),
    )
    print(
        "Document after vector-only update:",
        collection.get_document(101),
    )

    # Supplying metadata replaces the complete object. Fields omitted from the
    # replacement no longer exist.
    var replacement = Metadata()
    replacement.set("title", "MojoVec metadata")
    replacement.set("year", 2027)
    replacement.set("score", Float64(0.99))
    replacement.set("published", True)
    var replacements = List[Metadata]()
    replacements.append(replacement.copy())
    collection.update([101], [1.0, 0.1, 0.0, 0.0], replacements)

    # A vector query populates four aligned matrices and leaves BM25 `scores`
    # empty. Payloads use the same query/rank coordinates as the nearest ID.
    var results = collection.query([1.0, 0.0, 0.0, 0.0], n_results=1)
    var nearest_id = results.ids[0][0]
    print("\nNearest document")
    print("  ID:", nearest_id)
    print("  distance:", results.distances[0][0])
    print("  title:", results.metadatas[0][0].get_string("title"))
    print("  document:", results.documents[0][0])

    # Stored fields are immediately available to the typed filter API. Bitmap
    # indexes are maintained automatically; applications do not create them.
    var published = collection.query(
        [1.0, 0.0, 0.0, 0.0],
        where=Where.eq("published", True),
        n_results=2,
    )
    print("\nPublished result IDs")
    for record_id in published.ids[0]:
        if record_id >= 0:
            print(" ", record_id)

    # Metadata is written into the collection file together with IDs, deletion
    # state, vector storage, and the HNSW graph.
    collection.save(DATABASE_PATH)
    var loaded = Collection.load(DATABASE_PATH)
    print("\nLoaded from disk")
    print_record(loaded, 101)

    # Both active metadata snapshots survive a rebuild; deleted historical
    # snapshots are discarded with their vectors.
    var report = loaded.compact()
    print("\nCompaction performed:", report.performed)
    print("Reclaimed historical rows:", report.reclaimed_records)
    print_record(loaded, 101)
