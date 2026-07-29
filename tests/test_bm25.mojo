from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
)

from mojovec import Collection, Metadata, Where
from mojovec.api.text_analyzer import StandardBM25Analyzer


comptime DIMENSION = 4


def vector(base: Float32) -> List[Float32]:
    return [base, base + 1.0, base + 2.0, base + 3.0]


def append_vector(mut values: List[Float32], base: Float32):
    for component in vector(base):
        values.append(component)


def metadata(category: String) -> Metadata:
    var result = Metadata()
    result.set("category", category)
    return result^


def test_bm25_ranking_batching_and_payloads() raises:
    var collection = Collection(DIMENSION, quantized=False)
    var embeddings = List[Float32]()
    append_vector(embeddings, 1.0)
    append_vector(embeddings, 10.0)
    append_vector(embeddings, 20.0)
    var metadatas = List[Metadata]()
    metadatas.append(metadata("technical"))
    metadatas.append(metadata("technical"))
    metadatas.append(metadata("food"))
    collection.add(
        [10, 20, 30],
        embeddings,
        metadatas,
        [
            String("Mojo vector search search"),
            String("Vector database engine"),
            String("Cooking recipe"),
        ],
    )

    var results = collection.query(
        [String("VECTOR, search!"), String("cooking")],
        n_results=2,
    )
    assert_equal(len(results.ids), 2)
    assert_equal(len(results.distances), 0)
    assert_equal(len(results.scores), 2)
    assert_equal(results.ids[0][0], 10)
    assert_equal(results.ids[0][1], 20)
    assert_true(results.scores[0][0] > results.scores[0][1])
    assert_equal(
        results.metadatas[0][0].get_string("category"), "technical"
    )
    assert_equal(results.documents[0][0], "Mojo vector search search")

    assert_equal(results.ids[1][0], 30)
    assert_true(results.scores[1][0] > 0.0)
    assert_equal(results.ids[1][1], -1)
    assert_equal(results.scores[1][1], 0.0)
    assert_equal(results.metadatas[1][1].count(), 0)
    assert_equal(results.documents[1][1], "")


def test_bm25_unicode_case_and_punctuation() raises:
    var collection = Collection(DIMENSION, quantized=False)
    collection.add(
        [1, 2],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [
            String("«Быстрый» поиск — на Mojo"),
            String("Совсем другой документ"),
        ],
    )
    var results = collection.query(
        [String("ПОИСК mojo")], n_results=2
    )
    assert_equal(results.ids[0][0], 1)
    assert_true(results.scores[0][0] > 0.0)
    assert_equal(results.ids[0][1], -1)


def test_bm25_standard_analyzer_and_stopwords_without_stemming() raises:
    var analyzer = StandardBM25Analyzer()
    var tokens = analyzer.analyze(
        "The QUICK—БЫСТРЫЙ, café_42 🚀 и don't rock'n'roll"
    )
    assert_equal(len(tokens), 4)
    assert_equal(tokens[0], "quick")
    assert_equal(tokens[1], "быстрый")
    assert_equal(tokens[2], "café_42")
    assert_equal(tokens[3], "rock'n'roll")

    var collection = Collection(DIMENSION, quantized=False)
    collection.add(
        [1, 2],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [
            String("Running systems — быстрый поиск"),
            String("Run system — медленный поиск"),
        ],
    )

    # Stopword-only queries produce an empty padded result.
    var stopwords = collection.query(
        [String("THE and И на")], n_results=1
    )
    assert_equal(stopwords.ids[0][0], -1)
    assert_equal(stopwords.scores[0][0], 0.0)

    # Surface forms stay distinct because stemming is intentionally disabled.
    var exact_form = collection.query([String("RUNNING")], n_results=2)
    assert_equal(exact_form.ids[0][0], 1)
    assert_equal(exact_form.ids[0][1], -1)
    var other_form = collection.query([String("run")], n_results=2)
    assert_equal(other_form.ids[0][0], 2)
    assert_equal(other_form.ids[0][1], -1)


def test_bm25_top_k_ties_are_deterministic() raises:
    var collection = Collection(DIMENSION, quantized=False)
    var embeddings = List[Float32]()
    for record in range(5):
        append_vector(embeddings, Float32(record))
    collection.add(
        [1, 2, 3, 4, 5],
        embeddings,
        [
            String("shared"),
            String("shared"),
            String("shared"),
            String("shared"),
            String("shared"),
        ],
    )
    var results = collection.query([String("shared")], n_results=2)
    assert_equal(results.ids[0][0], 1)
    assert_equal(results.ids[0][1], 2)
    assert_equal(results.scores[0][0], results.scores[0][1])


def test_bm25_update_delete_and_empty_document() raises:
    var collection = Collection(DIMENSION, quantized=True)
    collection.add([10], vector(1.0), ["legacy keyword"])

    # A vector-only update inherits and reindexes the active document.
    collection.update([10], vector(2.0))
    assert_equal(
        collection.query([String("legacy")], n_results=1).ids[0][0],
        10,
    )

    # Replacing the document deactivates postings from the old version.
    collection.update([10], vector(3.0), ["modern keyword"])
    assert_equal(
        collection.query([String("legacy")], n_results=1).ids[0][0],
        -1,
    )
    assert_equal(
        collection.query([String("modern")], n_results=1).ids[0][0],
        10,
    )

    collection.add([20], vector(10.0), ["modern companion"])
    collection.delete([10])
    assert_equal(
        collection.query([String("keyword")], n_results=2).ids[0][0],
        -1,
    )
    assert_equal(
        collection.query([String("modern")], n_results=2).ids[0][0],
        20,
    )

    # Empty documents explicitly remove text from the active BM25 corpus.
    collection.update([20], vector(11.0), [""])
    var removed = collection.query([String("modern")], n_results=1)
    assert_equal(removed.ids[0][0], -1)
    assert_equal(removed.scores[0][0], 0.0)


def test_bm25_typed_where_filter() raises:
    var collection = Collection(DIMENSION, quantized=False)
    var metadatas = List[Metadata]()
    metadatas.append(metadata("public"))
    metadatas.append(metadata("private"))
    collection.add(
        [1, 2],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        metadatas,
        [
            String("vector database public guide"),
            String("vector database private notes"),
        ],
    )

    var results = collection.query(
        [String("vector database")],
        where=Where.eq("category", "private"),
        n_results=2,
    )
    assert_equal(results.ids[0][0], 2)
    assert_equal(results.metadatas[0][0].get_string("category"), "private")
    assert_equal(results.ids[0][1], -1)


def check_bm25_round_trip(quantized: Bool, path: String) raises:
    var collection = Collection(
        DIMENSION,
        M=8,
        ef_construction=48,
        ef_search=24,
        quantized=quantized,
    )
    collection.add([1], vector(1.0), ["old searchable text"])
    collection.update([1], vector(2.0), ["current searchable text"])
    collection.add([2], vector(10.0), ["deleted searchable text"])
    collection.delete([2])
    collection.save(path)

    var loaded = Collection.load(path)
    assert_equal(
        loaded.query([String("current")], n_results=2).ids[0][0],
        1,
    )
    assert_equal(
        loaded.query([String("old")], n_results=1).ids[0][0],
        -1,
    )
    assert_equal(
        loaded.query([String("deleted")], n_results=1).ids[0][0],
        -1,
    )

    var report = loaded.compact()
    assert_true(report.performed)
    assert_equal(
        loaded.query([String("current")], n_results=1).ids[0][0],
        1,
    )
    loaded.save(path + ".compacted")
    var reloaded = Collection.load(path + ".compacted")
    assert_equal(
        reloaded.query([String("current")], n_results=1).ids[0][0],
        1,
    )


def test_flat_bm25_serialization_and_compaction() raises:
    check_bm25_round_trip(
        False, "/tmp/mojovec_bm25_flat_v4.mojovec"
    )


def test_sq8_bm25_serialization_and_compaction() raises:
    check_bm25_round_trip(
        True, "/tmp/mojovec_bm25_sq8_v4.mojovec"
    )


def test_bm25_empty_inputs_and_validation() raises:
    var collection = Collection(DIMENSION, quantized=False)
    collection.add([1], vector(1.0))

    var empty_queries = List[String]()
    var empty_results = collection.query(empty_queries, n_results=3)
    assert_equal(len(empty_results.ids), 0)
    assert_equal(len(empty_results.scores), 0)

    var no_documents = collection.query([String("anything")], n_results=2)
    assert_equal(no_documents.ids[0][0], -1)
    assert_equal(no_documents.ids[0][1], -1)
    assert_equal(len(no_documents.documents), 0)
    assert_equal(len(no_documents.scores), 1)

    var invalid_failed = False
    try:
        _ = collection.query([String("anything")], n_results=0)
    except:
        invalid_failed = True
    assert_true(invalid_failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
