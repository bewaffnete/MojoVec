from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from mojovec import Collection, Metadata, Where
from mojovec.api.rrf import reciprocal_rank_fusion


comptime DIMENSION = 2


def category(value: String) -> Metadata:
    var metadata = Metadata()
    metadata.set("category", value)
    return metadata^


def hybrid_collection(quantized: Bool) raises -> Collection:
    var collection = Collection(
        DIMENSION,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=quantized,
    )
    var metadatas = List[Metadata]()
    metadatas.append(category("public"))
    metadatas.append(category("private"))
    metadatas.append(category("public"))
    collection.add(
        [10, 20, 30],
        [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
        metadatas,
        [
            String("unrelated"),
            String("target target target"),
            String("target"),
        ],
    )
    return collection^


def test_rrf_formula_and_ties() raises:
    var fused = reciprocal_rank_fusion(
        [0, 1, 2],
        [1, 3, 0],
        n_results=5,
        rrf_k=60,
    )
    assert_equal(fused.internal_ids[0], 1)
    assert_equal(fused.internal_ids[1], 0)
    assert_equal(fused.internal_ids[2], 3)
    assert_equal(fused.internal_ids[3], 2)
    assert_equal(fused.internal_ids[4], -1)
    assert_almost_equal(
        fused.scores[0],
        Float32(1.0 / 62.0 + 1.0 / 61.0),
        atol=1e-7,
    )
    assert_almost_equal(
        fused.scores[1],
        Float32(1.0 / 61.0 + 1.0 / 63.0),
        atol=1e-7,
    )

    # Equal one-source ranks use the smaller internal ID as a stable tie-break.
    var tied = reciprocal_rank_fusion([2], [1], n_results=2)
    assert_equal(tied.internal_ids[0], 1)
    assert_equal(tied.internal_ids[1], 2)
    assert_equal(tied.scores[0], tied.scores[1])


def check_hybrid_ranking_and_payloads(quantized: Bool) raises:
    var collection = hybrid_collection(quantized)
    var results = collection.query_hybrid(
        [0.0, 0.0, 2.0, 0.0],
        [String("target"), String("unrelated")],
        n_results=3,
    )

    assert_equal(len(results.ids), 2)
    assert_equal(len(results.distances), 0)
    assert_equal(len(results.scores), 2)

    # ID 20 is strong in both rankings and wins the first fusion.
    assert_equal(results.ids[0][0], 20)
    assert_equal(results.ids[0][1], 30)
    assert_equal(results.ids[0][2], 10)
    assert_true(results.scores[0][0] > results.scores[0][1])
    assert_equal(results.documents[0][0], "target target target")
    assert_equal(
        results.metadatas[0][0].get_string("category"), "private"
    )

    # Text rank pulls ID 10 ahead of the nearest vector for query two.
    assert_equal(results.ids[1][0], 10)
    assert_equal(results.ids[1][1], 30)
    assert_equal(results.documents[1][0], "unrelated")


def test_flat_hybrid_ranking_and_payloads() raises:
    check_hybrid_ranking_and_payloads(False)


def test_sq8_hybrid_ranking_and_payloads() raises:
    check_hybrid_ranking_and_payloads(True)


def test_hybrid_where_delete_and_vector_only_fallback() raises:
    var collection = hybrid_collection(False)
    var filtered = collection.query_hybrid(
        [0.0, 0.0],
        [String("target")],
        where=Where.eq("category", "public"),
        n_results=3,
    )
    assert_equal(filtered.ids[0][0], 30)
    assert_equal(filtered.ids[0][1], 10)
    assert_equal(filtered.ids[0][2], -1)

    collection.delete([30])
    var after_delete = collection.query_hybrid(
        [0.0, 0.0],
        [String("target")],
        n_results=3,
    )
    assert_equal(after_delete.ids[0][0], 20)
    for record_id in after_delete.ids[0]:
        assert_true(record_id != 30)

    # Stopword-only text contributes nothing, leaving the vector rank intact.
    var vector_only = collection.query_hybrid(
        [0.0, 0.0],
        [String("the и")],
        n_results=2,
    )
    assert_equal(vector_only.ids[0][0], 10)
    assert_equal(vector_only.ids[0][1], 20)


def test_hybrid_validation_and_empty_batch() raises:
    var collection = hybrid_collection(False)
    var empty_embeddings = List[Float32]()
    var empty_texts = List[String]()
    var empty = collection.query_hybrid(empty_embeddings, empty_texts)
    assert_equal(len(empty.ids), 0)
    assert_equal(len(empty.scores), 0)

    var empty_collection = Collection(DIMENSION, quantized=False)
    var no_matches = empty_collection.query_hybrid(
        [0.0, 0.0],
        [String("target")],
        n_results=2,
    )
    assert_equal(no_matches.ids[0][0], -1)
    assert_equal(no_matches.ids[0][1], -1)
    assert_equal(no_matches.scores[0][0], 0.0)

    var invalid_shape = False
    try:
        _ = collection.query_hybrid(
            [0.0],
            [String("target")],
        )
    except:
        invalid_shape = True
    assert_true(invalid_shape)

    var mismatched_batches = False
    try:
        _ = collection.query_hybrid(
            [0.0, 0.0],
            [String("target"), String("extra")],
        )
    except:
        mismatched_batches = True
    assert_true(mismatched_batches)

    var invalid_rrf_k = False
    try:
        _ = collection.query_hybrid(
            [0.0, 0.0],
            [String("target")],
            rrf_k=0,
        )
    except:
        invalid_rrf_k = True
    assert_true(invalid_rrf_k)

    var invalid_multiplier = False
    try:
        _ = collection.query_hybrid(
            [0.0, 0.0],
            [String("target")],
            candidate_multiplier=0,
        )
    except:
        invalid_multiplier = True
    assert_true(invalid_multiplier)

    var invalid_n_results = False
    try:
        _ = collection.query_hybrid(
            [0.0, 0.0],
            [String("target")],
            n_results=0,
        )
    except:
        invalid_n_results = True
    assert_true(invalid_n_results)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
