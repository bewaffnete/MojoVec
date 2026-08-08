import pytest

import mojovec


def _filter_collection():
    collection = mojovec.Collection(
        dimension=2,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=False,
    )
    collection.add(
        [1, 2, 3, 4, 5],
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ],
        metadatas=[
            {"kind": "a", "number": 1, "score": 1.5, "enabled": True},
            {"kind": "a", "number": 2, "score": 2.5, "enabled": False},
            {"kind": "b", "number": 3, "score": 3.5, "enabled": True},
            {"kind": "b", "number": 4, "score": 4.5, "enabled": False},
            {"other": "missing tested fields"},
        ],
        documents=["one", "two", "three", "four", "five"],
    )
    return collection


def _matching_ids(collection, where):
    result = collection.query(
        [[1.0, 0.0]],
        n_results=5,
        where=where,
    )
    return {record_id for record_id in result["ids"][0] if record_id >= 0}


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"kind": "a"}, {1, 2}),
        ({"kind": {"$eq": "b"}}, {3, 4}),
        ({"kind": {"$ne": "a"}}, {3, 4}),
        ({"number": {"$gt": 2}}, {3, 4}),
        ({"number": {"$gte": 2}}, {2, 3, 4}),
        ({"number": {"$lt": 3}}, {1, 2}),
        ({"number": {"$lte": 2}}, {1, 2}),
        ({"score": {"$gte": 3.5}}, {3, 4}),
        ({"kind": {"$in": ["a"]}}, {1, 2}),
        ({"kind": {"$nin": ["a"]}}, {3, 4}),
        ({"kind": {"$not_in": ["a"]}}, {3, 4}),
        ({"enabled": True}, {1, 3}),
        ({"kind": "a", "enabled": True}, {1}),
        ({"$not": {"kind": "a"}}, {3, 4, 5}),
        (
            {"$or": [{"number": {"$lt": 2}}, {"number": {"$gt": 3}}]},
            {1, 4},
        ),
        (
            {
                "$and": [
                    {"number": {"$gte": 2}},
                    {"number": {"$lte": 3}},
                ]
            },
            {2, 3},
        ),
    ],
)
def test_all_chroma_style_filter_operators(where, expected):
    assert _matching_ids(_filter_collection(), where) == expected


@pytest.mark.parametrize(
    ("where", "error_type", "message"),
    [
        ({}, ValueError, "where cannot be empty"),
        ({1: "value"}, TypeError, "keys must be strings"),
        ({"field": {}}, ValueError, "cannot be empty"),
        ({"field": None}, TypeError, "where values"),
        ({"field": {"$in": []}}, ValueError, "at least one value"),
        ({"field": {"$in": "abc"}}, TypeError, "requires a sequence"),
        ({"field": {"$in": [True, 1]}}, TypeError, "have one type"),
        ({"$and": []}, ValueError, "at least one condition"),
        ({"$or": "bad"}, TypeError, "requires a sequence"),
        ({"$not": []}, TypeError, "where must be a mapping"),
        ({"$xor": []}, ValueError, "unsupported logical"),
        ({"field": {"$unknown": 1}}, ValueError, "unsupported where"),
    ],
)
def test_malformed_where_expressions_fail_in_python(where, error_type, message):
    collection = mojovec.Collection(2)
    with pytest.raises(error_type, match=message):
        collection.query([[1.0, 0.0]], where=where)


@pytest.mark.parametrize(
    "where",
    [
        {"enabled": {"$gt": True}},
        {"kind": {"$lte": "b"}},
    ],
)
def test_ordered_filters_reject_non_numeric_values(where):
    collection = mojovec.Collection(2)
    with pytest.raises(Exception, match="Ordered where predicates"):
        collection.query([[1.0, 0.0]], where=where)


def test_bm25_positional_shorthand_and_batched_queries():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [1, 2, 3],
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.3]],
        documents=[
            "Python vector search reference",
            "Unicode поиск по векторам",
            "Hybrid retrieval tutorial",
        ],
    )

    keyword = collection.query(
        query_texts=["PYTHON REFERENCE", "ПОИСК ВЕКТОРАМ"],
        n_results=2,
    )
    shorthand = collection.query(["python reference"], n_results=2)

    assert keyword["ids"][0][0] == 1
    assert keyword["ids"][1][0] == 2
    assert shorthand["ids"][0][0] == 1
    assert keyword["distances"] == []
    assert len(keyword["scores"]) == 2
    assert all(len(row) == 2 for row in keyword["scores"])


def test_bm25_stopword_only_and_unknown_terms_return_padded_no_match():
    collection = mojovec.Collection(2)
    collection.add(
        [1],
        [[1.0, 0.0]],
        documents=["meaningful vector document"],
    )

    result = collection.query(
        query_texts=["the and", "term-not-present-anywhere"],
        n_results=2,
    )

    assert result["ids"] == [[-1, -1], [-1, -1]]
    assert result["documents"] == [["", ""], ["", ""]]


def test_document_replacement_and_delete_update_bm25_postings():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [1, 2],
        [[1.0, 0.0], [0.0, 1.0]],
        documents=["legacy uniqueterm", "stable document"],
    )
    assert collection.query(["uniqueterm"], 1)["ids"][0][0] == 1

    collection.update(
        [1],
        [[0.9, 0.1]],
        documents=["replacement freshterm"],
    )
    assert collection.query(["uniqueterm"], 1)["ids"][0][0] == -1
    assert collection.query(["freshterm"], 1)["ids"][0][0] == 1

    collection.delete([1])
    assert collection.query(["freshterm"], 1)["ids"][0][0] == -1


def test_hybrid_search_batches_and_applies_filter_to_both_sources():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [1, 2, 3],
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        metadatas=[
            {"visible": True},
            {"visible": True},
            {"visible": False},
        ],
        documents=["alpha text", "beta text", "alpha beta"],
    )

    result = collection.query_hybrid(
        [[1.0, 0.0], [0.0, 1.0]],
        ["alpha", "beta"],
        n_results=2,
        rrf_k=10,
        candidate_multiplier=2,
        where={"visible": True},
    )

    assert len(result["ids"]) == 2
    assert all(3 not in row for row in result["ids"])
    assert result["distances"] == []
    assert all(score > 0.0 for row in result["scores"] for score in row)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_embeddings": [[1.0, 0.0]], "query_texts": []},
        {
            "query_embeddings": [[1.0, 0.0]],
            "query_texts": ["text"],
            "n_results": 0,
        },
        {
            "query_embeddings": [[1.0, 0.0]],
            "query_texts": ["text"],
            "rrf_k": 0,
        },
        {
            "query_embeddings": [[1.0, 0.0]],
            "query_texts": ["text"],
            "candidate_multiplier": 0,
        },
        {
            "query_embeddings": [[1.0, 0.0]],
            "query_texts": ["text"],
            "candidate_multiplier": 2_049,
        },
    ],
)
def test_hybrid_search_validates_batch_and_ranking_parameters(kwargs):
    collection = mojovec.Collection(2)
    with pytest.raises(Exception):
        collection.query_hybrid(**kwargs)


@pytest.mark.parametrize("n_results", [0, 2_049])
def test_vector_and_text_search_validate_result_count(n_results):
    collection = mojovec.Collection(2)
    with pytest.raises(Exception, match="n_results"):
        collection.query([[1.0, 0.0]], n_results=n_results)
    with pytest.raises(Exception, match="n_results"):
        collection.query(query_texts=["text"], n_results=n_results)


def test_empty_vector_text_and_hybrid_batches_have_consistent_empty_shape():
    collection = mojovec.Collection(2)

    vector = collection.query([], n_results=2)
    text = collection.query(query_texts=[], n_results=2)
    hybrid = collection.query_hybrid([], [], n_results=2)

    expected = {
        "ids": [],
        "distances": [],
        "metadatas": [],
        "documents": [],
        "scores": [],
    }
    assert vector == expected
    assert text == expected
    assert hybrid == expected
