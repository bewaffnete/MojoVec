import pytest

import mojovec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimension": 0},
        {"dimension": -1},
        {"dimension": 65_537},
        {"dimension": 2, "M": 1},
        {"dimension": 2, "M": 1_001},
        {"dimension": 2, "ef_construction": 0},
        {"dimension": 2, "ef_construction": 2_049},
        {"dimension": 2, "ef_search": 0},
        {"dimension": 2, "ef_search": 2_049},
        {"dimension": 2, "metric": "manhattan"},
    ],
)
def test_constructor_rejects_invalid_index_configuration(kwargs):
    with pytest.raises(Exception):
        mojovec.Collection(**kwargs)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    ("metric", "expected_distances"),
    [
        ("l2", [0.0, 2.0, 4.0]),
        ("cosine", [0.0, 1.0, 2.0]),
        ("ip", [0.0, 1.0, 2.0]),
    ],
)
def test_metric_ranking_and_public_distance_semantics(
    quantized, metric, expected_distances
):
    collection = mojovec.Collection(
        dimension=2,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=quantized,
        metric=metric,
    )
    collection.add(
        [10, 20, 30],
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    )

    result = collection.query([[1.0, 0.0]], n_results=3)

    assert result["ids"][0] == [10, 20, 30]
    assert result["distances"][0] == pytest.approx(
        expected_distances, abs=1e-5
    )
    assert result["scores"] == []


@pytest.mark.parametrize("operation", ["add", "upsert"])
def test_duplicate_ids_reject_the_whole_batch(operation):
    collection = mojovec.Collection(2, quantized=False)

    with pytest.raises(Exception, match="(?i)duplicate"):
        getattr(collection, operation)(
            [10, 10],
            [[1.0, 0.0], [0.0, 1.0]],
            metadatas=[{"version": 1}, {"version": 2}],
            documents=["first", "second"],
        )

    assert collection.count() == 0
    assert collection.count_deleted() == 0


def test_duplicate_update_is_atomic_for_existing_record():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [10],
        [[1.0, 0.0]],
        metadatas=[{"version": 1}],
        documents=["original"],
    )

    with pytest.raises(Exception, match="(?i)duplicate"):
        collection.update(
            [10, 10],
            [[0.0, 1.0], [-1.0, 0.0]],
            metadatas=[{"version": 2}, {"version": 3}],
            documents=["second", "third"],
        )

    assert collection.count() == 1
    assert collection.count_deleted() == 0
    assert collection.get_metadata(10) == {"version": 1}
    assert collection.get_document(10) == "original"
    assert collection.query([[1.0, 0.0]], 1)["ids"][0][0] == 10


def test_add_existing_and_update_missing_leave_collection_unchanged():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [1],
        [[1.0, 0.0]],
        metadatas=[{"state": "original"}],
        documents=["original document"],
    )

    with pytest.raises(Exception):
        collection.add([1, 2], [[0.0, 1.0], [-1.0, 0.0]])
    with pytest.raises(Exception):
        collection.update([1, 999], [[0.0, 1.0], [-1.0, 0.0]])

    assert collection.count() == 1
    assert collection.count_deleted() == 0
    assert collection.get_metadata(1) == {"state": "original"}
    assert collection.get_document(1) == "original document"


def test_upsert_mixes_insert_and_replace_and_preserves_omitted_payloads():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [1],
        [[1.0, 0.0]],
        metadatas=[{"version": 1}],
        documents=["preserved"],
    )

    collection.upsert(
        [1, 2],
        [[0.9, 0.1], [0.0, 1.0]],
        metadatas=[{"version": 2}, {"version": 1}],
    )

    assert collection.count() == 2
    assert collection.count_deleted() == 1
    assert collection.get_metadata(1) == {"version": 2}
    assert collection.get_document(1) == "preserved"
    assert collection.get_metadata(2) == {"version": 1}
    with pytest.raises(Exception, match="does not have a document"):
        collection.get_document(2)


def test_empty_payloads_remove_only_the_supplied_payload_kind():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [1],
        [[1.0, 0.0]],
        metadatas=[{"category": "guide"}],
        documents=["searchable document"],
    )

    collection.update([1], [[0.9, 0.1]], metadatas=[{}])
    assert collection.get_metadata(1) == {}
    assert collection.get_document(1) == "searchable document"

    collection.update([1], [[0.8, 0.2]], documents=[""])
    assert collection.get_metadata(1) == {}
    with pytest.raises(Exception, match="does not have a document"):
        collection.get_document(1)


def test_python_metadata_crosses_the_binding_by_value():
    collection = mojovec.Collection(2, quantized=False)
    source = {
        "text": "value",
        "integer": 7,
        "floating": 1.25,
        "boolean": True,
    }
    collection.add([1], [[1.0, 0.0]], metadatas=[source])

    source["integer"] = 99
    first_read = collection.get_metadata(1)
    first_read["text"] = "mutated copy"

    assert collection.get_metadata(1) == {
        "text": "value",
        "integer": 7,
        "floating": 1.25,
        "boolean": True,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metadatas": []}, "metadatas length"),
        ({"documents": []}, "documents length"),
        ({"metadatas": [{"ok": True}], "documents": []}, "documents length"),
    ],
)
def test_payload_length_validation_happens_before_native_mutation(
    kwargs, message
):
    collection = mojovec.Collection(2)

    with pytest.raises(ValueError, match=message):
        collection.add([1], [[1.0, 0.0]], **kwargs)

    assert collection.count() == 0


def test_flat_embedding_shape_validation_is_atomic():
    collection = mojovec.Collection(3)

    with pytest.raises(ValueError, match="expected 6"):
        collection.add([1, 2], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="multiple of dimension 3"):
        collection.query([1.0, 0.0])

    assert collection.count() == 0


def test_query_result_padding_and_payload_alignment():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [1],
        [[0.0, 0.0]],
        metadatas=[{"payload": True}],
        documents=["first"],
    )
    collection.add([2], [[10.0, 0.0]])

    result = collection.query([[0.0, 0.0]], n_results=3)

    assert result["ids"][0] == [1, 2, -1]
    assert result["metadatas"][0] == [{"payload": True}, {}, {}]
    assert result["documents"][0] == ["first", "", ""]
    assert len(result["distances"][0]) == 3


def test_empty_mutation_batches_and_idempotent_delete_are_noops():
    collection = mojovec.Collection(2)

    collection.add([], [])
    collection.upsert([], [])
    collection.update([], [])
    collection.delete([])
    collection.delete([999])

    assert collection.count() == 0
    assert collection.count_deleted() == 0


def test_deleted_records_are_hidden_and_missing_payload_access_fails():
    collection = mojovec.Collection(2, quantized=False)
    collection.add(
        [1, 2],
        [[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"id": 1}, {"id": 2}],
        documents=["one", "two"],
    )
    collection.delete([1, 1, 999])

    assert collection.count() == 1
    assert collection.count_deleted() == 1
    assert collection.query([[1.0, 0.0]], 2)["ids"][0] == [2, -1]
    with pytest.raises(Exception):
        collection.get_metadata(1)
    with pytest.raises(Exception):
        collection.get_document(999)
