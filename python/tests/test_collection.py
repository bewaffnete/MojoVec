import pytest
import mojovec

def test_collection_happy_path():
    # dimension, nlist, M, efConstruction
    col = mojovec.Collection(16, 2, 8, 40)
    
    ids = [100, 200, 300]
    embeddings = []
    for i in range(3):
        for j in range(16):
            if i == 0:
                embeddings.append(j / 16.0)
            else:
                embeddings.append((i + j) / 10.0)
                
    col.upsert_batch(ids, embeddings)
    
    query = [j / 16.0 for j in range(16)]
    res = col.query_batch(query, 2)
    
    assert len(res["ids"]) == 1
    assert len(res["ids"][0]) == 2
    assert res["ids"][0][0] == 100
    assert res["distances"][0][0] < 0.001

def test_collection_save_load(tmp_path):
    col = mojovec.Collection(16, 2, 8, 40)
    ids = [100]
    embeddings = [j / 16.0 for j in range(16)]
    col.upsert_batch(ids, embeddings)
    
    path = str(tmp_path / "test_col.bin")
    col.save(path)
    
    loaded_col = mojovec.load(path)
    res = loaded_col.query_batch(embeddings, 1)
    
    assert res["ids"][0][0] == 100
    assert res["distances"][0][0] < 0.001

def test_invalid_upsert():
    col = mojovec.Collection(16, 2, 8, 40)
    
    # 2 IDs, but only 1 embedding (16 floats)
    ids = [1, 2]
    embeddings = [0.1] * 16
    
    with pytest.raises(Exception):
        col.upsert_batch(ids, embeddings)
        
def test_invalid_query():
    col = mojovec.Collection(16, 2, 8, 40)
    
    # Query with 15 floats instead of 16
    query = [0.1] * 15
    with pytest.raises(Exception):
        col.query_batch(query, 1)

def test_empty_query():
    col = mojovec.Collection(16, 2, 8, 40)
    res = col.query_batch([], 2)
    # Should return empty
    assert len(res["ids"]) == 0
    assert len(res["distances"]) == 0


@pytest.mark.parametrize("quantized", [False, True])
def test_stats_and_compaction(quantized):
    col = mojovec.Collection(4, 8, 48, 24, quantized)
    col.upsert_batch(
        [10, 20, 30],
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
    )
    col.upsert_batch([20], [0.0, 0.9, 0.1, 0.0])
    col.delete([30])

    before = col.stats()
    assert before == {
        "active_count": 2,
        "deleted_count": 2,
        "total_count": 4,
        "deleted_ratio": 0.5,
        "dimension": 4,
        "quantized": quantized,
        "M": 8,
        "ef_construction": 48,
        "ef_search": 24,
    }

    no_op = col.compact_if_needed(0.75)
    assert no_op["performed"] is False
    assert no_op["reclaimed_records"] == 0

    report = col.compact_if_needed(0.5)
    assert report["performed"] is True
    assert report["before"]["total_count"] == 4
    assert report["after"]["active_count"] == 2
    assert report["after"]["deleted_count"] == 0
    assert report["after"]["total_count"] == 2
    assert report["reclaimed_records"] == 2
    assert report["elapsed_seconds"] >= 0.0

    result = col.query_batch([0.0, 0.9, 0.1, 0.0], 1)
    assert result["ids"][0][0] == 20


def test_compact_if_needed_rejects_invalid_threshold():
    col = mojovec.Collection(4)
    with pytest.raises(Exception):
        col.compact_if_needed(-0.01)
    with pytest.raises(Exception):
        col.compact_if_needed(1.01)


@pytest.mark.parametrize("quantized", [False, True])
def test_metadata_storage_serialization_and_compaction(tmp_path, quantized):
    col = mojovec.Collection(4, 8, 48, 24, quantized)
    col.add_with_metadata(
        [10, 20],
        [
            1.0, 2.0, 3.0, 4.0,
            10.0, 11.0, 12.0, 13.0,
        ],
        [
            {
                "label": "first",
                "year": 2026,
                "score": 0.875,
                "published": True,
            },
            {
                "label": "second",
                "score": -12.5,
                "published": False,
            },
        ],
    )

    assert col.get_metadata(10) == {
        "label": "first",
        "year": 2026,
        "score": 0.875,
        "published": True,
    }

    # A vector-only update inherits metadata from the active version.
    col.upsert_batch([10], [2.0, 3.0, 4.0, 5.0])
    assert col.get_metadata(10)["label"] == "first"

    # An explicit metadata update replaces the complete metadata object.
    col.update_with_metadata(
        [10],
        [3.0, 4.0, 5.0, 6.0],
        [{"label": "replacement", "version": 2}],
    )
    assert col.get_metadata(10) == {
        "label": "replacement",
        "version": 2,
    }

    col.delete([20])
    report = col.compact()
    assert report["performed"] is True
    assert col.get_metadata(10)["label"] == "replacement"

    path = str(tmp_path / f"metadata_{quantized}.mojovec")
    col.save(path)
    loaded = mojovec.load(path)
    assert loaded.get_metadata(10) == {
        "label": "replacement",
        "version": 2,
    }


def test_metadata_rejects_unsupported_python_values():
    col = mojovec.Collection(4)
    with pytest.raises(Exception):
        col.add_with_metadata(
            [1],
            [1.0, 2.0, 3.0, 4.0],
            [{"tags": ["unsupported", "for now"]}],
        )


def test_keyword_constructor_and_collection_accessors():
    col = mojovec.Collection(
        dimension=4,
        M=12,
        ef_construction=64,
        ef_search=32,
        quantized=False,
        name="python-parity",
        metric="cosine",
    )

    assert col.name() == "python-parity"
    assert col.dimension() == 4
    assert col.storage_kind() == "flat"
    assert col.metric() == "cosine"
    assert col.is_quantized() is False
    assert col.count() == 0
    assert col.count_deleted() == 0
    assert "python-parity" in repr(col)

    col.set_ef_search(48)
    assert col.stats()["ef_search"] == 48

    # Preserve the pre-parity positional slot for metric.
    positional = mojovec.Collection(2, 8, 32, 16, False, "ip")
    assert positional.metric() == "ip"
    assert positional.name() == ""


@pytest.mark.parametrize("quantized", [False, True])
def test_managed_crud_documents_and_complete_query_results(quantized):
    col = mojovec.Collection(4, quantized=quantized)
    col.add(
        ids=[10, 20],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        metadatas=[
            {"category": "docs", "year": 2024},
            {"category": "code", "year": 2026},
        ],
        documents=["Mojo vector search guide", "Python bindings reference"],
    )

    assert col.get_document(10) == "Mojo vector search guide"
    result = col.query([[1.0, 0.0, 0.0, 0.0]], n_results=2)
    assert set(result) == {
        "ids",
        "distances",
        "metadatas",
        "documents",
        "scores",
    }
    assert result["ids"][0][0] == 10
    assert result["metadatas"][0][0]["category"] == "docs"
    assert result["documents"][0][0] == "Mojo vector search guide"
    assert result["scores"] == []

    col.update(
        [10],
        [[0.9, 0.1, 0.0, 0.0]],
        metadatas=[{"category": "updated", "year": 2027}],
        documents=["Updated Mojo guide"],
    )
    assert col.get_metadata(10)["category"] == "updated"
    assert col.get_document(10) == "Updated Mojo guide"

    col.upsert(
        [20],
        [[0.1, 0.9, 0.0, 0.0]],
        documents=["Updated Python reference"],
    )
    assert col.get_document(20) == "Updated Python reference"
    assert col.get_metadata(20)["category"] == "code"


def test_chroma_style_where_for_vector_search():
    col = mojovec.Collection(3, quantized=False)
    col.add(
        [1, 2, 3, 4],
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ],
        metadatas=[
            {"kind": "guide", "year": 2023, "published": True},
            {"kind": "guide", "year": 2025, "published": True},
            {"kind": "api", "year": 2026, "published": False},
            {"kind": "news", "year": 2027, "published": True},
        ],
    )

    result = col.query(
        [[1.0, 0.0, 0.0]],
        n_results=3,
        where={
            "$and": [
                {"year": {"$gte": 2025}},
                {"kind": {"$in": ["guide", "news"]}},
                {"published": {"$ne": False}},
            ]
        },
    )
    assert result["ids"][0][:2] == [2, 4]
    assert result["ids"][0][2] == -1

    negated = col.query(
        [[1.0, 0.0, 0.0]],
        n_results=2,
        where={"$not": {"kind": {"$nin": ["api", "news"]}}},
    )
    assert negated["ids"][0] == [3, 4]


def test_bm25_and_hybrid_search_return_scores_and_payloads():
    col = mojovec.Collection(3, quantized=False)
    col.add(
        [1, 2, 3],
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        metadatas=[
            {"lang": "en"},
            {"lang": "ru"},
            {"lang": "en"},
        ],
        documents=[
            "Mojo vector search performance",
            "Поиск по векторам",
            "Mojo Python API guide",
        ],
    )

    text = col.query(
        query_texts=["python api"],
        n_results=2,
        where={"lang": "en"},
    )
    assert text["ids"][0][0] == 3
    assert text["distances"] == []
    assert text["scores"][0][0] > 0.0
    assert text["documents"][0][0] == "Mojo Python API guide"

    hybrid = col.query_hybrid(
        [[1.0, 0.0, 0.0]],
        ["python api"],
        n_results=2,
        where={"lang": {"$eq": "en"}},
    )
    assert set(hybrid["ids"][0]) == {1, 3}
    assert hybrid["distances"] == []
    assert all(score > 0.0 for score in hybrid["scores"][0])


def test_query_rejects_ambiguous_or_invalid_inputs():
    col = mojovec.Collection(2)
    with pytest.raises(ValueError, match="either query_embeddings"):
        col.query([1.0, 0.0], query_texts=["text"])
    with pytest.raises(ValueError, match="required"):
        col.query()
    with pytest.raises(ValueError, match="unsupported where operator"):
        col.query([1.0, 0.0], where={"x": {"$contains": 1}})
    with pytest.raises(TypeError, match="where must be a mapping"):
        col.query([1.0, 0.0], where=["not", "a", "mapping"])
    with pytest.raises(TypeError, match="have one type"):
        col.query([1.0, 0.0], where={"x": {"$in": [1, "1"]}})


def test_ragged_embeddings_are_rejected():
    col = mojovec.Collection(dimension=4)
    with pytest.raises(ValueError, match="embedding at index 0 has dimension 3; expected 4"):
        col.add(ids=[101, 102], embeddings=[[1, 2, 3], [4, 5, 6, 7, 8]])
    with pytest.raises(ValueError, match="embedding at index 0 has dimension 2; expected 4"):
        col.query(query_embeddings=[[1, 2], [3, 4, 5, 6, 7, 8]])


def test_snapshot_load_mmap_and_wal_recovery(tmp_path):
    snapshot_path = tmp_path / "snapshot.mojovec"
    point_in_time_path = tmp_path / "point-in-time.mojovec"
    wal_path = tmp_path / "collection.wal"

    col = mojovec.Collection(
        dimension=4,
        quantized=False,
        name="durable-python",
    )
    col.add(
        [1],
        [[1.0, 0.0, 0.0, 0.0]],
        documents=["snapshot document"],
    )
    col.save(snapshot_path)

    mapped = mojovec.load(
        snapshot_path,
        memory_mapped=True,
        mmap_threshold_bytes=0,
    )
    assert mapped.is_memory_mapped() is True
    assert mapped.get_document(1) == "snapshot document"
    copied = mojovec.Collection.load(snapshot_path, memory_mapped=False)
    assert copied.is_memory_mapped() is False
    assert copied.count() == 1

    point_in_time = col.snapshot(
        point_in_time_path,
        memory_mapped=False,
    )
    assert point_in_time.is_memory_mapped() is False
    assert point_in_time.count() == 1

    col.enable_wal(wal_path, durability=mojovec.WAL_SYNC)
    assert col.wal_enabled() is True
    col.add(
        [2],
        [[0.0, 1.0, 0.0, 0.0]],
        documents=["WAL document"],
    )
    col.flush_wal()
    assert col.wal_sequence() == 1

    recovered = mojovec.recover(
        snapshot_path,
        wal_path,
        durability=mojovec.WAL_ASYNC,
        memory_mapped=False,
    )
    assert recovered.count() == 2
    assert recovered.get_document(2) == "WAL document"
    assert recovered.wal_enabled() is True
    recovered.disable_wal()
    assert recovered.wal_enabled() is False


def test_compact_if_needed_uses_public_default():
    col = mojovec.Collection(2)
    col.add([1, 2], [[1.0, 0.0], [0.0, 1.0]])
    col.delete([1])
    report = col.compact_if_needed()
    assert report["performed"] is True


def test_numpy_fast_paths_if_numpy_is_available():
    np = pytest.importorskip("numpy")
    col = mojovec.Collection(3, quantized=False)
    ids = np.asarray([1, 2], dtype=np.int64)
    embeddings = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    col.upsert_numpy(ids, embeddings)
    result = col.query_numpy(embeddings[:1], 2)
    assert result["ids"].shape == (1, 2)
    assert result["ids"][0, 0] == 1
    assert result["metadatas"] == []
