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
