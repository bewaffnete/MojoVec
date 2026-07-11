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
