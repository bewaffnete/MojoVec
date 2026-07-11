from mojovec.api import Client, Collection, CollectionIVFPQ, QueryResults
from std.collections import List

def assert_true(cond: Bool, msg: String = "Assertion failed") raises:
    if not cond:
        raise Error(msg)

def test_collection_hnsw() raises:
    var client = Client()
    var col = client.create_collection("test", 16)
    
    var ids = List[Int](capacity=3)
    ids.append(10)
    ids.append(20)
    ids.append(30)
    var embeddings = List[Float32](capacity=48)
    for i in range(48):
        # We make the first vector exactly match the query, 
        # so it will be the guaranteed nearest neighbor.
        if i < 16:
            embeddings.append(Float32(i) / 16.0)
        else:
            embeddings.append(Float32(16 - i % 16) / 16.0)
        
    col.add(ids, embeddings)
    
    var q = List[Float32](capacity=16)
    for i in range(16):
        q.append(Float32(i) / 16.0)
        
    var results = col.query(q, n_results=2)
    assert_true(len(results.ids) == 1, "Should have 1 query result list")
    assert_true(len(results.ids[0]) == 2, "Should return top 2 results")
    assert_true(results.ids[0][0] == 10, "First result should be ID 10")
    
    # Test Serialization
    col.save("test_hnsw_col.bin")
    var loaded = Collection.load("test_hnsw_col.bin")
    var res2 = loaded.query(q, n_results=2)
    assert_true(res2.ids[0][0] == 10, "Loaded col should return ID 10")

def test_collection_ivfpq() raises:
    var client = Client()
    # Very small parameters so training is fast and doesn't require huge dataset
    var col = client.create_ivfpq_collection("test_ivfpq", dimension=16, nlist=2, M=8)
    
    var ids = List[Int]()
    var embeddings = List[Float32]()
    # Generate 100 vectors to have enough data for K-Means and PQ training
    for i in range(100):
        ids.append(1000 + i)
        for j in range(16):
            if i == 0:
                embeddings.append(Float32(j) / 16.0) # Match query exactly
            else:
                embeddings.append(Float32((i + j) % 10) / 10.0)
            
    col.add(ids, embeddings)
    
    var q = List[Float32]()
    for j in range(16):
        q.append(Float32(j) / 16.0)
        
    var results = col.query(q, n_results=3)
    assert_true(len(results.ids) == 1, "Should have 1 query result list")
    assert_true(len(results.ids[0]) == 3, "Should return top 3 results")
    assert_true(results.ids[0][0] == 1000, "First result should be ID 1000")
    
    # Test Serialization
    col.save("test_ivfpq_col.bin")
    var loaded = CollectionIVFPQ.load("test_ivfpq_col.bin")
    var res2 = loaded.query(q, n_results=3)
    assert_true(res2.ids[0][0] == 1000, "Loaded IVFPQ should return same ID 1000")


    
def main() raises:
    print("Testing API Collection HNSW...")
    test_collection_hnsw()
    print("Testing API Collection IVFPQ...")
    test_collection_ivfpq()
    print("All API Unit Tests passed!")
