from std.testing import assert_equal, assert_not_equal, assert_true
from std.collections import List
from std.memory.span import Span
from mojovec.api.collection import Collection

def get_test_data() -> Tuple[List[Int], List[Float32]]:
    var ids = List[Int]()
    var embeddings = List[Float32]()
    for i in range(100):
        ids.append(i)
        for j in range(16):
            embeddings.append(Float32(i))
    return ids^, embeddings^

def test_soft_delete() raises:
    var collection = Collection(16)
    var data = get_test_data()
    collection.add(data[0], data[1])
    
    # Query for id 50
    var query = List[Float32]()
    for j in range(16):
        query.append(50.0)
        
    var results1 = collection.query(query, n_results=5)
    assert_equal(results1.ids[0][0], 50, "Nearest neighbor should be 50")
    
    # Delete id 50
    var delete_list = List[Int]()
    delete_list.append(50)
    collection.delete(delete_list)
    assert_equal(collection.count_deleted(), 1, "Count deleted should be 1")
    
    # Query again
    var results2 = collection.query(query, n_results=5)
    assert_not_equal(results2.ids[0][0], 50, "Nearest neighbor should no longer be 50")
    
    # Verify 50 is nowhere in top 5
    for id in results2.ids[0]:
        assert_not_equal(id, 50, "50 should not be in the results at all")
        
def test_save_load_deleted() raises:
    var collection = Collection(16)
    var data = get_test_data()
    collection.add(data[0], data[1])
    
    var delete_list = List[Int]()
    delete_list.append(10)
    delete_list.append(20)
    collection.delete(delete_list)
    
    var path = "test_deleted.mojovec"
    collection.save(path)
    
    var loaded = Collection.load(path)
    assert_equal(loaded.count_deleted(), 2, "Loaded collection should have 2 deleted items")
    
    # Verify 10 and 20 are excluded from results
    var query10 = List[Float32]()
    var query20 = List[Float32]()
    for j in range(16):
        query10.append(10.0)
        query20.append(20.0)
        
    var r10 = loaded.query(query10, n_results=5)
    var r20 = loaded.query(query20, n_results=5)
    
    assert_not_equal(r10.ids[0][0], 10, "10 should be deleted in loaded model")
    assert_not_equal(r20.ids[0][0], 20, "20 should be deleted in loaded model")

def main() raises:
    test_soft_delete()
    test_save_load_deleted()
    print("All Soft Delete tests passed!")
