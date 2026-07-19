from std.testing import assert_true, assert_equal, TestSuite
from mojovec import Client, Collection
from std.collections import List

def test_soft_delete_basic() raises:
    var client = Client()
    var col = client.create_collection("test_soft", 4)
    
    var ids = List[Int]()
    var embeddings = List[Float32]()
    
    ids.append(100)
    for _ in range(4): embeddings.append(1.0)
    
    ids.append(200)
    for _ in range(4): embeddings.append(2.0)
    
    ids.append(300)
    for _ in range(4): embeddings.append(3.0)
    
    col.add(ids, embeddings)
    
    var q = List[Float32]()
    for _ in range(4): q.append(2.0)
    
    var res1 = col.query(q, n_results=1)
    assert_equal(res1.ids[0][0], 200, "Should find 200 as closest initially")
    
    var del_ids = List[Int]()
    del_ids.append(200)
    col.delete(del_ids)
    
    assert_equal(col.count_deleted(), 1, "Count deleted should be 1")
    
    var res2 = col.query(q, n_results=1)
    assert_true(res2.ids[0][0] != 200, "Should NOT find 200 after deletion")
    assert_true(res2.ids[0][0] == 100 or res2.ids[0][0] == 300, "Should fallback to next closest")

def test_soft_delete_all() raises:
    var client = Client()
    var col = client.create_collection("test_soft_all", 4)
    
    var ids = List[Int]()
    var embeddings = List[Float32]()
    
    for j in range(10):
        ids.append(j)
        for _ in range(4): embeddings.append(Float32(j))
    
    col.add(ids, embeddings)
    col.delete(ids)
    assert_equal(col.count_deleted(), 10, "Count deleted should be 10")
    
    var q = List[Float32]()
    for _ in range(4): q.append(0.0)
    
    var res = col.query(q, n_results=3)
    assert_equal(res.ids[0][0], -1, "Should be padded with -1 since everything is deleted")
    assert_equal(res.ids[0][1], -1, "Should be padded with -1")
    assert_equal(res.ids[0][2], -1, "Should be padded with -1")

def test_soft_delete_restore() raises:
    var client = Client()
    var col = client.create_collection("test_soft_restore", 4)
    
    var ids = List[Int]()
    var embeddings = List[Float32]()
    
    ids.append(100)
    for _ in range(4): embeddings.append(1.0)
    
    col.add(ids, embeddings)
    
    var q = List[Float32]()
    for _ in range(4): q.append(1.0)
    
    var del_ids = List[Int]()
    del_ids.append(100)
    col.delete(del_ids)
    
    var res = col.query(q, n_results=1)
    assert_equal(res.ids[0][0], -1, "Should be -1 after delete")
    
    # Add same ID again with new data.
    # In MojoVec's current implementation, adding the same ID adds a new row,
    # but the old one remains deleted. Since query loops over results and uses `_user_ids[i]`,
    # the new row (which is not deleted) should be returned!
    var ids2 = List[Int]()
    ids2.append(100)
    var embeddings2 = List[Float32]()
    for _ in range(4): embeddings2.append(1.0)
    
    col.add(ids2, embeddings2)
    var res2 = col.query(q, n_results=1)
    assert_equal(res2.ids[0][0], 100, "Should find 100 after re-adding it")

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
