from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from mojovec.api.collection import Collection


comptime DIMENSION = 8


def make_ids() -> List[Int]:
    return [10, 20, 30]


def make_embeddings() -> List[Float32]:
    var embeddings = List[Float32](capacity=3 * DIMENSION)
    for record in range(3):
        for dim in range(DIMENSION):
            embeddings.append(Float32(record * 10 + dim))
    return embeddings^


def make_query(base: Int) -> List[Float32]:
    var query = List[Float32](capacity=DIMENSION)
    for dim in range(DIMENSION):
        query.append(Float32(base + dim))
    return query^


def check_storage(quantized: Bool, path: String) raises:
    var collection = Collection(
        DIMENSION,
        M=8,
        ef_construction=40,
        ef_search=16,
        quantized=quantized,
        name="records",
    )
    assert_equal(collection.name(), "records")
    assert_equal(collection.count(), 0)
    if quantized:
        assert_true(collection.is_quantized())
    else:
        assert_false(collection.is_quantized())

    collection.add(make_ids(), make_embeddings())
    assert_equal(collection.count(), 3)

    var duplicate_failed = False
    try:
        collection.add([10], make_query(100))
    except:
        duplicate_failed = True
    assert_true(duplicate_failed, "add must reject an existing ID")

    var result = collection.query(make_query(0), n_results=1)
    assert_equal(result.ids[0][0], 10)

    collection.upsert([10], make_query(100))
    assert_equal(collection.count(), 3)
    assert_equal(collection.count_deleted(), 1)
    var updated = collection.query(make_query(100), n_results=1)
    assert_equal(updated.ids[0][0], 10)

    var managed_result = collection.query(make_query(100), n_results=1)
    assert_equal(managed_result.ids[0][0], 10)

    collection.delete([20])
    assert_equal(collection.count(), 2)

    collection.save(path)
    var loaded = Collection.load(path)
    assert_equal(loaded.name(), "records")
    assert_equal(loaded.count(), 2)
    assert_equal(loaded.is_quantized(), quantized)
    var loaded_result = loaded.query(make_query(100), n_results=1)
    assert_equal(loaded_result.ids[0][0], 10)


def test_flat_collection() raises:
    check_storage(False, "test_collection_flat.mojovec")


def test_sq8_collection() raises:
    check_storage(True, "test_collection_sq8.mojovec")


def test_update_requires_existing_id() raises:
    var collection = Collection(DIMENSION, quantized=False)
    var failed = False
    try:
        collection.update([999], make_query(0))
    except:
        failed = True
    assert_true(failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
