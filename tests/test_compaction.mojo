from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from mojovec import Collection


comptime DIMENSION = 8


def make_vector(base: Int) -> List[Float32]:
    var vector = List[Float32](capacity=DIMENSION)
    for dim in range(DIMENSION):
        vector.append(Float32(base + dim))
    return vector^


def make_dataset(count: Int) -> Tuple[List[Int], List[Float32]]:
    var ids = List[Int](capacity=count)
    var embeddings = List[Float32](capacity=count * DIMENSION)
    for record in range(count):
        ids.append((record + 1) * 10)
        for dim in range(DIMENSION):
            embeddings.append(Float32(record * 10 + dim))
    return ids^, embeddings^


def check_compaction(quantized: Bool, path: String) raises:
    var collection = Collection(
        DIMENSION,
        M=8,
        ef_construction=48,
        ef_search=24,
        quantized=quantized,
        name="compact_records",
    )
    var data = make_dataset(5)
    collection.add(data[0], data[1])

    # An update appends a replacement and leaves the old internal row deleted.
    collection.upsert([20], make_vector(200))
    collection.delete([30])

    var before = collection.stats()
    assert_equal(before.active_count, 4)
    assert_equal(before.deleted_count, 2)
    assert_equal(before.total_count, 6)
    assert_almost_equal(before.deleted_ratio, 2.0 / 6.0, atol=1e-9)
    assert_equal(before.dimension, DIMENSION)
    assert_equal(before.quantized, quantized)
    assert_equal(before.M, 8)
    assert_equal(before.ef_construction, 48)
    assert_equal(before.ef_search, 24)

    var before_result = collection.query(make_vector(200), n_results=1)
    assert_equal(before_result.ids[0][0], 20)

    var report = collection.compact()
    assert_true(report.performed)
    assert_equal(report.before.total_count, 6)
    assert_equal(report.after.active_count, 4)
    assert_equal(report.after.deleted_count, 0)
    assert_equal(report.after.total_count, 4)
    assert_equal(report.reclaimed_records, 2)
    assert_true(report.elapsed_seconds >= 0.0)

    var after = collection.stats()
    assert_equal(after.active_count, 4)
    assert_equal(after.deleted_count, 0)
    assert_equal(after.total_count, 4)
    assert_equal(after.M, 8)
    assert_equal(after.ef_construction, 48)
    assert_equal(after.ef_search, 24)
    assert_equal(collection.name(), "compact_records")

    var after_result = collection.query(make_vector(200), n_results=1)
    assert_equal(after_result.ids[0][0], 20)
    var deleted_result = collection.query(make_vector(20), n_results=4)
    for result_id in deleted_result.ids[0]:
        assert_true(result_id != 30)

    # Compacted state and graph parameters must survive serialization.
    collection.save(path)
    var loaded = Collection.load(path)
    var loaded_stats = loaded.stats()
    assert_equal(loaded_stats.active_count, 4)
    assert_equal(loaded_stats.deleted_count, 0)
    assert_equal(loaded_stats.total_count, 4)
    assert_equal(loaded_stats.M, 8)
    assert_equal(loaded_stats.ef_construction, 48)
    assert_equal(loaded_stats.ef_search, 24)
    var loaded_result = loaded.query(make_vector(200), n_results=1)
    assert_equal(loaded_result.ids[0][0], 20)

    # Running compact again without garbage is intentionally a no-op.
    var no_op = collection.compact()
    assert_false(no_op.performed)
    assert_equal(no_op.reclaimed_records, 0)


def test_flat_compaction() raises:
    check_compaction(False, "/tmp/mojovec_compact_flat.mojovec")


def test_sq8_compaction() raises:
    check_compaction(True, "/tmp/mojovec_compact_sq8.mojovec")


def test_compact_if_needed_threshold() raises:
    var collection = Collection(DIMENSION, quantized=False)
    var data = make_dataset(10)
    collection.add(data[0], data[1])

    collection.delete([10])
    var below_threshold = collection.compact_if_needed()
    assert_false(below_threshold.performed)
    assert_equal(collection.stats().total_count, 10)

    collection.delete([20])
    var at_threshold = collection.compact_if_needed()
    assert_true(at_threshold.performed)
    assert_equal(at_threshold.reclaimed_records, 2)
    assert_equal(collection.stats().active_count, 8)
    assert_equal(collection.stats().deleted_count, 0)
    assert_equal(collection.stats().total_count, 8)


def test_compact_all_deleted() raises:
    var collection = Collection(DIMENSION, quantized=True)
    var data = make_dataset(2)
    collection.add(data[0], data[1])
    collection.delete(data[0])

    var report = collection.compact()
    assert_true(report.performed)
    assert_equal(report.reclaimed_records, 2)
    assert_equal(collection.stats().active_count, 0)
    assert_equal(collection.stats().deleted_count, 0)
    assert_equal(collection.stats().total_count, 0)

    var result = collection.query(make_vector(0), n_results=2)
    assert_equal(result.ids[0][0], -1)
    assert_equal(result.ids[0][1], -1)


def test_compact_if_needed_validates_threshold() raises:
    var collection = Collection(DIMENSION, quantized=False)

    var negative_failed = False
    try:
        _ = collection.compact_if_needed(-0.01)
    except:
        negative_failed = True
    assert_true(negative_failed)

    var above_one_failed = False
    try:
        _ = collection.compact_if_needed(1.01)
    except:
        above_one_failed = True
    assert_true(above_one_failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
