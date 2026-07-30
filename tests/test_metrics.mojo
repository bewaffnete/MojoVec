from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from mojovec import Client, Collection, WAL_SYNC


def _check_inner_product(quantized: Bool) raises:
    var collection = Client().create_collection(
        "inner_product",
        dimension=2,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=quantized,
        metric="ip",
    )
    collection.add(
        [10, 20, 30],
        [
            Float32(1.0), Float32(0.0),
            Float32(0.0), Float32(1.0),
            Float32(2.0), Float32(2.0),
        ],
    )

    var results = collection.query(
        [Float32(1.0), Float32(0.0)],
        n_results=3,
    )
    assert_equal(collection.metric(), "ip")
    assert_equal(results.ids[0][0], 30)
    assert_equal(results.ids[0][1], 10)
    assert_equal(results.ids[0][2], 20)
    # Chroma-style IP distance is 1 - dot product.
    assert_almost_equal(results.distances[0][0], -1.0, atol=1e-5)
    assert_almost_equal(results.distances[0][1], 0.0, atol=1e-5)
    assert_almost_equal(results.distances[0][2], 1.0, atol=1e-5)


def test_inner_product_flat() raises:
    _check_inner_product(False)


def test_inner_product_sq8() raises:
    _check_inner_product(True)


def _check_cosine(quantized: Bool, path: String) raises:
    var collection = Client().create_collection(
        "cosine",
        dimension=2,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=quantized,
        metric="cosine",
    )
    var embeddings = [
        Float32(2.0), Float32(0.0),
        Float32(0.0), Float32(3.0),
        Float32(1.0), Float32(1.0),
    ]
    collection.add([10, 20, 30], embeddings)

    # Managed inputs are borrowed, not normalized in place.
    assert_almost_equal(embeddings[0], 2.0, atol=1e-6)
    assert_almost_equal(embeddings[3], 3.0, atol=1e-6)

    var results = collection.query(
        [Float32(10.0), Float32(0.0)],
        n_results=3,
    )
    assert_equal(collection.metric(), "cosine")
    assert_equal(results.ids[0][0], 10)
    assert_equal(results.ids[0][1], 30)
    assert_equal(results.ids[0][2], 20)
    assert_almost_equal(results.distances[0][0], 0.0, atol=1e-5)
    assert_almost_equal(
        results.distances[0][1],
        1.0 - 0.7071067811865476,
        atol=1e-5,
    )
    assert_almost_equal(results.distances[0][2], 1.0, atol=1e-5)

    # Metric semantics survive graph rebuilding and mmap serialization.
    collection.delete([20])
    _ = collection.compact()
    assert_equal(collection.metric(), "cosine")
    collection.save(path)
    var loaded = Collection.load(path, mmap_threshold_bytes=0)
    assert_true(loaded.is_memory_mapped())
    assert_equal(loaded.metric(), "cosine")
    var loaded_results = loaded.query(
        [Float32(5.0), Float32(0.0)],
        n_results=2,
    )
    assert_equal(loaded_results.ids[0][0], 10)
    assert_equal(loaded_results.ids[0][1], 30)
    assert_almost_equal(
        loaded_results.distances[0][0],
        0.0,
        atol=1e-5,
    )


def test_cosine_flat_round_trip() raises:
    _check_cosine(False, "/tmp/mojovec_metric_cosine_flat.bin")


def test_cosine_sq8_round_trip() raises:
    _check_cosine(True, "/tmp/mojovec_metric_cosine_sq8.bin")


def test_metric_validation_and_zero_cosine_vectors() raises:
    var invalid_metric_failed = False
    try:
        _ = Collection(2, metric="angular")
    except:
        invalid_metric_failed = True
    assert_true(invalid_metric_failed)

    var collection = Collection(2, quantized=False, metric="cosine")
    var zero_add_failed = False
    try:
        collection.add(
            [1],
            [Float32(0.0), Float32(0.0)],
        )
    except:
        zero_add_failed = True
    assert_true(zero_add_failed)
    assert_equal(collection.count(), 0)

    collection.add(
        [1],
        [Float32(1.0), Float32(0.0)],
    )
    var zero_query_failed = False
    try:
        _ = collection.query(
            [Float32(0.0), Float32(0.0)],
            n_results=1,
        )
    except:
        zero_query_failed = True
    assert_true(zero_query_failed)


def test_cosine_wal_recovery_preserves_metric() raises:
    var snapshot_path = "/tmp/mojovec_metric_cosine_wal.mojovec"
    var wal_path = "/tmp/mojovec_metric_cosine_wal.log"
    var empty_wal = open(wal_path, "w")
    empty_wal.close()

    var collection = Collection(
        2,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=False,
        name="cosine_wal",
        metric="cosine",
    )
    collection.add(
        [10],
        [Float32(2.0), Float32(0.0)],
    )
    collection.save(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    collection.add(
        [20],
        [Float32(1.0), Float32(1.0)],
    )
    collection.disable_wal()

    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        durability=WAL_SYNC,
        memory_mapped=False,
    )
    assert_equal(recovered.metric(), "cosine")
    var results = recovered.query(
        [Float32(5.0), Float32(0.0)],
        n_results=2,
    )
    assert_equal(results.ids[0][0], 10)
    assert_equal(results.ids[0][1], 20)
    assert_almost_equal(results.distances[0][0], 0.0, atol=1e-5)
    recovered.disable_wal()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
