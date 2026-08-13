from std.collections import List
from std.memory import alloc
from std.memory.span import Span
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from mojovec import Client, CollectionIVFPQ
from mojovec.core.types import METRIC_L2
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_ivf_pq import IndexIVFPQ
from mojovec.io.atomic_file import remove_file_best_effort


comptime DIMENSION = 8
comptime VECTOR_COUNT = 256


def make_ids(first: Int = 10_000) -> List[Int]:
    var ids = List[Int](capacity=VECTOR_COUNT)
    for index in range(VECTOR_COUNT):
        ids.append(first + index)
    return ids^


def make_vectors() -> List[Float32]:
    var vectors = List[Float32](capacity=VECTOR_COUNT * DIMENSION)
    for row in range(VECTOR_COUNT):
        for column in range(DIMENSION):
            var value = Float32((row * 17 + column * 29) % 251) / 251.0
            if column == row % DIMENSION:
                value += 2.0
            vectors.append(value)
    return vectors^


def first_vector(vectors: List[Float32]) -> List[Float32]:
    var query = List[Float32](capacity=DIMENSION)
    for index in range(DIMENSION):
        query.append(vectors[index])
    return query^


def test_ivfpq_managed_lifecycle_and_stats() raises:
    var collection = CollectionIVFPQ(
        DIMENSION,
        nlist=8,
        M=2,
        nprobe=4,
        name="compressed",
        metric="l2",
    )
    assert_false(collection.is_trained())
    assert_equal(collection.name(), "compressed")
    assert_equal(collection.dimension(), DIMENSION)
    assert_equal(collection.metric(), "l2")

    var ids = make_ids()
    var vectors = make_vectors()
    collection.train(vectors)
    assert_true(collection.is_trained())
    collection.add(ids, vectors)

    var stats = collection.stats()
    assert_equal(stats.count, VECTOR_COUNT)
    assert_equal(stats.nlist, 8)
    assert_equal(stats.M, 2)
    assert_equal(stats.nprobe, 4)
    assert_equal(stats.code_size_bytes, 2)

    var results = collection.query(first_vector(vectors), n_results=5)
    assert_equal(results.ids[0][0], 10_000)
    assert_true(results.distances[0][0] < 1e-4)
    assert_equal(len(results.metadatas), 0)
    assert_equal(len(results.documents), 0)
    assert_equal(len(results.scores), 0)

    collection.set_nprobe(8)
    assert_equal(collection.nprobe(), 8)


def test_ivfpq_auto_train_and_duplicate_ids_are_rejected() raises:
    var collection = CollectionIVFPQ(DIMENSION, nlist=8, M=2)
    var ids = make_ids()
    var vectors = make_vectors()
    collection.add(ids, vectors)
    assert_true(collection.is_trained())
    assert_equal(collection.count(), VECTOR_COUNT)

    var duplicate_failed = False
    try:
        collection.add([10_000], first_vector(vectors))
    except:
        duplicate_failed = True
    assert_true(duplicate_failed)
    assert_equal(collection.count(), VECTOR_COUNT)


def test_ivfpq_validates_configuration_and_training_size() raises:
    var invalid_shape = False
    try:
        _ = CollectionIVFPQ(10, nlist=8, M=3)
    except:
        invalid_shape = True
    assert_true(invalid_shape)

    var invalid_probe = False
    try:
        _ = CollectionIVFPQ(8, nlist=4, M=2, nprobe=5)
    except:
        invalid_probe = True
    assert_true(invalid_probe)

    var collection = CollectionIVFPQ(8, nlist=4, M=2)
    var too_small = List[Float32](length=255 * 8, fill=0.25)
    var training_failed = False
    try:
        collection.train(too_small)
    except:
        training_failed = True
    assert_true(training_failed)
    assert_false(collection.is_trained())


def test_ivfpq_cosine_and_inner_product_metrics() raises:
    var ids = make_ids()
    var vectors = make_vectors()
    var query = first_vector(vectors)

    var cosine = CollectionIVFPQ(
        DIMENSION, nlist=8, M=2, nprobe=8, metric="cosine"
    )
    cosine.add(ids, vectors)
    var cosine_results = cosine.query(query, n_results=1)
    assert_equal(cosine_results.ids[0][0], 10_000)
    assert_true(cosine_results.distances[0][0] < 1e-4)

    # Give the first vector a uniquely dominant inner product.
    for component in range(DIMENSION):
        vectors[component] *= 20.0
        query[component] = vectors[component]
    var inner_product = CollectionIVFPQ(
        DIMENSION, nlist=8, M=2, nprobe=8, metric="ip"
    )
    inner_product.add(ids, vectors)
    var ip_results = inner_product.query(query, n_results=1)
    assert_equal(ip_results.ids[0][0], 10_000)
    assert_true(ip_results.distances[0][0] < 0.0)


def test_ivfpq_save_load_round_trip() raises:
    var path = String("/tmp/mojovec_test_ivfpq.bin")
    remove_file_best_effort(path)
    var client = Client()
    var collection = client.create_ivfpq_collection(
        "persistent",
        DIMENSION,
        nlist=8,
        M=2,
        nprobe=4,
        metric="cosine",
    )
    var ids = make_ids(first=50_000)
    var vectors = make_vectors()
    collection.add(ids, vectors)
    var expected = collection.query(first_vector(vectors), n_results=3)
    collection.save(path)

    var loaded = CollectionIVFPQ.load(path)
    assert_equal(loaded.name(), "persistent")
    assert_equal(loaded.metric(), "cosine")
    assert_equal(loaded.count(), VECTOR_COUNT)
    assert_equal(loaded.nprobe(), 4)
    var actual = loaded.query(first_vector(vectors), n_results=3)
    for rank in range(3):
        assert_equal(actual.ids[0][rank], expected.ids[0][rank])
        assert_almost_equal(
            actual.distances[0][rank],
            expected.distances[0][rank],
            atol=1e-6,
        )
    remove_file_best_effort(path)


def test_ivfpq_untrained_snapshot_round_trip() raises:
    var path = String("/tmp/mojovec_test_ivfpq_untrained.bin")
    remove_file_best_effort(path)
    var collection = CollectionIVFPQ(
        DIMENSION,
        nlist=8,
        M=2,
        nprobe=4,
        name="empty",
        metric="ip",
    )
    collection.save(path)
    var loaded = CollectionIVFPQ.load(path)
    assert_false(loaded.is_trained())
    assert_equal(loaded.count(), 0)
    assert_equal(loaded.metric(), "ip")
    assert_equal(loaded.nprobe(), 4)
    remove_file_best_effort(path)


def test_ivfpq_low_level_filter_excludes_internal_ids() raises:
    var vectors = make_vectors()
    var index = IndexIVFPQ[IndexFlat](
        IndexFlat(DIMENSION, METRIC_L2), DIMENSION, 8, 2, METRIC_L2
    )
    index.train(Span[Float32](vectors))
    index.add(Span[Float32](vectors))
    index.nprobe = 8

    var query = first_vector(vectors)
    var distances = List[Float32](unsafe_uninit_length=1)
    var labels = List[Int](unsafe_uninit_length=1)
    var exclusion = List[UInt8](length=VECTOR_COUNT, fill=0)
    exclusion[0] = 1
    var distance_span = Span[mut=True, Float32](distances)
    var label_span = Span[mut=True, Int](labels)
    index.search(
        Span[Float32](query),
        1,
        distance_span,
        label_span,
        Span[UInt8](exclusion),
    )
    assert_true(labels[0] != 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
