from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from mojovec import Collection, Metadata


comptime DIMENSION = 4


def vector(base: Float32) -> List[Float32]:
    return [base, base + 1.0, base + 2.0, base + 3.0]


def append_vector(mut values: List[Float32], base: Float32):
    for component in vector(base):
        values.append(component)


def make_collection(quantized: Bool) raises -> Collection:
    var collection = Collection(
        DIMENSION,
        M=8,
        ef_construction=48,
        ef_search=24,
        quantized=quantized,
        name="mapped_records",
    )
    var embeddings = List[Float32]()
    append_vector(embeddings, 1.0)
    append_vector(embeddings, 10.0)
    append_vector(embeddings, 20.0)

    var first = Metadata()
    first.set("category", "guide")
    var second = Metadata()
    second.set("category", "reference")
    var third = Metadata()
    third.set("category", "guide")
    var metadatas = List[Metadata]()
    metadatas.append(first^)
    metadatas.append(second^)
    metadatas.append(third^)

    collection.add(
        [10, 20, 30],
        embeddings,
        metadatas,
        [
            String("Mojo memory mapped index"),
            String("Vector database reference"),
            String("Unrelated document"),
        ],
    )
    return collection^


def check_mapped_round_trip(quantized: Bool, path: String) raises:
    var collection = make_collection(quantized)
    collection.save(path)

    # A zero threshold forces mmap even for this deliberately tiny fixture.
    # load() closes its FileHandle before returning; the mapping owns the file
    # pages independently and therefore remains queryable here.
    var mapped = Collection.load(path, mmap_threshold_bytes=0)
    assert_true(mapped.is_memory_mapped())
    assert_equal(mapped.name(), "mapped_records")
    assert_equal(mapped.is_quantized(), quantized)
    assert_equal(mapped.count(), 3)

    var vector_result = mapped.query(vector(1.0), n_results=1)
    assert_equal(vector_result.ids[0][0], 10)
    assert_equal(
        vector_result.metadatas[0][0].get_string("category"), "guide"
    )
    assert_equal(
        vector_result.documents[0][0], "Mojo memory mapped index"
    )

    var text_result = mapped.query(
        [String("memory mapped")], n_results=1
    )
    assert_equal(text_result.ids[0][0], 10)

    # Runtime scalar tuning and soft deletion do not touch mapped arrays.
    mapped.set_ef_search(32)
    mapped.delete([30])
    assert_true(mapped.is_memory_mapped())

    # The first graph mutation transparently copies mapped arrays to owned
    # heap memory. Callers keep using the normal Collection API.
    mapped.add([40], vector(40.0), ["new mapped record"])
    assert_false(mapped.is_memory_mapped())
    assert_equal(mapped.count(), 3)
    assert_equal(mapped.query(vector(40.0), n_results=1).ids[0][0], 40)


def test_flat_mapped_round_trip_and_detach() raises:
    check_mapped_round_trip(False, "/tmp/mojovec_test_mmap_flat.bin")


def test_sq8_mapped_round_trip_and_detach() raises:
    check_mapped_round_trip(True, "/tmp/mojovec_test_mmap_sq8.bin")


def test_mmap_policy_and_resave() raises:
    var collection = make_collection(False)
    collection.save("/tmp/mojovec_test_mmap_policy.bin")

    # The default 64 MiB threshold keeps small files on the heap.
    var automatic = Collection.load("/tmp/mojovec_test_mmap_policy.bin")
    assert_false(automatic.is_memory_mapped())

    # Explicit opt-out wins even when the threshold would force mmap.
    var copied = Collection.load(
        "/tmp/mojovec_test_mmap_policy.bin",
        memory_mapped=False,
        mmap_threshold_bytes=0,
    )
    assert_false(copied.is_memory_mapped())

    # A mapped collection can be saved again without first materializing it.
    var mapped = Collection.load(
        "/tmp/mojovec_test_mmap_policy.bin",
        mmap_threshold_bytes=0,
    )
    mapped.save("/tmp/mojovec_test_mmap_resaved.bin")
    assert_true(mapped.is_memory_mapped())
    var reloaded = Collection.load(
        "/tmp/mojovec_test_mmap_resaved.bin",
        mmap_threshold_bytes=0,
    )
    assert_true(reloaded.is_memory_mapped())
    assert_equal(reloaded.query(vector(10.0), n_results=1).ids[0][0], 20)

    var invalid_threshold = False
    try:
        _ = Collection.load(
            "/tmp/mojovec_test_mmap_policy.bin",
            mmap_threshold_bytes=-1,
        )
    except:
        invalid_threshold = True
    assert_true(invalid_threshold)


def check_empty_mapped_collection_can_grow(
    quantized: Bool, path: String
) raises:
    var empty = Collection(
        DIMENSION,
        M=8,
        quantized=quantized,
        name="empty",
    )
    empty.save(path)
    var mapped = Collection.load(
        path,
        mmap_threshold_bytes=0,
    )
    assert_true(mapped.is_memory_mapped())
    assert_equal(mapped.is_quantized(), quantized)
    assert_equal(mapped.count(), 0)

    mapped.add([1], vector(1.0))
    assert_false(mapped.is_memory_mapped())
    assert_equal(mapped.count(), 1)
    assert_equal(mapped.query(vector(1.0), n_results=1).ids[0][0], 1)


def test_empty_mapped_collections_can_grow() raises:
    check_empty_mapped_collection_can_grow(
        False, "/tmp/mojovec_test_mmap_empty_flat.bin"
    )
    check_empty_mapped_collection_can_grow(
        True, "/tmp/mojovec_test_mmap_empty_sq8.bin"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
