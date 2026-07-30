from std.collections import List
from std.algorithm import parallelize
from std.os import SEEK_END, SEEK_SET
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from mojovec import Collection, Metadata
from mojovec.io.atomic_file import atomic_replace
from mojovec.io.fault_injection import (
    SNAPSHOT_FAULT_AFTER_HEADER,
    SNAPSHOT_FAULT_AFTER_PAYLOAD,
    SNAPSHOT_FAULT_AFTER_PUBLISH,
    SNAPSHOT_FAULT_BEFORE_PUBLISH,
)


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
    assert_equal(vector_result.metadatas[0][0].get_string("category"), "guide")
    assert_equal(vector_result.documents[0][0], "Mojo memory mapped index")

    var text_result = mapped.query([String("memory mapped")], n_results=1)
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
    # Atomic rename keeps the old inode alive, so overwriting the same path is
    # safe even while this collection continues to query its existing mmap.
    mapped.save("/tmp/mojovec_test_mmap_policy.bin")
    assert_true(mapped.is_memory_mapped())
    assert_equal(mapped.query(vector(1.0), n_results=1).ids[0][0], 10)
    var reloaded = Collection.load(
        "/tmp/mojovec_test_mmap_policy.bin",
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


def test_point_in_time_snapshots_survive_new_publications() raises:
    var writer = make_collection(False)
    var first = writer.snapshot(
        "/tmp/mojovec_test_snapshot.bin",
        mmap_threshold_bytes=0,
    )
    assert_true(first.is_memory_mapped())
    assert_equal(first.count(), 3)
    assert_equal(first.query(vector(1.0), n_results=1).ids[0][0], 10)

    writer.upsert([10], vector(100.0), ["updated document"])
    writer.add([40], vector(40.0), ["new document"])
    var second = writer.snapshot(
        "/tmp/mojovec_test_snapshot.bin",
        mmap_threshold_bytes=0,
    )

    # The first reader remains pinned to the previous inode and state.
    assert_true(first.is_memory_mapped())
    assert_equal(first.count(), 3)
    assert_equal(first.query(vector(1.0), n_results=1).ids[0][0], 10)
    assert_equal(first.get_document(10), "Mojo memory mapped index")

    # Newly acquired readers observe the atomically published generation.
    assert_true(second.is_memory_mapped())
    assert_equal(second.count(), 4)
    assert_equal(second.query(vector(40.0), n_results=1).ids[0][0], 40)
    assert_equal(second.get_document(10), "updated document")


def test_queries_continue_during_atomic_publication() raises:
    var writer = make_collection(False)
    var reader = writer.snapshot(
        "/tmp/mojovec_test_concurrent_snapshot.bin",
        mmap_threshold_bytes=0,
    )
    writer.add([40], vector(40.0), ["new document"])
    var failures = List[Int](length=2, fill=0)

    @parameter
    def work(worker: Int):
        try:
            if worker == 0:
                for _ in range(100):
                    var result = reader.query(vector(1.0), n_results=1)
                    if result.ids[0][0] != 10:
                        failures[0] = 1
            else:
                writer.save("/tmp/mojovec_test_concurrent_snapshot.bin")
        except:
            failures[worker] = 1

    parallelize[work](2, 2)
    assert_equal(failures[0], 0)
    assert_equal(failures[1], 0)
    assert_true(reader.is_memory_mapped())
    assert_equal(reader.count(), 3)

    var latest = Collection.load(
        "/tmp/mojovec_test_concurrent_snapshot.bin",
        mmap_threshold_bytes=0,
    )
    assert_equal(latest.count(), 4)
    assert_equal(latest.query(vector(40.0), n_results=1).ids[0][0], 40)


def test_failed_atomic_replacement_preserves_destination() raises:
    var writer = make_collection(False)
    writer.save("/tmp/mojovec_test_atomic_failure.bin")

    var replacement_failed = False
    try:
        atomic_replace(
            "/tmp/mojovec_missing_atomic_source_for_test",
            "/tmp/mojovec_test_atomic_failure.bin",
        )
    except:
        replacement_failed = True
    assert_true(replacement_failed)

    var preserved = Collection.load(
        "/tmp/mojovec_test_atomic_failure.bin",
        mmap_threshold_bytes=0,
    )
    assert_equal(preserved.count(), 3)
    assert_equal(preserved.query(vector(1.0), n_results=1).ids[0][0], 10)


def test_snapshot_checksum_rejects_payload_and_trailer_corruption() raises:
    var path = "/tmp/mojovec_test_snapshot_checksum.bin"
    var writer = make_collection(False)
    writer.save(path)

    # A valid checksummed snapshot remains eligible for mmap.
    var valid = Collection.load(path, mmap_threshold_bytes=0)
    assert_true(valid.is_memory_mapped())
    assert_equal(valid.count(), 3)

    # Corrupt a payload byte. Validation happens before parsing or mapping it.
    var payload_file = open(path, "rw")
    _ = payload_file.seek(32, SEEK_SET)
    var original_payload = payload_file.read_bytes(1)
    _ = payload_file.seek(32, SEEK_SET)
    var corrupted_payload = [original_payload[0] ^ UInt8(0xFF)]
    payload_file.write_all(corrupted_payload)
    payload_file.close()

    var payload_failed = False
    try:
        _ = Collection.load(path, mmap_threshold_bytes=0)
    except:
        payload_failed = True
    assert_true(payload_failed)

    writer.save(path)
    var trailer_file = open(path, "rw")
    var file_size = Int(trailer_file.seek(0, SEEK_END))
    _ = trailer_file.seek(UInt64(file_size - 1), SEEK_SET)
    var original_trailer = trailer_file.read_bytes(1)
    _ = trailer_file.seek(UInt64(file_size - 1), SEEK_SET)
    var corrupted_trailer = [original_trailer[0] ^ UInt8(0xFF)]
    trailer_file.write_all(corrupted_trailer)
    trailer_file.close()

    var trailer_failed = False
    try:
        _ = Collection.load(path, mmap_threshold_bytes=0)
    except:
        trailer_failed = True
    assert_true(trailer_failed)


def test_snapshot_fault_injection_preserves_atomic_generations() raises:
    var path = "/tmp/mojovec_test_snapshot_faults.bin"
    var writer = make_collection(False)
    writer.save(path)
    writer.add([40], vector(40.0))

    for fault_point in [
        SNAPSHOT_FAULT_AFTER_HEADER,
        SNAPSHOT_FAULT_AFTER_PAYLOAD,
        SNAPSHOT_FAULT_BEFORE_PUBLISH,
    ]:
        var failed = False
        try:
            writer._save_with_fault(path, fault_point)
        except:
            failed = True
        assert_true(failed)

        # Every pre-publication failure leaves the prior complete generation.
        var preserved = Collection.load(path, mmap_threshold_bytes=0)
        assert_equal(preserved.count(), 3)
        assert_equal(
            preserved.query(vector(1.0), n_results=1).ids[0][0],
            10,
        )

    var post_publish_failed = False
    try:
        writer._save_with_fault(path, SNAPSHOT_FAULT_AFTER_PUBLISH)
    except:
        post_publish_failed = True
    assert_true(post_publish_failed)

    # Rename installs only a complete checksummed generation.
    var published = Collection.load(path, mmap_threshold_bytes=0)
    assert_equal(published.count(), 4)
    assert_equal(
        published.query(vector(40.0), n_results=1).ids[0][0],
        40,
    )


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
