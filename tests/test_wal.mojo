from std.collections import List
from std.os import SEEK_SET, mkdir, rmdir
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from mojovec import (
    Collection,
    Metadata,
    WAL_ASYNC,
    WAL_SYNC,
)
from mojovec.io.fault_injection import (
    BATCH_FAULT_AFTER_PAYLOAD_PREPARE,
    BATCH_FAULT_AFTER_VECTOR_PREPARE,
    BATCH_FAULT_AFTER_WAL_APPEND,
    BATCH_FAULT_DURING_PAYLOAD_PREPARE,
    SNAPSHOT_FAULT_AFTER_CHECKPOINT_SNAPSHOT,
    SNAPSHOT_FAULT_AFTER_WAL_TEMPORARY,
)
from mojovec.io.atomic_file import (
    atomic_temporary_path,
    remove_file_best_effort,
)


comptime DIMENSION = 4


def vector(base: Float32) -> List[Float32]:
    return [base, base + 1.0, base + 2.0, base + 3.0]


def vectors(first: Float32, second: Float32) -> List[Float32]:
    var result = vector(first)
    for value in vector(second):
        result.append(value)
    return result^


def metadata(label: String, year: Int) -> Metadata:
    var value = Metadata()
    value.set("label", label)
    value.set("year", year)
    value.set("score", Float64(0.875))
    value.set("published", True)
    return value^


def empty_file(path: String) raises:
    var file = open(path, "w")
    file.close()


def create_temporary_directory(prefix: String) raises -> String:
    var path = atomic_temporary_path(prefix)
    mkdir(path)
    return path^


def remove_empty_directory(path: String) raises:
    rmdir(path)


def make_snapshot(snapshot_path: String, quantized: Bool = False) raises:
    var collection = Collection(
        DIMENSION,
        M=8,
        ef_construction=32,
        ef_search=16,
        quantized=quantized,
        name="wal-tests",
    )
    collection.save(snapshot_path)


def test_async_wal_recovers_batched_crud_and_payloads() raises:
    var snapshot_path = "test_wal_async.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_async.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_ASYNC)
    assert_true(collection.wal_enabled())
    assert_equal(collection.wal_sequence(), 0)

    var metadatas = List[Metadata]()
    metadatas.append(metadata("first", 2025))
    metadatas.append(metadata("second", 2026))
    collection.add(
        [10, 20],
        vectors(1.0, 10.0),
        metadatas,
        ["first document", "second document"],
    )
    # One public batch produces one WAL sequence, not one per vector.
    assert_equal(collection.wal_sequence(), 1)

    var replacements = List[Metadata]()
    replacements.append(metadata("updated", 2030))
    collection.upsert(
        [10],
        vector(100.0),
        replacements,
        ["updated document"],
    )
    collection.delete([20])
    assert_equal(collection.wal_sequence(), 3)
    collection.flush_wal()
    collection.disable_wal()
    assert_false(collection.wal_enabled())

    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        durability=WAL_ASYNC,
        memory_mapped=False,
    )
    assert_true(recovered.wal_enabled())
    assert_equal(recovered.wal_sequence(), 3)
    assert_equal(recovered.count(), 1)
    assert_equal(recovered.get_metadata(10).get_string("label"), "updated")
    assert_equal(recovered.get_metadata(10).get_int("year"), 2030)
    assert_equal(recovered.get_document(10), "updated document")
    var nearest = recovered.query(vector(100.0), n_results=1)
    assert_equal(nearest.ids[0][0], 10)
    recovered.disable_wal()


def test_sync_checkpoint_rotates_wal_after_snapshot() raises:
    var snapshot_path = "test_wal_checkpoint.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_checkpoint.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    collection.add([1], vector(1.0))
    collection.checkpoint(snapshot_path)
    assert_equal(collection.wal_sequence(), 1)
    collection.add([2], vector(20.0))
    assert_equal(collection.wal_sequence(), 2)
    collection.disable_wal()

    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        durability=WAL_SYNC,
        memory_mapped=False,
    )
    assert_equal(recovered.count(), 2)
    assert_equal(recovered.wal_sequence(), 2)
    assert_equal(recovered.query(vector(1.0), n_results=1).ids[0][0], 1)
    assert_equal(recovered.query(vector(20.0), n_results=1).ids[0][0], 2)
    recovered.disable_wal()


def test_checkpoint_fault_after_snapshot_keeps_wal_recoverable() raises:
    var snapshot_path = "test_wal_checkpoint_fault.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_checkpoint_fault.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    collection.add([1], vector(1.0))

    var checkpoint_failed = False
    try:
        collection._checkpoint_with_fault(
            snapshot_path,
            SNAPSHOT_FAULT_AFTER_CHECKPOINT_SNAPSHOT,
        )
    except:
        checkpoint_failed = True
    assert_true(checkpoint_failed)
    collection.disable_wal()

    # The snapshot covers sequence 1 and the unrotated WAL still contains it.
    # Recovery must skip that frame instead of applying it twice.
    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        durability=WAL_SYNC,
        memory_mapped=False,
    )
    assert_equal(recovered.count(), 1)
    assert_equal(recovered.wal_sequence(), 1)
    assert_equal(recovered.query(vector(1.0), n_results=1).ids[0][0], 1)
    recovered.disable_wal()


def test_failed_wal_rotation_removes_temporary_file() raises:
    var directory = create_temporary_directory(
        "/tmp/mojovec_wal_rotation_cleanup"
    )
    var snapshot_path = String(directory, "/snapshot.mojovec")
    var wal_path = String(directory, "/collection.wal")
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    collection.add([1], vector(1.0))

    var failed = False
    try:
        collection._checkpoint_with_fault(
            snapshot_path,
            SNAPSHOT_FAULT_AFTER_WAL_TEMPORARY,
        )
    except:
        failed = True
    assert_true(failed)
    collection.disable_wal()

    remove_file_best_effort(snapshot_path)
    remove_file_best_effort(wal_path)
    remove_empty_directory(directory)


def test_recovery_skips_records_already_in_snapshot() raises:
    var snapshot_path = "test_wal_idempotent.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_idempotent.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    collection.add([7], vector(7.0))
    # Simulates a crash after the new snapshot is durable but before WAL
    # rotation. Recovery must not apply sequence 1 a second time.
    collection.save(snapshot_path)
    collection.disable_wal()

    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        memory_mapped=False,
    )
    assert_equal(recovered.count(), 1)
    assert_equal(recovered.wal_sequence(), 1)
    assert_equal(recovered.query(vector(7.0), n_results=1).ids[0][0], 7)
    recovered.disable_wal()


def test_recovery_discards_incomplete_tail_before_resuming() raises:
    var snapshot_path = "test_wal_tail.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_tail.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_ASYNC)
    collection.add([1], vector(1.0))
    collection.flush_wal()
    collection.disable_wal()

    # A process may die while writing the next frame. A short uncommitted
    # tail is ignored and physically truncated before appends resume.
    var file = open(wal_path, "a")
    var incomplete_tail = [UInt8(1), UInt8(2), UInt8(3), UInt8(4), UInt8(5)]
    file.write_all(incomplete_tail)
    file.close()

    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        durability=WAL_ASYNC,
        memory_mapped=False,
    )
    assert_equal(recovered.count(), 1)
    recovered.add([2], vector(20.0))
    assert_equal(recovered.wal_sequence(), 2)
    recovered.flush_wal()
    recovered.disable_wal()

    var recovered_again = Collection.recover(
        snapshot_path,
        wal_path,
        memory_mapped=False,
    )
    assert_equal(recovered_again.count(), 2)
    assert_equal(recovered_again.wal_sequence(), 2)
    recovered_again.disable_wal()


def test_failed_mutation_does_not_enter_wal() raises:
    var snapshot_path = "test_wal_validation.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_validation.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    collection.add([1], vector(1.0))
    var failed = False
    try:
        collection.add([1], vector(2.0))
    except:
        failed = True
    assert_true(failed)
    assert_equal(collection.wal_sequence(), 1)
    collection.disable_wal()

    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        memory_mapped=False,
    )
    assert_equal(recovered.count(), 1)
    assert_equal(recovered.wal_sequence(), 1)
    recovered.disable_wal()


def test_faulted_payload_batches_preserve_live_and_wal_state() raises:
    var snapshot_path = "test_wal_atomic_batch.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_atomic_batch.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path, memory_mapped=False)
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    var initial_metadatas = List[Metadata]()
    initial_metadatas.append(metadata("original-one", 2025))
    initial_metadatas.append(metadata("original-two", 2026))
    collection.add(
        [1, 2],
        vectors(1.0, 20.0),
        initial_metadatas,
        ["original first document", "original second document"],
    )
    assert_equal(collection.wal_sequence(), 1)

    var replacement_metadatas = List[Metadata]()
    replacement_metadatas.append(metadata("replacement-one", 2030))
    replacement_metadatas.append(metadata("new-three", 2031))
    var replacement_documents = List[String]()
    replacement_documents.append("replacement first document")
    replacement_documents.append("new third document")
    for fault_point in [
        BATCH_FAULT_AFTER_VECTOR_PREPARE,
        BATCH_FAULT_DURING_PAYLOAD_PREPARE,
        BATCH_FAULT_AFTER_PAYLOAD_PREPARE,
        BATCH_FAULT_AFTER_WAL_APPEND,
    ]:
        var failed = False
        try:
            collection._upsert_with_fault(
                [1, 3],
                vectors(100.0, 300.0),
                replacement_metadatas,
                replacement_documents,
                fault_point,
            )
        except:
            failed = True
        assert_true(failed)

        # Neither the managed payloads nor the graph publish a partial row.
        assert_equal(collection.count(), 2)
        assert_equal(collection.count_deleted(), 0)
        assert_equal(collection.stats().total_count, 2)
        assert_equal(
            collection.get_metadata(1).get_string("label"),
            "original-one",
        )
        assert_equal(
            collection.get_document(1),
            "original first document",
        )
        assert_equal(
            collection.query(vector(1.0), n_results=1).ids[0][0],
            1,
        )
        var new_id_visible = True
        try:
            _ = collection.get_metadata(3)
        except:
            new_id_visible = False
        assert_false(new_id_visible)
        assert_equal(collection.wal_sequence(), 1)

    # The after-WAL rollback leaves the writer usable for the next batch.
    collection.upsert(
        [1, 3],
        vectors(100.0, 300.0),
        replacement_metadatas,
        replacement_documents,
    )
    assert_equal(collection.wal_sequence(), 2)
    collection.disable_wal()

    # Recovery observes only the successful retry, never a faulted frame.
    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        durability=WAL_SYNC,
        memory_mapped=False,
    )
    assert_equal(recovered.count(), 3)
    assert_equal(recovered.wal_sequence(), 2)
    assert_equal(
        recovered.get_metadata(1).get_string("label"),
        "replacement-one",
    )
    assert_equal(recovered.get_document(3), "new third document")
    recovered.disable_wal()


def test_sq8_collection_recovers_through_same_wal_api() raises:
    var snapshot_path = "test_wal_sq8.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_sq8.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path, quantized=True)

    var collection = Collection.load(snapshot_path)
    assert_true(collection.is_quantized())
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    collection.add([10, 20], vectors(1.0, 20.0))
    collection.disable_wal()

    var recovered = Collection.recover(
        snapshot_path,
        wal_path,
        memory_mapped=False,
    )
    assert_true(recovered.is_quantized())
    assert_equal(recovered.count(), 2)
    assert_equal(recovered.wal_sequence(), 1)
    assert_equal(recovered.query(vector(1.0), n_results=1).ids[0][0], 10)
    recovered.disable_wal()


def test_committed_checksum_corruption_is_rejected() raises:
    var snapshot_path = "test_wal_checksum.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_checksum.log"
    empty_file(wal_path)
    make_snapshot(snapshot_path)

    var collection = Collection.load(snapshot_path)
    collection.enable_wal(wal_path, durability=WAL_SYNC)
    collection.add([1], vector(1.0))
    collection.disable_wal()

    # Header is seven Int fields plus the nine-byte collection name. Change one
    # payload byte while leaving the final commit marker intact.
    var payload_offset = 7 * 8 + String("wal-tests").byte_length() + 8 * 8
    var file = open(wal_path, "rw")
    _ = file.seek(UInt64(payload_offset), SEEK_SET)
    var original = file.read_bytes(1)
    _ = file.seek(UInt64(payload_offset), SEEK_SET)
    var corrupted = [original[0] ^ UInt8(0xFF)]
    file.write_all(corrupted)
    file.close()

    var failed = False
    try:
        _ = Collection.recover(
            snapshot_path,
            wal_path,
            memory_mapped=False,
        )
    except:
        failed = True
    assert_true(failed)


def test_wal_is_bound_to_one_collection_incarnation() raises:
    var first_snapshot = "test_wal_identity_first.mojovec"
    var second_snapshot = "test_wal_identity_second.mojovec"
    var wal_path = "/tmp/mojovec_test_wal_identity.log"
    empty_file(wal_path)
    make_snapshot(first_snapshot)
    make_snapshot(second_snapshot)

    var first = Collection.load(first_snapshot)
    first.enable_wal(wal_path, durability=WAL_SYNC)
    first.add([1], vector(1.0))
    first.disable_wal()

    # Name, dimension, and storage kind intentionally match. The persistent
    # collection identity still prevents replaying a valid WAL into another
    # collection created separately.
    var failed = False
    try:
        _ = Collection.recover(
            second_snapshot,
            wal_path,
            memory_mapped=False,
        )
    except:
        failed = True
    assert_true(failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
