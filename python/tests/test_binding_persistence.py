import pytest

import mojovec


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_snapshot_round_trip_preserves_collection_contract(
    tmp_path, quantized, metric
):
    path = tmp_path / f"{metric}-{quantized}.mojovec"
    collection = mojovec.Collection(
        dimension=3,
        M=8,
        ef_construction=48,
        ef_search=32,
        quantized=quantized,
        metric=metric,
        name="round-trip",
    )
    collection.add(
        [1, 2, 3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        metadatas=[{"rank": 1}, {"rank": 2}, {"rank": 3}],
        documents=["first", "second", "third"],
    )
    collection.delete([2])
    collection.save(path)

    loaded = mojovec.Collection.load(path, memory_mapped=False)

    assert loaded.name() == "round-trip"
    assert loaded.dimension() == 3
    assert loaded.metric() == metric
    assert loaded.is_quantized() is quantized
    assert loaded.storage_kind() == ("sq8" if quantized else "flat")
    assert loaded.count() == 2
    assert loaded.count_deleted() == 1
    assert loaded.get_metadata(1) == {"rank": 1}
    assert loaded.get_document(3) == "third"
    assert loaded.query([[1.0, 0.0, 0.0]], 2)["ids"][0] == [1, 3]


@pytest.mark.parametrize("quantized", [False, True])
def test_mmap_stays_active_for_reads_and_detaches_on_growth(tmp_path, quantized):
    path = tmp_path / f"mapped-{quantized}.mojovec"
    collection = mojovec.Collection(2, quantized=quantized)
    collection.add([1, 2], [[1.0, 0.0], [0.0, 1.0]])
    collection.save(path)

    mapped = mojovec.load(path, memory_mapped=True, mmap_threshold_bytes=0)
    assert mapped.is_memory_mapped() is True

    mapped.query([[1.0, 0.0]], 1)
    mapped.set_ef_search(32)
    mapped.delete([2])
    assert mapped.is_memory_mapped() is True

    mapped.add([3], [[-1.0, 0.0]])
    assert mapped.is_memory_mapped() is False
    assert mapped.count() == 2


def test_snapshot_reader_is_point_in_time_and_independent(tmp_path):
    path = tmp_path / "point-in-time.mojovec"
    writer = mojovec.Collection(2, quantized=False, name="writer")
    writer.add(
        [1],
        [[1.0, 0.0]],
        metadatas=[{"version": 1}],
        documents=["version one"],
    )

    reader = writer.snapshot(path, memory_mapped=True, mmap_threshold_bytes=0)
    writer.update(
        [1],
        [[0.0, 1.0]],
        metadatas=[{"version": 2}],
        documents=["version two"],
    )
    writer.add([2], [[1.0, 0.0]], documents=["writer only"])

    assert reader.is_memory_mapped() is True
    assert reader.count() == 1
    assert reader.get_metadata(1) == {"version": 1}
    assert reader.get_document(1) == "version one"
    assert writer.count() == 2
    assert writer.get_metadata(1) == {"version": 2}
    with pytest.raises(Exception):
        reader.get_document(2)


def test_snapshot_checksum_rejects_payload_corruption(tmp_path):
    path = tmp_path / "corrupt.mojovec"
    collection = mojovec.Collection(4)
    collection.add([1], [[1.0, 2.0, 3.0, 4.0]])
    collection.save(path)

    corrupted = bytearray(path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0x5A
    path.write_bytes(corrupted)

    with pytest.raises(Exception, match="checksum"):
        mojovec.load(path, memory_mapped=False)


@pytest.mark.parametrize("contents", [b"", b"not a mojovec snapshot"])
def test_loader_rejects_empty_and_invalid_files(tmp_path, contents):
    path = tmp_path / "invalid.mojovec"
    path.write_bytes(contents)

    with pytest.raises(Exception):
        mojovec.load(path)


def test_loader_rejects_missing_file(tmp_path):
    with pytest.raises(Exception):
        mojovec.load(tmp_path / "missing.mojovec")


def test_wal_replays_all_mutation_kinds_and_checkpoint_rotates(tmp_path):
    snapshot = tmp_path / "base.mojovec"
    wal = tmp_path / "changes.wal"
    collection = mojovec.Collection(
        2,
        quantized=True,
        metric="cosine",
        name="durable",
    )
    collection.save(snapshot)
    collection.enable_wal(wal, durability=mojovec.WAL_ASYNC)

    collection.add(
        [1],
        [[1.0, 0.0]],
        metadatas=[{"version": 1}],
        documents=["first"],
    )
    collection.upsert(
        [1, 2],
        [[0.9, 0.1], [0.0, 1.0]],
        metadatas=[{"version": 2}, {"version": 1}],
        documents=["updated", "second"],
    )
    collection.update(
        [2],
        [[0.1, 0.9]],
        documents=["second updated"],
    )
    collection.delete([1])
    assert collection.wal_sequence() == 4
    collection.flush_wal()
    collection.disable_wal()

    recovered = mojovec.recover(
        snapshot,
        wal,
        durability=mojovec.WAL_ASYNC,
        memory_mapped=False,
    )
    assert recovered.wal_enabled() is True
    assert recovered.wal_sequence() == 4
    assert recovered.count() == 1
    assert recovered.count_deleted() == 3
    assert recovered.get_metadata(2) == {"version": 1}
    assert recovered.get_document(2) == "second updated"

    recovered.checkpoint(snapshot)
    recovered.add([3], [[-1.0, 0.0]], documents=["after checkpoint"])
    assert recovered.wal_sequence() == 5
    recovered.flush_wal()
    recovered.disable_wal()

    second_recovery = mojovec.Collection.recover(
        snapshot,
        wal,
        durability=mojovec.WAL_SYNC,
        memory_mapped=False,
    )
    assert second_recovery.wal_sequence() == 5
    assert second_recovery.count() == 2
    assert second_recovery.get_document(3) == "after checkpoint"
    second_recovery.disable_wal()


def test_recovery_rejects_wal_from_another_collection(tmp_path):
    first_snapshot = tmp_path / "first.mojovec"
    second_snapshot = tmp_path / "second.mojovec"
    wal = tmp_path / "first.wal"

    first = mojovec.Collection(2, name="same-shape")
    second = mojovec.Collection(2, name="same-shape")
    first.save(first_snapshot)
    second.save(second_snapshot)
    first.enable_wal(wal, durability=mojovec.WAL_SYNC)
    first.add([1], [[1.0, 0.0]])
    first.disable_wal()

    with pytest.raises(Exception, match="different collection"):
        mojovec.recover(second_snapshot, wal, memory_mapped=False)


@pytest.mark.parametrize("durability", [0, 3, -1])
def test_enable_wal_rejects_unknown_durability(tmp_path, durability):
    collection = mojovec.Collection(2)
    collection.save(tmp_path / "base.mojovec")

    with pytest.raises(Exception):
        collection.enable_wal(tmp_path / "bad.wal", durability=durability)

    assert collection.wal_enabled() is False
