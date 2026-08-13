import math
import os
import subprocess
import sys

import pytest

import mojovec


DIMENSION = 8
COUNT = 256


def vectors():
    return [
        [
            ((row * 17 + column * 29) % 251) / 251.0
            + (2.0 if column == row % DIMENSION else 0.0)
            for column in range(DIMENSION)
        ]
        for row in range(COUNT)
    ]


def test_ivfpq_python_lifecycle_and_stats():
    collection = mojovec.IVFPQCollection(
        DIMENSION,
        nlist=8,
        pq_subvectors=2,
        nprobe=4,
        metric="l2",
        name="compressed",
    )
    rows = vectors()
    ids = list(range(10_000, 10_000 + COUNT))

    assert not collection.is_trained()
    collection.train(rows)
    collection.add(ids, rows)

    assert collection.is_trained()
    assert collection.count() == COUNT
    assert collection.name() == "compressed"
    assert collection.metric() == "l2"
    assert collection.dimension() == DIMENSION
    assert collection.nlist() == 8
    assert collection.pq_subvectors() == 2
    assert collection.nprobe() == 4
    assert collection.stats() == {
        "count": COUNT,
        "dimension": DIMENSION,
        "nlist": 8,
        "M": 2,
        "nprobe": 4,
        "trained": True,
        "metric": "l2",
        "code_size_bytes": 2,
    }

    result = collection.query([rows[0]], n_results=3)
    assert result["ids"][0][0] == 10_000
    assert result["distances"][0][0] < 1e-4

    collection.set_nprobe(8)
    assert collection.nprobe() == 8


def test_ivfpq_python_auto_training_validation_and_duplicates():
    with pytest.raises(Exception, match="dimension"):
        mojovec.IVFPQCollection(0, nlist=8, pq_subvectors=2)
    with pytest.raises(Exception, match="divide dimension"):
        mojovec.IVFPQCollection(10, nlist=8, pq_subvectors=3)

    collection = mojovec.IVFPQCollection(
        DIMENSION, nlist=8, pq_subvectors=2
    )
    rows = vectors()
    ids = list(range(COUNT))
    collection.add(ids, rows)
    assert collection.is_trained()

    with pytest.raises(Exception, match="already exists"):
        collection.add([0], [rows[0]])
    with pytest.raises(Exception, match="nprobe"):
        collection.set_nprobe(9)
    with pytest.raises(ValueError, match="expected 8"):
        collection.add([999], [[1.0, 2.0]])

    with pytest.raises(Exception, match="at least max"):
        mojovec.IVFPQCollection(
            DIMENSION, nlist=8, pq_subvectors=2
        ).train(rows[:255])
    with pytest.raises(Exception, match="finite"):
        mojovec.IVFPQCollection(
            DIMENSION, nlist=8, pq_subvectors=2
        ).train([[math.nan] * DIMENSION for _ in range(COUNT)])


@pytest.mark.parametrize("metric", ["cosine", "ip"])
def test_ivfpq_python_similarity_metrics(metric):
    rows = vectors()
    query = rows[0].copy()
    if metric == "ip":
        rows[0] = [component * 20.0 for component in rows[0]]
        query = rows[0]
    collection = mojovec.IVFPQCollection(
        DIMENSION,
        nlist=8,
        pq_subvectors=2,
        nprobe=8,
        metric=metric,
    )
    collection.add(list(range(COUNT)), rows)
    result = collection.query(query, n_results=1)
    assert result["ids"] == [[0]]
    if metric == "cosine":
        assert result["distances"][0][0] < 1e-4
    else:
        assert result["distances"][0][0] < 0.0


def test_ivfpq_python_save_load(tmp_path):
    path = tmp_path / "index.ivfpq"
    rows = vectors()
    collection = mojovec.IVFPQCollection(
        DIMENSION,
        nlist=8,
        pq_subvectors=2,
        nprobe=4,
        metric="cosine",
        name="persistent",
    )
    collection.add(list(range(COUNT)), rows)
    expected = collection.query(rows[0], n_results=3)
    collection.save(path)

    loaded = mojovec.IVFPQCollection.load(path)
    assert loaded.name() == "persistent"
    assert loaded.metric() == "cosine"
    assert loaded.count() == COUNT
    assert loaded.nprobe() == 4
    assert loaded.query(rows[0], n_results=3) == expected

    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(Exception, match="checksum"):
        mojovec.IVFPQCollection.load(path)


def test_ivfpq_python_untrained_save_load(tmp_path):
    path = tmp_path / "empty.ivfpq"
    collection = mojovec.IVFPQCollection(
        DIMENSION,
        nlist=8,
        pq_subvectors=2,
        nprobe=4,
        metric="ip",
        name="empty",
    )
    collection.save(path)
    loaded = mojovec.IVFPQCollection.load(path)
    assert not loaded.is_trained()
    assert loaded.count() == 0
    assert loaded.metric() == "ip"
    assert loaded.nprobe() == 4


def test_ivfpq_python_repr_and_signature_are_public():
    collection = mojovec.IVFPQCollection(
        8, nlist=4, pq_subvectors=2, name="docs"
    )
    rendered = repr(collection)
    assert "IVFPQCollection" in rendered
    assert "name='docs'" in rendered
    assert mojovec.IVFPQCollection.__module__ == "mojovec"


def test_ivfpq_numpy_fast_path():
    np = pytest.importorskip("numpy")
    rows = np.asarray(vectors(), dtype=np.float32)
    ids = np.arange(COUNT, dtype=np.int64)
    collection = mojovec.IVFPQCollection(
        DIMENSION, nlist=8, pq_subvectors=2, nprobe=8
    )
    collection.train_numpy(rows)
    collection.add_numpy(ids, rows)
    result = collection.query_numpy(rows[:2], n_results=2)
    assert result["ids"][0][0] == 0
    assert result["ids"][1][0] == 1

    with pytest.raises(TypeError, match="float32"):
        collection.query_numpy(rows.astype(np.float64))
    with pytest.raises(TypeError, match="int64"):
        mojovec.IVFPQCollection(
            DIMENSION, nlist=8, pq_subvectors=2
        ).add_numpy(ids.astype(np.int32), rows)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="glibc heap validation is Linux-specific",
)
def test_ivfpq_fresh_process_shutdown_is_clean():
    script = """
import mojovec

rows = [
    [((row * 17 + column * 29) % 251) / 251.0 for column in range(8)]
    for row in range(256)
]
for _ in range(4):
    collection = mojovec.IVFPQCollection(
        8, nlist=8, pq_subvectors=2, nprobe=8
    )
    collection.add(list(range(256)), rows)
    assert collection.query(rows[0], n_results=3)["ids"][0]
    del collection
print(mojovec.native_backend())
"""
    environment = os.environ.copy()
    environment["MALLOC_CHECK_"] = "3"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
