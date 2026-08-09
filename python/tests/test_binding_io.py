import csv
import json
from pathlib import Path
import sys
import types

import pytest

import mojovec
from mojovec.io import (
    ImportBatch,
    iter_file_batches,
    read_fvecs,
    read_ivecs,
)


np = pytest.importorskip("numpy")


def _assert_two_record_collection(collection):
    query = [1.0] + [0.0] * (collection.dimension() - 1)
    result = collection.query([query], n_results=1)
    assert result["ids"] == [[10]]
    assert result["documents"] == [["first document"]]


def test_csv_reader_ingests_json_vector_documents_and_metadata(tmp_path):
    path = tmp_path / "records.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["id", "embedding", "text", "category"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": 10,
                "embedding": "[1, 0, 0]",
                "text": "first document",
                "category": "alpha",
            }
        )
        writer.writerow(
            {
                "id": 20,
                "embedding": "[0, 1, 0]",
                "text": "second document",
                "category": "beta",
            }
        )

    collection = mojovec.Collection(3, quantized=False)
    imported = collection.add_from(
        path,
        id_column="id",
        embedding_column="embedding",
        document_column="text",
        metadata_columns=["category"],
        batch_size=1,
    )

    assert imported == 2
    assert collection.count() == 2
    assert collection.get_metadata(10) == {"category": "alpha"}
    _assert_two_record_collection(collection)


def test_tsv_reader_accepts_one_scalar_column_per_dimension(tmp_path):
    path = tmp_path / "records.tsv"
    path.write_text(
        "id\tx\ty\tz\ttext\n"
        "10\t1\t0\t0\tfirst document\n"
        "20\t0\t1\t0\tsecond document\n",
        encoding="utf-8",
    )
    collection = mojovec.Collection(3, quantized=False)
    collection.add_from(
        path,
        id_column="id",
        embedding_columns=["x", "y", "z"],
        document_column="text",
    )
    _assert_two_record_collection(collection)


@pytest.mark.parametrize("file_format", ["json", "jsonl"])
def test_json_readers_preserve_typed_payloads(tmp_path, file_format):
    rows = [
        {
            "id": 10,
            "embedding": [1.0, 0.0, 0.0],
            "text": "first document",
            "year": 2026,
            "published": True,
        },
        {
            "id": 20,
            "embedding": [0.0, 1.0, 0.0],
            "text": "second document",
            "year": 2025,
            "published": False,
        },
    ]
    path = tmp_path / f"records.{file_format}"
    if file_format == "json":
        path.write_text(json.dumps({"data": rows}), encoding="utf-8")
    else:
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    collection = mojovec.Collection(3, quantized=False)
    collection.add_from(
        path,
        id_column="id",
        document_column="text",
        metadata_columns=["year", "published"],
        batch_size=1,
    )
    assert collection.get_metadata(10) == {
        "year": 2026,
        "published": True,
    }
    _assert_two_record_collection(collection)


def test_npy_reader_generates_ids_and_upserts_through_fast_path(tmp_path):
    path = tmp_path / "vectors.npy"
    np.save(
        path,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    collection = mojovec.Collection(2, quantized=False)
    assert collection.upsert_from(path, id_start=100, batch_size=1) == 2
    result = collection.query([[1.0, 0.0]], n_results=1)
    assert result["ids"] == [[100]]


def test_npz_reader_uses_named_arrays_and_payloads(tmp_path):
    path = tmp_path / "records.npz"
    np.savez(
        path,
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        record_ids=np.asarray([10, 20], dtype=np.int64),
        texts=np.asarray(["first document", "second document"]),
        year=np.asarray([2026, 2025], dtype=np.int64),
    )
    collection = mojovec.Collection(2, quantized=False)
    collection.add_from(
        path,
        embeddings_key="vectors",
        ids_key="record_ids",
        documents_key="texts",
        metadata_columns=["year"],
        batch_size=1,
    )
    assert collection.get_metadata(10) == {"year": 2026}
    _assert_two_record_collection(collection)


def _write_vecs(path: Path, values, *, integer: bool):
    values = np.asarray(values, dtype=np.int32 if integer else np.float32)
    header = np.full((len(values), 1), values.shape[1], dtype=np.int32)
    if integer:
        rows = np.concatenate([header, values], axis=1).astype("<i4")
    else:
        rows = np.empty((len(values), values.shape[1] + 1), dtype="<i4")
        rows[:, 0] = values.shape[1]
        rows[:, 1:] = values.astype("<f4").view("<i4")
    rows.tofile(path)


def test_fvecs_and_ivecs_readers_validate_and_ingest(tmp_path):
    fvecs = tmp_path / "vectors.fvecs"
    ivecs = tmp_path / "neighbors.ivecs"
    _write_vecs(fvecs, [[1.0, 0.0], [0.0, 1.0]], integer=False)
    _write_vecs(ivecs, [[4, 8], [15, 16]], integer=True)

    assert read_fvecs(fvecs).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert read_ivecs(ivecs).tolist() == [[4, 8], [15, 16]]

    collection = mojovec.Collection(2, quantized=False)
    assert collection.add_from(fvecs, id_start=50, batch_size=1) == 2
    assert collection.query([[1.0, 0.0]], 1)["ids"] == [[50]]
    with pytest.raises(Exception, match="already exists"):
        collection.add_from(fvecs, id_start=50, batch_size=1)


def test_malformed_vecs_and_wrong_dimensions_are_rejected(tmp_path):
    malformed = tmp_path / "bad.fvecs"
    np.asarray([2, 0, 0, 3, 0, 0], dtype="<i4").tofile(malformed)
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        read_fvecs(malformed)

    path = tmp_path / "wrong.npy"
    np.save(path, np.zeros((2, 4), dtype=np.float32))
    collection = mojovec.Collection(3)
    with pytest.raises(ValueError, match="dimension is 4; expected 3"):
        collection.add_from(path)
    assert collection.count() == 0


def test_file_readers_reject_fractional_and_overflowing_ids(tmp_path):
    path = tmp_path / "bad_ids.jsonl"
    path.write_text(
        json.dumps({"id": 1.5, "embedding": [1.0, 0.0]}) + "\n",
        encoding="utf-8",
    )
    collection = mojovec.Collection(2)
    with pytest.raises(ValueError, match="must be an integer"):
        collection.add_from(path, id_column="id")

    vectors = tmp_path / "vectors.npy"
    np.save(vectors, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="outside the int64 range"):
        collection.add_from(vectors, id_start=(1 << 63) - 1)


def test_parquet_reader_imports_columns_in_batches(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": [10, 20],
            "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "text": ["first document", "second document"],
            "year": [2026, 2025],
        }
    )
    pq.write_table(table, path)

    collection = mojovec.Collection(3, quantized=False)
    collection.add_from(
        path,
        id_column="id",
        document_column="text",
        metadata_columns=["year"],
        batch_size=1,
    )
    assert collection.get_metadata(10) == {"year": 2026}
    _assert_two_record_collection(collection)


@pytest.mark.parametrize("stream", [False, True])
def test_arrow_file_and_stream_readers(tmp_path, stream):
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / ("records.stream.arrow" if stream else "records.arrow")
    batch = pa.record_batch(
        [
            pa.array([10, 20]),
            pa.array([[1.0, 0.0], [0.0, 1.0]]),
            pa.array(["first document", "second document"]),
        ],
        names=["id", "embedding", "text"],
    )
    with pa.OSFile(str(path), "wb") as sink:
        writer_factory = pa.ipc.new_stream if stream else pa.ipc.new_file
        with writer_factory(sink, batch.schema) as writer:
            writer.write_batch(batch)

    collection = mojovec.Collection(2, quantized=False)
    collection.add_from(
        path,
        file_format="arrow",
        id_column="id",
        document_column="text",
        batch_size=1,
    )
    _assert_two_record_collection(collection)


def test_huggingface_adapter_is_batched_and_forwards_loader_options(monkeypatch):
    calls = []
    fake = types.ModuleType("datasets")

    def load_dataset(dataset, config, **kwargs):
        calls.append((dataset, config, kwargs))
        return [
            {"id": 10, "vector": [1.0, 0.0], "text": "first document"},
            {"id": 20, "vector": [0.0, 1.0], "text": "second document"},
        ]

    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)

    collection = mojovec.Collection(2, quantized=False)
    imported = collection.add_huggingface(
        "owner/dataset",
        config="english",
        split="validation",
        id_column="id",
        embedding_column="vector",
        document_column="text",
        batch_size=1,
        load_kwargs={"revision": "abc123"},
    )

    assert imported == 2
    assert calls == [
        (
            "owner/dataset",
            "english",
            {
                "split": "validation",
                "streaming": True,
                "revision": "abc123",
            },
        )
    ]
    _assert_two_record_collection(collection)


def test_public_batch_reader_exposes_contiguous_arrays(tmp_path):
    path = tmp_path / "vectors.npy"
    np.save(path, np.eye(3, dtype=np.float32))
    batches = list(iter_file_batches(path, dimension=3, batch_size=2))
    assert all(isinstance(batch, ImportBatch) for batch in batches)
    assert [len(batch) for batch in batches] == [2, 1]
    assert all(batch.ids.dtype == np.int64 for batch in batches)
    assert all(batch.embeddings.dtype == np.float32 for batch in batches)
    assert all(batch.embeddings.flags.c_contiguous for batch in batches)
