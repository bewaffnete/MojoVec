"""Runtime-independent batched readers for external vector datasets.

The module deliberately lives above the native Mojo core. File readers produce
contiguous numeric batches and managed Python payloads, then use the regular
``Collection.add`` or ``Collection.upsert`` API. NumPy is required for numeric
formats; PyArrow and Hugging Face Datasets are imported only by their adapters.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from os import PathLike
from pathlib import Path
from numbers import Integral
from typing import Any, Literal


FileFormat = Literal[
    "csv",
    "tsv",
    "json",
    "jsonl",
    "parquet",
    "arrow",
    "npy",
    "npz",
    "fvecs",
    "ivecs",
]
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


@dataclass(slots=True)
class ImportBatch:
    """One validated batch ready for collection ingestion.

    Attributes
    ----------
    ids
        Contiguous one-dimensional ``numpy.int64`` array.
    embeddings
        Contiguous two-dimensional ``numpy.float32`` array.
    metadatas
        Optional list of scalar metadata mappings aligned with ``ids``.
    documents
        Optional list of strings aligned with ``ids``.
    """

    ids: Any
    embeddings: Any
    metadatas: list[dict[str, Any]] | None = None
    documents: list[str] | None = None

    def __len__(self) -> int:
        return int(len(self.ids))


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "This MojoVec reader requires NumPy; install mojovec[io]."
        ) from error
    return np


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Parquet and Arrow readers require PyArrow; install "
            "mojovec[arrow]."
        ) from error
    return pa, pq


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _infer_format(path: str | PathLike[str], file_format: str | None) -> str:
    if file_format is not None:
        normalized = file_format.lower().lstrip(".")
    else:
        normalized = Path(path).suffix.lower().lstrip(".")
    aliases = {
        "ndjson": "jsonl",
        "feather": "arrow",
        "ipc": "arrow",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {
        "csv",
        "tsv",
        "json",
        "jsonl",
        "parquet",
        "arrow",
        "npy",
        "npz",
        "fvecs",
        "ivecs",
    }
    if normalized not in supported:
        raise ValueError(
            f"unsupported dataset format {normalized!r}; "
            f"expected one of {sorted(supported)}"
        )
    return normalized


def _required_columns(
    *,
    id_column: str | None,
    embedding_column: str,
    embedding_columns: Sequence[str] | None,
    document_column: str | None,
    metadata_columns: Sequence[str] | None,
) -> list[str]:
    columns: list[str] = []
    for column in (
        [id_column] if id_column is not None else []
    ) + list(embedding_columns or [embedding_column]) + (
        [document_column] if document_column is not None else []
    ) + list(metadata_columns or []):
        if column is not None and column not in columns:
            columns.append(column)
    return columns


def _column(row: Mapping[str, Any], name: str) -> Any:
    if name not in row:
        raise ValueError(f"missing required column {name!r}")
    return row[name]


def _record_id(value: Any, *, row_index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"ID at row {row_index} must be an integer")
    if isinstance(value, str):
        try:
            result = int(value.strip())
        except ValueError as error:
            raise ValueError(
                f"ID at row {row_index} must be an integer"
            ) from error
    elif isinstance(value, Integral):
        result = int(value)
    else:
        raise ValueError(f"ID at row {row_index} must be an integer")
    if result < _INT64_MIN or result > _INT64_MAX:
        raise ValueError(f"ID at row {row_index} is outside the int64 range")
    return result


def _generated_id(id_start: int, row_index: int) -> int:
    return _record_id(id_start + row_index, row_index=row_index)


def _embedding_from_value(value: Any, *, row_index: int) -> Any:
    np = _numpy()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"embedding at row {row_index} is empty")
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"embedding at row {row_index} is not valid JSON"
                ) from error
        else:
            separator = "," if "," in stripped else " "
            value = np.fromstring(stripped, dtype=np.float32, sep=separator)
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"embedding at row {row_index} is not numeric"
        ) from error
    if vector.ndim != 1:
        raise ValueError(f"embedding at row {row_index} must be one-dimensional")
    return vector


def _metadata_value(value: Any, *, column: str, row_index: int) -> Any:
    np = _numpy()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if not isinstance(value, (str, bool, int, float)):
        raise ValueError(
            f"metadata column {column!r} at row {row_index} must contain "
            "String, Int, Float, Bool, or null"
        )
    return value


def _rows_to_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_offset: int,
    dimension: int | None,
    id_column: str | None,
    embedding_column: str,
    embedding_columns: Sequence[str] | None,
    document_column: str | None,
    metadata_columns: Sequence[str] | None,
    id_start: int,
) -> ImportBatch:
    np = _numpy()
    ids: list[int] = []
    vectors: list[Any] = []
    documents = [] if document_column is not None else None
    metadatas = [] if metadata_columns else None
    expected_dimension = dimension

    for local_index, row in enumerate(rows):
        row_index = row_offset + local_index
        if id_column is None:
            record_id = _generated_id(id_start, row_index)
        else:
            record_id = _record_id(
                _column(row, id_column), row_index=row_index
            )
        ids.append(record_id)

        if embedding_columns:
            try:
                vector = np.asarray(
                    [float(_column(row, name)) for name in embedding_columns],
                    dtype=np.float32,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"embedding columns at row {row_index} must be numeric"
                ) from error
        else:
            vector = _embedding_from_value(
                _column(row, embedding_column), row_index=row_index
            )
        if expected_dimension is None:
            expected_dimension = int(vector.size)
        if vector.size != expected_dimension:
            raise ValueError(
                f"embedding at row {row_index} has dimension {vector.size}; "
                f"expected {expected_dimension}"
            )
        vectors.append(vector)

        if documents is not None:
            document = _column(row, document_column)  # type: ignore[arg-type]
            documents.append("" if document is None else str(document))
        if metadatas is not None:
            metadata: dict[str, Any] = {}
            for name in metadata_columns or ():
                value = _metadata_value(
                    _column(row, name), column=name, row_index=row_index
                )
                if value is not None:
                    metadata[name] = value
            metadatas.append(metadata)

    if not rows:
        return ImportBatch(
            np.empty(0, dtype=np.int64),
            np.empty((0, dimension or 0), dtype=np.float32),
            metadatas,
            documents,
        )
    return ImportBatch(
        np.ascontiguousarray(ids, dtype=np.int64),
        np.ascontiguousarray(np.stack(vectors), dtype=np.float32),
        metadatas,
        documents,
    )


def _mapping_batches(
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_size: int,
    dimension: int | None,
    id_column: str | None,
    embedding_column: str,
    embedding_columns: Sequence[str] | None,
    document_column: str | None,
    metadata_columns: Sequence[str] | None,
    id_start: int,
) -> Iterator[ImportBatch]:
    batch: list[Mapping[str, Any]] = []
    row_offset = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_offset + len(batch)} must be an object")
        batch.append(row)
        if len(batch) == batch_size:
            yield _rows_to_batch(
                batch,
                row_offset=row_offset,
                dimension=dimension,
                id_column=id_column,
                embedding_column=embedding_column,
                embedding_columns=embedding_columns,
                document_column=document_column,
                metadata_columns=metadata_columns,
                id_start=id_start,
            )
            row_offset += len(batch)
            batch = []
    if batch:
        yield _rows_to_batch(
            batch,
            row_offset=row_offset,
            dimension=dimension,
            id_column=id_column,
            embedding_column=embedding_column,
            embedding_columns=embedding_columns,
            document_column=document_column,
            metadata_columns=metadata_columns,
            id_start=id_start,
        )


def _normalize_matrix(values: Any, *, dimension: int | None, name: str) -> Any:
    np = _numpy()
    array = np.asarray(values)
    if array.ndim == 1:
        if dimension is None:
            raise ValueError(f"dimension is required for a flat {name} array")
        if array.size % dimension != 0:
            raise ValueError(
                f"flat {name} array length must be a multiple of dimension"
            )
        array = array.reshape((-1, dimension))
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(
            f"{name} dimension is {array.shape[1]}; expected {dimension}"
        )
    try:
        return np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _array_batches(
    embeddings: Any,
    *,
    dimension: int | None,
    batch_size: int,
    id_start: int,
    ids: Any = None,
    documents: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[ImportBatch]:
    np = _numpy()
    matrix = _normalize_matrix(
        embeddings, dimension=dimension, name="embeddings"
    )
    count = int(matrix.shape[0])
    if ids is None:
        if count:
            _generated_id(id_start, count - 1)
        id_values = np.arange(id_start, id_start + count, dtype=np.int64)
    else:
        id_values = np.asarray(ids)
        if id_values.ndim != 1 or len(id_values) != count:
            raise ValueError("ids must be one-dimensional and match embeddings")
        if id_values.dtype.kind not in {"i", "u"}:
            raise ValueError("ids must contain integers")
        if id_values.dtype.kind == "u" and id_values.size:
            if int(id_values.max()) > _INT64_MAX:
                raise ValueError("ids contain a value outside the int64 range")
        try:
            id_values = np.asarray(id_values, dtype=np.int64)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("ids must contain integers") from error
    if documents is not None and len(documents) != count:
        raise ValueError("documents length must match embeddings")
    for name, values in (metadata or {}).items():
        if len(values) != count:
            raise ValueError(
                f"metadata array {name!r} length must match embeddings"
            )

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch_documents = None
        if documents is not None:
            batch_documents = [
                "" if value is None else str(value)
                for value in documents[start:stop]
            ]
        batch_metadatas = None
        if metadata:
            batch_metadatas = []
            for index in range(start, stop):
                item: dict[str, Any] = {}
                for name, values in metadata.items():
                    value = _metadata_value(
                        values[index], column=name, row_index=index
                    )
                    if value is not None:
                        item[name] = value
                batch_metadatas.append(item)
        yield ImportBatch(
            np.ascontiguousarray(id_values[start:stop], dtype=np.int64),
            np.ascontiguousarray(matrix[start:stop], dtype=np.float32),
            batch_metadatas,
            batch_documents,
        )


def _vecs_matrix(path: str | PathLike[str], *, integer: bool) -> Any:
    np = _numpy()
    if Path(path).stat().st_size == 0:
        dtype = np.int32 if integer else np.float32
        return np.empty((0, 0), dtype=dtype)
    raw = np.memmap(path, dtype="<i4", mode="r")
    dimension = int(raw[0])
    if dimension <= 0:
        raise ValueError("fvecs/ivecs dimension must be positive")
    width = dimension + 1
    if raw.size % width != 0:
        raise ValueError("truncated or malformed fvecs/ivecs file")
    rows = raw.reshape((-1, width))
    if not np.all(rows[:, 0] == dimension):
        raise ValueError("fvecs/ivecs rows have inconsistent dimensions")
    values = rows[:, 1:]
    if integer:
        return values
    return values.view("<f4")


def read_fvecs(path: str | PathLike[str]) -> Any:
    """Read a standard ``.fvecs`` file into a float32 matrix."""
    np = _numpy()
    return np.ascontiguousarray(_vecs_matrix(path, integer=False))


def read_ivecs(path: str | PathLike[str]) -> Any:
    """Read a standard ``.ivecs`` file into an int32 matrix."""
    np = _numpy()
    return np.ascontiguousarray(_vecs_matrix(path, integer=True))


def iter_file_batches(
    path: str | PathLike[str],
    *,
    file_format: str | None = None,
    dimension: int | None = None,
    id_column: str | None = None,
    embedding_column: str = "embedding",
    embedding_columns: Sequence[str] | None = None,
    document_column: str | None = None,
    metadata_columns: Sequence[str] | None = None,
    batch_size: int = 8192,
    id_start: int = 0,
    embeddings_key: str = "embeddings",
    ids_key: str = "ids",
    documents_key: str = "documents",
) -> Iterator[ImportBatch]:
    """Yield validated batches from a supported local dataset file.

    CSV/TSV, JSON/JSONL, Parquet, and Arrow use named columns. A single
    embedding column may contain an array, JSON array string, comma-separated
    string, or whitespace-separated string. ``embedding_columns`` selects
    one scalar column per dimension instead.

    NPY and fvecs/ivecs contain embeddings only and receive generated IDs.
    NPZ reads arrays from ``embeddings_key`` and optionally ``ids_key`` and
    ``documents_key``; names in ``metadata_columns`` select aligned NPZ arrays.
    """
    _validate_batch_size(batch_size)
    normalized = _infer_format(path, file_format)

    if normalized in {"csv", "tsv"}:
        delimiter = "," if normalized == "csv" else "\t"
        with open(path, "r", encoding="utf-8", newline="") as input_file:
            yield from _mapping_batches(
                csv.DictReader(input_file, delimiter=delimiter),
                batch_size=batch_size,
                dimension=dimension,
                id_column=id_column,
                embedding_column=embedding_column,
                embedding_columns=embedding_columns,
                document_column=document_column,
                metadata_columns=metadata_columns,
                id_start=id_start,
            )
        return

    if normalized == "jsonl":
        def json_lines() -> Iterator[Mapping[str, Any]]:
            with open(path, "r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid JSON on line {line_number}"
                        ) from error
                    if not isinstance(row, Mapping):
                        raise ValueError(
                            f"JSONL line {line_number} must contain an object"
                        )
                    yield row

        yield from _mapping_batches(
            json_lines(),
            batch_size=batch_size,
            dimension=dimension,
            id_column=id_column,
            embedding_column=embedding_column,
            embedding_columns=embedding_columns,
            document_column=document_column,
            metadata_columns=metadata_columns,
            id_start=id_start,
        )
        return

    if normalized == "json":
        with open(path, "r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
        if isinstance(payload, Mapping) and isinstance(payload.get("data"), list):
            payload = payload["data"]
        elif isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be an object, list, or {data: [...]}")
        yield from _mapping_batches(
            payload,
            batch_size=batch_size,
            dimension=dimension,
            id_column=id_column,
            embedding_column=embedding_column,
            embedding_columns=embedding_columns,
            document_column=document_column,
            metadata_columns=metadata_columns,
            id_start=id_start,
        )
        return

    if normalized in {"parquet", "arrow"}:
        pa, pq = _pyarrow()
        columns = _required_columns(
            id_column=id_column,
            embedding_column=embedding_column,
            embedding_columns=embedding_columns,
            document_column=document_column,
            metadata_columns=metadata_columns,
        )
        if normalized == "parquet":
            batches = pq.ParquetFile(path).iter_batches(
                batch_size=batch_size, columns=columns
            )
            rows: Iterable[Mapping[str, Any]] = (
                row for batch in batches for row in batch.to_pylist()
            )
        else:
            def arrow_rows() -> Iterator[Mapping[str, Any]]:
                with pa.memory_map(str(path), "r") as source:
                    try:
                        reader = pa.ipc.open_file(source)
                        record_batches = (
                            reader.get_batch(index)
                            for index in range(reader.num_record_batches)
                        )
                    except pa.ArrowInvalid:
                        source.seek(0)
                        reader = pa.ipc.open_stream(source)
                        record_batches = iter(reader)
                    for record_batch in record_batches:
                        yield from record_batch.select(columns).to_pylist()

            rows = arrow_rows()
        yield from _mapping_batches(
            rows,
            batch_size=batch_size,
            dimension=dimension,
            id_column=id_column,
            embedding_column=embedding_column,
            embedding_columns=embedding_columns,
            document_column=document_column,
            metadata_columns=metadata_columns,
            id_start=id_start,
        )
        return

    np = _numpy()
    if normalized == "npy":
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
        yield from _array_batches(
            embeddings,
            dimension=dimension,
            batch_size=batch_size,
            id_start=id_start,
        )
        return

    if normalized == "npz":
        with np.load(path, allow_pickle=False) as archive:
            if embeddings_key not in archive:
                raise ValueError(f"NPZ archive has no {embeddings_key!r} array")
            metadata = {
                name: archive[name]
                for name in metadata_columns or ()
                if name in archive
            }
            missing = [
                name for name in metadata_columns or () if name not in archive
            ]
            if missing:
                raise ValueError(f"NPZ archive has no metadata arrays {missing}")
            yield from _array_batches(
                archive[embeddings_key],
                dimension=dimension,
                batch_size=batch_size,
                id_start=id_start,
                ids=archive[ids_key] if ids_key in archive else None,
                documents=(
                    archive[documents_key]
                    if documents_key in archive
                    else None
                ),
                metadata=metadata,
            )
        return

    matrix = _vecs_matrix(path, integer=normalized == "ivecs")
    yield from _array_batches(
        matrix,
        dimension=dimension,
        batch_size=batch_size,
        id_start=id_start,
    )


def iter_huggingface_batches(
    dataset: str,
    *,
    split: str = "train",
    config: str | None = None,
    dimension: int | None = None,
    id_column: str | None = None,
    embedding_column: str = "embedding",
    embedding_columns: Sequence[str] | None = None,
    document_column: str | None = None,
    metadata_columns: Sequence[str] | None = None,
    batch_size: int = 8192,
    id_start: int = 0,
    streaming: bool = True,
    load_kwargs: Mapping[str, Any] | None = None,
) -> Iterator[ImportBatch]:
    """Yield batches from a Hugging Face dataset split.

    Requires the optional ``datasets`` package. ``load_kwargs`` is forwarded
    to ``datasets.load_dataset`` for options such as ``revision``, ``token``,
    ``cache_dir``, or ``data_files``.
    """
    _validate_batch_size(batch_size)
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Hugging Face import requires datasets; install "
            "mojovec[huggingface]."
        ) from error
    source = load_dataset(
        dataset,
        config,
        split=split,
        streaming=streaming,
        **dict(load_kwargs or {}),
    )
    yield from _mapping_batches(
        source,
        batch_size=batch_size,
        dimension=dimension,
        id_column=id_column,
        embedding_column=embedding_column,
        embedding_columns=embedding_columns,
        document_column=document_column,
        metadata_columns=metadata_columns,
        id_start=id_start,
    )


def ingest_batches(
    collection: Any,
    batches: Iterable[ImportBatch],
    *,
    operation: Literal["add", "upsert"],
) -> int:
    """Apply imported batches and return the number of processed records.

    Each individual collection call is atomic. A multi-batch import is not a
    transaction: if a later batch fails, earlier committed batches remain.
    """
    if operation not in {"add", "upsert"}:
        raise ValueError("operation must be 'add' or 'upsert'")
    count = 0
    for batch in batches:
        if len(batch) == 0:
            continue
        if batch.metadatas is None and batch.documents is None:
            getattr(collection, f"{operation}_numpy")(
                batch.ids, batch.embeddings
            )
        else:
            getattr(collection, operation)(
                batch.ids.tolist(),
                batch.embeddings,
                metadatas=batch.metadatas,
                documents=batch.documents,
            )
        count += len(batch)
    return count


__all__ = [
    "FileFormat",
    "ImportBatch",
    "ingest_batches",
    "iter_file_batches",
    "iter_huggingface_batches",
    "read_fvecs",
    "read_ivecs",
]
