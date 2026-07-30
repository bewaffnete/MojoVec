"""Managed Python API for MojoVec.

The public layer deliberately keeps Python argument handling out of the native
search path. Vector indexing and search still execute entirely in Mojo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from os import PathLike
from typing import Any, TypedDict

from . import _native


WAL_ASYNC = 1
WAL_SYNC = 2
DEFAULT_MMAP_THRESHOLD_BYTES = 64 * 1024 * 1024
__version__ = "0.6.0"

Metadata = Mapping[str, str | int | float | bool]
Where = Mapping[str, Any]


class QueryResult(TypedDict):
    ids: list[list[int]]
    distances: list[list[float]]
    metadatas: list[list[dict[str, str | int | float | bool]]]
    documents: list[list[str]]
    scores: list[list[float]]


def _flatten_embeddings(
    embeddings: Sequence[Real] | Sequence[Sequence[Real]],
) -> Sequence[Real]:
    if len(embeddings) == 0:
        return []
    first = embeddings[0]
    if isinstance(first, Real):
        # Keep the established flat-list path allocation-free on the Python
        # side; the native boundary already converts each scalar to Float32.
        return embeddings  # type: ignore[return-value]
    return [
        float(value)
        for row in embeddings  # type: ignore[assignment]
        for value in row
    ]


def _validate_payload_lengths(
    ids: Sequence[int],
    metadatas: Sequence[Metadata] | None,
    documents: Sequence[str] | None,
) -> None:
    if metadatas is not None and len(metadatas) != len(ids):
        raise ValueError("metadatas length must match ids length")
    if documents is not None and len(documents) != len(ids):
        raise ValueError("documents length must match ids length")


def _compile_predicate(operation: str, key: str, value: Any) -> Any:
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError("where values must be str, int, float, or bool")
    return _native._where_predicate(operation, key, value)


def _scalar_kind(value: Any) -> type[Any]:
    # bool is an int subclass in Python but a separate MetadataValue kind.
    return bool if isinstance(value, bool) else type(value)


def _combine(operation: str, conditions: Sequence[Any]) -> Any:
    if not conditions:
        raise ValueError(f"${operation} requires at least one condition")
    return _native._where_combine(operation, list(conditions))


def _compile_field(key: str, expression: Any) -> Any:
    if not isinstance(expression, Mapping):
        return _compile_predicate("eq", key, expression)
    if not expression:
        raise ValueError(f"where expression for {key!r} cannot be empty")

    conditions: list[Any] = []
    operators = {
        "$eq": "eq",
        "$ne": "ne",
        "$gt": "gt",
        "$gte": "gte",
        "$lt": "lt",
        "$lte": "lte",
    }
    for operator, value in expression.items():
        if operator in operators:
            conditions.append(_compile_predicate(operators[operator], key, value))
        elif operator in ("$in", "$nin", "$not_in"):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{operator} requires a sequence of scalar values")
            if not value:
                raise ValueError(f"{operator} requires at least one value")
            first_kind = _scalar_kind(value[0])
            if first_kind not in (str, int, float, bool) or any(
                _scalar_kind(item) is not first_kind for item in value
            ):
                raise TypeError(
                    f"{operator} values must be scalar and have one type"
                )
            predicate = "eq" if operator == "$in" else "ne"
            combine = "any" if operator == "$in" else "all"
            conditions.append(
                _combine(
                    combine,
                    [_compile_predicate(predicate, key, item) for item in value],
                )
            )
        else:
            raise ValueError(f"unsupported where operator: {operator!r}")
    return conditions[0] if len(conditions) == 1 else _combine("all", conditions)


def _compile_where(where: Where) -> Any:
    if not isinstance(where, Mapping):
        raise TypeError("where must be a mapping")
    if not where:
        raise ValueError("where cannot be empty")

    conditions: list[Any] = []
    for key, expression in where.items():
        if not isinstance(key, str):
            raise TypeError("where keys must be strings")
        if key in ("$and", "$or"):
            if (
                isinstance(expression, (str, bytes))
                or not isinstance(expression, Sequence)
            ):
                raise TypeError(f"{key} requires a sequence of where mappings")
            operation = "all" if key == "$and" else "any"
            conditions.append(
                _combine(operation, [_compile_where(item) for item in expression])
            )
        elif key == "$not":
            conditions.append(_combine("not", [_compile_where(expression)]))
        elif key.startswith("$"):
            raise ValueError(f"unsupported logical where operator: {key!r}")
        else:
            conditions.append(_compile_field(key, expression))
    return conditions[0] if len(conditions) == 1 else _combine("all", conditions)


class Collection:
    """A managed Flat or SQ8 HNSW collection."""

    def __init__(
        self,
        dimension: int,
        M: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16,
        quantized: bool = True,
        metric: str = "l2",
        name: str = "",
    ) -> None:
        self._inner = _native._Collection(
            dimension,
            M,
            ef_construction,
            ef_search,
            quantized,
            name,
            metric,
        )

    @classmethod
    def _from_native(cls, inner: Any) -> Collection:
        collection = cls.__new__(cls)
        collection._inner = inner
        return collection

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        memory_mapped: bool = True,
        mmap_threshold_bytes: int = DEFAULT_MMAP_THRESHOLD_BYTES,
    ) -> Collection:
        return load(path, memory_mapped, mmap_threshold_bytes)

    @classmethod
    def recover(
        cls,
        snapshot_path: str | PathLike[str],
        wal_path: str | PathLike[str],
        durability: int = WAL_ASYNC,
        memory_mapped: bool = True,
        mmap_threshold_bytes: int = DEFAULT_MMAP_THRESHOLD_BYTES,
    ) -> Collection:
        return recover(
            snapshot_path,
            wal_path,
            durability,
            memory_mapped,
            mmap_threshold_bytes,
        )

    def __repr__(self) -> str:
        return (
            f"Collection(name={self.name()!r}, dimension={self.dimension()}, "
            f"storage={self.storage_kind()!r}, metric={self.metric()!r}, "
            f"count={self.count()})"
        )

    def name(self) -> str:
        return self._inner.name()

    def dimension(self) -> int:
        return self._inner.dimension()

    def storage_kind(self) -> str:
        return "sq8" if self._inner.is_quantized() else "flat"

    def metric(self) -> str:
        return self._inner.metric()

    def is_quantized(self) -> bool:
        return self._inner.is_quantized()

    def count(self) -> int:
        return self._inner.count()

    def count_deleted(self) -> int:
        return self._inner.count_deleted()

    def get_metadata(self, record_id: int) -> dict[str, Any]:
        return self._inner.get_metadata(record_id)

    def get_document(self, record_id: int) -> str:
        return self._inner.get_document(record_id)

    def stats(self) -> dict[str, Any]:
        return self._inner.stats()

    def is_memory_mapped(self) -> bool:
        return self._inner.is_memory_mapped()

    def set_ef_search(self, ef_search: int) -> None:
        self._inner.set_ef_search(ef_search)

    def _write(
        self,
        operation: str,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata] | None,
        documents: Sequence[str] | None,
    ) -> None:
        _validate_payload_lengths(ids, metadatas, documents)
        flat = _flatten_embeddings(embeddings)
        expected = len(ids) * self.dimension()
        if len(flat) != expected:
            raise ValueError(
                f"embeddings contain {len(flat)} values; expected {expected}"
            )

        method = getattr(self._inner, operation)
        if metadatas is not None and documents is not None:
            getattr(self._inner, f"{operation}_with_payloads")(
                ids, flat, metadatas, documents
            )
        elif metadatas is not None:
            getattr(self._inner, f"{operation}_with_metadata")(
                ids, flat, metadatas
            )
        elif documents is not None:
            getattr(self._inner, f"{operation}_with_documents")(
                ids, flat, documents
            )
        else:
            method(ids, flat)

    def add(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata] | None = None,
        documents: Sequence[str] | None = None,
    ) -> None:
        self._write("add", ids, embeddings, metadatas, documents)

    def upsert(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata] | None = None,
        documents: Sequence[str] | None = None,
    ) -> None:
        self._write("upsert", ids, embeddings, metadatas, documents)

    def update(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata] | None = None,
        documents: Sequence[str] | None = None,
    ) -> None:
        self._write("update", ids, embeddings, metadatas, documents)

    def add_with_metadata(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata],
    ) -> None:
        self.add(ids, embeddings, metadatas=metadatas)

    def upsert_with_metadata(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata],
    ) -> None:
        self.upsert(ids, embeddings, metadatas=metadatas)

    def update_with_metadata(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata],
    ) -> None:
        self.update(ids, embeddings, metadatas=metadatas)

    def delete(self, ids: Sequence[int]) -> None:
        self._inner.delete(ids)

    def query(
        self,
        query_embeddings: (
            Sequence[Real] | Sequence[Sequence[Real]] | Sequence[str] | None
        ) = None,
        n_results: int = 10,
        *,
        query_texts: Sequence[str] | None = None,
        where: Where | None = None,
    ) -> QueryResult:
        if (
            query_texts is None
            and query_embeddings is not None
            and len(query_embeddings) > 0
            and isinstance(query_embeddings[0], str)
        ):
            query_texts = query_embeddings  # type: ignore[assignment]
            query_embeddings = None
        compiled = _compile_where(where) if where is not None else None
        if query_texts is not None:
            if query_embeddings is not None:
                raise ValueError(
                    "query accepts either query_embeddings or query_texts; "
                    "use query_hybrid() to combine both"
                )
            if compiled is None:
                return self._inner.query_text(query_texts, n_results)
            return self._inner.query_text_where(
                query_texts, n_results, compiled
            )
        if query_embeddings is None:
            raise ValueError("query_embeddings or query_texts is required")
        flat = _flatten_embeddings(query_embeddings)
        if compiled is None:
            return self._inner.query_vector(flat, n_results)
        return self._inner.query_vector_where(flat, n_results, compiled)

    def query_hybrid(
        self,
        query_embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        query_texts: Sequence[str],
        n_results: int = 10,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        where: Where | None = None,
    ) -> QueryResult:
        flat = _flatten_embeddings(query_embeddings)
        if where is None:
            return self._inner.query_hybrid(
                flat,
                query_texts,
                n_results,
                rrf_k,
                candidate_multiplier,
            )
        return self._inner.query_hybrid_where(
            flat,
            query_texts,
            n_results,
            rrf_k,
            candidate_multiplier,
            _compile_where(where),
        )

    def compact(self) -> dict[str, Any]:
        return self._inner.compact()

    def compact_if_needed(
        self, deleted_ratio: float = 0.25
    ) -> dict[str, Any]:
        return self._inner.compact_if_needed(deleted_ratio)

    def save(self, path: str | PathLike[str]) -> None:
        self._inner.save(str(path))

    def snapshot(
        self,
        path: str | PathLike[str],
        memory_mapped: bool = True,
        mmap_threshold_bytes: int = DEFAULT_MMAP_THRESHOLD_BYTES,
    ) -> Collection:
        return Collection._from_native(
            self._inner.snapshot(
                str(path), memory_mapped, mmap_threshold_bytes
            )
        )

    def wal_enabled(self) -> bool:
        return self._inner.wal_enabled()

    def wal_sequence(self) -> int:
        return self._inner.wal_sequence()

    def enable_wal(
        self,
        path: str | PathLike[str],
        durability: int = WAL_ASYNC,
    ) -> None:
        self._inner.enable_wal(str(path), durability)

    def disable_wal(self) -> None:
        self._inner.disable_wal()

    def flush_wal(self) -> None:
        self._inner.flush_wal()

    def checkpoint(self, path: str | PathLike[str]) -> None:
        self._inner.checkpoint(str(path))

    def upsert_numpy(
        self,
        ids: Any,
        embeddings: Any,
        metadatas: Sequence[Metadata] | None = None,
        documents: Sequence[str] | None = None,
    ) -> None:
        if metadatas is not None or documents is not None:
            self.upsert(ids.tolist(), embeddings, metadatas, documents)
            return
        import numpy as np

        ids_array = np.asarray(ids)
        embeddings_array = np.asarray(embeddings)
        if ids_array.dtype != np.int64 or ids_array.ndim != 1:
            raise TypeError("ids must be a one-dimensional numpy.int64 array")
        if (
            embeddings_array.dtype != np.float32
            or embeddings_array.ndim not in (1, 2)
            or not embeddings_array.flags.c_contiguous
            or not ids_array.flags.c_contiguous
        ):
            raise TypeError(
                "embeddings must be a contiguous numpy.float32 array"
            )
        if embeddings_array.size != ids_array.size * self.dimension():
            raise ValueError("embeddings shape does not match ids and dimension")
        self._inner.upsert_numpy(ids_array, embeddings_array)

    def query_numpy(self, query_embeddings: Any, n_results: int = 10) -> QueryResult:
        import numpy as np

        embeddings_array = np.asarray(query_embeddings)
        if embeddings_array.dtype != np.float32:
            raise TypeError("query_embeddings must have numpy.float32 dtype")
        if embeddings_array.ndim == 1:
            if embeddings_array.size % self.dimension() != 0:
                raise ValueError(
                    "query embeddings length must be a multiple of dimension"
                )
            embeddings_array = embeddings_array.reshape(
                (-1, self.dimension())
            )
        if (
            embeddings_array.ndim != 2
            or embeddings_array.shape[1] != self.dimension()
            or not embeddings_array.flags.c_contiguous
        ):
            raise ValueError(
                "query_embeddings must be a contiguous (n, dimension) array"
            )
        result = self._inner.query_numpy(embeddings_array, n_results)
        result["metadatas"] = []
        result["documents"] = []
        result["scores"] = []
        return result

    def upsert_batch(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
    ) -> None:
        self.upsert(ids, embeddings)

    def query_batch(
        self,
        query_embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        n_results: int = 10,
    ) -> QueryResult:
        return self.query(query_embeddings, n_results)

    upsert_batch_numpy = upsert_numpy
    query_batch_numpy = query_numpy


def load(
    path: str | PathLike[str],
    memory_mapped: bool = True,
    mmap_threshold_bytes: int = DEFAULT_MMAP_THRESHOLD_BYTES,
) -> Collection:
    return Collection._from_native(
        _native._load(str(path), memory_mapped, mmap_threshold_bytes)
    )


def recover(
    snapshot_path: str | PathLike[str],
    wal_path: str | PathLike[str],
    durability: int = WAL_ASYNC,
    memory_mapped: bool = True,
    mmap_threshold_bytes: int = DEFAULT_MMAP_THRESHOLD_BYTES,
) -> Collection:
    return Collection._from_native(
        _native._recover(
            str(snapshot_path),
            str(wal_path),
            durability,
            memory_mapped,
            mmap_threshold_bytes,
        )
    )


__all__ = [
    "Collection",
    "DEFAULT_MMAP_THRESHOLD_BYTES",
    "Metadata",
    "QueryResult",
    "WAL_ASYNC",
    "WAL_SYNC",
    "Where",
    "__version__",
    "load",
    "recover",
]
