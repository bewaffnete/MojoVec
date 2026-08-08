"""Documented Python facade over the native Mojo collection."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from os import PathLike
from typing import Any

from ._runtime import _native
from ._types import (
    DEFAULT_MMAP_THRESHOLD_BYTES,
    WAL_ASYNC,
    Metadata,
    QueryResult,
    Where,
)
from ._where import _compile_where


def _flatten_embeddings(
    embeddings: Sequence[Real] | Sequence[Sequence[Real]],
    dimension: int | None = None,
) -> Sequence[Real]:
    if len(embeddings) == 0:
        return []
    first = embeddings[0]
    if isinstance(first, Real):
        if dimension is not None and len(embeddings) % dimension != 0:
            raise ValueError(
                f"flat embeddings contain {len(embeddings)} values; "
                f"expected a multiple of dimension {dimension}"
            )
        return embeddings  # type: ignore[return-value]

    flat: list[float] = []
    for index, row in enumerate(embeddings):  # type: ignore[union-attr]
        if isinstance(row, Real):
            raise ValueError("embeddings must be homogenously flat or 2D")
        row_len = len(row)  # type: ignore[arg-type]
        if dimension is not None and row_len != dimension:
            raise ValueError(
                f"embedding at index {index} has dimension {row_len}; "
                f"expected {dimension}"
            )
        elif dimension is None and index > 0 and row_len != len(embeddings[0]):  # type: ignore[arg-type]
            raise ValueError(
                f"embedding at index {index} has dimension {row_len}; "
                f"expected {len(embeddings[0])}"
            )
        flat.extend(float(value) for value in row)  # type: ignore[union-attr]
    return flat


def _validate_payload_lengths(
    ids: Sequence[int],
    metadatas: Sequence[Metadata] | None,
    documents: Sequence[str] | None,
) -> None:
    if metadatas is not None and len(metadatas) != len(ids):
        raise ValueError("metadatas length must match ids length")
    if documents is not None and len(documents) != len(ids):
        raise ValueError("documents length must match ids length")


class Collection:
    """Managed Flat or SQ8 HNSW collection.

    Parameters
    ----------
    dimension : int
        Number of components in every stored and query vector.
    M : int, default=32
        Maximum HNSW neighbor count.
    ef_construction : int, default=40
        Candidate-list size used while constructing the graph.
    ef_search : int, default=16
        Candidate-list size used by vector queries.
    quantized : bool, default=True
        Store vectors as SQ8 when true or exact Float32 when false.
    metric : {"l2", "cosine", "ip"}, default="l2"
        Distance metric. Returned distances are always smaller-is-better.
    name : str, default=""
        Optional collection name preserved by persistence and compaction.

    Notes
    -----
    The Python object owns the native collection. No manual allocation,
    pointer management, or memory-release call is required.

    Concurrent ``query`` and ``query_hybrid`` calls are supported on the same
    unchanged collection. Native read-only search releases the Python GIL.
    Mutations, persistence, compaction, and configuration changes must not run
    concurrently with queries or other mutations on the same object; use
    external synchronization or independent :meth:`snapshot` readers.
    """

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
        """Load a collection snapshot.

        This is the class-oriented equivalent of :func:`mojovec.load`.
        """
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
        """Recover a snapshot by replaying its write-ahead log.

        This is the class-oriented equivalent of :func:`mojovec.recover`.
        """
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
        """Return the collection name."""
        return self._inner.name()

    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._inner.dimension()

    def storage_kind(self) -> str:
        """Return ``"sq8"`` or ``"flat"`` for the active vector storage."""
        return "sq8" if self._inner.is_quantized() else "flat"

    def metric(self) -> str:
        """Return the configured distance metric."""
        return self._inner.metric()

    def is_quantized(self) -> bool:
        """Return whether vectors use SQ8 storage."""
        return self._inner.is_quantized()

    def count(self) -> int:
        """Return the number of active records."""
        return self._inner.count()

    def count_deleted(self) -> int:
        """Return the number of soft-deleted or superseded records."""
        return self._inner.count_deleted()

    def get_metadata(self, record_id: int) -> dict[str, Any]:
        """Return a copy of the metadata stored for ``record_id``."""
        return self._inner.get_metadata(record_id)

    def get_document(self, record_id: int) -> str:
        """Return the document stored for ``record_id``."""
        return self._inner.get_document(record_id)

    def stats(self) -> dict[str, Any]:
        """Return collection size, deletion ratio, graph, and storage settings."""
        return self._inner.stats()

    def is_memory_mapped(self) -> bool:
        """Return whether immutable vector and graph regions remain mapped."""
        return self._inner.is_memory_mapped()

    def set_ef_search(self, ef_search: int) -> None:
        """Set the HNSW search candidate-list size for subsequent queries."""
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
        flat = _flatten_embeddings(embeddings, self.dimension())
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
        """Insert new records.

        Parameters
        ----------
        ids : sequence of int
            Unique record identifiers that do not already exist.
        embeddings : sequence of float or sequence of vector rows
            Flat row-major values or nested vectors with ``dimension`` values
            per ID.
        metadatas : sequence of mappings, optional
            One scalar metadata mapping per ID.
        documents : sequence of str, optional
            One document per ID.

        Raises
        ------
        ValueError
            If payload lengths or embedding dimensions do not match, a batch
            contains duplicate IDs, or an ID already exists.
        """
        self._write("add", ids, embeddings, metadatas, documents)

    def upsert(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata] | None = None,
        documents: Sequence[str] | None = None,
    ) -> None:
        """Insert missing IDs and replace existing records atomically.

        Validation, payload preparation, and WAL failures leave both the live
        collection and its committed WAL sequence unchanged.
        Vector-only replacements preserve existing metadata and documents.
        Supplying either payload replaces that payload in full.
        """
        self._write("upsert", ids, embeddings, metadatas, documents)

    def update(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata] | None = None,
        documents: Sequence[str] | None = None,
    ) -> None:
        """Replace existing records and reject IDs that are not present.

        Vector-only updates preserve existing metadata and documents.
        Supplying either payload replaces that payload in full.
        """
        self._write("update", ids, embeddings, metadatas, documents)

    def add_with_metadata(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata],
    ) -> None:
        """Compatibility wrapper for :meth:`add` with metadata."""
        self.add(ids, embeddings, metadatas=metadatas)

    def upsert_with_metadata(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata],
    ) -> None:
        """Compatibility wrapper for :meth:`upsert` with metadata."""
        self.upsert(ids, embeddings, metadatas=metadatas)

    def update_with_metadata(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        metadatas: Sequence[Metadata],
    ) -> None:
        """Compatibility wrapper for :meth:`update` with metadata."""
        self.update(ids, embeddings, metadatas=metadatas)

    def delete(self, ids: Sequence[int]) -> None:
        """Soft-delete active records and ignore unknown IDs."""
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
        """Search by vectors or BM25 text.

        Parameters
        ----------
        query_embeddings : sequence, optional
            A single vector, nested vector rows, or—only for compatibility—a
            sequence of strings interpreted as ``query_texts``.
        n_results : int, default=10
            Maximum number of neighbors returned per query.
        query_texts : sequence of str, optional
            BM25 queries. Cannot be combined with ``query_embeddings`` here;
            use :meth:`query_hybrid` for combined retrieval.
        where : mapping, optional
            Chroma-style metadata filter supporting comparison, membership,
            ``$and``, ``$or``, and ``$not`` expressions.

        Returns
        -------
        QueryResult
            Five aligned batched fields: ``ids``, ``distances``,
            ``metadatas``, ``documents``, and ``scores``. Vector queries fill
            distances; BM25 queries fill scores.
        """
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
        flat = _flatten_embeddings(query_embeddings, self.dimension())
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
        """Combine vector and BM25 rankings using reciprocal rank fusion.

        Parameters
        ----------
        query_embeddings : sequence
            One vector per text query, as flat or nested values.
        query_texts : sequence of str
            One text query per vector query.
        n_results : int, default=10
            Maximum fused results per query.
        rrf_k : int, default=60
            Reciprocal-rank-fusion smoothing constant.
        candidate_multiplier : int, default=4
            Number of candidates retrieved from each source relative to
            ``n_results``.
        where : mapping, optional
            Metadata filter applied to both retrieval paths.

        Returns
        -------
        QueryResult
            Batched results with fused values in ``scores``.
        """
        flat = _flatten_embeddings(query_embeddings, self.dimension())
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
        """Rebuild immediately and reclaim deleted or superseded records."""
        return self._inner.compact()

    def compact_if_needed(
        self, deleted_ratio: float = 0.25
    ) -> dict[str, Any]:
        """Compact when the deleted-record ratio reaches ``deleted_ratio``."""
        return self._inner.compact_if_needed(deleted_ratio)

    def save(self, path: str | PathLike[str]) -> None:
        """Atomically save a checksummed collection snapshot to ``path``."""
        self._inner.save(str(path))

    def snapshot(
        self,
        path: str | PathLike[str],
        memory_mapped: bool = True,
        mmap_threshold_bytes: int = DEFAULT_MMAP_THRESHOLD_BYTES,
    ) -> Collection:
        """Publish and return an independent point-in-time collection view."""
        return Collection._from_native(
            self._inner.snapshot(
                str(path), memory_mapped, mmap_threshold_bytes
            )
        )

    def wal_enabled(self) -> bool:
        """Return whether write-ahead logging is enabled."""
        return self._inner.wal_enabled()

    def wal_sequence(self) -> int:
        """Return the latest WAL sequence number."""
        return self._inner.wal_sequence()

    def enable_wal(
        self,
        path: str | PathLike[str],
        durability: int = WAL_ASYNC,
    ) -> None:
        """Enable write-ahead logging at ``path``.

        Use :data:`mojovec.WAL_ASYNC` for throughput or
        :data:`mojovec.WAL_SYNC` for synchronous durability.
        """
        self._inner.enable_wal(str(path), durability)

    def disable_wal(self) -> None:
        """Stop recording future mutations in the active WAL."""
        self._inner.disable_wal()

    def flush_wal(self) -> None:
        """Flush pending asynchronous WAL records to durable storage."""
        self._inner.flush_wal()

    def checkpoint(self, path: str | PathLike[str]) -> None:
        """Save a checkpoint and rotate the active WAL safely."""
        self._inner.checkpoint(str(path))

    def upsert_numpy(
        self,
        ids: Any,
        embeddings: Any,
        metadatas: Sequence[Metadata] | None = None,
        documents: Sequence[str] | None = None,
    ) -> None:
        """Upsert contiguous NumPy arrays through the zero-copy fast path.

        Parameters
        ----------
        ids : numpy.ndarray
            Contiguous one-dimensional ``int64`` array.
        embeddings : numpy.ndarray
            Contiguous ``float32`` array with shape ``(n, dimension)`` or an
            equivalent flat row-major array.
        metadatas, documents : sequence, optional
            Payloads use the managed path because Python objects cannot cross
            the numeric zero-copy boundary.
        """
        if metadatas is not None or documents is not None:
            self.upsert(ids.tolist(), embeddings, metadatas, documents)
            return
        import numpy as np

        ids_array = np.asarray(ids)
        embeddings_array = np.asarray(embeddings)
        if (
            ids_array.dtype != np.int64
            or ids_array.ndim != 1
            or not ids_array.flags.c_contiguous
        ):
            raise TypeError(
                "ids must be a contiguous one-dimensional numpy.int64 array"
            )
        if (
            embeddings_array.dtype != np.float32
            or embeddings_array.ndim not in (1, 2)
            or not embeddings_array.flags.c_contiguous
        ):
            raise TypeError(
                "embeddings must be a contiguous numpy.float32 array"
            )
        if embeddings_array.size != ids_array.size * self.dimension():
            raise ValueError("embeddings shape does not match ids and dimension")
        self._inner.upsert_numpy(ids_array, embeddings_array)

    def query_numpy(
        self, query_embeddings: Any, n_results: int = 10
    ) -> QueryResult:
        """Query contiguous ``float32`` NumPy vectors without list conversion."""
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
        """Compatibility alias for vector-only :meth:`upsert`."""
        self.upsert(ids, embeddings)

    def query_batch(
        self,
        query_embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        n_results: int = 10,
    ) -> QueryResult:
        """Compatibility alias for vector :meth:`query`."""
        return self.query(query_embeddings, n_results)

    upsert_batch_numpy = upsert_numpy
    query_batch_numpy = query_numpy


def load(
    path: str | PathLike[str],
    memory_mapped: bool = True,
    mmap_threshold_bytes: int = DEFAULT_MMAP_THRESHOLD_BYTES,
) -> Collection:
    """Load a collection snapshot.

    Parameters
    ----------
    path : path-like
        Snapshot created by :meth:`Collection.save` or
        :meth:`Collection.checkpoint`.
    memory_mapped : bool, default=True
        Keep eligible immutable vector and graph regions mapped read-only.
    mmap_threshold_bytes : int, default=67108864
        Minimum eligible mapped region size. Use zero to force mmap for small
        snapshots.

    Returns
    -------
    Collection
        A fully managed loaded collection.
    """
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
    """Load a snapshot and replay committed WAL mutations.

    Parameters
    ----------
    snapshot_path : path-like
        Base collection snapshot.
    wal_path : path-like
        Write-ahead log associated with the snapshot.
    durability : int, default=WAL_ASYNC
        WAL durability mode used after recovery.
    memory_mapped : bool, default=True
        Keep eligible snapshot regions memory-mapped.
    mmap_threshold_bytes : int, default=67108864
        Minimum eligible mapped region size.

    Returns
    -------
    Collection
        Recovered managed collection with WAL recording active.
    """
    return Collection._from_native(
        _native._recover(
            str(snapshot_path),
            str(wal_path),
            durability,
            memory_mapped,
            mmap_threshold_bytes,
        )
    )
