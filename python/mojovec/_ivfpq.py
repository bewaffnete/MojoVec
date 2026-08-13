"""Documented Python facade for the managed IVF-PQ collection."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from os import PathLike
from typing import Any, TypedDict

from ._collection import _flatten_embeddings, _validated_numpy_vectors
from ._runtime import _native
from ._types import QueryResult


class IVFPQStats(TypedDict):
    """Shape, training state, and active probing configuration."""

    count: int
    dimension: int
    nlist: int
    M: int
    nprobe: int
    trained: bool
    metric: str
    code_size_bytes: int


class IVFPQCollection:
    """Compressed IVF-PQ vector collection.

    Parameters
    ----------
    dimension:
        Number of Float32 components per vector. It must be divisible by
        ``pq_subvectors``.
    nlist:
        Number of coarse inverted lists.
    pq_subvectors:
        Number of byte-sized PQ codes stored per vector.
    nprobe:
        Number of coarse lists searched. ``None`` selects ``min(10, nlist)``.
    metric:
        ``"l2"``, ``"cosine"``, or ``"ip"``.
    name:
        Optional collection name preserved by :meth:`save`.

    Notes
    -----
    Training requires at least ``max(nlist, 256)`` representative vectors.
    :meth:`add` automatically trains on its first batch when needed.
    """

    def __init__(
        self,
        dimension: int,
        nlist: int = 100,
        pq_subvectors: int = 16,
        nprobe: int | None = None,
        metric: str = "l2",
        name: str = "",
    ) -> None:
        native_nprobe = 0 if nprobe is None else nprobe
        self._inner = _native._IVFPQCollection(
            dimension,
            nlist,
            pq_subvectors,
            native_nprobe,
            name,
            metric,
        )

    @classmethod
    def _from_native(cls, inner: Any) -> IVFPQCollection:
        collection = cls.__new__(cls)
        collection._inner = inner
        return collection

    @classmethod
    def load(cls, path: str | PathLike[str]) -> IVFPQCollection:
        """Load an owned IVF-PQ snapshot."""
        return cls._from_native(_native._load_ivfpq(str(path)))

    def __repr__(self) -> str:
        return (
            f"IVFPQCollection(name={self.name()!r}, "
            f"dimension={self.dimension()}, metric={self.metric()!r}, "
            f"count={self.count()}, nlist={self.nlist()}, "
            f"pq_subvectors={self.pq_subvectors()}, "
            f"nprobe={self.nprobe()}, trained={self.is_trained()})"
        )

    def name(self) -> str:
        """Return the collection name."""
        return self._inner.name()

    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._inner.dimension()

    def metric(self) -> str:
        """Return ``"l2"``, ``"cosine"``, or ``"ip"``."""
        return self._inner.metric()

    def count(self) -> int:
        """Return the number of indexed vectors."""
        return self._inner.count()

    def is_trained(self) -> bool:
        """Return whether coarse and PQ codebooks have been trained."""
        return self._inner.is_trained()

    def nlist(self) -> int:
        """Return the number of coarse inverted lists."""
        return self._inner.nlist()

    def pq_subvectors(self) -> int:
        """Return the number of byte-sized codes stored per vector."""
        return self._inner.pq_subvectors()

    def nprobe(self) -> int:
        """Return the number of lists searched per query."""
        return self._inner.nprobe()

    def set_nprobe(self, nprobe: int) -> None:
        """Set the recall/speed tradeoff between 1 and ``nlist``."""
        self._inner.set_nprobe(nprobe)

    def stats(self) -> IVFPQStats:
        """Return collection shape, training state, and code size."""
        return self._inner.stats()

    def train(
        self,
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
    ) -> None:
        """Train codebooks on representative vectors without adding them."""
        flat = _flatten_embeddings(embeddings, self.dimension())
        self._inner.train(flat)

    def add(
        self,
        ids: Sequence[int],
        embeddings: Sequence[Real] | Sequence[Sequence[Real]],
    ) -> None:
        """Add unique IDs, automatically training on the first batch."""
        flat = _flatten_embeddings(embeddings, self.dimension())
        expected = len(ids) * self.dimension()
        if len(flat) != expected:
            raise ValueError(
                f"embeddings contain {len(flat)} values; expected {expected}"
            )
        self._inner.add(ids, flat)

    def train_numpy(self, embeddings: Any) -> None:
        """Train directly from a contiguous ``float32`` NumPy matrix."""
        import numpy as np

        array = np.asarray(embeddings)
        if (
            array.dtype != np.float32
            or array.ndim != 2
            or array.shape[1] != self.dimension()
            or not array.flags.c_contiguous
        ):
            raise TypeError(
                "embeddings must be a contiguous (n, dimension) "
                "numpy.float32 array"
            )
        self._inner.train_numpy(array)

    def add_numpy(self, ids: Any, embeddings: Any) -> None:
        """Add contiguous ``int64`` IDs and ``float32`` vectors zero-copy."""
        ids_array, embeddings_array = _validated_numpy_vectors(
            ids, embeddings, self.dimension()
        )
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(
                (-1, self.dimension())
            )
        self._inner.add_numpy(ids_array, embeddings_array)

    def query(
        self,
        query_embeddings: Sequence[Real] | Sequence[Sequence[Real]],
        n_results: int = 10,
    ) -> QueryResult:
        """Return approximate nearest neighbors for one or more vectors."""
        flat = _flatten_embeddings(query_embeddings, self.dimension())
        return self._inner.query(flat, n_results)

    def query_numpy(
        self, query_embeddings: Any, n_results: int = 10
    ) -> QueryResult:
        """Query a contiguous ``float32`` NumPy matrix without conversion."""
        import numpy as np

        array = np.asarray(query_embeddings)
        if array.dtype != np.float32:
            raise TypeError("query_embeddings must have numpy.float32 dtype")
        if array.ndim == 1:
            if array.size % self.dimension() != 0:
                raise ValueError(
                    "query embeddings length must be a multiple of dimension"
                )
            array = array.reshape((-1, self.dimension()))
        if (
            array.ndim != 2
            or array.shape[1] != self.dimension()
            or not array.flags.c_contiguous
        ):
            raise ValueError(
                "query_embeddings must be a contiguous (n, dimension) array"
            )
        return self._inner.query_numpy(array, n_results)

    def save(self, path: str | PathLike[str]) -> None:
        """Atomically persist a checksummed owned IVF-PQ snapshot."""
        self._inner.save(str(path))
