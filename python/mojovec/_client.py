"""Client interface for creating, loading, and managing collections."""

from __future__ import annotations

from os import PathLike
from typing import Any

from ._collection import Collection, load, recover
from ._ivfpq import IVFPQCollection


class Client:
    """High-level client entry point for MojoVec collections.

    The client provides a unified interface for instantiating new vector
    collections, compressed IVF-PQ collections, as well as loading existing
    snapshots and recovering collections from write-ahead logs.
    """

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Client()"

    def create_collection(
        self,
        name: str,
        dimension: int,
        M: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16,
        quantized: bool = True,
        metric: str = "l2",
        **kwargs: Any,
    ) -> Collection:
        """Create and return a new managed HNSW collection.

        Parameters
        ----------
        name : str
            Name of the collection.
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
        **kwargs : Any
            Additional keyword arguments passed to :class:`Collection`.

        Returns
        -------
        Collection
            A newly initialized managed vector collection.
        """
        return Collection(
            dimension=dimension,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            quantized=quantized,
            metric=metric,
            name=name,
            **kwargs,
        )

    def create_ivfpq_collection(
        self,
        name: str,
        dimension: int,
        nlist: int = 100,
        M: int = 16,
        nprobe: int = 0,
        metric: str = "l2",
    ) -> IVFPQCollection:
        """Create and return a new compressed IVF-PQ collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        dimension : int
            Number of components in every vector. Must be divisible by ``M``.
        nlist : int, default=100
            Number of coarse inverted lists.
        M : int, default=16
            Number of byte-sized PQ codes (subvectors) stored per vector.
        nprobe : int, default=0
            Number of coarse lists searched. 0 selects default (min(10, nlist)).
        metric : {"l2", "cosine", "ip"}, default="l2"
            Distance metric.

        Returns
        -------
        IVFPQCollection
            A newly initialized IVF-PQ vector collection.
        """
        return IVFPQCollection(
            dimension=dimension,
            nlist=nlist,
            pq_subvectors=M,
            nprobe=nprobe,
            metric=metric,
            name=name,
        )

    def load(self, path: str | PathLike[str]) -> Collection:
        """Load an existing collection snapshot from disk.

        Parameters
        ----------
        path : str | PathLike[str]
            Path to the saved collection snapshot.

        Returns
        -------
        Collection
            The loaded vector collection.
        """
        return load(path)

    def recover(
        self,
        path: str | PathLike[str],
        wal_directory: str | PathLike[str],
    ) -> Collection:
        """Recover a collection from a snapshot and its write-ahead log.

        Parameters
        ----------
        path : str | PathLike[str]
            Path to the base collection snapshot.
        wal_directory : str | PathLike[str]
            Path to the write-ahead log file or directory.

        Returns
        -------
        Collection
            Recovered managed collection with WAL recording active.
        """
        return recover(path, wal_directory)
