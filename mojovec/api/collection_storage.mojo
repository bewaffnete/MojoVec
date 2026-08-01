"""Flat/SQ8 HNSW storage dispatch for the managed Collection API."""

from std.io.file import FileHandle
from std.memory.span import Span
from std.utils import Variant

from mojovec.api.collection_codec import _index_metric
from mojovec.core.types import STORAGE_SQ8, MetricType, StorageKind
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_flat_sq8 import IndexFlatSQ8
from mojovec.index.index_hnsw import IndexHNSW
from mojovec.io.serialization import (
    read_index_hnsw_mmap,
    read_index_hnsw_sq8_mmap,
    write_index_hnsw_mmap,
    write_index_hnsw_sq8_mmap,
)


comptime FlatHNSW = IndexHNSW[IndexFlat]
comptime SQ8HNSW = IndexHNSW[IndexFlatSQ8]
comptime HNSWStorage = Variant[FlatHNSW, SQ8HNSW]


def _create_hnsw_storage(
    dimension: Int,
    storage_kind: StorageKind,
    metric_type: MetricType,
    M: Int,
    ef_construction: Int,
    ef_search: Int,
) raises -> HNSWStorage:
    """Constructs the selected concrete index once at collection creation."""
    var storage_metric = _index_metric(metric_type)
    if storage_kind == STORAGE_SQ8:
        var storage = IndexFlatSQ8(dimension, storage_metric)
        var index = SQ8HNSW(
            storage^,
            dimension,
            storage_metric,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        return HNSWStorage(index^)

    var storage = IndexFlat(dimension, storage_metric)
    var index = FlatHNSW(
        storage^,
        dimension,
        storage_metric,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    return HNSWStorage(index^)


def _storage_M(storage: HNSWStorage, kind: StorageKind) -> Int:
    if kind == STORAGE_SQ8:
        return storage.unsafe_get[SQ8HNSW]().hnsw.M
    return storage.unsafe_get[FlatHNSW]().hnsw.M


def _storage_ef_construction(storage: HNSWStorage, kind: StorageKind) -> Int:
    if kind == STORAGE_SQ8:
        return storage.unsafe_get[SQ8HNSW]().hnsw.efConstruction
    return storage.unsafe_get[FlatHNSW]().hnsw.efConstruction


def _storage_ef_search(storage: HNSWStorage, kind: StorageKind) -> Int:
    if kind == STORAGE_SQ8:
        return storage.unsafe_get[SQ8HNSW]().hnsw.efSearch
    return storage.unsafe_get[FlatHNSW]().hnsw.efSearch


def _storage_is_memory_mapped(storage: HNSWStorage, kind: StorageKind) -> Bool:
    if kind == STORAGE_SQ8:
        return (
            storage.unsafe_get[SQ8HNSW]().storage.is_memory_mapped()
            and storage.unsafe_get[SQ8HNSW]().hnsw.is_memory_mapped()
        )
    return (
        storage.unsafe_get[FlatHNSW]().storage.is_memory_mapped()
        and storage.unsafe_get[FlatHNSW]().hnsw.is_memory_mapped()
    )


def _storage_vector(
    storage: HNSWStorage,
    kind: StorageKind,
    internal_id: Int,
    dimension: Int,
) -> Span[Float32, MutUntrackedOrigin]:
    if kind == STORAGE_SQ8:
        return Span[Float32, MutUntrackedOrigin](
            ptr=storage.unsafe_get[SQ8HNSW]().storage.get_vector(internal_id),
            length=dimension,
        )
    return Span[Float32, MutUntrackedOrigin](
        ptr=storage.unsafe_get[FlatHNSW]().storage.get_vector(internal_id),
        length=dimension,
    )


def _storage_add(
    mut storage: HNSWStorage,
    kind: StorageKind,
    embeddings: Span[Float32, _],
):
    if kind == STORAGE_SQ8:
        storage.unsafe_get[SQ8HNSW]().add(embeddings)
    else:
        storage.unsafe_get[FlatHNSW]().add(embeddings)


def _storage_set_ef_search(
    mut storage: HNSWStorage, kind: StorageKind, ef: Int
):
    if kind == STORAGE_SQ8:
        storage.unsafe_get[SQ8HNSW]().hnsw.efSearch = ef
    else:
        storage.unsafe_get[FlatHNSW]().hnsw.efSearch = ef


def _storage_search(
    storage: HNSWStorage,
    kind: StorageKind,
    queries: Span[Float32, _],
    n_results: Int,
    mut distances: Span[mut=True, Float32, _],
    mut labels: Span[mut=True, Int, _],
    deleted: Span[UInt8, _],
):
    if kind == STORAGE_SQ8:
        storage.unsafe_get[SQ8HNSW]().search(
            queries, n_results, distances, labels, deleted
        )
    else:
        storage.unsafe_get[FlatHNSW]().search(
            queries, n_results, distances, labels, deleted
        )


def _write_storage(
    mut file: FileHandle,
    storage: HNSWStorage,
    kind: StorageKind,
) raises:
    if kind == STORAGE_SQ8:
        write_index_hnsw_sq8_mmap(file, storage.unsafe_get[SQ8HNSW]())
    else:
        write_index_hnsw_mmap(file, storage.unsafe_get[FlatHNSW]())


def _read_storage(
    mut file: FileHandle,
    file_size: Int,
    kind: StorageKind,
    metric_type: MetricType,
    expected_count: Int,
    use_mmap: Bool,
) raises -> HNSWStorage:
    var expected_metric = _index_metric(metric_type)
    if kind == STORAGE_SQ8:
        var index = read_index_hnsw_sq8_mmap(file, file_size)
        if (
            index.metric_type != expected_metric
            or index.storage.metric_type != expected_metric
        ):
            raise Error("Collection and SQ8 index metrics differ.")
        if not use_mmap:
            index.storage._detach_mapped()
            index.hnsw._detach_mapped()
        if index.ntotal != expected_count:
            raise Error("Collection metadata and SQ8 index size differ.")
        return HNSWStorage(index^)

    var index = read_index_hnsw_mmap(file, file_size)
    if (
        index.metric_type != expected_metric
        or index.storage.metric_type != expected_metric
    ):
        raise Error("Collection and Flat index metrics differ.")
    if not use_mmap:
        index.storage._detach_mapped()
        index.hnsw._detach_mapped()
    if index.ntotal != expected_count:
        raise Error("Collection metadata and Flat index size differ.")
    return HNSWStorage(index^)
