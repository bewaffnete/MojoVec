"""Allocation-aware vector query execution shared by managed frontends."""

from std.collections import List
from std.math import sqrt
from std.memory.span import Span

from mojovec.api.collection_codec import METRIC_COSINE
from mojovec.api.collection_storage import HNSWStorage, _storage_search
from mojovec.core.types import METRIC_L2, MetricType, StorageKind


def _normalize_vectors(
    vectors: Span[Float32, _], dimension: Int
) raises -> List[Float32]:
    """Returns row-wise L2-normalized vectors for cosine search."""
    var normalized = List[Float32](unsafe_uninit_length=len(vectors))
    var vector_count = len(vectors) // dimension
    for vector_index in range(vector_count):
        var offset = vector_index * dimension
        var squared_norm: Float64 = 0.0
        for component in range(dimension):
            var value = Float64(vectors[offset + component])
            squared_norm += value * value
        if not (squared_norm > 0.0 and squared_norm <= Float64.MAX):
            raise Error("Cosine embeddings must be finite non-zero vectors.")
        var inverse_norm = 1.0 / sqrt(squared_norm)
        for component in range(dimension):
            normalized[offset + component] = Float32(
                Float64(vectors[offset + component]) * inverse_norm
            )
    return normalized^


def _search_collection_index(
    storage: HNSWStorage,
    storage_kind: StorageKind,
    metric_type: MetricType,
    dimension: Int,
    queries: Span[Float32, _],
    n_results: Int,
    mut distances: Span[mut=True, Float32, _],
    mut labels: Span[mut=True, Int, _],
    deleted: Span[UInt8, _],
) raises:
    if metric_type == METRIC_COSINE:
        var normalized = _normalize_vectors(queries, dimension)
        _storage_search(
            storage,
            storage_kind,
            Span[Float32](normalized),
            n_results,
            distances,
            labels,
            deleted,
        )
        return
    _storage_search(
        storage,
        storage_kind,
        queries,
        n_results,
        distances,
        labels,
        deleted,
    )


def _convert_similarity_distances(
    metric_type: MetricType,
    labels: Span[Int, _],
    mut distances: Span[mut=True, Float32, _],
    count: Int,
):
    if metric_type == METRIC_L2:
        return
    for index in range(count):
        if labels[index] >= 0:
            distances[index] = 1.0 - distances[index]


def _query_collection_into(
    storage: HNSWStorage,
    storage_kind: StorageKind,
    metric_type: MetricType,
    dimension: Int,
    user_ids: List[Int],
    is_deleted: List[UInt8],
    deleted_count: Int,
    query_embeddings: Span[Float32, _],
    n_results: Int,
    mut ids: Span[mut=True, Int, _],
    mut distances: Span[mut=True, Float32, _],
) raises:
    if dimension <= 0:
        raise Error("Collection dimension must be positive.")
    if len(query_embeddings) % dimension != 0:
        raise Error("Query embeddings length must be a multiple of dimension.")
    if n_results <= 0 or n_results > 2048:
        raise Error("n_results must be between 1 and 2048.")

    var num_queries = len(query_embeddings) // dimension
    var output_size = num_queries * n_results
    if len(ids) < output_size or len(distances) < output_size:
        raise Error("Output buffers are smaller than query_count * n_results.")
    if num_queries == 0:
        return

    if deleted_count == 0:
        var empty_filter = Span[UInt8, MutUntrackedOrigin]()
        _search_collection_index(
            storage,
            storage_kind,
            metric_type,
            dimension,
            query_embeddings,
            n_results,
            distances,
            ids,
            empty_filter,
        )
    else:
        var deleted = Span[mut=False, UInt8](is_deleted)
        _search_collection_index(
            storage,
            storage_kind,
            metric_type,
            dimension,
            query_embeddings,
            n_results,
            distances,
            ids,
            deleted,
        )

    _convert_similarity_distances(metric_type, ids, distances, output_size)
    for index in range(output_size):
        var internal_id = ids[index]
        ids[index] = (
            user_ids[internal_id] if internal_id >= 0
            and internal_id < len(user_ids) else -1
        )


def _query_collection_into_filtered(
    storage: HNSWStorage,
    storage_kind: StorageKind,
    metric_type: MetricType,
    dimension: Int,
    user_ids: List[Int],
    query_embeddings: Span[Float32, _],
    n_results: Int,
    mut ids: Span[mut=True, Int, _],
    mut distances: Span[mut=True, Float32, _],
    exclusion: Span[UInt8, _],
) raises:
    if dimension <= 0:
        raise Error("Collection dimension must be positive.")
    if len(query_embeddings) % dimension != 0:
        raise Error("Query embeddings length must be a multiple of dimension.")
    if n_results <= 0 or n_results > 2048:
        raise Error("n_results must be between 1 and 2048.")

    var num_queries = len(query_embeddings) // dimension
    var output_size = num_queries * n_results
    if len(ids) < output_size or len(distances) < output_size:
        raise Error("Output buffers are smaller than query_count * n_results.")
    if len(exclusion) != len(user_ids):
        raise Error("Where filter size does not match collection size.")
    if num_queries == 0:
        return

    _search_collection_index(
        storage,
        storage_kind,
        metric_type,
        dimension,
        query_embeddings,
        n_results,
        distances,
        ids,
        exclusion,
    )
    _convert_similarity_distances(metric_type, ids, distances, output_size)
    for index in range(output_size):
        var internal_id = ids[index]
        ids[index] = (
            user_ids[internal_id] if internal_id >= 0
            and internal_id < len(user_ids) else -1
        )
