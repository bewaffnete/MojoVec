"""BM25 and reciprocal-rank-fusion query execution for Collection."""

from std.collections import Dict, List
from std.memory.span import Span

from mojovec.api.bm25 import BM25Index
from mojovec.api.collection_results import _build_ranked_query_results
from mojovec.api.collection_storage import HNSWStorage
from mojovec.api.collection_vector_query import _search_collection_index
from mojovec.api.metadata import Metadata
from mojovec.api.results import QueryResults
from mojovec.api.rrf import reciprocal_rank_fusion
from mojovec.core.types import MetricType, StorageKind


def _query_bm25_index(
    bm25: BM25Index,
    query_texts: List[String],
    n_results: Int,
    exclusion: List[UInt8],
    user_ids: List[Int],
    is_deleted: List[UInt8],
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
    document_by_internal: Dict[Int, Int],
    documents: List[String],
) raises -> QueryResults:
    if n_results <= 0 or n_results > 2048:
        raise Error("n_results must be between 1 and 2048.")
    if len(query_texts) == 0:
        return QueryResults(
            List[List[Int]](),
            List[List[Float32]](),
            List[List[Metadata]](),
            List[List[String]](),
            List[List[Float32]](),
        )

    var ids_storage = List[Int](capacity=len(query_texts) * n_results)
    var scores_storage = List[Float32](capacity=len(query_texts) * n_results)
    for query_text in query_texts:
        var result = bm25.search(query_text, n_results, exclusion)
        for rank in range(n_results):
            ids_storage.append(result.internal_ids[rank])
            scores_storage.append(result.scores[rank])

    return _build_ranked_query_results(
        ids_storage,
        scores_storage,
        len(query_texts),
        n_results,
        user_ids,
        is_deleted,
        metadata_by_internal,
        metadatas,
        document_by_internal,
        documents,
    )


def _query_hybrid_index(
    storage: HNSWStorage,
    storage_kind: StorageKind,
    metric_type: MetricType,
    dimension: Int,
    bm25: BM25Index,
    user_ids: List[Int],
    is_deleted: List[UInt8],
    deleted_count: Int,
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
    document_by_internal: Dict[Int, Int],
    documents: List[String],
    query_embeddings: List[Float32],
    query_texts: List[String],
    n_results: Int,
    rrf_k: Int,
    candidate_multiplier: Int,
    exclusion: List[UInt8],
) raises -> QueryResults:
    if dimension <= 0:
        raise Error("Collection dimension must be positive.")
    if len(query_embeddings) % dimension != 0:
        raise Error("Query embeddings length must be a multiple of dimension.")
    if n_results <= 0 or n_results > 2048:
        raise Error("n_results must be between 1 and 2048.")
    if rrf_k <= 0:
        raise Error("rrf_k must be positive.")
    if candidate_multiplier <= 0 or candidate_multiplier > 2048:
        raise Error("candidate_multiplier must be between 1 and 2048.")

    var num_queries = len(query_embeddings) // dimension
    if num_queries != len(query_texts):
        raise Error("Hybrid query embedding and text batch sizes must match.")
    if num_queries == 0:
        return QueryResults(
            List[List[Int]](),
            List[List[Float32]](),
            List[List[Metadata]](),
            List[List[String]](),
            List[List[Float32]](),
        )
    if len(exclusion) > 0 and len(exclusion) != len(user_ids):
        raise Error("Where filter size does not match collection size.")

    var candidate_count = min(n_results * candidate_multiplier, 2048)
    var candidate_storage_size = num_queries * candidate_count
    var vector_ids = List[Int](unsafe_uninit_length=candidate_storage_size)
    var vector_distances = List[Float32](
        unsafe_uninit_length=candidate_storage_size
    )
    var queries = Span[Float32](
        ptr=query_embeddings.unsafe_ptr(), length=len(query_embeddings)
    )
    var vector_id_span = Span[mut=True, Int](vector_ids)
    var vector_distance_span = Span[mut=True, Float32](vector_distances)

    if len(exclusion) > 0:
        var filter_span = Span[UInt8](exclusion)
        _search_collection_index(
            storage,
            storage_kind,
            metric_type,
            dimension,
            queries,
            candidate_count,
            vector_distance_span,
            vector_id_span,
            filter_span,
        )
    elif deleted_count > 0:
        var deleted_span = Span[mut=False, UInt8](is_deleted)
        _search_collection_index(
            storage,
            storage_kind,
            metric_type,
            dimension,
            queries,
            candidate_count,
            vector_distance_span,
            vector_id_span,
            deleted_span,
        )
    else:
        var empty_filter = Span[UInt8, MutUntrackedOrigin]()
        _search_collection_index(
            storage,
            storage_kind,
            metric_type,
            dimension,
            queries,
            candidate_count,
            vector_distance_span,
            vector_id_span,
            empty_filter,
        )

    var fused_ids = List[Int](capacity=num_queries * n_results)
    var fused_scores = List[Float32](capacity=num_queries * n_results)
    for query_index in range(num_queries):
        var vector_row = List[Int](capacity=candidate_count)
        var vector_offset = query_index * candidate_count
        for rank in range(candidate_count):
            vector_row.append(vector_ids[vector_offset + rank])
        var text_row = bm25.search(
            query_texts[query_index], candidate_count, exclusion
        )
        var fused = reciprocal_rank_fusion(
            vector_row, text_row.internal_ids, n_results, rrf_k
        )
        for rank in range(n_results):
            fused_ids.append(fused.internal_ids[rank])
            fused_scores.append(fused.scores[rank])

    return _build_ranked_query_results(
        fused_ids,
        fused_scores,
        num_queries,
        n_results,
        user_ids,
        is_deleted,
        metadata_by_internal,
        metadatas,
        document_by_internal,
        documents,
    )
