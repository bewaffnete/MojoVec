"""Managed QueryResults materialization for vector and ranked searches."""

from std.collections import Dict, List

from mojovec.api.metadata import Metadata
from mojovec.api.results import QueryResults


def _result_metadata(
    internal_id: Int,
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
) raises -> Metadata:
    if internal_id in metadata_by_internal:
        return metadatas[metadata_by_internal[internal_id]].copy()
    return Metadata()


def _result_document(
    internal_id: Int,
    document_by_internal: Dict[Int, Int],
    documents: List[String],
) raises -> String:
    if internal_id in document_by_internal:
        return documents[document_by_internal[internal_id]].copy()
    return String("")


def _build_vector_query_results(
    ids_storage: List[Int],
    distances_storage: List[Float32],
    num_queries: Int,
    n_results: Int,
    id_to_internal: Dict[Int, Int],
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
    document_by_internal: Dict[Int, Int],
    documents: List[String],
) raises -> QueryResults:
    """Materializes aligned vector results and optional payload rows."""
    var include_metadatas = len(metadatas) > 0
    var include_documents = len(documents) > 0

    # Preserve the vector-only fast path: it performs no ID-map lookup and no
    # payload branch for each neighbor.
    if not include_metadatas and not include_documents:
        var vector_ids = List[List[Int]](capacity=num_queries)
        var vector_distances = List[List[Float32]](capacity=num_queries)
        for query_index in range(num_queries):
            var row_ids = List[Int](capacity=n_results)
            var row_distances = List[Float32](capacity=n_results)
            for rank in range(n_results):
                var offset = query_index * n_results + rank
                row_ids.append(ids_storage[offset])
                row_distances.append(distances_storage[offset])
            vector_ids.append(row_ids^)
            vector_distances.append(row_distances^)
        return QueryResults(
            vector_ids^,
            vector_distances^,
            List[List[Metadata]](),
            List[List[String]](),
            List[List[Float32]](),
        )

    var all_ids = List[List[Int]](capacity=num_queries)
    var all_distances = List[List[Float32]](capacity=num_queries)
    var all_metadatas = List[List[Metadata]]()
    var all_documents = List[List[String]]()
    if include_metadatas:
        all_metadatas = List[List[Metadata]](capacity=num_queries)
    if include_documents:
        all_documents = List[List[String]](capacity=num_queries)

    for query_index in range(num_queries):
        var row_ids = List[Int](capacity=n_results)
        var row_distances = List[Float32](capacity=n_results)
        var row_metadatas = List[Metadata]()
        var row_documents = List[String]()
        if include_metadatas:
            row_metadatas = List[Metadata](capacity=n_results)
        if include_documents:
            row_documents = List[String](capacity=n_results)

        for rank in range(n_results):
            var offset = query_index * n_results + rank
            var record_id = ids_storage[offset]
            row_ids.append(record_id)
            row_distances.append(distances_storage[offset])

            var internal_id = -1
            if record_id >= 0 and record_id in id_to_internal:
                internal_id = id_to_internal[record_id]
            if include_metadatas:
                row_metadatas.append(
                    _result_metadata(
                        internal_id, metadata_by_internal, metadatas
                    ) if internal_id
                    >= 0 else Metadata()
                )
            if include_documents:
                row_documents.append(
                    _result_document(
                        internal_id, document_by_internal, documents
                    ) if internal_id
                    >= 0 else String("")
                )

        all_ids.append(row_ids^)
        all_distances.append(row_distances^)
        if include_metadatas:
            all_metadatas.append(row_metadatas^)
        if include_documents:
            all_documents.append(row_documents^)

    return QueryResults(
        all_ids^,
        all_distances^,
        all_metadatas^,
        all_documents^,
        List[List[Float32]](),
    )


def _build_ranked_query_results(
    internal_ids_storage: List[Int],
    scores_storage: List[Float32],
    num_queries: Int,
    n_results: Int,
    user_ids: List[Int],
    is_deleted: List[UInt8],
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
    document_by_internal: Dict[Int, Int],
    documents: List[String],
) raises -> QueryResults:
    """Materializes ranked scores and aligned collection payloads."""
    var include_metadatas = len(metadatas) > 0
    var include_documents = len(documents) > 0
    var all_ids = List[List[Int]](capacity=num_queries)
    var all_scores = List[List[Float32]](capacity=num_queries)
    var all_metadatas = List[List[Metadata]]()
    var all_documents = List[List[String]]()
    if include_metadatas:
        all_metadatas = List[List[Metadata]](capacity=num_queries)
    if include_documents:
        all_documents = List[List[String]](capacity=num_queries)

    for query_index in range(num_queries):
        var row_ids = List[Int](capacity=n_results)
        var row_scores = List[Float32](capacity=n_results)
        var row_metadatas = List[Metadata]()
        var row_documents = List[String]()
        if include_metadatas:
            row_metadatas = List[Metadata](capacity=n_results)
        if include_documents:
            row_documents = List[String](capacity=n_results)

        for rank in range(n_results):
            var offset = query_index * n_results + rank
            var internal_id = internal_ids_storage[offset]
            var is_valid = (
                internal_id >= 0
                and internal_id < len(user_ids)
                and is_deleted[internal_id] == 0
            )
            row_ids.append(user_ids[internal_id] if is_valid else -1)
            row_scores.append(scores_storage[offset])
            if include_metadatas:
                row_metadatas.append(
                    _result_metadata(
                        internal_id, metadata_by_internal, metadatas
                    ) if is_valid else Metadata()
                )
            if include_documents:
                row_documents.append(
                    _result_document(
                        internal_id, document_by_internal, documents
                    ) if is_valid else String("")
                )

        all_ids.append(row_ids^)
        all_scores.append(row_scores^)
        if include_metadatas:
            all_metadatas.append(row_metadatas^)
        if include_documents:
            all_documents.append(row_documents^)

    return QueryResults(
        all_ids^,
        List[List[Float32]](),
        all_metadatas^,
        all_documents^,
        all_scores^,
    )
