"""Preparation of fallible payload work for atomic collection batches."""

from std.collections import Dict, List
from std.memory.span import Span

from mojovec.api.bm25 import BM25Index
from mojovec.api.metadata import Metadata
from mojovec.io.fault_injection import (
    BATCH_FAULT_DURING_PAYLOAD_PREPARE,
    inject_batch_fault,
)


struct PreparedBatchPayloads(Movable):
    """Owned payload values whose remaining commit operations cannot raise."""

    var previous_internal_ids: List[Int]
    var metadatas: List[Metadata]
    var documents: List[String]
    var document_tokens: List[List[String]]

    def __init__(out self):
        self.previous_internal_ids = List[Int]()
        self.metadatas = List[Metadata]()
        self.documents = List[String]()
        self.document_tokens = List[List[String]]()

    def __init__(
        out self,
        var previous_internal_ids: List[Int],
        var metadatas: List[Metadata],
        var documents: List[String],
        var document_tokens: List[List[String]],
    ):
        self.previous_internal_ids = previous_internal_ids^
        self.metadatas = metadatas^
        self.documents = documents^
        self.document_tokens = document_tokens^


def _prepare_batch_payloads(
    ids: Span[Int, _],
    id_to_internal: Dict[Int, Int],
    metadata_by_internal: Dict[Int, Int],
    stored_metadatas: List[Metadata],
    document_by_internal: Dict[Int, Int],
    stored_documents: List[String],
    bm25: BM25Index,
    metadatas: List[Metadata],
    has_metadatas: Bool,
    documents: List[String],
    has_documents: Bool,
    fault_point: Int,
) raises -> PreparedBatchPayloads:
    """Materializes inheritance and tokenization without changing collection state."""
    var previous_internal_ids = List[Int](capacity=len(ids))
    var prepared_metadatas = List[Metadata](capacity=len(ids))
    var prepared_documents = List[String](capacity=len(ids))
    var prepared_document_tokens = List[List[String]](capacity=len(ids))

    for index in range(len(ids)):
        var previous = -1
        if ids[index] in id_to_internal:
            previous = id_to_internal[ids[index]]
        previous_internal_ids.append(previous)

        var prepared_metadata = Metadata()
        if has_metadatas:
            prepared_metadata = metadatas[index].copy()
        elif previous >= 0 and previous in metadata_by_internal:
            prepared_metadata = stored_metadatas[
                metadata_by_internal[previous]
            ].copy()
        prepared_metadatas.append(prepared_metadata^)

        var prepared_document = String("")
        if has_documents:
            prepared_document = documents[index].copy()
        elif previous >= 0 and previous in document_by_internal:
            prepared_document = stored_documents[
                document_by_internal[previous]
            ].copy()

        var tokens = List[String]()
        if prepared_document.byte_length() > 0:
            tokens = bm25._prepare_document(prepared_document)
        prepared_documents.append(prepared_document^)
        prepared_document_tokens.append(tokens^)

        # With at least two records this deterministically exercises a fault
        # after one complete payload row has been prepared.
        inject_batch_fault(
            fault_point,
            BATCH_FAULT_DURING_PAYLOAD_PREPARE,
        )

    return PreparedBatchPayloads(
        previous_internal_ids^,
        prepared_metadatas^,
        prepared_documents^,
        prepared_document_tokens^,
    )
