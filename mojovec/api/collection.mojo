from std.collections import Dict, List, Optional
from std.memory.span import Span
from std.time import perf_counter_ns

from mojovec.api.bm25 import BM25Index
from mojovec.api.collection_codec import (
    METRIC_COSINE,
    _metric_name,
    _parse_metric,
)
from mojovec.api.collection_batch import (
    PreparedBatchPayloads,
    _prepare_batch_payloads,
)
from mojovec.api.collection_filter import _build_where_filter
from mojovec.api.metadata import Metadata
from mojovec.api.metadata_bitmap import MetadataBitmapIndex
from mojovec.api.results import CollectionStats, CompactReport, QueryResults
from mojovec.api.collection_results import (
    _build_vector_query_results,
)
from mojovec.api.collection_ranked_query import (
    _query_bm25_index,
    _query_hybrid_index,
)
from mojovec.api.collection_storage import (
    HNSWStorage,
    _create_hnsw_storage,
    _storage_M,
    _storage_add,
    _storage_ef_construction,
    _storage_ef_search,
    _storage_is_memory_mapped,
    _storage_set_ef_search,
    _storage_vector,
)
from mojovec.api.collection_vector_query import (
    _normalize_vectors,
    _query_collection_into,
    _query_collection_into_filtered,
)
from mojovec.api.rrf import (
    RRF_DEFAULT_CANDIDATE_MULTIPLIER,
    RRF_DEFAULT_K,
)
from mojovec.api.where import Where
from mojovec.core.types import (
    MetricType,
    STORAGE_FLAT,
    STORAGE_SQ8,
    StorageKind,
)
from mojovec.core.validation import (
    _validate_hnsw_parameters,
    _validate_vector_dimension,
)
from mojovec.io.wal import (
    WAL_ASYNC,
    WAL_OPERATION_DELETE,
    WAL_OPERATION_WRITE,
    WAL_SYNC,
    WalDurability,
    WalReader,
    WriteAheadLog,
)
from mojovec.io.collection_snapshot import (
    _read_collection_snapshot,
    _save_collection_snapshot,
)
from mojovec.io.fault_injection import (
    BATCH_FAULT_AFTER_PAYLOAD_PREPARE,
    BATCH_FAULT_AFTER_VECTOR_PREPARE,
    BATCH_FAULT_AFTER_WAL_APPEND,
    BATCH_FAULT_DURING_PAYLOAD_PREPARE,
    BATCH_FAULT_NONE,
    inject_batch_fault,
)


comptime DEFAULT_MMAP_THRESHOLD_BYTES = 64 * 1024 * 1024
comptime COMPACTION_MAX_BATCH_VECTORS = 16_384
comptime COMPACTION_MAX_BATCH_COMPONENTS = 2_097_152


struct Collection(Movable, Writable):
    """A Chroma-style vector and document collection with HNSW and BM25."""

    var _name: String
    var _dimension: Int
    var _storage_kind: StorageKind
    var _metric_type: MetricType
    var _hnsw: HNSWStorage
    var _user_ids: List[Int]
    var _is_deleted: List[UInt8]
    var _metadata_by_internal: Dict[Int, Int]
    var _metadatas: List[Metadata]
    var _metadata_index: MetadataBitmapIndex
    var _document_by_internal: Dict[Int, Int]
    var _documents: List[String]
    var _bm25: BM25Index
    var _id_to_internal: Dict[Int, Int]
    var _identity: Int
    var _wal: Optional[WriteAheadLog]
    var _applied_sequence: Int
    var _replaying_wal: Bool

    def __init__(
        out self,
        dimension: Int,
        M: Int = 32,
        ef_construction: Int = 40,
        ef_search: Int = 16,
        quantized: Bool = True,
        name: String = "",
        metric: String = "l2",
    ) raises:
        # Validate every allocation-sensitive value before constructing either
        # the Flat or SQ8 storage backend.
        _validate_vector_dimension(dimension)
        _validate_hnsw_parameters(M, ef_construction, ef_search)
        self._name = name.copy()
        self._dimension = dimension
        self._storage_kind = STORAGE_SQ8 if quantized else STORAGE_FLAT
        self._metric_type = _parse_metric(metric)
        self._hnsw = _create_hnsw_storage(
            dimension,
            self._storage_kind,
            self._metric_type,
            M,
            ef_construction,
            ef_search,
        )

        self._user_ids = List[Int]()
        self._is_deleted = List[UInt8]()
        self._metadata_by_internal = Dict[Int, Int]()
        self._metadatas = List[Metadata]()
        self._metadata_index = MetadataBitmapIndex()
        self._document_by_internal = Dict[Int, Int]()
        self._documents = List[String]()
        self._bm25 = BM25Index()
        self._id_to_internal = Dict[Int, Int]()
        self._identity = Int(perf_counter_ns())
        self._wal = None
        self._applied_sequence = 0
        self._replaying_wal = False

    def __init__(out self, *, deinit take: Self):
        self._name = take._name^
        self._dimension = take._dimension
        self._storage_kind = take._storage_kind
        self._metric_type = take._metric_type
        self._hnsw = take._hnsw^
        self._user_ids = take._user_ids^
        self._is_deleted = take._is_deleted^
        self._metadata_by_internal = take._metadata_by_internal^
        self._metadatas = take._metadatas^
        self._metadata_index = take._metadata_index^
        self._document_by_internal = take._document_by_internal^
        self._documents = take._documents^
        self._bm25 = take._bm25^
        self._id_to_internal = take._id_to_internal^
        self._identity = take._identity
        self._wal = take._wal^
        self._applied_sequence = take._applied_sequence
        self._replaying_wal = take._replaying_wal

    def write_to[W: Writer](self, mut writer: W):
        var storage = "sq8" if self.is_quantized() else "flat"
        writer.write(
            "Collection(name=",
            self._name,
            ", dimension=",
            self._dimension,
            ", storage=",
            storage,
            ", metric=",
            _metric_name(self._metric_type),
            ", count=",
            self.count(),
            ")",
        )

    def name(self) -> String:
        return self._name.copy()

    def dimension(self) -> Int:
        return self._dimension

    def storage_kind(self) -> StorageKind:
        return self._storage_kind

    def metric(self) -> String:
        """Returns `"l2"`, `"cosine"`, or `"ip"`."""
        return _metric_name(self._metric_type)

    def is_quantized(self) -> Bool:
        return self._storage_kind == STORAGE_SQ8

    def count(self) -> Int:
        """Returns the number of active records."""
        return len(self._id_to_internal)

    def count_deleted(self) -> Int:
        return len(self._user_ids) - len(self._id_to_internal)

    def get_metadata(self, record_id: Int) raises -> Metadata:
        """Returns an owned copy of the active record's metadata."""
        if record_id not in self._id_to_internal:
            raise Error("Cannot get metadata for an ID that does not exist.")
        var internal_id = self._id_to_internal[record_id]
        return self._metadata_for_internal(internal_id)

    def _metadata_for_internal(self, internal_id: Int) raises -> Metadata:
        if internal_id in self._metadata_by_internal:
            var metadata_index = self._metadata_by_internal[internal_id]
            return self._metadatas[metadata_index].copy()
        return Metadata()

    def _store_metadata(
        mut self, internal_id: Int, metadata: Metadata
    ):
        if metadata.count() == 0:
            return
        self._metadata_by_internal[internal_id] = len(self._metadatas)
        self._metadatas.append(metadata.copy())
        self._metadata_index.add(internal_id, metadata)

    def get_document(self, record_id: Int) raises -> String:
        """Returns an owned copy of an active record's document."""
        if record_id not in self._id_to_internal:
            raise Error("Cannot get document for an ID that does not exist.")
        var internal_id = self._id_to_internal[record_id]
        if internal_id not in self._document_by_internal:
            raise Error("Record does not have a document.")
        return self._documents[
            self._document_by_internal[internal_id]
        ].copy()

    def _document_for_internal(self, internal_id: Int) raises -> String:
        if internal_id in self._document_by_internal:
            return self._documents[
                self._document_by_internal[internal_id]
            ].copy()
        return String("")

    def _store_document(
        mut self, internal_id: Int, document: String
    ) raises:
        # Empty strings represent an absent document in managed results and
        # allow update/upsert to remove a previous document explicitly.
        if document.byte_length() == 0:
            return
        var tokens = self._bm25._prepare_document(document)
        self._store_document_prepared(internal_id, document, tokens)

    def _store_document_prepared(
        mut self,
        internal_id: Int,
        document: String,
        tokens: List[String],
    ):
        """Commits a document whose fallible text analysis is complete."""
        if document.byte_length() == 0:
            return
        self._document_by_internal[internal_id] = len(self._documents)
        self._documents.append(document.copy())
        if (
            internal_id >= len(self._is_deleted)
            or self._is_deleted[internal_id] == 0
        ):
            self._bm25._add_prepared(internal_id, tokens)

    def stats(self) -> CollectionStats:
        """Returns active/deleted counts and the current HNSW configuration."""
        var active_count = self.count()
        var total_count = len(self._user_ids)
        var deleted_count = total_count - active_count
        var deleted_ratio: Float64 = 0.0
        if total_count > 0:
            deleted_ratio = Float64(deleted_count) / Float64(total_count)

        var M = _storage_M(self._hnsw, self._storage_kind)
        var ef_construction = _storage_ef_construction(
            self._hnsw, self._storage_kind
        )
        var ef_search = _storage_ef_search(
            self._hnsw, self._storage_kind
        )

        return CollectionStats(
            active_count,
            deleted_count,
            total_count,
            deleted_ratio,
            self._dimension,
            self.is_quantized(),
            M,
            ef_construction,
            ef_search,
        )

    def is_memory_mapped(self) -> Bool:
        """Returns whether vector and graph arrays currently use file mappings."""
        return _storage_is_memory_mapped(self._hnsw, self._storage_kind)

    def _get_vector(
        self, internal_id: Int
    ) -> Span[Float32, MutUntrackedOrigin]:
        return _storage_vector(
            self._hnsw,
            self._storage_kind,
            internal_id,
            self._dimension,
        )

    def compact(mut self) raises -> CompactReport:
        """
        Rebuilds the HNSW index from active records and removes old versions.

        The replacement is installed only after the complete new index has
        been built. If rebuilding raises, the original collection is left
        unchanged. Live vectors are copied in bounded batches.
        """
        var before = self.stats()
        if before.deleted_count == 0:
            return CompactReport(False, before.copy(), before^, 0, 0.0)

        var started = perf_counter_ns()
        var rebuilt = Collection(
            before.dimension,
            M=before.M,
            ef_construction=before.ef_construction,
            ef_search=before.ef_search,
            quantized=before.quantized,
            name=self._name,
            metric=self.metric(),
        )
        var batch_size = max(
            1,
            min(
                COMPACTION_MAX_BATCH_VECTORS,
                COMPACTION_MAX_BATCH_COMPONENTS // self._dimension,
            ),
        )
        var batch_ids = List[Int](capacity=batch_size)
        var batch_embeddings = List[Float32](
            capacity=batch_size * self._dimension
        )
        var batch_metadatas = List[Metadata](capacity=batch_size)
        var batch_documents = List[String](capacity=batch_size)

        for internal_id in range(len(self._user_ids)):
            if self._is_deleted[internal_id] > 0:
                continue

            batch_ids.append(self._user_ids[internal_id])
            batch_metadatas.append(
                self._metadata_for_internal(internal_id)
            )
            batch_documents.append(
                self._document_for_internal(internal_id)
            )
            var vector = self._get_vector(internal_id)
            for dim in range(self._dimension):
                batch_embeddings.append(vector[dim])

            if len(batch_ids) == batch_size:
                rebuilt.add(
                    batch_ids,
                    batch_embeddings,
                    batch_metadatas,
                    batch_documents,
                )
                batch_ids.clear()
                batch_embeddings.clear()
                batch_metadatas.clear()
                batch_documents.clear()

        if len(batch_ids) > 0:
            rebuilt.add(
                batch_ids,
                batch_embeddings,
                batch_metadatas,
                batch_documents,
            )

        rebuilt._identity = self._identity
        rebuilt._applied_sequence = self._applied_sequence
        if self._wal:
            rebuilt._wal = self._wal.take()
        self = rebuilt^

        var after = self.stats()
        var elapsed_seconds = (
            Float64(perf_counter_ns() - started) / 1_000_000_000.0
        )
        var reclaimed_records = before.total_count - after.total_count
        return CompactReport(
            True,
            before^,
            after^,
            reclaimed_records,
            elapsed_seconds,
        )

    def compact_if_needed(
        mut self, deleted_ratio: Float64 = 0.20
    ) raises -> CompactReport:
        """
        Compacts when deleted records occupy at least `deleted_ratio`.

        The threshold is inclusive and must be between 0.0 and 1.0. A
        collection without deleted records is always a no-op.
        """
        if deleted_ratio < 0.0 or deleted_ratio > 1.0:
            raise Error("deleted_ratio must be between 0.0 and 1.0.")

        var before = self.stats()
        if (
            before.deleted_count == 0
            or before.deleted_ratio < deleted_ratio
        ):
            return CompactReport(False, before.copy(), before^, 0, 0.0)
        return self.compact()

    def _validate_shape(
        self, ids: Span[Int, _], embeddings: Span[Float32, _]
    ) raises:
        if self._dimension <= 0:
            raise Error("Collection dimension must be positive.")
        if len(embeddings) != len(ids) * self._dimension:
            raise Error(
                "Embeddings length must equal len(ids) * collection dimension."
            )

    def _validate_unique_batch(self, ids: Span[Int, _]) raises:
        var seen = Dict[Int, Bool]()
        for i in range(len(ids)):
            var record_id = ids[i]
            if record_id in seen:
                raise Error("Duplicate IDs are not allowed in one operation.")
            seen[record_id] = True

    def _normalize_vectors(
        self,
        vectors: Span[Float32, _],
    ) raises -> List[Float32]:
        return _normalize_vectors(vectors, self._dimension)

    def _add_prepared_to_index(
        mut self,
        embeddings: Span[Float32, _],
    ):
        _storage_add(self._hnsw, self._storage_kind, embeddings)

    def _append_records(
        mut self,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
        replace_existing: Bool,
        metadatas: List[Metadata],
        has_metadatas: Bool,
        documents: List[String],
        has_documents: Bool,
    ) raises:
        self._append_records_with_fault(
            ids,
            embeddings,
            replace_existing,
            metadatas,
            has_metadatas,
            documents,
            has_documents,
            BATCH_FAULT_NONE,
        )

    def _append_records_with_fault(
        mut self,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
        replace_existing: Bool,
        metadatas: List[Metadata],
        has_metadatas: Bool,
        documents: List[String],
        has_documents: Bool,
        fault_point: Int,
    ) raises:
        self._validate_shape(ids, embeddings)
        self._validate_unique_batch(ids)
        if has_metadatas and len(metadatas) != len(ids):
            raise Error("Metadata length must equal len(ids).")
        if has_documents and len(documents) != len(ids):
            raise Error("Documents length must equal len(ids).")
        if len(ids) == 0:
            return
        if not replace_existing:
            for i in range(len(ids)):
                if ids[i] in self._id_to_internal:
                    raise Error("ID already exists; use upsert or update.")

        # Validate and normalize cosine rows before WAL or managed state is
        # changed. The caller's managed List remains untouched.
        var normalized_embeddings = List[Float32]()
        if self._metric_type == METRIC_COSINE:
            normalized_embeddings = self._normalize_vectors(embeddings)
        inject_batch_fault(
            fault_point,
            BATCH_FAULT_AFTER_VECTOR_PREPARE,
        )

        # New vector-only records have no payload work. Keep that dominant
        # ingestion path allocation-free apart from the graph and ID arrays.
        var payload_free_insert = not has_metadatas and not has_documents
        if payload_free_insert and replace_existing:
            for index in range(len(ids)):
                if ids[index] in self._id_to_internal:
                    payload_free_insert = False
                    break

        # Materialize inherited payloads and tokenize every document before
        # WAL or live state changes. The remaining commit path cannot raise.
        var prepared = PreparedBatchPayloads()
        if payload_free_insert:
            inject_batch_fault(
                fault_point,
                BATCH_FAULT_DURING_PAYLOAD_PREPARE,
            )
        else:
            prepared = _prepare_batch_payloads(
                ids,
                self._id_to_internal,
                self._metadata_by_internal,
                self._metadatas,
                self._document_by_internal,
                self._documents,
                self._bm25,
                metadatas,
                has_metadatas,
                documents,
                has_documents,
                fault_point,
            )
        inject_batch_fault(
            fault_point,
            BATCH_FAULT_AFTER_PAYLOAD_PREPARE,
        )

        var wal_sequence = self._applied_sequence
        var wal_byte_size = 0
        var wrote_wal = False
        if not self._replaying_wal and self._wal:
            wal_byte_size = self._wal[].byte_size()
            wal_sequence = self._wal[].append_write(
                ids,
                embeddings,
                replace_existing,
                metadatas,
                has_metadatas,
                documents,
                has_documents,
            )
            wrote_wal = True

        var fault_after_wal = False
        try:
            inject_batch_fault(
                fault_point,
                BATCH_FAULT_AFTER_WAL_APPEND,
            )
        except:
            fault_after_wal = True
        if fault_after_wal:
            if wrote_wal:
                self._wal[].rollback_last_append(
                    wal_byte_size,
                    wal_sequence,
                )
            raise Error("Injected atomic batch failure after WAL append.")

        # From this point through publication, every operation is non-raising.
        if self._metric_type == METRIC_COSINE:
            self._add_prepared_to_index(
                Span[Float32](normalized_embeddings)
            )
        else:
            self._add_prepared_to_index(embeddings)

        if payload_free_insert:
            for i in range(len(ids)):
                var internal_id = len(self._user_ids)
                self._user_ids.append(ids[i])
                self._is_deleted.append(0)
                self._id_to_internal[ids[i]] = internal_id
        else:
            self._commit_prepared_payloads(ids, prepared)

        if not self._replaying_wal and self._wal:
            self._applied_sequence = wal_sequence

    def _commit_prepared_payloads(
        mut self,
        ids: Span[Int, _],
        prepared: PreparedBatchPayloads,
    ):
        """Publishes previously prepared payload rows without raising."""
        for i in range(len(ids)):
            var record_id = ids[i]
            var previous = prepared.previous_internal_ids[i]
            if previous >= 0:
                self._is_deleted[previous] = 1
                self._bm25.deactivate(previous)

            var internal_id = len(self._user_ids)
            self._user_ids.append(record_id)
            self._is_deleted.append(0)
            self._store_metadata(internal_id, prepared.metadatas[i])
            self._store_document_prepared(
                internal_id,
                prepared.documents[i],
                prepared.document_tokens[i],
            )
            self._id_to_internal[record_id] = internal_id

    def add(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        """
        Adds records and rejects IDs that already exist.

        The collection borrows managed Lists for the duration of the call.
        Callers never allocate or free MojoVec-owned memory manually.
        """
        self._add_from_spans(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
        )

    def add(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        metadatas: List[Metadata],
    ) raises:
        """Adds records together with one metadata object per ID."""
        self._append_records(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
            replace_existing=False,
            metadatas=metadatas,
            has_metadatas=True,
            documents=List[String](),
            has_documents=False,
        )

    def add(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        documents: List[String],
    ) raises:
        """Adds records together with one document per ID."""
        self._append_records(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
            replace_existing=False,
            metadatas=List[Metadata](),
            has_metadatas=False,
            documents=documents,
            has_documents=True,
        )

    def add(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        metadatas: List[Metadata],
        documents: List[String],
    ) raises:
        """Adds records with aligned metadata and documents."""
        self._append_records(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
            replace_existing=False,
            metadatas=metadatas,
            has_metadatas=True,
            documents=documents,
            has_documents=True,
        )

    def _add_from_spans(
        mut self,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
    ) raises:
        """Internal zero-copy bridge used by managed frontends."""
        self._append_records(
            ids,
            embeddings,
            replace_existing=False,
            metadatas=List[Metadata](),
            has_metadatas=False,
            documents=List[String](),
            has_documents=False,
        )

    def upsert(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        """
        Atomically inserts records or replaces active records with matching IDs.

        Inputs and collection storage use automatic lifetime management.
        """
        self._upsert_from_spans(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
        )

    def upsert(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        metadatas: List[Metadata],
    ) raises:
        """Inserts or replaces records and explicitly replaces metadata."""
        self._append_records(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
            replace_existing=True,
            metadatas=metadatas,
            has_metadatas=True,
            documents=List[String](),
            has_documents=False,
        )

    def upsert(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        documents: List[String],
    ) raises:
        """Inserts or replaces records and explicitly replaces documents."""
        self._append_records(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
            replace_existing=True,
            metadatas=List[Metadata](),
            has_metadatas=False,
            documents=documents,
            has_documents=True,
        )

    def upsert(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        metadatas: List[Metadata],
        documents: List[String],
    ) raises:
        """Inserts or replaces records with metadata and documents."""
        self._append_records(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
            replace_existing=True,
            metadatas=metadatas,
            has_metadatas=True,
            documents=documents,
            has_documents=True,
        )

    def _upsert_from_spans(
        mut self,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
    ) raises:
        """Internal zero-copy bridge used by managed frontends."""
        self._append_records(
            ids,
            embeddings,
            replace_existing=True,
            metadatas=List[Metadata](),
            has_metadatas=False,
            documents=List[String](),
            has_documents=False,
        )

    def _upsert_with_fault(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        metadatas: List[Metadata],
        documents: List[String],
        fault_point: Int,
    ) raises:
        """Test-only entry point for deterministic atomic batch failures."""
        self._append_records_with_fault(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](
                ptr=embeddings.unsafe_ptr(), length=len(embeddings)
            ),
            replace_existing=True,
            metadatas=metadatas,
            has_metadatas=True,
            documents=documents,
            has_documents=True,
            fault_point=fault_point,
        )

    def _validate_existing_ids(self, ids: Span[Int, _]) raises:
        for i in range(len(ids)):
            if ids[i] not in self._id_to_internal:
                raise Error("Cannot update an ID that does not exist.")

    def update(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        """Updates existing IDs and rejects missing IDs."""
        var ids_span = Span[Int](ptr=ids.unsafe_ptr(), length=len(ids))
        var embeddings_span = Span[Float32](
            ptr=embeddings.unsafe_ptr(), length=len(embeddings)
        )
        self._validate_shape(ids_span, embeddings_span)
        self._validate_unique_batch(ids_span)
        self._validate_existing_ids(ids_span)
        self._append_records(
            ids_span,
            embeddings_span,
            replace_existing=True,
            metadatas=List[Metadata](),
            has_metadatas=False,
            documents=List[String](),
            has_documents=False,
        )

    def update(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        metadatas: List[Metadata],
    ) raises:
        """Updates existing records and explicitly replaces their metadata."""
        var ids_span = Span[Int](ptr=ids.unsafe_ptr(), length=len(ids))
        var embeddings_span = Span[Float32](
            ptr=embeddings.unsafe_ptr(), length=len(embeddings)
        )
        self._validate_shape(ids_span, embeddings_span)
        self._validate_unique_batch(ids_span)
        self._validate_existing_ids(ids_span)
        self._append_records(
            ids_span,
            embeddings_span,
            replace_existing=True,
            metadatas=metadatas,
            has_metadatas=True,
            documents=List[String](),
            has_documents=False,
        )

    def update(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        documents: List[String],
    ) raises:
        """Updates existing records and explicitly replaces documents."""
        var ids_span = Span[Int](ptr=ids.unsafe_ptr(), length=len(ids))
        var embeddings_span = Span[Float32](
            ptr=embeddings.unsafe_ptr(), length=len(embeddings)
        )
        self._validate_shape(ids_span, embeddings_span)
        self._validate_unique_batch(ids_span)
        self._validate_existing_ids(ids_span)
        self._append_records(
            ids_span,
            embeddings_span,
            replace_existing=True,
            metadatas=List[Metadata](),
            has_metadatas=False,
            documents=documents,
            has_documents=True,
        )

    def update(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
        metadatas: List[Metadata],
        documents: List[String],
    ) raises:
        """Updates existing records with metadata and documents."""
        var ids_span = Span[Int](ptr=ids.unsafe_ptr(), length=len(ids))
        var embeddings_span = Span[Float32](
            ptr=embeddings.unsafe_ptr(), length=len(embeddings)
        )
        self._validate_shape(ids_span, embeddings_span)
        self._validate_unique_batch(ids_span)
        self._validate_existing_ids(ids_span)
        self._append_records(
            ids_span,
            embeddings_span,
            replace_existing=True,
            metadatas=metadatas,
            has_metadatas=True,
            documents=documents,
            has_documents=True,
        )

    def delete(mut self, ids: List[Int]) raises:
        """Soft-deletes active records by ID."""
        if len(ids) == 0:
            return
        var wal_sequence = self._applied_sequence
        if not self._replaying_wal and self._wal:
            wal_sequence = self._wal[].append_delete(
                Span[Int](ptr=ids.unsafe_ptr(), length=len(ids))
            )
        for i in range(len(ids)):
            var internal_id = self._id_to_internal.pop(ids[i], -1)
            if internal_id >= 0:
                self._is_deleted[internal_id] = 1
                self._bm25.deactivate(internal_id)
        if not self._replaying_wal and self._wal:
            self._applied_sequence = wal_sequence

    def set_ef_search(mut self, ef: Int) raises:
        if ef <= 0 or ef > 2048:
            raise Error("ef_search must be between 1 and 2048.")
        _storage_set_ef_search(self._hnsw, self._storage_kind, ef)

    def _query_into(
        self,
        query_embeddings: Span[Float32, _],
        n_results: Int,
        mut ids: Span[mut=True, Int, _],
        mut distances: Span[mut=True, Float32, _],
    ) raises:
        """Internal zero-copy search bridge used by managed frontends."""
        _query_collection_into(
            self._hnsw,
            self._storage_kind,
            self._metric_type,
            self._dimension,
            self._user_ids,
            self._is_deleted,
            self.count_deleted(),
            query_embeddings,
            n_results,
            ids,
            distances,
        )

    def _where_filter(self, where: Where) raises -> List[UInt8]:
        return _build_where_filter(
            where,
            self._user_ids,
            self._is_deleted,
            self._metadata_by_internal,
            self._metadatas,
            self._metadata_index,
        )

    def _query_into_filtered(
        self,
        query_embeddings: Span[Float32, _],
        n_results: Int,
        mut ids: Span[mut=True, Int, _],
        mut distances: Span[mut=True, Float32, _],
        exclusion: Span[UInt8, _],
    ) raises:
        _query_collection_into_filtered(
            self._hnsw,
            self._storage_kind,
            self._metric_type,
            self._dimension,
            self._user_ids,
            query_embeddings,
            n_results,
            ids,
            distances,
            exclusion,
        )

    def _build_query_results(
        self,
        ids_storage: List[Int],
        distances_storage: List[Float32],
        num_queries: Int,
        n_results: Int,
    ) raises -> QueryResults:
        return _build_vector_query_results(
            ids_storage,
            distances_storage,
            num_queries,
            n_results,
            self._id_to_internal,
            self._metadata_by_internal,
            self._metadatas,
            self._document_by_internal,
            self._documents,
        )

    def query(
        self,
        query_embeddings: List[Float32],
        n_results: Int = 10,
    ) raises -> QueryResults:
        """
        Runs embedding queries and returns automatically managed results.

        QueryResults owns aligned ID, distance, metadata, and document Lists;
        its ranked-search `scores` field is empty. Callers do not allocate
        output buffers or release result memory.
        """
        if self._dimension <= 0:
            raise Error("Collection dimension must be positive.")
        if len(query_embeddings) % self._dimension != 0:
            raise Error(
                "Query embeddings length must be a multiple of dimension."
            )
        if n_results <= 0 or n_results > 2048:
            raise Error("n_results must be between 1 and 2048.")

        var num_queries = len(query_embeddings) // self._dimension
        if num_queries == 0:
            return QueryResults(
                List[List[Int]](),
                List[List[Float32]](),
                List[List[Metadata]](),
                List[List[String]](),
                List[List[Float32]](),
            )

        var output_size = num_queries * n_results
        var ids_storage = List[Int](unsafe_uninit_length=output_size)
        var distances_storage = List[Float32](
            unsafe_uninit_length=output_size
        )
        var queries = Span[Float32](
            ptr=query_embeddings.unsafe_ptr(),
            length=len(query_embeddings),
        )
        var ids = Span[mut=True, Int](ids_storage)
        var distances = Span[mut=True, Float32](distances_storage)
        self._query_into(queries, n_results, ids, distances)

        return self._build_query_results(
            ids_storage,
            distances_storage,
            num_queries,
            n_results,
        )

    def query(
        self,
        query_embeddings: List[Float32],
        where: Where,
        n_results: Int = 10,
    ) raises -> QueryResults:
        """Queries nearest neighbors restricted by a typed metadata filter."""
        if self._dimension <= 0:
            raise Error("Collection dimension must be positive.")
        if len(query_embeddings) % self._dimension != 0:
            raise Error(
                "Query embeddings length must be a multiple of dimension."
            )
        if n_results <= 0 or n_results > 2048:
            raise Error("n_results must be between 1 and 2048.")

        var num_queries = len(query_embeddings) // self._dimension
        if num_queries == 0:
            return QueryResults(
                List[List[Int]](),
                List[List[Float32]](),
                List[List[Metadata]](),
                List[List[String]](),
                List[List[Float32]](),
            )

        var output_size = num_queries * n_results
        var ids_storage = List[Int](unsafe_uninit_length=output_size)
        var distances_storage = List[Float32](
            unsafe_uninit_length=output_size
        )
        var exclusion_storage = self._where_filter(where)
        var queries = Span[Float32](
            ptr=query_embeddings.unsafe_ptr(),
            length=len(query_embeddings),
        )
        var ids = Span[mut=True, Int](ids_storage)
        var distances = Span[mut=True, Float32](distances_storage)
        var exclusion = Span[UInt8](exclusion_storage)
        self._query_into_filtered(
            queries,
            n_results,
            ids,
            distances,
            exclusion,
        )

        return self._build_query_results(
            ids_storage,
            distances_storage,
            num_queries,
            n_results,
        )

    def _query_bm25(
        self,
        query_texts: List[String],
        n_results: Int,
        exclusion: List[UInt8],
    ) raises -> QueryResults:
        return _query_bm25_index(
            self._bm25,
            query_texts,
            n_results,
            exclusion,
            self._user_ids,
            self._is_deleted,
            self._metadata_by_internal,
            self._metadatas,
            self._document_by_internal,
            self._documents,
        )

    def query(
        self,
        query_texts: List[String],
        n_results: Int = 10,
    ) raises -> QueryResults:
        """
        Runs BM25 full-text queries over active collection documents.

        Results are ordered by descending `scores`; `distances` is empty.
        The shared analyzer applies Unicode lowercase and word boundaries,
        then removes bundled English and Russian stopwords without stemming.
        """
        return self._query_bm25(
            query_texts,
            n_results,
            List[UInt8](),
        )

    def query(
        self,
        query_texts: List[String],
        where: Where,
        n_results: Int = 10,
    ) raises -> QueryResults:
        """Runs BM25 queries restricted by a typed metadata filter."""
        return self._query_bm25(
            query_texts,
            n_results,
            self._where_filter(where),
        )

    def _query_hybrid(
        self,
        query_embeddings: List[Float32],
        query_texts: List[String],
        n_results: Int,
        rrf_k: Int,
        candidate_multiplier: Int,
        exclusion: List[UInt8],
    ) raises -> QueryResults:
        return _query_hybrid_index(
            self._hnsw,
            self._storage_kind,
            self._metric_type,
            self._dimension,
            self._bm25,
            self._user_ids,
            self._is_deleted,
            self.count_deleted(),
            self._metadata_by_internal,
            self._metadatas,
            self._document_by_internal,
            self._documents,
            query_embeddings,
            query_texts,
            n_results,
            rrf_k,
            candidate_multiplier,
            exclusion,
        )

    def query_hybrid(
        self,
        query_embeddings: List[Float32],
        query_texts: List[String],
        n_results: Int = 10,
        rrf_k: Int = RRF_DEFAULT_K,
        candidate_multiplier: Int = RRF_DEFAULT_CANDIDATE_MULTIPLIER,
    ) raises -> QueryResults:
        """
        Fuses matching vector and BM25 query batches with reciprocal rank fusion.

        Each source contributes `1 / (rrf_k + rank)` using one-based ranks.
        The result `scores` contain the fused score; `distances` is empty.
        """
        return self._query_hybrid(
            query_embeddings,
            query_texts,
            n_results,
            rrf_k,
            candidate_multiplier,
            List[UInt8](),
        )

    def query_hybrid(
        self,
        query_embeddings: List[Float32],
        query_texts: List[String],
        where: Where,
        n_results: Int = 10,
        rrf_k: Int = RRF_DEFAULT_K,
        candidate_multiplier: Int = RRF_DEFAULT_CANDIDATE_MULTIPLIER,
    ) raises -> QueryResults:
        """Runs RRF hybrid search restricted by a typed metadata filter."""
        return self._query_hybrid(
            query_embeddings,
            query_texts,
            n_results,
            rrf_k,
            candidate_multiplier,
            self._where_filter(where),
        )

    def wal_enabled(self) -> Bool:
        """Returns whether mutations are appended to a write-ahead log."""
        return Bool(self._wal)

    def wal_sequence(self) -> Int:
        """Returns the latest WAL sequence applied to this collection."""
        return self._applied_sequence

    def enable_wal(
        mut self,
        path: String,
        durability: WalDurability = WAL_ASYNC,
    ) raises:
        """
        Enables an optional append-only WAL for subsequent mutations.

        A non-empty WAL must be opened through `recover()` so committed
        records cannot be skipped accidentally.
        """
        if self._wal:
            raise Error("WAL is already enabled.")
        var wal = WriteAheadLog.create(
            path,
            self._dimension,
            self._storage_kind,
            self._identity,
            self._name,
            self._applied_sequence,
            durability,
        )
        self._wal = wal^

    def disable_wal(mut self):
        """Stops logging new mutations without deleting the WAL file."""
        if self._wal:
            _ = self._wal.take()

    def flush_wal(mut self) raises:
        """Makes all WAL frames written so far durable."""
        if not self._wal:
            raise Error("WAL is not enabled.")
        self._wal[].flush()

    def checkpoint(mut self, path: String) raises:
        """
        Atomically saves all applied operations and rotates the WAL.

        The snapshot is durable before old WAL frames are discarded.
        """
        from mojovec.io.fault_injection import SNAPSHOT_FAULT_NONE

        self._checkpoint_with_fault(path, SNAPSHOT_FAULT_NONE)

    def _checkpoint_with_fault(
        mut self,
        path: String,
        fault_point: Int,
    ) raises:
        """Checkpoint implementation with deterministic durability faults."""
        from mojovec.io.fault_injection import (
            SNAPSHOT_FAULT_AFTER_CHECKPOINT_SNAPSHOT,
            inject_snapshot_fault,
        )

        self._save_with_fault(path, fault_point)
        inject_snapshot_fault(
            fault_point,
            SNAPSHOT_FAULT_AFTER_CHECKPOINT_SNAPSHOT,
        )
        if self._wal:
            self._wal[].reset(self._applied_sequence, fault_point)

    def save(mut self, path: String) raises:
        """
        Atomically saves a complete collection.

        A synchronized temporary file is renamed over `path`, so readers see
        either the previous complete snapshot or the new complete snapshot.
        Existing mmap readers retain their previous file generation.
        """
        from mojovec.io.fault_injection import SNAPSHOT_FAULT_NONE

        self._save_with_fault(path, SNAPSHOT_FAULT_NONE)

    def _save_with_fault(
        mut self,
        path: String,
        fault_point: Int,
    ) raises:
        """Atomic save implementation with deterministic durability faults."""
        _save_collection_snapshot(
            path,
            fault_point,
            self._name,
            self._dimension,
            self._storage_kind,
            self._metric_type,
            self._identity,
            self._applied_sequence,
            self._user_ids,
            self._is_deleted,
            self._metadata_by_internal,
            self._metadatas,
            self._document_by_internal,
            self._documents,
            self._hnsw,
        )

    def snapshot(
        mut self,
        path: String,
        memory_mapped: Bool = True,
        mmap_threshold_bytes: Int = DEFAULT_MMAP_THRESHOLD_BYTES,
    ) raises -> Collection:
        """
        Atomically publishes and returns an independent point-in-time view.

        The returned collection can serve concurrent readers while the
        original collection continues to receive writes from one writer.
        """
        self.checkpoint(path)
        return Collection.load(
            path,
            memory_mapped=memory_mapped,
            mmap_threshold_bytes=mmap_threshold_bytes,
        )

    @staticmethod
    def recover(
        snapshot_path: String,
        wal_path: String,
        durability: WalDurability = WAL_ASYNC,
        memory_mapped: Bool = True,
        mmap_threshold_bytes: Int = DEFAULT_MMAP_THRESHOLD_BYTES,
    ) raises -> Collection:
        """Loads a snapshot, replays committed WAL frames, and resumes WAL."""
        var collection = Collection.load(
            snapshot_path,
            memory_mapped=memory_mapped,
            mmap_threshold_bytes=mmap_threshold_bytes,
        )
        var reader = WalReader.open(wal_path)
        if (
            reader.header.dimension != collection._dimension
            or reader.header.storage_kind != collection._storage_kind
            or reader.header.identity != collection._identity
            or reader.header.name != collection._name
        ):
            raise Error("WAL belongs to a different collection.")
        if reader.header.base_sequence > collection._applied_sequence:
            raise Error("Snapshot is older than the retained WAL base.")

        var expected_sequence = reader.header.base_sequence + 1
        var last_sequence = reader.header.base_sequence
        while True:
            var optional_record = reader.next()
            if not optional_record:
                break
            var record = optional_record.take()
            if record.sequence != expected_sequence:
                raise Error("WAL sequence is not contiguous.")
            expected_sequence += 1
            last_sequence = record.sequence
            if record.sequence <= collection._applied_sequence:
                continue
            if record.sequence != collection._applied_sequence + 1:
                raise Error("WAL does not continue the snapshot sequence.")

            collection._replaying_wal = True
            if record.operation == WAL_OPERATION_WRITE:
                collection._append_records(
                    Span[Int](
                        ptr=record.ids.unsafe_ptr(),
                        length=len(record.ids),
                    ),
                    Span[Float32](
                        ptr=record.embeddings.unsafe_ptr(),
                        length=len(record.embeddings),
                    ),
                    record.replace_existing,
                    record.metadatas,
                    record.has_metadatas,
                    record.documents,
                    record.has_documents,
                )
            elif record.operation == WAL_OPERATION_DELETE:
                collection.delete(record.ids)
            else:
                raise Error("Unsupported WAL operation.")
            collection._replaying_wal = False
            collection._applied_sequence = record.sequence

        if collection._applied_sequence > last_sequence:
            raise Error("WAL ends before the snapshot sequence.")
        var wal = WriteAheadLog.resume(
            wal_path,
            collection._dimension,
            collection._storage_kind,
            collection._identity,
            collection._name,
            durability,
            last_sequence + 1,
            reader.valid_bytes,
        )
        collection._wal = wal^
        return collection^

    @staticmethod
    def load(
        path: String,
        memory_mapped: Bool = True,
        mmap_threshold_bytes: Int = DEFAULT_MMAP_THRESHOLD_BYTES,
    ) raises -> Collection:
        """Loads Flat/SQ8, mapping index arrays when the threshold is met.
        """
        var snapshot = _read_collection_snapshot(
            path, memory_mapped, mmap_threshold_bytes
        )
        var collection = Collection(
            snapshot.dimension,
            quantized=snapshot.storage_kind == STORAGE_SQ8,
            name=snapshot.name,
            metric=_metric_name(snapshot.metric_type),
        )
        collection._identity = snapshot.identity
        collection._applied_sequence = snapshot.applied_sequence
        collection._user_ids = snapshot.user_ids.take()
        collection._is_deleted = snapshot.is_deleted.take()
        for internal_id in range(len(collection._user_ids)):
            if collection._is_deleted[internal_id] == 0:
                collection._id_to_internal[
                    collection._user_ids[internal_id]
                ] = internal_id
        for index in range(len(snapshot.metadata_internal_ids)):
            collection._store_metadata(
                snapshot.metadata_internal_ids[index],
                snapshot.metadatas[index],
            )
        for index in range(len(snapshot.document_internal_ids)):
            collection._store_document(
                snapshot.document_internal_ids[index],
                snapshot.documents[index],
            )
        collection._hnsw = snapshot.storage.take()
        return collection^
