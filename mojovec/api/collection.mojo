from std.collections import Dict, List
from std.io.file import FileHandle
from std.memory import alloc
from std.memory.span import Span
from std.os import SEEK_END, SEEK_SET
from std.time import perf_counter_ns
from std.utils import Variant

from mojovec.api.bm25 import BM25Index
from mojovec.api.metadata import (
    METADATA_BOOL,
    METADATA_FLOAT,
    METADATA_INT,
    METADATA_STRING,
    Metadata,
    MetadataValue,
)
from mojovec.api.metadata_bitmap import (
    BITMAP_WORD_BITS,
    MetadataBitmapIndex,
)
from mojovec.api.results import CollectionStats, CompactReport, QueryResults
from mojovec.api.rrf import (
    RRF_DEFAULT_CANDIDATE_MULTIPLIER,
    RRF_DEFAULT_K,
    reciprocal_rank_fusion,
)
from mojovec.api.where import Where, WhereNode
from mojovec.core.types import (
    METRIC_L2,
    STORAGE_FLAT,
    STORAGE_SQ8,
    StorageKind,
)
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_flat_sq8 import IndexFlatSQ8
from mojovec.index.index_hnsw import IndexHNSW


comptime FlatHNSW = IndexHNSW[IndexFlat]
comptime SQ8HNSW = IndexHNSW[IndexFlatSQ8]
comptime HNSWStorage = Variant[FlatHNSW, SQ8HNSW]

comptime COLLECTION_MAGIC_V1 = 1129270348
comptime COLLECTION_MAGIC_V2 = 1129270349
comptime COLLECTION_FORMAT_VERSION = 5
comptime COLLECTION_FORMAT_VERSION_WITH_METADATA = 3
comptime COLLECTION_FORMAT_VERSION_WITH_DOCUMENTS = 4
comptime COLLECTION_FORMAT_VERSION_WITH_MMAP_INDEX = 5
comptime DEFAULT_MMAP_THRESHOLD_BYTES = 64 * 1024 * 1024
comptime COMPACTION_MAX_BATCH_VECTORS = 16_384
comptime COMPACTION_MAX_BATCH_COMPONENTS = 2_097_152


def _write_string(mut file: FileHandle, value: String) raises:
    from mojovec.io.serialization import write_int

    write_int(file, value.byte_length())
    file.write_bytes(value.as_bytes())


def _read_string(mut file: FileHandle) raises -> String:
    from mojovec.io.serialization import check_size_limit, read_int

    var size = read_int(file)
    check_size_limit(size, 1_048_576)
    var data = file.read_bytes(size)
    return String(from_utf8=data)


def _write_float64(mut file: FileHandle, value: Float64) raises:
    var storage = alloc[Float64](1)
    storage[0] = value
    file.write_bytes(
        Span[UInt8](ptr=storage.bitcast[UInt8](), length=8)
    )
    storage.free()


def _read_float64(mut file: FileHandle) raises -> Float64:
    var data = file.read_bytes(8)
    var value = data.unsafe_ptr().bitcast[Float64]()[0]
    _ = len(data)
    return value


def _write_metadata(mut file: FileHandle, metadata: Metadata) raises:
    from mojovec.io.serialization import write_bool, write_int

    write_int(file, metadata.count())
    for index in range(metadata.count()):
        _write_string(file, metadata._key_at(index))
        var value = metadata._value_at(index)
        write_int(file, value.kind())
        if value.kind() == METADATA_STRING:
            _write_string(file, value.as_string())
        elif value.kind() == METADATA_INT:
            write_int(file, value.as_int())
        elif value.kind() == METADATA_FLOAT:
            _write_float64(file, value.as_float())
        elif value.kind() == METADATA_BOOL:
            write_bool(file, value.as_bool())
        else:
            raise Error("Unsupported metadata value kind.")


def _read_metadata(mut file: FileHandle) raises -> Metadata:
    from mojovec.io.serialization import (
        check_size_limit,
        read_bool,
        read_int,
    )

    var field_count = read_int(file)
    check_size_limit(field_count, 65_536)
    var metadata = Metadata()
    for _ in range(field_count):
        var key = _read_string(file)
        var kind = read_int(file)
        if kind == METADATA_STRING:
            metadata.set(key, _read_string(file))
        elif kind == METADATA_INT:
            metadata.set(key, read_int(file))
        elif kind == METADATA_FLOAT:
            metadata.set(key, _read_float64(file))
        elif kind == METADATA_BOOL:
            metadata.set(key, read_bool(file))
        else:
            raise Error("Invalid metadata value kind.")
    return metadata^


struct Collection(Movable, Writable):
    """A Chroma-style vector and document collection with HNSW and BM25."""

    var _name: String
    var _dimension: Int
    var _storage_kind: StorageKind
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

    def __init__(
        out self,
        dimension: Int,
        M: Int = 32,
        ef_construction: Int = 40,
        ef_search: Int = 16,
        quantized: Bool = True,
        name: String = "",
    ):
        self._name = name.copy()
        self._dimension = dimension
        self._storage_kind = STORAGE_SQ8 if quantized else STORAGE_FLAT

        if quantized:
            var storage = IndexFlatSQ8(dimension, METRIC_L2)
            var index = SQ8HNSW(storage^, dimension, METRIC_L2, M=M)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            self._hnsw = HNSWStorage(index^)
        else:
            var storage = IndexFlat(dimension, METRIC_L2)
            var index = FlatHNSW(storage^, dimension, METRIC_L2, M=M)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            self._hnsw = HNSWStorage(index^)

        self._user_ids = List[Int]()
        self._is_deleted = List[UInt8]()
        self._metadata_by_internal = Dict[Int, Int]()
        self._metadatas = List[Metadata]()
        self._metadata_index = MetadataBitmapIndex()
        self._document_by_internal = Dict[Int, Int]()
        self._documents = List[String]()
        self._bm25 = BM25Index()
        self._id_to_internal = Dict[Int, Int]()

    def __init__(out self, *, deinit take: Self):
        self._name = take._name^
        self._dimension = take._dimension
        self._storage_kind = take._storage_kind
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

    def write_to[W: Writer](self, mut writer: W):
        var storage = "sq8" if self.is_quantized() else "flat"
        writer.write(
            "Collection(name=",
            self._name,
            ", dimension=",
            self._dimension,
            ", storage=",
            storage,
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
    ) raises:
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
        self._document_by_internal[internal_id] = len(self._documents)
        self._documents.append(document.copy())
        if (
            internal_id >= len(self._is_deleted)
            or self._is_deleted[internal_id] == 0
        ):
            self._bm25.add(internal_id, document)

    def stats(self) -> CollectionStats:
        """Returns active/deleted counts and the current HNSW configuration."""
        var active_count = self.count()
        var total_count = len(self._user_ids)
        var deleted_count = total_count - active_count
        var deleted_ratio: Float64 = 0.0
        if total_count > 0:
            deleted_ratio = Float64(deleted_count) / Float64(total_count)

        var M: Int
        var ef_construction: Int
        var ef_search: Int
        if self._storage_kind == STORAGE_SQ8:
            M = self._hnsw.unsafe_get[SQ8HNSW]().hnsw.M
            ef_construction = (
                self._hnsw.unsafe_get[SQ8HNSW]().hnsw.efConstruction
            )
            ef_search = self._hnsw.unsafe_get[SQ8HNSW]().hnsw.efSearch
        else:
            M = self._hnsw.unsafe_get[FlatHNSW]().hnsw.M
            ef_construction = (
                self._hnsw.unsafe_get[FlatHNSW]().hnsw.efConstruction
            )
            ef_search = self._hnsw.unsafe_get[FlatHNSW]().hnsw.efSearch

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
        if self._storage_kind == STORAGE_SQ8:
            return (
                self._hnsw.unsafe_get[SQ8HNSW]().storage.is_memory_mapped()
                and self._hnsw.unsafe_get[SQ8HNSW]().hnsw.is_memory_mapped()
            )
        return (
            self._hnsw.unsafe_get[FlatHNSW]().storage.is_memory_mapped()
            and self._hnsw.unsafe_get[FlatHNSW]().hnsw.is_memory_mapped()
        )

    def _get_vector(
        self, internal_id: Int
    ) -> Span[Float32, MutUntrackedOrigin]:
        if self._storage_kind == STORAGE_SQ8:
            return Span[Float32, MutUntrackedOrigin](
                ptr=self._hnsw.unsafe_get[
                    SQ8HNSW
                ]().storage.get_vector(internal_id),
                length=self._dimension,
            )
        return Span[Float32, MutUntrackedOrigin](
            ptr=self._hnsw.unsafe_get[
                FlatHNSW
            ]().storage.get_vector(internal_id),
            length=self._dimension,
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

    def _add_to_index(mut self, embeddings: Span[Float32, _]):
        if self._storage_kind == STORAGE_SQ8:
            self._hnsw.unsafe_get[SQ8HNSW]().add(embeddings)
        else:
            self._hnsw.unsafe_get[FlatHNSW]().add(embeddings)

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
        self._validate_shape(ids, embeddings)
        self._validate_unique_batch(ids)
        if has_metadatas and len(metadatas) != len(ids):
            raise Error("Metadata length must equal len(ids).")
        if has_documents and len(documents) != len(ids):
            raise Error("Documents length must equal len(ids).")
        if len(ids) == 0:
            return

        for i in range(len(ids)):
            var record_id = ids[i]
            var previous = -1
            if record_id in self._id_to_internal:
                if not replace_existing:
                    raise Error("ID already exists; use upsert or update.")
                previous = self._id_to_internal[record_id]
                self._is_deleted[previous] = 1
                self._bm25.deactivate(previous)

            var internal_id = len(self._user_ids)
            self._user_ids.append(record_id)
            self._is_deleted.append(0)
            if has_metadatas:
                self._store_metadata(internal_id, metadatas[i])
            elif (
                previous >= 0
                and previous in self._metadata_by_internal
            ):
                var previous_metadata_index = (
                    self._metadata_by_internal[previous]
                )
                var inherited_metadata = (
                    self._metadatas[previous_metadata_index].copy()
                )
                self._store_metadata(internal_id, inherited_metadata)
            if has_documents:
                self._store_document(internal_id, documents[i])
            elif (
                previous >= 0
                and previous in self._document_by_internal
            ):
                var previous_document_index = (
                    self._document_by_internal[previous]
                )
                var inherited_document = (
                    self._documents[previous_document_index].copy()
                )
                self._store_document(
                    internal_id,
                    inherited_document,
                )
            self._id_to_internal[record_id] = internal_id

        self._add_to_index(embeddings)

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
        Inserts records or replaces active records with matching IDs.

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

    def delete(mut self, ids: List[Int]):
        """Soft-deletes active records by ID."""
        for i in range(len(ids)):
            var internal_id = self._id_to_internal.pop(ids[i], -1)
            if internal_id >= 0:
                self._is_deleted[internal_id] = 1
                self._bm25.deactivate(internal_id)

    def set_ef_search(mut self, ef: Int) raises:
        if ef <= 0 or ef > 2048:
            raise Error("ef_search must be between 1 and 2048.")
        if self._storage_kind == STORAGE_SQ8:
            self._hnsw.unsafe_get[SQ8HNSW]().hnsw.efSearch = ef
        else:
            self._hnsw.unsafe_get[FlatHNSW]().hnsw.efSearch = ef

    def _search_index(
        self,
        queries: Span[Float32, _],
        n_results: Int,
        mut distances: Span[mut=True, Float32, _],
        mut labels: Span[mut=True, Int, _],
        deleted: Span[UInt8, _],
    ):
        if self._storage_kind == STORAGE_SQ8:
            self._hnsw.unsafe_get[SQ8HNSW]().search(
                queries, n_results, distances, labels, deleted
            )
        else:
            self._hnsw.unsafe_get[FlatHNSW]().search(
                queries, n_results, distances, labels, deleted
            )

    def _query_into(
        self,
        query_embeddings: Span[Float32, _],
        n_results: Int,
        mut ids: Span[mut=True, Int, _],
        mut distances: Span[mut=True, Float32, _],
    ) raises:
        """Internal zero-copy search bridge used by managed frontends."""
        if self._dimension <= 0:
            raise Error("Collection dimension must be positive.")
        if len(query_embeddings) % self._dimension != 0:
            raise Error(
                "Query embeddings length must be a multiple of dimension."
            )
        if n_results <= 0 or n_results > 2048:
            raise Error("n_results must be between 1 and 2048.")

        var num_queries = len(query_embeddings) // self._dimension
        var output_size = num_queries * n_results
        if len(ids) < output_size or len(distances) < output_size:
            raise Error(
                "Output buffers are smaller than query_count * n_results."
            )
        if num_queries == 0:
            return

        # Keep the common append-only path completely filter-free. Passing an
        # all-zero deletion bitmap still selects the filtered HNSW kernel and
        # adds a random bitmap load for every candidate.
        if self.count_deleted() == 0:
            var empty_filter = Span[UInt8, MutUntrackedOrigin]()
            self._search_index(
                query_embeddings,
                n_results,
                distances,
                ids,
                empty_filter,
            )
        else:
            var deleted = Span[mut=False, UInt8](self._is_deleted)
            self._search_index(
                query_embeddings,
                n_results,
                distances,
                ids,
                deleted,
            )

        for i in range(output_size):
            var internal_id = ids[i]
            if internal_id >= 0 and internal_id < len(self._user_ids):
                ids[i] = self._user_ids[internal_id]
            else:
                ids[i] = -1

    def _empty_bitmap(self) -> List[UInt64]:
        var word_count = (
            len(self._user_ids) + BITMAP_WORD_BITS - 1
        ) // BITMAP_WORD_BITS
        var result = List[UInt64](unsafe_uninit_length=word_count)
        for index in range(word_count):
            result[index] = 0
        return result^

    def _scan_predicate_bitmap(
        self, node: WhereNode
    ) raises -> List[UInt64]:
        var result = self._empty_bitmap()
        var key = node.key()
        for internal_id in range(len(self._user_ids)):
            if internal_id not in self._metadata_by_internal:
                continue
            var metadata_index = self._metadata_by_internal[internal_id]
            if not self._metadatas[metadata_index].contains(key):
                continue
            var value = self._metadatas[metadata_index].get(key)
            if node.matches(value):
                var word_index = internal_id // BITMAP_WORD_BITS
                result[word_index] |= (
                    UInt64(1)
                    << UInt64(internal_id % BITMAP_WORD_BITS)
                )
        return result^

    def _predicate_bitmap(
        self, node: WhereNode
    ) raises -> List[UInt64]:
        var key = node.key()
        if not self._metadata_index.contains_field(key):
            return self._empty_bitmap()
        if self._metadata_index.can_evaluate(key):
            return self._metadata_index.evaluate(
                node, len(self._user_ids)
            )
        return self._scan_predicate_bitmap(node)

    def _where_filter(self, where: Where) raises -> List[UInt8]:
        """
        Builds the HNSW exclusion mask from typed bitmap predicates.

        Metadata bitmaps contain matching internal IDs. Logical instructions
        combine complete packed UInt64 words before one final conversion to
        the byte mask required by HNSW. Deleted rows are always excluded.
        """
        var nodes = where.nodes()
        if len(nodes) == 0:
            raise Error("Where expression cannot be empty.")

        # Validate the postfix expression and calculate its maximum bitmap
        # stack depth so compound filters allocate only the required slots.
        var depth = 0
        var max_depth = 0
        for node_index in range(len(nodes)):
            if nodes[node_index].is_predicate():
                depth += 1
                max_depth = max(max_depth, depth)
            elif nodes[node_index].is_not():
                if depth < 1:
                    raise Error("Invalid Where expression.")
            else:
                var arity = nodes[node_index].arity()
                if arity <= 0 or depth < arity:
                    raise Error("Invalid Where expression.")
                depth = depth - arity + 1
        if depth != 1:
            raise Error("Invalid Where expression.")

        var word_count = (
            len(self._user_ids) + BITMAP_WORD_BITS - 1
        ) // BITMAP_WORD_BITS
        var stack_words = List[UInt64](
            unsafe_uninit_length=max_depth * word_count
        )
        var stack_size = 0
        for node_index in range(len(nodes)):
            if nodes[node_index].is_predicate():
                var predicate = self._predicate_bitmap(nodes[node_index])
                var destination = stack_size * word_count
                for word_index in range(word_count):
                    stack_words[destination + word_index] = (
                        predicate[word_index]
                    )
                stack_size += 1
            elif nodes[node_index].is_not():
                var destination = (stack_size - 1) * word_count
                for word_index in range(word_count):
                    stack_words[destination + word_index] = ~stack_words[
                        destination + word_index
                    ]
            else:
                var arity = nodes[node_index].arity()
                var first = stack_size - arity
                var destination = first * word_count
                for word_index in range(word_count):
                    var combined = stack_words[
                        destination + word_index
                    ]
                    for operand in range(1, arity):
                        var operand_word = stack_words[
                            (first + operand) * word_count + word_index
                        ]
                        if nodes[node_index].is_all():
                            combined &= operand_word
                        elif nodes[node_index].is_any():
                            combined |= operand_word
                        else:
                            raise Error("Invalid Where expression.")
                    stack_words[destination + word_index] = combined
                stack_size = first + 1

        # NOT may set unused high bits in the final word. Mask them so the
        # packed result always describes exactly the collection's row count.
        var remainder = len(self._user_ids) % BITMAP_WORD_BITS
        if word_count > 0 and remainder > 0:
            var valid_bits = (
                UInt64(1) << UInt64(remainder)
            ) - UInt64(1)
            stack_words[word_count - 1] &= valid_bits

        var exclusion = List[UInt8](
            unsafe_uninit_length=len(self._user_ids)
        )
        for internal_id in range(len(self._user_ids)):
            var word = stack_words[internal_id // BITMAP_WORD_BITS]
            var mask = UInt64(1) << UInt64(
                internal_id % BITMAP_WORD_BITS
            )
            var matches = (word & mask) != 0
            exclusion[internal_id] = (
                1
                if self._is_deleted[internal_id] > 0 or not matches
                else 0
            )
        return exclusion^

    def _query_into_filtered(
        self,
        query_embeddings: Span[Float32, _],
        n_results: Int,
        mut ids: Span[mut=True, Int, _],
        mut distances: Span[mut=True, Float32, _],
        exclusion: Span[UInt8, _],
    ) raises:
        if self._dimension <= 0:
            raise Error("Collection dimension must be positive.")
        if len(query_embeddings) % self._dimension != 0:
            raise Error(
                "Query embeddings length must be a multiple of dimension."
            )
        if n_results <= 0 or n_results > 2048:
            raise Error("n_results must be between 1 and 2048.")

        var num_queries = len(query_embeddings) // self._dimension
        var output_size = num_queries * n_results
        if len(ids) < output_size or len(distances) < output_size:
            raise Error(
                "Output buffers are smaller than query_count * n_results."
            )
        if len(exclusion) != len(self._user_ids):
            raise Error("Where filter size does not match collection size.")
        if num_queries == 0:
            return

        self._search_index(
            query_embeddings,
            n_results,
            distances,
            ids,
            exclusion,
        )
        for index in range(output_size):
            var internal_id = ids[index]
            if internal_id >= 0 and internal_id < len(self._user_ids):
                ids[index] = self._user_ids[internal_id]
            else:
                ids[index] = -1

    def _build_query_results(
        self,
        ids_storage: List[Int],
        distances_storage: List[Float32],
        num_queries: Int,
        n_results: Int,
    ) raises -> QueryResults:
        """
        Materializes aligned managed result rows.

        Metadata and document rows are omitted entirely when the collection
        has no values of that kind, preserving the lightweight vector-only
        query path.
        """
        var include_metadatas = len(self._metadatas) > 0
        var include_documents = len(self._documents) > 0

        # Preserve the original vector-only materialization path exactly:
        # no ID-map lookup and no payload branch for every returned neighbor.
        if not include_metadatas and not include_documents:
            var vector_ids = List[List[Int]](capacity=num_queries)
            var vector_distances = List[List[Float32]](
                capacity=num_queries
            )
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
                if record_id >= 0 and record_id in self._id_to_internal:
                    internal_id = self._id_to_internal[record_id]

                if include_metadatas:
                    if internal_id >= 0:
                        row_metadatas.append(
                            self._metadata_for_internal(internal_id)
                        )
                    else:
                        row_metadatas.append(Metadata())

                if include_documents:
                    if internal_id >= 0:
                        row_documents.append(
                            self._document_for_internal(internal_id)
                        )
                    else:
                        row_documents.append("")

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

    def query(
        mut self,
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
        mut self,
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

    def _build_scored_query_results(
        self,
        internal_ids_storage: List[Int],
        scores_storage: List[Float32],
        num_queries: Int,
        n_results: Int,
    ) raises -> QueryResults:
        """Materializes ranked scores and aligned collection payloads."""
        var include_metadatas = len(self._metadatas) > 0
        var include_documents = len(self._documents) > 0
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
                    and internal_id < len(self._user_ids)
                    and self._is_deleted[internal_id] == 0
                )
                row_ids.append(
                    self._user_ids[internal_id] if is_valid else -1
                )
                row_scores.append(scores_storage[offset])

                if include_metadatas:
                    if is_valid:
                        row_metadatas.append(
                            self._metadata_for_internal(internal_id)
                        )
                    else:
                        row_metadatas.append(Metadata())
                if include_documents:
                    if is_valid:
                        row_documents.append(
                            self._document_for_internal(internal_id)
                        )
                    else:
                        row_documents.append("")

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

    def _query_bm25(
        self,
        query_texts: List[String],
        n_results: Int,
        exclusion: List[UInt8],
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

        var ids_storage = List[Int](
            capacity=len(query_texts) * n_results
        )
        var scores_storage = List[Float32](
            capacity=len(query_texts) * n_results
        )
        for query_text in query_texts:
            var result = self._bm25.search(
                query_text, n_results, exclusion
            )
            for rank in range(n_results):
                ids_storage.append(result.internal_ids[rank])
                scores_storage.append(result.scores[rank])

        return self._build_scored_query_results(
            ids_storage,
            scores_storage,
            len(query_texts),
            n_results,
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
        mut self,
        query_embeddings: List[Float32],
        query_texts: List[String],
        n_results: Int,
        rrf_k: Int,
        candidate_multiplier: Int,
        exclusion: List[UInt8],
    ) raises -> QueryResults:
        if self._dimension <= 0:
            raise Error("Collection dimension must be positive.")
        if len(query_embeddings) % self._dimension != 0:
            raise Error(
                "Query embeddings length must be a multiple of dimension."
            )
        if n_results <= 0 or n_results > 2048:
            raise Error("n_results must be between 1 and 2048.")
        if rrf_k <= 0:
            raise Error("rrf_k must be positive.")
        if candidate_multiplier <= 0 or candidate_multiplier > 2048:
            raise Error("candidate_multiplier must be between 1 and 2048.")

        var num_queries = len(query_embeddings) // self._dimension
        if num_queries != len(query_texts):
            raise Error(
                "Hybrid query embedding and text batch sizes must match."
            )
        if num_queries == 0:
            return QueryResults(
                List[List[Int]](),
                List[List[Float32]](),
                List[List[Metadata]](),
                List[List[String]](),
                List[List[Float32]](),
            )
        if len(exclusion) > 0 and len(exclusion) != len(self._user_ids):
            raise Error("Where filter size does not match collection size.")

        var candidate_count = n_results * candidate_multiplier
        if candidate_count > 2048:
            candidate_count = 2048
        var candidate_storage_size = num_queries * candidate_count
        var vector_ids = List[Int](
            unsafe_uninit_length=candidate_storage_size
        )
        var vector_distances = List[Float32](
            unsafe_uninit_length=candidate_storage_size
        )
        var queries = Span[Float32](
            ptr=query_embeddings.unsafe_ptr(),
            length=len(query_embeddings),
        )
        var vector_id_span = Span[mut=True, Int](vector_ids)
        var vector_distance_span = Span[mut=True, Float32](
            vector_distances
        )

        if len(exclusion) > 0:
            var filter_span = Span[UInt8](exclusion)
            self._search_index(
                queries,
                candidate_count,
                vector_distance_span,
                vector_id_span,
                filter_span,
            )
        elif self.count_deleted() > 0:
            var deleted_span = Span[mut=False, UInt8](self._is_deleted)
            self._search_index(
                queries,
                candidate_count,
                vector_distance_span,
                vector_id_span,
                deleted_span,
            )
        else:
            var empty_filter = Span[UInt8, MutUntrackedOrigin]()
            self._search_index(
                queries,
                candidate_count,
                vector_distance_span,
                vector_id_span,
                empty_filter,
            )

        var fused_ids = List[Int](
            capacity=num_queries * n_results
        )
        var fused_scores = List[Float32](
            capacity=num_queries * n_results
        )
        for query_index in range(num_queries):
            var vector_row = List[Int](capacity=candidate_count)
            var vector_offset = query_index * candidate_count
            for rank in range(candidate_count):
                vector_row.append(vector_ids[vector_offset + rank])

            var text_row = self._bm25.search(
                query_texts[query_index],
                candidate_count,
                exclusion,
            )
            var fused = reciprocal_rank_fusion(
                vector_row,
                text_row.internal_ids,
                n_results,
                rrf_k,
            )
            for rank in range(n_results):
                fused_ids.append(fused.internal_ids[rank])
                fused_scores.append(fused.scores[rank])

        return self._build_scored_query_results(
            fused_ids,
            fused_scores,
            num_queries,
            n_results,
        )

    def query_hybrid(
        mut self,
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
        mut self,
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

    def save(mut self, path: String) raises:
        """Saves payloads, storage kind, vectors, and the HNSW graph."""
        from mojovec.io.serialization import (
            write_index_hnsw_mmap,
            write_index_hnsw_sq8_mmap,
            write_int,
        )

        var file = open(path, "w")
        write_int(file, COLLECTION_MAGIC_V2)
        write_int(file, COLLECTION_FORMAT_VERSION)
        _write_string(file, self._name)
        write_int(file, self._dimension)
        write_int(file, self._storage_kind)

        var num_ids = len(self._user_ids)
        write_int(file, num_ids)
        if num_ids > 0:
            file.write_bytes(
                Span[UInt8](
                    ptr=self._user_ids.unsafe_ptr().bitcast[UInt8](),
                    length=num_ids * 8,
                )
            )
            file.write_bytes(
                Span[UInt8](ptr=self._is_deleted.unsafe_ptr(), length=num_ids)
            )

        # Metadata is sparse: collections that do not use it pay no
        # per-record object or file-format overhead.
        write_int(file, len(self._metadatas))
        for internal_id in range(num_ids):
            if internal_id in self._metadata_by_internal:
                write_int(file, internal_id)
                var metadata_index = self._metadata_by_internal[internal_id]
                _write_metadata(file, self._metadatas[metadata_index])

        # Documents use the same sparse representation as metadata. Empty
        # strings are reserved for a missing document and are never persisted.
        write_int(file, len(self._documents))
        for internal_id in range(num_ids):
            if internal_id in self._document_by_internal:
                write_int(file, internal_id)
                var document_index = self._document_by_internal[internal_id]
                _write_string(file, self._documents[document_index])

        if self._storage_kind == STORAGE_SQ8:
            write_index_hnsw_sq8_mmap(
                file, self._hnsw.unsafe_get[SQ8HNSW]()
            )
        else:
            write_index_hnsw_mmap(
                file, self._hnsw.unsafe_get[FlatHNSW]()
            )
        file.close()

    @staticmethod
    def load(
        path: String,
        memory_mapped: Bool = True,
        mmap_threshold_bytes: Int = DEFAULT_MMAP_THRESHOLD_BYTES,
    ) raises -> Collection:
        """Loads Flat/SQ8, mapping V5 index arrays when the threshold is met.
        """
        from mojovec.io.serialization import (
            check_size_limit,
            read_index_hnsw,
            read_index_hnsw_mmap,
            read_index_hnsw_sq8,
            read_index_hnsw_sq8_mmap,
            read_int,
        )

        if mmap_threshold_bytes < 0:
            raise Error("mmap_threshold_bytes cannot be negative.")
        var file = open(path, "r")
        var file_size = Int(file.seek(0, SEEK_END))
        _ = file.seek(0, SEEK_SET)
        var magic = read_int(file)
        var name = String("")
        var dimension: Int
        var storage_kind: StorageKind = STORAGE_SQ8
        var version = 1

        if magic == COLLECTION_MAGIC_V1:
            dimension = read_int(file)
        elif magic == COLLECTION_MAGIC_V2:
            version = read_int(file)
            if version < 2 or version > COLLECTION_FORMAT_VERSION:
                raise Error("Unsupported Collection format version.")
            name = _read_string(file)
            dimension = read_int(file)
            storage_kind = read_int(file)
            if storage_kind != STORAGE_FLAT and storage_kind != STORAGE_SQ8:
                raise Error("Invalid Collection storage kind.")
        else:
            raise Error("Invalid Collection magic.")

        check_size_limit(dimension, 65_536)
        var num_ids = read_int(file)
        check_size_limit(num_ids, 1_000_000_000)
        var collection = Collection(
            dimension,
            quantized=storage_kind == STORAGE_SQ8,
            name=name,
        )

        if num_ids > 0:
            var ids_data = file.read_bytes(num_ids * 8)
            var ids_source = ids_data.unsafe_ptr().bitcast[Int]()
            var deleted_data = file.read_bytes(num_ids)
            var deleted_source = deleted_data.unsafe_ptr()
            for i in range(num_ids):
                var record_id = ids_source[i]
                var is_deleted = deleted_source[i]
                collection._user_ids.append(record_id)
                collection._is_deleted.append(is_deleted)
                if is_deleted == 0:
                    collection._id_to_internal[record_id] = i
            _ = len(ids_data)
            _ = len(deleted_data)

        if version >= COLLECTION_FORMAT_VERSION_WITH_METADATA:
            var metadata_count = read_int(file)
            check_size_limit(metadata_count, num_ids)
            for _ in range(metadata_count):
                var internal_id = read_int(file)
                if (
                    internal_id < 0
                    or internal_id >= num_ids
                    or internal_id in collection._metadata_by_internal
                ):
                    raise Error("Invalid metadata internal ID.")
                var metadata = _read_metadata(file)
                if metadata.count() == 0:
                    raise Error("Sparse metadata entries cannot be empty.")
                collection._store_metadata(internal_id, metadata)

        if version >= COLLECTION_FORMAT_VERSION_WITH_DOCUMENTS:
            var document_count = read_int(file)
            check_size_limit(document_count, num_ids)
            for _ in range(document_count):
                var internal_id = read_int(file)
                if (
                    internal_id < 0
                    or internal_id >= num_ids
                    or internal_id in collection._document_by_internal
                ):
                    raise Error("Invalid document internal ID.")
                var document = _read_string(file)
                if document.byte_length() == 0:
                    raise Error("Sparse document entries cannot be empty.")
                collection._store_document(internal_id, document)

        var use_mmap = (
            memory_mapped
            and version >= COLLECTION_FORMAT_VERSION_WITH_MMAP_INDEX
            and file_size >= mmap_threshold_bytes
        )
        if storage_kind == STORAGE_SQ8:
            var index: SQ8HNSW
            if version >= COLLECTION_FORMAT_VERSION_WITH_MMAP_INDEX:
                index = read_index_hnsw_sq8_mmap(file, file_size)
                if not use_mmap:
                    index.storage._detach_mapped()
                    index.hnsw._detach_mapped()
            else:
                index = read_index_hnsw_sq8(file)
            if index.ntotal != num_ids:
                raise Error("Collection metadata and SQ8 index size differ.")
            collection._hnsw.set[SQ8HNSW](index^)
        else:
            var index: FlatHNSW
            if version >= COLLECTION_FORMAT_VERSION_WITH_MMAP_INDEX:
                index = read_index_hnsw_mmap(file, file_size)
                if not use_mmap:
                    index.storage._detach_mapped()
                    index.hnsw._detach_mapped()
            else:
                index = read_index_hnsw(file)
            if index.ntotal != num_ids:
                raise Error("Collection metadata and Flat index size differ.")
            collection._hnsw.set[FlatHNSW](index^)

        file.close()
        return collection^
