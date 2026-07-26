from std.collections import Dict, List
from std.io.file import FileHandle
from std.memory import alloc
from std.memory.span import Span
from std.utils import Variant

from mojovec.api.results import QueryResults
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
comptime COLLECTION_FORMAT_VERSION = 2


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


struct Collection(Movable, Writable):
    """A Chroma-style embedding collection backed by Flat or SQ8 HNSW."""

    var _name: String
    var _dimension: Int
    var _storage_kind: StorageKind
    var _hnsw: HNSWStorage
    var _user_ids: List[Int]
    var _is_deleted: List[UInt8]
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
        self._id_to_internal = Dict[Int, Int]()

    def __init__(out self, *, deinit take: Self):
        self._name = take._name^
        self._dimension = take._dimension
        self._storage_kind = take._storage_kind
        self._hnsw = take._hnsw^
        self._user_ids = take._user_ids^
        self._is_deleted = take._is_deleted^
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
    ) raises:
        self._validate_shape(ids, embeddings)
        self._validate_unique_batch(ids)
        if len(ids) == 0:
            return

        for i in range(len(ids)):
            var record_id = ids[i]
            if record_id in self._id_to_internal:
                if not replace_existing:
                    raise Error("ID already exists; use upsert or update.")
                var previous = self._id_to_internal[record_id]
                self._is_deleted[previous] = 1

            var internal_id = len(self._user_ids)
            self._user_ids.append(record_id)
            self._is_deleted.append(0)
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

    def _add_from_spans(
        mut self,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
    ) raises:
        """Internal zero-copy bridge used by managed frontends."""
        self._append_records(ids, embeddings, replace_existing=False)

    def upsert(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        """
        Inserts records or replaces active records with matching IDs.

        Inputs and collection storage use automatic lifetime management.
        """
        self._upsert_from_spans(
            Span[Int](ptr=ids.unsafe_ptr(), length=len(ids)),
            Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings)),
        )

    def _upsert_from_spans(
        mut self,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
    ) raises:
        """Internal zero-copy bridge used by managed frontends."""
        self._append_records(ids, embeddings, replace_existing=True)

    def update(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        """Updates existing IDs and rejects missing IDs."""
        var ids_span = Span[Int](ptr=ids.unsafe_ptr(), length=len(ids))
        var embeddings_span = Span[Float32](
            ptr=embeddings.unsafe_ptr(), length=len(embeddings)
        )
        self._validate_shape(ids_span, embeddings_span)
        self._validate_unique_batch(ids_span)
        for i in range(len(ids_span)):
            if ids_span[i] not in self._id_to_internal:
                raise Error("Cannot update an ID that does not exist.")
        self._append_records(ids_span, embeddings_span, replace_existing=True)

    def delete(mut self, ids: List[Int]):
        """Soft-deletes active records by ID."""
        for i in range(len(ids)):
            var internal_id = self._id_to_internal.pop(ids[i], -1)
            if internal_id >= 0:
                self._is_deleted[internal_id] = 1

    def set_ef_search(mut self, ef: Int) raises:
        if ef <= 0 or ef > 2048:
            raise Error("ef_search must be between 1 and 2048.")
        if self._storage_kind == STORAGE_SQ8:
            self._hnsw.unsafe_get[SQ8HNSW]().hnsw.efSearch = ef
        else:
            self._hnsw.unsafe_get[FlatHNSW]().hnsw.efSearch = ef

    def _search_index(
        mut self,
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
        mut self,
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

        var deleted_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
            self._is_deleted.unsafe_ptr()
        )
        var deleted = Span[UInt8, MutAnyOrigin](
            ptr=deleted_ptr,
            length=len(self._is_deleted),
        )
        self._search_index(query_embeddings, n_results, distances, ids, deleted)

        for i in range(output_size):
            var internal_id = ids[i]
            if internal_id >= 0 and internal_id < len(self._user_ids):
                ids[i] = self._user_ids[internal_id]
            else:
                ids[i] = -1

    def query(
        mut self,
        query_embeddings: List[Float32],
        n_results: Int = 10,
    ) raises -> QueryResults:
        """
        Runs embedding queries and returns automatically managed results.

        QueryResults owns its ID and distance Lists. Callers do not allocate
        output buffers and never need to release result memory manually.
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
            return QueryResults(List[List[Int]](), List[List[Float32]]())

        var ids_ptr = alloc[Int](num_queries * n_results)
        var distances_ptr = alloc[Float32](num_queries * n_results)
        var queries = Span[Float32](
            ptr=query_embeddings.unsafe_ptr(),
            length=len(query_embeddings),
        )
        var ids = Span[mut=True, Int](
            ptr=ids_ptr, length=num_queries * n_results
        )
        var distances = Span[mut=True, Float32](
            ptr=distances_ptr, length=num_queries * n_results
        )
        self._query_into(queries, n_results, ids, distances)

        var all_ids = List[List[Int]](capacity=num_queries)
        var all_distances = List[List[Float32]](capacity=num_queries)
        for i in range(num_queries):
            var row_ids = List[Int](capacity=n_results)
            var row_distances = List[Float32](capacity=n_results)
            for j in range(n_results):
                var offset = i * n_results + j
                row_ids.append(ids_ptr[offset])
                row_distances.append(distances_ptr[offset])
            all_ids.append(row_ids^)
            all_distances.append(row_distances^)

        ids_ptr.free()
        distances_ptr.free()
        return QueryResults(all_ids^, all_distances^)

    def save(mut self, path: String) raises:
        """Saves metadata, storage kind, vectors, and HNSW graph."""
        from mojovec.io.serialization import (
            write_index_hnsw,
            write_index_hnsw_sq8,
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

        if self._storage_kind == STORAGE_SQ8:
            write_index_hnsw_sq8(file, self._hnsw.unsafe_get[SQ8HNSW]())
        else:
            write_index_hnsw(file, self._hnsw.unsafe_get[FlatHNSW]())
        file.close()

    @staticmethod
    def load(path: String) raises -> Collection:
        """Loads both the legacy SQ8 format and the versioned Flat/SQ8 format.
        """
        from mojovec.io.serialization import (
            check_size_limit,
            read_index_hnsw,
            read_index_hnsw_sq8,
            read_int,
        )

        var file = open(path, "r")
        var magic = read_int(file)
        var name = String("")
        var dimension: Int
        var storage_kind: StorageKind = STORAGE_SQ8

        if magic == COLLECTION_MAGIC_V1:
            dimension = read_int(file)
        elif magic == COLLECTION_MAGIC_V2:
            var version = read_int(file)
            if version != COLLECTION_FORMAT_VERSION:
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

        if storage_kind == STORAGE_SQ8:
            var index = read_index_hnsw_sq8(file)
            if index.ntotal != num_ids:
                raise Error("Collection metadata and SQ8 index size differ.")
            collection._hnsw.set[SQ8HNSW](index^)
        else:
            var index = read_index_hnsw(file)
            if index.ntotal != num_ids:
                raise Error("Collection metadata and Flat index size differ.")
            collection._hnsw.set[FlatHNSW](index^)

        file.close()
        return collection^
