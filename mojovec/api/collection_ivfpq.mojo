"""Managed high-level collection backed by an IVF-PQ index."""

from std.collections import Dict, List
from std.memory import OwnedPointer
from std.memory.span import Span
from std.math import max, min
from std.os import SEEK_CUR, SEEK_END, SEEK_SET

from mojovec.api.collection_codec import (
    METRIC_COSINE,
    _index_metric,
    _metric_name,
    _parse_metric,
    _read_string,
    _write_string,
)
from mojovec.api.collection_vector_query import (
    _normalize_vectors,
    _validate_finite_vectors,
)
from mojovec.api.metadata import Metadata
from mojovec.api.results import IVFPQStats, QueryResults
from mojovec.core.types import METRIC_L2, MetricType
from mojovec.core.validation import _validate_vector_dimension
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_ivf_pq import IndexIVFPQ


comptime COLLECTION_IVFPQ_MAGIC = 0x43495650
comptime COLLECTION_IVFPQ_VERSION = 1
comptime MAX_IVF_NLIST = 1_000_000
comptime MAX_IVF_NPROBE = 65_536
comptime MAX_QUERY_RESULTS = 2_048
comptime PQ_CENTROIDS = 256


def _validate_ivfpq_parameters(
    dimension: Int,
    nlist: Int,
    M: Int,
    nprobe: Int,
) raises:
    _validate_vector_dimension(dimension)
    if nlist <= 0 or nlist > MAX_IVF_NLIST:
        raise Error("nlist must be between 1 and 1000000.")
    if M <= 0 or M > dimension or dimension % M != 0:
        raise Error("M must divide dimension and be between 1 and dimension.")
    if nprobe <= 0 or nprobe > nlist or nprobe > MAX_IVF_NPROBE:
        raise Error("nprobe must be between 1 and nlist.")


struct CollectionIVFPQ(Movable, Writable):
    """A managed IVF-PQ collection with compact eight-bit subvector codes.

    Training learns ``nlist`` coarse centroids and 256 centroids for each of
    the ``M`` product-quantizer subvectors. ``dimension`` must be divisible by
    ``M``. Search probes ``nprobe`` coarse lists and returns Chroma-style
    smaller-is-better distances for L2, cosine, and inner product.
    """

    var _name: String
    var _dimension: Int
    var _metric_type: MetricType
    var _ivfpq: OwnedPointer[IndexIVFPQ[IndexFlat]]
    var _user_ids: List[Int]
    var _id_to_internal: Dict[Int, Int]

    def __init__(
        out self,
        dimension: Int,
        nlist: Int = 100,
        M: Int = 16,
        nprobe: Int = 0,
        name: String = "",
        metric: String = "l2",
    ) raises:
        var effective_nprobe = min(10, nlist) if nprobe == 0 else nprobe
        _validate_ivfpq_parameters(dimension, nlist, M, effective_nprobe)
        var metric_type = _parse_metric(metric)
        self._name = name.copy()
        self._dimension = dimension
        self._metric_type = metric_type
        # Cosine vectors are normalized by this collection. Their inner
        # product coarse ordering is therefore equivalent to L2 ordering,
        # while IP collections retain the correct maximum-dot-product probes.
        self._ivfpq = OwnedPointer(
            IndexIVFPQ[IndexFlat](
                IndexFlat(dimension, _index_metric(metric_type)),
                dimension,
                nlist,
                M,
                _index_metric(metric_type),
            )
        )
        self._ivfpq[].nprobe = effective_nprobe
        self._user_ids = List[Int]()
        self._id_to_internal = Dict[Int, Int]()

    def __init__(out self, *, deinit take: Self):
        self._name = take._name^
        self._dimension = take._dimension
        self._metric_type = take._metric_type
        self._ivfpq = take._ivfpq^
        self._user_ids = take._user_ids^
        self._id_to_internal = take._id_to_internal^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "CollectionIVFPQ(name=",
            self._name,
            ", dimension=",
            self._dimension,
            ", metric=",
            self.metric(),
            ", count=",
            self.count(),
            ", nlist=",
            self.nlist(),
            ", M=",
            self.pq_subvectors(),
            ", nprobe=",
            self.nprobe(),
            ", trained=",
            self.is_trained(),
            ")",
        )

    def name(self) -> String:
        return self._name.copy()

    def dimension(self) -> Int:
        return self._dimension

    def metric(self) -> String:
        return _metric_name(self._metric_type)

    def count(self) -> Int:
        return len(self._user_ids)

    def nlist(self) -> Int:
        return self._ivfpq[].nlist

    def pq_subvectors(self) -> Int:
        return self._ivfpq[].M

    def nprobe(self) -> Int:
        return self._ivfpq[].nprobe

    def is_trained(self) -> Bool:
        return self._ivfpq[].is_trained

    def set_nprobe(mut self, nprobe: Int) raises:
        if nprobe <= 0 or nprobe > self.nlist():
            raise Error("nprobe must be between 1 and nlist.")
        self._ivfpq[].nprobe = nprobe

    def stats(self) -> IVFPQStats:
        return IVFPQStats(
            count=self.count(),
            dimension=self._dimension,
            nlist=self.nlist(),
            M=self.pq_subvectors(),
            nprobe=self.nprobe(),
            trained=self.is_trained(),
            metric=self.metric(),
            code_size_bytes=self.pq_subvectors(),
        )

    def train(mut self, embeddings: List[Float32]) raises:
        """Trains coarse and PQ codebooks on representative vectors."""
        self._train_from_span(Span[Float32](embeddings))

    def _train_from_span(
        mut self, embeddings: Span[Float32, _]
    ) raises:
        if len(embeddings) % self._dimension != 0:
            raise Error(
                "Embeddings length must be a multiple of collection dimension."
            )
        var num_vectors = len(embeddings) // self._dimension
        var minimum = max(self.nlist(), PQ_CENTROIDS)
        if num_vectors < minimum:
            raise Error(
                "IVF-PQ training requires at least max(nlist, 256) vectors."
            )
        if self.is_trained():
            return
        if self._metric_type == METRIC_COSINE:
            var normalized = _normalize_vectors(
                embeddings, self._dimension
            )
            self._ivfpq[].train(Span[Float32](normalized))
            return
        _validate_finite_vectors(embeddings)
        self._ivfpq[].train(embeddings)

    def _validate_new_ids(self, ids: Span[Int, _]) raises:
        var batch = Dict[Int, Bool]()
        for record_id in ids:
            if record_id in batch:
                raise Error("Duplicate IDs are not allowed within an add batch.")
            if record_id in self._id_to_internal:
                raise Error("Cannot add an ID that already exists.")
            batch[record_id] = True

    def add(
        mut self,
        ids: List[Int],
        embeddings: List[Float32],
    ) raises:
        """Adds unique IDs, auto-training from the first batch if necessary."""
        self._add_from_spans(
            Span[Int](ids),
            Span[Float32](embeddings),
        )

    def _add_from_spans(
        mut self,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
    ) raises:
        var num_vectors = len(ids)
        if len(embeddings) != num_vectors * self._dimension:
            raise Error("Embeddings length must equal len(ids) * dimension.")
        if num_vectors == 0:
            return
        self._validate_new_ids(ids)
        if not self.is_trained():
            self._train_from_span(embeddings)

        if self._metric_type == METRIC_COSINE:
            var normalized = _normalize_vectors(
                embeddings, self._dimension
            )
            self._ivfpq[].add(Span[Float32](normalized))
        else:
            _validate_finite_vectors(embeddings)
            self._ivfpq[].add(embeddings)
        for record_id in ids:
            var internal_id = len(self._user_ids)
            self._user_ids.append(record_id)
            self._id_to_internal[record_id] = internal_id

    def query(
        self,
        query_embeddings: List[Float32],
        n_results: Int = 10,
    ) raises -> QueryResults:
        """Searches one or more flattened query vectors."""
        return self._query_from_span(
            Span[Float32](query_embeddings), n_results
        )

    def _query_from_span(
        self,
        query_embeddings: Span[Float32, _],
        n_results: Int = 10,
    ) raises -> QueryResults:
        if len(query_embeddings) % self._dimension != 0:
            raise Error(
                "Query embeddings length must be a multiple of dimension."
            )
        if n_results <= 0 or n_results > MAX_QUERY_RESULTS:
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

        if self._metric_type == METRIC_COSINE:
            var normalized = _normalize_vectors(
                query_embeddings, self._dimension
            )
            return self._query_prepared(
                Span[Float32](normalized), num_queries, n_results
            )
        _validate_finite_vectors(query_embeddings)
        return self._query_prepared(
            query_embeddings, num_queries, n_results
        )

    def _query_prepared(
        self,
        prepared: Span[Float32, _],
        num_queries: Int,
        n_results: Int,
    ) raises -> QueryResults:
        var output_size = num_queries * n_results
        var distances_storage = List[Float32](
            unsafe_uninit_length=output_size
        )
        var labels_storage = List[Int](unsafe_uninit_length=output_size)
        var distance_span = Span[mut=True, Float32](distances_storage)
        var label_span = Span[mut=True, Int](labels_storage)
        self._ivfpq[].search(
            prepared,
            n_results,
            distance_span,
            label_span,
        )

        var all_ids = List[List[Int]](capacity=num_queries)
        var all_distances = List[List[Float32]](capacity=num_queries)
        for query_index in range(num_queries):
            var query_ids = List[Int](capacity=n_results)
            var query_distances = List[Float32](capacity=n_results)
            for rank in range(n_results):
                var offset = query_index * n_results + rank
                var internal_id = labels_storage[offset]
                var distance = distances_storage[offset]
                if internal_id >= 0 and internal_id < len(self._user_ids):
                    query_ids.append(self._user_ids[internal_id])
                    if self._metric_type != METRIC_L2:
                        distance = 1.0 - distance
                    query_distances.append(distance)
                else:
                    query_ids.append(-1)
                    query_distances.append(distance)
            all_ids.append(query_ids^)
            all_distances.append(query_distances^)

        return QueryResults(
            all_ids^,
            all_distances^,
            List[List[Metadata]](),
            List[List[String]](),
            List[List[Float32]](),
        )

    def save(self, path: String) raises:
        """Atomically saves a checksummed owned IVF-PQ snapshot."""
        from mojovec.io.serialization import write_index_ivf_pq, write_int
        from mojovec.io.atomic_file import (
            atomic_replace,
            atomic_temporary_path,
            remove_file_best_effort,
            sync_parent_directory,
        )
        from mojovec.io.snapshot_file import append_snapshot_checksum

        var temporary_path = atomic_temporary_path(path)
        try:
            var file = open(temporary_path, "w")
            write_int(file, COLLECTION_IVFPQ_MAGIC)
            write_int(file, COLLECTION_IVFPQ_VERSION)
            write_int(file, self._dimension)
            _write_string(file, self._name)
            _write_string(file, self.metric())
            write_int(file, len(self._user_ids))
            if len(self._user_ids) > 0:
                file.write_bytes(
                    Span[UInt8](
                        ptr=self._user_ids.unsafe_ptr().bitcast[UInt8](),
                        length=len(self._user_ids) * 8,
                    )
                )
            write_index_ivf_pq(file, self._ivfpq[])
            file.close()
            append_snapshot_checksum(temporary_path)
            atomic_replace(temporary_path, path)
            sync_parent_directory(path)
        except error:
            remove_file_best_effort(temporary_path)
            raise error^

    @staticmethod
    def load(path: String) raises -> CollectionIVFPQ:
        """Loads an owned IVF-PQ collection snapshot."""
        from mojovec.io.serialization import (
            checked_byte_count,
            check_size_limit,
            read_index_ivf_pq,
            read_int,
        )
        from mojovec.io.snapshot_file import (
            SNAPSHOT_CHECKSUM_TRAILER_BYTES,
            validate_snapshot_checksum,
        )

        var file = open(path, "r")
        var total_size = Int(file.seek(0, SEEK_END))
        if total_size < SNAPSHOT_CHECKSUM_TRAILER_BYTES:
            raise Error("IVF-PQ snapshot checksum trailer is missing.")
        var payload_size = validate_snapshot_checksum(file, total_size)
        _ = file.seek(0, SEEK_SET)
        if read_int(file) != COLLECTION_IVFPQ_MAGIC:
            raise Error("Invalid CollectionIVFPQ snapshot magic.")
        if read_int(file) != COLLECTION_IVFPQ_VERSION:
            raise Error("Unsupported CollectionIVFPQ snapshot version.")
        var dimension = read_int(file)
        _validate_vector_dimension(dimension)
        var name = _read_string(file)
        var metric = _read_string(file)
        var metric_type = _parse_metric(metric)
        var num_ids = read_int(file)
        check_size_limit(num_ids, 1_000_000_000)
        var ids_bytes = checked_byte_count(num_ids, 8)
        var ids_offset = Int(file.seek(0, SEEK_CUR))
        if ids_bytes > payload_size - ids_offset:
            raise Error("IVF-PQ ID data exceeds the snapshot payload.")

        var ids = List[Int](capacity=num_ids)
        if num_ids > 0:
            var data = file.read_bytes(ids_bytes)
            if len(data) != ids_bytes:
                raise Error("Unexpected end of IVF-PQ ID data.")
            var source = data.unsafe_ptr().bitcast[Int]()
            for index in range(num_ids):
                ids.append(source[index])
            _ = len(data)

        var loaded_index = read_index_ivf_pq(file)
        if Int(file.seek(0, SEEK_CUR)) != payload_size:
            raise Error("IVF-PQ snapshot contains unexpected payload data.")
        file.close()
        _validate_ivfpq_parameters(
            dimension,
            loaded_index.nlist,
            loaded_index.M,
            loaded_index.nprobe,
        )
        if loaded_index.d != dimension or loaded_index.ntotal != num_ids:
            raise Error("IVF-PQ snapshot dimensions or record count mismatch.")
        if loaded_index.metric_type != _index_metric(metric_type):
            raise Error("IVF-PQ snapshot metric mismatch.")
        if (
            loaded_index.quantizer.d != dimension
            or (
                loaded_index.is_trained
                and loaded_index.quantizer.ntotal != loaded_index.nlist
            )
            or (
                not loaded_index.is_trained
                and loaded_index.quantizer.ntotal != 0
            )
            or loaded_index.quantizer.metric_type
            != loaded_index.metric_type
        ):
            raise Error("IVF-PQ coarse quantizer does not match its index.")

        var seen_internal_ids = List[UInt8](length=num_ids, fill=0)
        var indexed_count = 0
        for list_no in range(loaded_index.nlist):
            var list_ids = loaded_index.invlists.get_ids(list_no)
            indexed_count += len(list_ids)
            for internal_id in list_ids:
                if internal_id < 0 or internal_id >= num_ids:
                    raise Error("IVF-PQ snapshot contains an invalid internal ID.")
                if seen_internal_ids[internal_id] != 0:
                    raise Error("IVF-PQ snapshot contains a duplicate internal ID.")
                seen_internal_ids[internal_id] = 1
        if indexed_count != num_ids:
            raise Error("IVF-PQ inverted-list count does not match its IDs.")

        var collection = CollectionIVFPQ(
            dimension,
            loaded_index.nlist,
            loaded_index.M,
            loaded_index.nprobe,
            name,
            metric,
        )
        collection._ivfpq = OwnedPointer(loaded_index^)

        for internal_id in range(len(ids)):
            var record_id = ids[internal_id]
            if record_id in collection._id_to_internal:
                raise Error("IVF-PQ snapshot contains duplicate IDs.")
            collection._user_ids.append(record_id)
            collection._id_to_internal[record_id] = internal_id
        return collection^
