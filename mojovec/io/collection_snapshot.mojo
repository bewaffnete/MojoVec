"""Atomic Collection snapshot encoding and checksum publication."""

from std.collections import Dict, List, Optional
from std.io.file import FileHandle
from std.memory.span import Span
from std.os import SEEK_CUR, SEEK_END, SEEK_SET

from mojovec.api.collection_codec import (
    METRIC_COSINE,
    _read_metadata,
    _read_string,
    _write_metadata,
    _write_string,
)
from mojovec.api.collection_storage import (
    HNSWStorage,
    _read_storage,
    _write_storage,
)
from mojovec.api.metadata import Metadata
from mojovec.core.types import (
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    STORAGE_FLAT,
    STORAGE_SQ8,
    MetricType,
    StorageKind,
)
from mojovec.core.validation import _validate_vector_dimension
from mojovec.io.atomic_file import (
    atomic_replace,
    atomic_temporary_path,
    remove_file_best_effort,
    sync_parent_directory,
)
from mojovec.io.fault_injection import (
    SNAPSHOT_FAULT_AFTER_HEADER,
    SNAPSHOT_FAULT_AFTER_PAYLOAD,
    SNAPSHOT_FAULT_AFTER_PUBLISH,
    SNAPSHOT_FAULT_BEFORE_PUBLISH,
    inject_snapshot_fault,
)
from mojovec.io.serialization import (
    checked_byte_count,
    check_size_limit,
    read_int,
    write_int,
)
from mojovec.io.snapshot_file import (
    SNAPSHOT_CHECKSUM_TRAILER_BYTES,
    append_snapshot_checksum,
    validate_snapshot_checksum,
)


comptime COLLECTION_MAGIC = 1129270349
comptime COLLECTION_FORMAT = 8
comptime MAX_COLLECTION_RECORDS = 10_000_000
comptime MAX_COLLECTION_SNAPSHOT_BYTES = 274_877_906_944


def _remaining_snapshot_bytes(
    mut file: FileHandle,
    payload_size: Int,
) raises -> Int:
    """Returns validated bytes remaining in the checksummed payload."""
    var position = Int(file.seek(0, SEEK_CUR))
    if position < 0 or position > payload_size:
        raise Error("Snapshot cursor is outside the checksummed payload.")
    return payload_size - position


def _validate_sparse_entry_count(
    count: Int,
    record_count: Int,
    remaining_bytes: Int,
) raises:
    """Rejects impossible sparse counts before reserving managed memory."""
    check_size_limit(count, record_count)
    # Every sparse entry contains at least an internal ID and a count/length.
    if checked_byte_count(count, 16) > remaining_bytes:
        raise Error("Sparse entry count exceeds the remaining snapshot size.")


@fieldwise_init
struct CollectionSnapshot(Movable):
    """Validated snapshot data independent of the public Collection type."""

    var name: String
    var dimension: Int
    var storage_kind: StorageKind
    var metric_type: MetricType
    var identity: Int
    var applied_sequence: Int
    var user_ids: Optional[List[Int]]
    var is_deleted: Optional[List[UInt8]]
    var metadata_internal_ids: List[Int]
    var metadatas: List[Metadata]
    var document_internal_ids: List[Int]
    var documents: List[String]
    var storage: Optional[HNSWStorage]


def _write_collection_snapshot(
    path: String,
    fault_point: Int,
    name: String,
    dimension: Int,
    storage_kind: StorageKind,
    metric_type: MetricType,
    identity: Int,
    applied_sequence: Int,
    user_ids: List[Int],
    is_deleted: List[UInt8],
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
    document_by_internal: Dict[Int, Int],
    documents: List[String],
    storage: HNSWStorage,
) raises:
    """Writes one complete payload; the caller publishes it atomically."""
    var num_ids = len(user_ids)
    check_size_limit(num_ids, MAX_COLLECTION_RECORDS)
    if len(is_deleted) != num_ids:
        raise Error("Collection user IDs and deletion flags differ in size.")
    var file = open(path, "w")
    write_int(file, COLLECTION_MAGIC)
    write_int(file, COLLECTION_FORMAT)
    _write_string(file, name)
    write_int(file, dimension)
    write_int(file, storage_kind)
    write_int(file, metric_type)
    write_int(file, identity)
    write_int(file, applied_sequence)
    inject_snapshot_fault(fault_point, SNAPSHOT_FAULT_AFTER_HEADER)

    write_int(file, num_ids)
    if num_ids > 0:
        file.write_bytes(
            Span[UInt8](
                ptr=user_ids.unsafe_ptr().bitcast[UInt8](),
                length=num_ids * 8,
            )
        )
        file.write_bytes(
            Span[UInt8](ptr=is_deleted.unsafe_ptr(), length=num_ids)
        )

    write_int(file, len(metadatas))
    for internal_id in range(num_ids):
        if internal_id in metadata_by_internal:
            write_int(file, internal_id)
            _write_metadata(file, metadatas[metadata_by_internal[internal_id]])

    write_int(file, len(documents))
    for internal_id in range(num_ids):
        if internal_id in document_by_internal:
            write_int(file, internal_id)
            _write_string(file, documents[document_by_internal[internal_id]])

    _write_storage(file, storage, storage_kind)
    if Int(file.seek(0, SEEK_END)) > MAX_COLLECTION_SNAPSHOT_BYTES:
        raise Error("Collection snapshot exceeds the 256 GiB safety limit.")
    inject_snapshot_fault(fault_point, SNAPSHOT_FAULT_AFTER_PAYLOAD)
    file.close()


def _save_collection_snapshot(
    path: String,
    fault_point: Int,
    name: String,
    dimension: Int,
    storage_kind: StorageKind,
    metric_type: MetricType,
    identity: Int,
    applied_sequence: Int,
    user_ids: List[Int],
    is_deleted: List[UInt8],
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
    document_by_internal: Dict[Int, Int],
    documents: List[String],
    storage: HNSWStorage,
) raises:
    """Writes, checksums, and atomically publishes a collection snapshot."""
    var temporary_path = atomic_temporary_path(path)
    try:
        _write_collection_snapshot(
            temporary_path,
            fault_point,
            name,
            dimension,
            storage_kind,
            metric_type,
            identity,
            applied_sequence,
            user_ids,
            is_deleted,
            metadata_by_internal,
            metadatas,
            document_by_internal,
            documents,
            storage,
        )
        append_snapshot_checksum(temporary_path)
        inject_snapshot_fault(fault_point, SNAPSHOT_FAULT_BEFORE_PUBLISH)
        atomic_replace(temporary_path, path)
        inject_snapshot_fault(fault_point, SNAPSHOT_FAULT_AFTER_PUBLISH)
        sync_parent_directory(path)
    except error:
        # After a successful rename this path no longer exists, so the same
        # cleanup is safe for both pre- and post-publication failures.
        remove_file_best_effort(temporary_path)
        raise error^


def _read_collection_snapshot(
    path: String,
    memory_mapped: Bool,
    mmap_threshold_bytes: Int,
) raises -> CollectionSnapshot:
    """Reads and validates a snapshot without depending on Collection."""
    if mmap_threshold_bytes < 0:
        raise Error("mmap_threshold_bytes cannot be negative.")
    var file = open(path, "r")
    var total_size = Int(file.seek(0, SEEK_END))
    if (
        total_size < SNAPSHOT_CHECKSUM_TRAILER_BYTES
        or total_size
        > MAX_COLLECTION_SNAPSHOT_BYTES + SNAPSHOT_CHECKSUM_TRAILER_BYTES
    ):
        raise Error("Collection snapshot size is outside safety limits.")
    var file_size = validate_snapshot_checksum(file, total_size)
    _ = file.seek(0, SEEK_SET)
    if read_int(file) != COLLECTION_MAGIC:
        raise Error("Invalid Collection magic.")
    if read_int(file) != COLLECTION_FORMAT:
        raise Error("Unsupported Collection format version.")

    var name = _read_string(file)
    var dimension = read_int(file)
    var storage_kind: StorageKind = read_int(file)
    if storage_kind != STORAGE_FLAT and storage_kind != STORAGE_SQ8:
        raise Error("Invalid Collection storage kind.")
    var metric_type: MetricType = read_int(file)
    if (
        metric_type != METRIC_L2
        and metric_type != METRIC_COSINE
        and metric_type != METRIC_INNER_PRODUCT
    ):
        raise Error("Invalid Collection metric.")
    var identity = read_int(file)
    var applied_sequence = read_int(file)
    if identity <= 0 or applied_sequence < 0:
        raise Error("Invalid Collection durability state.")

    _validate_vector_dimension(dimension)
    var num_ids = read_int(file)
    check_size_limit(num_ids, MAX_COLLECTION_RECORDS)
    var record_bytes = checked_byte_count(num_ids, 9)
    if record_bytes > _remaining_snapshot_bytes(file, file_size):
        raise Error("Collection record count exceeds the snapshot size.")
    var user_ids = List[Int](capacity=num_ids)
    var is_deleted = List[UInt8](capacity=num_ids)
    if num_ids > 0:
        var ids_byte_count = checked_byte_count(num_ids, 8)
        var ids_data = file.read_bytes(ids_byte_count)
        if len(ids_data) != ids_byte_count:
            raise Error("Collection user ID array is truncated.")
        var ids_source = ids_data.unsafe_ptr().bitcast[Int]()
        var deleted_data = file.read_bytes(num_ids)
        if len(deleted_data) != num_ids:
            raise Error("Collection deletion array is truncated.")
        var deleted_source = deleted_data.unsafe_ptr()
        var active_ids = Dict[Int, Bool]()
        for index in range(num_ids):
            var user_id = ids_source[index]
            var deleted = deleted_source[index]
            if deleted > 1:
                raise Error("Collection deletion flags must be 0 or 1.")
            if deleted == 0:
                if user_id in active_ids:
                    raise Error("Active Collection user IDs must be unique.")
                active_ids[user_id] = True
            user_ids.append(user_id)
            is_deleted.append(deleted)
        _ = len(ids_data)
        _ = len(deleted_data)

    var metadata_count = read_int(file)
    _validate_sparse_entry_count(
        metadata_count,
        num_ids,
        _remaining_snapshot_bytes(file, file_size),
    )
    var metadata_internal_ids = List[Int](capacity=metadata_count)
    var metadatas = List[Metadata](capacity=metadata_count)
    var previous_metadata_id = -1
    for _ in range(metadata_count):
        var internal_id = read_int(file)
        if (
            internal_id < 0
            or internal_id >= num_ids
            or internal_id <= previous_metadata_id
        ):
            raise Error("Invalid metadata internal ID.")
        var metadata = _read_metadata(file)
        if metadata.count() == 0:
            raise Error("Sparse metadata entries cannot be empty.")
        previous_metadata_id = internal_id
        metadata_internal_ids.append(internal_id)
        metadatas.append(metadata^)

    var document_count = read_int(file)
    _validate_sparse_entry_count(
        document_count,
        num_ids,
        _remaining_snapshot_bytes(file, file_size),
    )
    var document_internal_ids = List[Int](capacity=document_count)
    var documents = List[String](capacity=document_count)
    var previous_document_id = -1
    for _ in range(document_count):
        var internal_id = read_int(file)
        if (
            internal_id < 0
            or internal_id >= num_ids
            or internal_id <= previous_document_id
        ):
            raise Error("Invalid document internal ID.")
        var document = _read_string(file)
        if document.byte_length() == 0:
            raise Error("Sparse document entries cannot be empty.")
        previous_document_id = internal_id
        document_internal_ids.append(internal_id)
        documents.append(document^)

    var storage = _read_storage(
        file,
        file_size,
        storage_kind,
        metric_type,
        dimension,
        num_ids,
        memory_mapped and file_size >= mmap_threshold_bytes,
    )
    if Int(file.seek(0, SEEK_CUR)) != file_size:
        raise Error("Collection snapshot contains trailing payload data.")
    file.close()
    return CollectionSnapshot(
        name^,
        dimension,
        storage_kind,
        metric_type,
        identity,
        applied_sequence,
        Optional[List[Int]](user_ids^),
        Optional[List[UInt8]](is_deleted^),
        metadata_internal_ids^,
        metadatas^,
        document_internal_ids^,
        documents^,
        Optional[HNSWStorage](storage^),
    )
