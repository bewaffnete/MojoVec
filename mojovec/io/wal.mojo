from std.collections import List, Optional
from std.ffi import external_call
from std.io.file import FileHandle
from std.memory.span import Span
from std.os import SEEK_CUR, SEEK_END, SEEK_SET
from std.os.path import exists, getsize

from mojovec.api.metadata import (
    METADATA_BOOL,
    METADATA_FLOAT,
    METADATA_INT,
    METADATA_STRING,
    Metadata,
)
from mojovec.io.atomic_file import (
    atomic_replace,
    atomic_temporary_path,
    sync_file,
    sync_parent_directory,
)


comptime WalDurability = Int
comptime WAL_ASYNC: Int = 1
comptime WAL_SYNC: Int = 2

comptime WAL_FILE_MAGIC = 0x4D4F4A4F57414C31
comptime WAL_HEADER_COMMIT = 0x4D4F4A4F57414C48
comptime WAL_FRAME_MAGIC = 0x4D4F4A4F57414C46
comptime WAL_FRAME_COMMIT = 0x4D4F4A4F57414C43
comptime WAL_OPERATION_WRITE = 1
comptime WAL_OPERATION_DELETE = 2
comptime WAL_FRAME_HEADER_BYTES = 8 * 8
comptime WAL_FRAME_TRAILER_BYTES = 2 * 8
comptime WAL_MIN_FRAME_BYTES = (
    WAL_FRAME_HEADER_BYTES + WAL_FRAME_TRAILER_BYTES
)
comptime WAL_CHECKSUM_SEED: UInt64 = 0x9E3779B97F4A7C15


def _rotate_left(value: UInt64, amount: Int) -> UInt64:
    return (value << UInt64(amount)) | (value >> UInt64(64 - amount))


def _checksum_update(checksum: UInt64, bytes: Span[UInt8, _]) -> UInt64:
    """Mixes aligned 64-bit chunks without copying the payload."""
    var result = checksum
    var pointer = bytes.unsafe_ptr()
    var words = len(bytes) // 8
    for index in range(words):
        var address = Int(pointer) + index * 8
        var word = UnsafePointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=address
        )[0]
        result ^= word
        result = _rotate_left(result, 27) ^ _rotate_left(word, 17)
        result ^= result >> 23

    var tail: UInt64 = 0
    var tail_start = words * 8
    for index in range(tail_start, len(bytes)):
        tail |= UInt64(pointer[index]) << UInt64((index - tail_start) * 8)
    result ^= tail
    result ^= UInt64(len(bytes)) << 32
    return _rotate_left(result, 11)


def _append_int(mut destination: List[UInt8], value: Int):
    var storage = InlineArray[Int, 1](uninitialized=True)
    storage[0] = value
    var bytes = storage.unsafe_ptr().bitcast[UInt8]()
    for index in range(8):
        destination.append(bytes[index])


def _append_uint64(mut destination: List[UInt8], value: UInt64):
    var storage = InlineArray[UInt64, 1](uninitialized=True)
    storage[0] = value
    var bytes = storage.unsafe_ptr().bitcast[UInt8]()
    for index in range(8):
        destination.append(bytes[index])


def _decode_int(data: List[UInt8], byte_offset: Int) -> Int:
    return (data.unsafe_ptr() + byte_offset).bitcast[Int]()[0]


def _write_and_hash(
    mut file: FileHandle,
    bytes: Span[UInt8, _],
    checksum: UInt64,
) raises -> UInt64:
    file.write_all(bytes)
    return _checksum_update(checksum, bytes)


def _write_int_and_hash(
    mut file: FileHandle, value: Int, checksum: UInt64
) raises -> UInt64:
    var bytes = List[UInt8](capacity=8)
    _append_int(bytes, value)
    return _write_and_hash(file, bytes, checksum)


def _read_exact(mut file: FileHandle, count: Int) raises -> List[UInt8]:
    if count < 0:
        raise Error("Negative WAL field size.")
    var bytes = file.read_bytes(count)
    if len(bytes) != count:
        raise Error("Unexpected end of WAL record.")
    return bytes^


def _read_int_and_hash(
    mut file: FileHandle, mut checksum: UInt64
) raises -> Int:
    var bytes = _read_exact(file, 8)
    checksum = _checksum_update(checksum, bytes)
    return _decode_int(bytes, 0)


def _write_string_and_hash(
    mut file: FileHandle,
    value: String,
    checksum: UInt64,
) raises -> UInt64:
    var result = _write_int_and_hash(file, value.byte_length(), checksum)
    if value.byte_length() > 0:
        result = _write_and_hash(file, value.as_bytes(), result)
    return result


def _read_string_and_hash(
    mut file: FileHandle, mut checksum: UInt64
) raises -> String:
    var size = _read_int_and_hash(file, checksum)
    if size < 0 or size > 1_073_741_824:
        raise Error("Invalid WAL string size.")
    var bytes = _read_exact(file, size)
    checksum = _checksum_update(checksum, bytes)
    return String(from_utf8=bytes)


def _metadata_encoded_size(metadata: Metadata) raises -> Int:
    var size = 8
    for field in range(metadata.count()):
        var key = metadata._key_at(field)
        var value = metadata._value_at(field)
        size += 8 + key.byte_length() + 8
        if value.kind() == METADATA_STRING:
            var string_value = value.as_string()
            size += 8 + string_value.byte_length()
        elif value.kind() == METADATA_INT:
            size += 8
        elif value.kind() == METADATA_FLOAT:
            size += 8
        elif value.kind() == METADATA_BOOL:
            size += 1
        else:
            raise Error("Unsupported metadata kind in WAL.")
    return size


def _write_metadata_and_hash(
    mut file: FileHandle,
    metadata: Metadata,
    checksum: UInt64,
) raises -> UInt64:
    var result = _write_int_and_hash(file, metadata.count(), checksum)
    for field in range(metadata.count()):
        var key = metadata._key_at(field)
        var value = metadata._value_at(field)
        result = _write_string_and_hash(file, key, result)
        result = _write_int_and_hash(file, value.kind(), result)
        if value.kind() == METADATA_STRING:
            result = _write_string_and_hash(file, value.as_string(), result)
        elif value.kind() == METADATA_INT:
            result = _write_int_and_hash(file, value.as_int(), result)
        elif value.kind() == METADATA_FLOAT:
            var storage = InlineArray[Float64, 1](uninitialized=True)
            storage[0] = value.as_float()
            var bytes = Span[UInt8](
                ptr=storage.unsafe_ptr().bitcast[UInt8](), length=8
            )
            result = _write_and_hash(file, bytes, result)
        elif value.kind() == METADATA_BOOL:
            var storage = InlineArray[UInt8, 1](uninitialized=True)
            storage[0] = 1 if value.as_bool() else 0
            result = _write_and_hash(
                file,
                Span[UInt8](ptr=storage.unsafe_ptr(), length=1),
                result,
            )
        else:
            raise Error("Unsupported metadata kind in WAL.")
    return result


def _read_metadata_and_hash(
    mut file: FileHandle, mut checksum: UInt64
) raises -> Metadata:
    var count = _read_int_and_hash(file, checksum)
    if count < 0 or count > 1_000_000:
        raise Error("Invalid WAL metadata field count.")
    var metadata = Metadata()
    for _ in range(count):
        var key = _read_string_and_hash(file, checksum)
        var kind = _read_int_and_hash(file, checksum)
        if kind == METADATA_STRING:
            metadata.set(key, _read_string_and_hash(file, checksum))
        elif kind == METADATA_INT:
            metadata.set(key, _read_int_and_hash(file, checksum))
        elif kind == METADATA_FLOAT:
            var bytes = _read_exact(file, 8)
            checksum = _checksum_update(checksum, bytes)
            metadata.set(key, bytes.unsafe_ptr().bitcast[Float64]()[0])
        elif kind == METADATA_BOOL:
            var bytes = _read_exact(file, 1)
            checksum = _checksum_update(checksum, bytes)
            if bytes[0] > 1:
                raise Error("Invalid WAL Bool metadata value.")
            metadata.set(key, bytes[0] == 1)
        else:
            raise Error("Invalid WAL metadata kind.")
    return metadata^


def _write_file_header(
    path: String,
    dimension: Int,
    storage_kind: Int,
    identity: Int,
    base_sequence: Int,
    name: String,
) raises:
    var file = open(path, "w")
    var header = List[UInt8](capacity=7 * 8 + name.byte_length())
    _append_int(header, WAL_FILE_MAGIC)
    _append_int(header, dimension)
    _append_int(header, storage_kind)
    _append_int(header, identity)
    _append_int(header, base_sequence)
    _append_int(header, name.byte_length())
    for byte in name.as_bytes():
        header.append(byte)
    _append_int(header, WAL_HEADER_COMMIT)
    file.write_all(header)
    sync_file(file)
    file.close()


struct WalHeader(Movable):
    var dimension: Int
    var storage_kind: Int
    var identity: Int
    var base_sequence: Int
    var name: String
    var byte_size: Int

    def __init__(
        out self,
        dimension: Int,
        storage_kind: Int,
        identity: Int,
        base_sequence: Int,
        name: String,
        byte_size: Int,
    ):
        self.dimension = dimension
        self.storage_kind = storage_kind
        self.identity = identity
        self.base_sequence = base_sequence
        self.name = name.copy()
        self.byte_size = byte_size


def read_wal_header(mut file: FileHandle) raises -> WalHeader:
    var fixed = _read_exact(file, 6 * 8)
    if _decode_int(fixed, 0) != WAL_FILE_MAGIC:
        raise Error("Invalid WAL file magic.")
    var dimension = _decode_int(fixed, 8)
    var storage_kind = _decode_int(fixed, 16)
    var identity = _decode_int(fixed, 24)
    var base_sequence = _decode_int(fixed, 32)
    var name_size = _decode_int(fixed, 40)
    if dimension <= 0 or dimension > 65_536:
        raise Error("Invalid WAL dimension.")
    if identity <= 0 or base_sequence < 0:
        raise Error("Invalid WAL durability state.")
    if name_size < 0 or name_size > 1_048_576:
        raise Error("Invalid WAL collection name size.")
    var name_bytes = _read_exact(file, name_size)
    var commit = _read_exact(file, 8)
    if _decode_int(commit, 0) != WAL_HEADER_COMMIT:
        raise Error("Incomplete WAL file header.")
    return WalHeader(
        dimension,
        storage_kind,
        identity,
        base_sequence,
        String(from_utf8=name_bytes),
        7 * 8 + name_size,
    )


struct WalRecord(Movable):
    var sequence: Int
    var operation: Int
    var replace_existing: Bool
    var has_metadatas: Bool
    var has_documents: Bool
    var ids: List[Int]
    var embeddings: List[Float32]
    var metadatas: List[Metadata]
    var documents: List[String]

    def __init__(
        out self,
        sequence: Int,
        operation: Int,
        replace_existing: Bool,
        has_metadatas: Bool,
        has_documents: Bool,
        var ids: List[Int],
        var embeddings: List[Float32],
        var metadatas: List[Metadata],
        var documents: List[String],
    ):
        self.sequence = sequence
        self.operation = operation
        self.replace_existing = replace_existing
        self.has_metadatas = has_metadatas
        self.has_documents = has_documents
        self.ids = ids^
        self.embeddings = embeddings^
        self.metadatas = metadatas^
        self.documents = documents^


struct WalReader(Movable):
    var file: FileHandle
    var file_size: Int
    var header: WalHeader
    var finished: Bool
    var valid_bytes: Int

    def __init__(
        out self,
        var file: FileHandle,
        file_size: Int,
        var header: WalHeader,
    ):
        self.file = file^
        self.file_size = file_size
        self.header = header^
        self.finished = False
        self.valid_bytes = self.header.byte_size

    @staticmethod
    def open(path: String) raises -> Self:
        var file = open(path, "r")
        var file_size = Int(file.seek(0, SEEK_END))
        _ = file.seek(0, SEEK_SET)
        var header = read_wal_header(file)
        return Self(file^, file_size, header^)

    def next(mut self) raises -> Optional[WalRecord]:
        if self.finished:
            return None
        var frame_start = Int(self.file.seek(0, SEEK_CUR))
        var remaining = self.file_size - frame_start
        if remaining == 0:
            self.finished = True
            return None
        if remaining < WAL_FRAME_HEADER_BYTES:
            self.finished = True
            return None

        var header_bytes = _read_exact(self.file, WAL_FRAME_HEADER_BYTES)
        var frame_magic = _decode_int(header_bytes, 0)
        var frame_size = _decode_int(header_bytes, 8)
        var sequence = _decode_int(header_bytes, 16)
        var operation = _decode_int(header_bytes, 24)
        var replace_existing = _decode_int(header_bytes, 32)
        var flags = _decode_int(header_bytes, 40)
        var id_count = _decode_int(header_bytes, 48)
        var embedding_count = _decode_int(header_bytes, 56)

        if frame_magic != WAL_FRAME_MAGIC:
            raise Error("Invalid WAL frame magic.")
        if frame_size < WAL_MIN_FRAME_BYTES:
            raise Error("Invalid WAL frame size.")
        if frame_size > remaining:
            self.finished = True
            return None
        if sequence <= 0:
            raise Error("Invalid WAL sequence.")
        if (
            operation != WAL_OPERATION_WRITE
            and operation != WAL_OPERATION_DELETE
        ):
            raise Error("Invalid WAL operation.")
        if replace_existing != 0 and replace_existing != 1:
            raise Error("Invalid WAL replace flag.")
        if flags < 0 or flags > 3:
            raise Error("Invalid WAL payload flags.")
        if id_count < 0 or id_count > 1_000_000_000:
            raise Error("Invalid WAL ID count.")
        if embedding_count < 0 or embedding_count > 2_000_000_000:
            raise Error("Invalid WAL embedding count.")
        if operation == WAL_OPERATION_WRITE:
            if embedding_count != id_count * self.header.dimension:
                raise Error("WAL embedding shape does not match IDs.")
        elif embedding_count != 0 or flags != 0:
            raise Error("Delete WAL record contains unexpected payloads.")
        var fixed_payload_size = id_count * 8 + embedding_count * 4
        if fixed_payload_size > frame_size - WAL_MIN_FRAME_BYTES:
            raise Error("WAL payload sizes exceed the frame boundary.")

        var checksum = _checksum_update(WAL_CHECKSUM_SEED, header_bytes)
        var ids = List[Int](unsafe_uninit_length=id_count)
        if id_count > 0:
            var id_bytes = Span[UInt8](
                ptr=ids.unsafe_ptr().bitcast[UInt8](),
                length=id_count * 8,
            )
            if self.file.read(id_bytes) != len(id_bytes):
                raise Error("Unexpected end of WAL IDs.")
            checksum = _checksum_update(checksum, id_bytes)

        var embeddings = List[Float32](unsafe_uninit_length=embedding_count)
        if embedding_count > 0:
            var embedding_bytes = Span[UInt8](
                ptr=embeddings.unsafe_ptr().bitcast[UInt8](),
                length=embedding_count * 4,
            )
            if self.file.read(embedding_bytes) != len(embedding_bytes):
                raise Error("Unexpected end of WAL embeddings.")
            checksum = _checksum_update(checksum, embedding_bytes)

        var has_metadatas = (flags & 1) != 0
        var has_documents = (flags & 2) != 0
        var metadatas = List[Metadata](capacity=id_count)
        if has_metadatas:
            for _ in range(id_count):
                metadatas.append(_read_metadata_and_hash(self.file, checksum))

        var documents = List[String](capacity=id_count)
        if has_documents:
            for _ in range(id_count):
                documents.append(_read_string_and_hash(self.file, checksum))

        var expected_trailer = frame_start + frame_size - 16
        if Int(self.file.seek(0, SEEK_CUR)) != expected_trailer:
            raise Error("WAL frame length does not match its payload.")
        var trailer = _read_exact(self.file, WAL_FRAME_TRAILER_BYTES)
        var stored_checksum = trailer.unsafe_ptr().bitcast[UInt64]()[0]
        var commit = _decode_int(trailer, 8)
        if commit != WAL_FRAME_COMMIT:
            if frame_start + frame_size == self.file_size:
                self.finished = True
                return None
            raise Error("Corrupted committed WAL frame.")
        if stored_checksum != checksum:
            raise Error("WAL frame checksum mismatch.")
        self.valid_bytes = frame_start + frame_size

        return WalRecord(
            sequence,
            operation,
            replace_existing == 1,
            has_metadatas,
            has_documents,
            ids^,
            embeddings^,
            metadatas^,
            documents^,
        )


def _metadata_payload_size(
    metadatas: List[Metadata], has_metadatas: Bool
) raises -> Int:
    if not has_metadatas:
        return 0
    var size = 0
    for metadata in metadatas:
        size += _metadata_encoded_size(metadata)
    return size


def _document_payload_size(documents: List[String], has_documents: Bool) -> Int:
    if not has_documents:
        return 0
    var size = 0
    for document in documents:
        size += 8 + document.byte_length()
    return size


struct WriteAheadLog(Movable):
    var path: String
    var file: FileHandle
    var dimension: Int
    var storage_kind: Int
    var identity: Int
    var name: String
    var durability: WalDurability
    var next_sequence: Int
    var usable: Bool

    def __init__(
        out self,
        path: String,
        var file: FileHandle,
        dimension: Int,
        storage_kind: Int,
        identity: Int,
        name: String,
        durability: WalDurability,
        next_sequence: Int,
    ):
        self.path = path.copy()
        self.file = file^
        self.dimension = dimension
        self.storage_kind = storage_kind
        self.identity = identity
        self.name = name.copy()
        self.durability = durability
        self.next_sequence = next_sequence
        self.usable = True

    def __init__(out self, *, deinit move: Self):
        self.path = move.path^
        self.file = move.file^
        self.dimension = move.dimension
        self.storage_kind = move.storage_kind
        self.identity = move.identity
        self.name = move.name^
        self.durability = move.durability
        self.next_sequence = move.next_sequence
        self.usable = move.usable

    @staticmethod
    def create(
        path: String,
        dimension: Int,
        storage_kind: Int,
        identity: Int,
        name: String,
        base_sequence: Int,
        durability: WalDurability,
    ) raises -> Self:
        if durability != WAL_ASYNC and durability != WAL_SYNC:
            raise Error("Unsupported WAL durability mode.")
        if exists(path) and getsize(path) > 0:
            raise Error(
                "WAL already contains data; recover it instead of replacing it."
            )
        _write_file_header(
            path,
            dimension,
            storage_kind,
            identity,
            base_sequence,
            name,
        )
        sync_parent_directory(path)
        var file = open(path, "a")
        return Self(
            path,
            file^,
            dimension,
            storage_kind,
            identity,
            name,
            durability,
            base_sequence + 1,
        )

    @staticmethod
    def resume(
        path: String,
        dimension: Int,
        storage_kind: Int,
        identity: Int,
        name: String,
        durability: WalDurability,
        next_sequence: Int,
        valid_bytes: Int,
    ) raises -> Self:
        if durability != WAL_ASYNC and durability != WAL_SYNC:
            raise Error("Unsupported WAL durability mode.")
        var file = open(path, "a")
        if external_call["ftruncate", Int](file.handle, valid_bytes) != 0:
            raise Error("Cannot discard an incomplete WAL tail.")
        sync_file(file)
        return Self(
            path,
            file^,
            dimension,
            storage_kind,
            identity,
            name,
            durability,
            next_sequence,
        )

    def _append(
        mut self,
        operation: Int,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
        replace_existing: Bool,
        metadatas: List[Metadata],
        has_metadatas: Bool,
        documents: List[String],
        has_documents: Bool,
    ) raises -> Int:
        if has_metadatas and len(metadatas) != len(ids):
            raise Error("WAL metadata count must equal ID count.")
        if has_documents and len(documents) != len(ids):
            raise Error("WAL document count must equal ID count.")
        var metadata_size = _metadata_payload_size(metadatas, has_metadatas)
        var document_size = _document_payload_size(documents, has_documents)
        if not self.usable:
            raise Error(
                "WAL append previously failed; recover or checkpoint before retrying."
            )
        var frame_size = (
            WAL_FRAME_HEADER_BYTES
            + len(ids) * 8
            + len(embeddings) * 4
            + metadata_size
            + document_size
            + WAL_FRAME_TRAILER_BYTES
        )
        var sequence = self.next_sequence
        var flags = (1 if has_metadatas else 0) | (2 if has_documents else 0)
        var header = List[UInt8](capacity=WAL_FRAME_HEADER_BYTES)
        _append_int(header, WAL_FRAME_MAGIC)
        _append_int(header, frame_size)
        _append_int(header, sequence)
        _append_int(header, operation)
        _append_int(header, 1 if replace_existing else 0)
        _append_int(header, flags)
        _append_int(header, len(ids))
        _append_int(header, len(embeddings))

        var frame_start = getsize(self.path)
        self.usable = False
        var append_failed = False
        try:
            var checksum = _write_and_hash(
                self.file, header, WAL_CHECKSUM_SEED
            )
            if len(ids) > 0:
                checksum = _write_and_hash(
                    self.file,
                    Span[UInt8](
                        ptr=ids.unsafe_ptr().bitcast[UInt8](),
                        length=len(ids) * 8,
                    ),
                    checksum,
                )
            if len(embeddings) > 0:
                checksum = _write_and_hash(
                    self.file,
                    Span[UInt8](
                        ptr=embeddings.unsafe_ptr().bitcast[UInt8](),
                        length=len(embeddings) * 4,
                    ),
                    checksum,
                )
            if has_metadatas:
                for metadata in metadatas:
                    checksum = _write_metadata_and_hash(
                        self.file, metadata, checksum
                    )
            if has_documents:
                for document in documents:
                    checksum = _write_string_and_hash(
                        self.file, document, checksum
                    )

            var trailer = List[UInt8](capacity=WAL_FRAME_TRAILER_BYTES)
            _append_uint64(trailer, checksum)
            _append_int(trailer, WAL_FRAME_COMMIT)
            self.file.write_all(trailer)
            if self.durability == WAL_SYNC:
                sync_file(self.file)
        except:
            append_failed = True

        if append_failed:
            self.rollback_last_append(frame_start, sequence)
            raise Error("WAL append failed; incomplete frame was rolled back.")

        self.next_sequence = sequence + 1
        self.usable = True
        return sequence

    def append_write(
        mut self,
        ids: Span[Int, _],
        embeddings: Span[Float32, _],
        replace_existing: Bool,
        metadatas: List[Metadata],
        has_metadatas: Bool,
        documents: List[String],
        has_documents: Bool,
    ) raises -> Int:
        return self._append(
            WAL_OPERATION_WRITE,
            ids,
            embeddings,
            replace_existing,
            metadatas,
            has_metadatas,
            documents,
            has_documents,
        )

    def append_delete(mut self, ids: Span[Int, _]) raises -> Int:
        return self._append(
            WAL_OPERATION_DELETE,
            ids,
            Span[Float32, MutUntrackedOrigin](),
            False,
            List[Metadata](),
            False,
            List[String](),
            False,
        )

    def byte_size(self) raises -> Int:
        """Returns the current append boundary for one transactional frame."""
        return getsize(self.path)

    def rollback_last_append(
        mut self,
        byte_size: Int,
        sequence: Int,
    ) raises:
        """Removes a completed frame that has not been applied in memory."""
        self.usable = False
        if external_call["ftruncate", Int](self.file.handle, byte_size) != 0:
            raise Error("Cannot roll back the last WAL frame.")
        _ = self.file.seek(0, SEEK_END)
        sync_file(self.file)
        self.next_sequence = sequence
        self.usable = True

    def flush(mut self) raises:
        sync_file(self.file)

    def reset(mut self, base_sequence: Int) raises:
        var temporary_path = atomic_temporary_path(self.path)
        _write_file_header(
            temporary_path,
            self.dimension,
            self.storage_kind,
            self.identity,
            base_sequence,
            self.name,
        )
        var replacement = open(temporary_path, "a")
        atomic_replace(temporary_path, self.path)
        self.file.close()
        self.file = replacement^
        self.next_sequence = base_sequence + 1
        self.usable = True
        sync_parent_directory(self.path)
