from std.io.file import FileHandle
from std.collections.span import Span
from std.os import SEEK_END, SEEK_SET

from mojovec.io.atomic_file import sync_file


comptime SNAPSHOT_CHECKSUM_MAGIC: UInt64 = 0x4D4F4A4F534E4150
comptime SNAPSHOT_CHECKSUM_SEED: UInt64 = 0xD6E8FEB86659FD93
comptime SNAPSHOT_CHECKSUM_TRAILER_BYTES = 16
comptime SNAPSHOT_CHECKSUM_CHUNK_BYTES = 8 * 1024 * 1024


@always_inline
def _rotate_left(value: UInt64, amount: Int) -> UInt64:
    return (value << UInt64(amount)) | (value >> UInt64(64 - amount))


def _checksum_update(
    checksum: UInt64,
    bytes: Span[UInt8, _],
) -> UInt64:
    """Updates the snapshot checksum from one aligned streaming chunk."""
    var result = checksum
    var pointer = bytes.unsafe_ptr()
    var words = len(bytes) // 8
    for index in range(words):
        var address = Int(pointer) + index * 8
        var word = Pointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=address
        )[unsafe_offset=0]
        result ^= word
        result = _rotate_left(result, 27) ^ _rotate_left(word, 17)
        result ^= result >> 23

    var tail: UInt64 = 0
    var tail_start = words * 8
    for index in range(tail_start, len(bytes)):
        tail |= UInt64(pointer[unsafe_offset=index]) << UInt64((index - tail_start) * 8)
    result ^= tail
    result ^= UInt64(len(bytes)) << 32
    return _rotate_left(result, 11)


def _checksum_prefix(mut file: FileHandle, byte_count: Int) raises -> UInt64:
    if byte_count < 0:
        raise Error("Snapshot checksum byte count cannot be negative.")
    _ = file.seek(0, SEEK_SET)
    var remaining = byte_count
    var checksum = SNAPSHOT_CHECKSUM_SEED
    while remaining > 0:
        var chunk_size = min(remaining, SNAPSHOT_CHECKSUM_CHUNK_BYTES)
        var chunk = file.read_bytes(chunk_size)
        if len(chunk) != chunk_size:
            raise Error("Unexpected end of snapshot while checksumming.")
        checksum = _checksum_update(checksum, chunk)
        remaining -= chunk_size
    return checksum


def _write_uint64(mut file: FileHandle, value: UInt64) raises:
    var storage = InlineArray[UInt64, 1](uninitialized=True)
    storage[0] = value
    file.write_all(
        Span[UInt8](
            unsafe_ptr=storage.unsafe_ptr().unsafe_bitcast[UInt8](),
            length=8,
        )
    )


def append_snapshot_checksum(path: String) raises:
    """Appends and synchronizes a checksum trailer to a complete payload."""
    var reader = open(path, "r")
    var payload_size = Int(reader.seek(0, SEEK_END))
    var checksum = _checksum_prefix(reader, payload_size)
    reader.close()

    var writer = open(path, "a")
    _write_uint64(writer, SNAPSHOT_CHECKSUM_MAGIC)
    _write_uint64(writer, checksum)
    sync_file(writer)
    writer.close()


def validate_snapshot_checksum(
    mut file: FileHandle,
    total_size: Int,
) raises -> Int:
    """Validates a trailer and returns the payload size excluding it."""
    if total_size < SNAPSHOT_CHECKSUM_TRAILER_BYTES:
        raise Error("Snapshot checksum trailer is missing.")
    var payload_size = total_size - SNAPSHOT_CHECKSUM_TRAILER_BYTES
    _ = file.seek(payload_size, SEEK_SET)
    var trailer = file.read_bytes(SNAPSHOT_CHECKSUM_TRAILER_BYTES)
    if len(trailer) != SNAPSHOT_CHECKSUM_TRAILER_BYTES:
        raise Error("Snapshot checksum trailer is truncated.")
    var values = trailer.unsafe_ptr().unsafe_bitcast[UInt64]()
    var magic = values[unsafe_offset=0]
    var expected = values[unsafe_offset=1]
    _ = len(trailer)
    if magic != SNAPSHOT_CHECKSUM_MAGIC:
        raise Error("Invalid snapshot checksum trailer.")

    var actual = _checksum_prefix(file, payload_size)
    if actual != expected:
        raise Error("Snapshot checksum mismatch.")
    return payload_size
