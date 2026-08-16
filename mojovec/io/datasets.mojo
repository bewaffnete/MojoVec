"""Native readers for common vector-dataset interchange formats.

The readers return managed Lists that can be passed directly to Collection.add
or Collection.upsert. They intentionally do not depend on Python. Ecosystem
formats that require a substantial runtime (Hugging Face, Parquet, Arrow and
compressed NPZ) are provided by the Python package instead.
"""

from std.collections import List
from std.os import SEEK_END, SEEK_SET


comptime _MAX_DATASET_BYTES = 1_099_511_627_776  # 1 TiB
comptime _MAX_DATASET_ROWS = 100_000_000
comptime _MAX_VECTOR_DIMENSION = 65_536


@fieldwise_init
struct VectorDataset(Movable):
    """A dense float32 matrix and the user IDs assigned to its rows."""

    var ids: List[Int]
    var embeddings: List[Float32]
    var dimension: Int

    def count(self) -> Int:
        return len(self.ids)


@fieldwise_init
struct IntVectorDataset(Movable):
    """A dense int32 matrix, normally used for ivecs ground-truth IDs."""

    var values: List[Int]
    var dimension: Int

    def count(self) -> Int:
        if self.dimension == 0:
            return 0
        return len(self.values) // self.dimension


def _read_dataset_file(path: String) raises -> List[UInt8]:
    var file = open(path, "r")
    var size = Int(file.seek(0, SEEK_END))
    if size < 0 or size > _MAX_DATASET_BYTES:
        file.close()
        raise Error("Dataset file exceeds the supported size limit.")
    _ = file.seek(0, SEEK_SET)
    var data = file.read_bytes(size)
    file.close()
    if len(data) != size:
        raise Error("Unexpected end of dataset file.")
    return data^


@always_inline
def _read_i32(data: List[UInt8], offset: Int) raises -> Int:
    if offset < 0 or offset % 4 != 0 or offset > len(data) - 4:
        raise Error("Unexpected end of dataset file.")
    return Int(data.unsafe_ptr().unsafe_offset(offset).unsafe_bitcast[Int32]()[unsafe_offset=0])


@always_inline
def _read_f32(data: List[UInt8], offset: Int) raises -> Float32:
    if offset < 0 or offset % 4 != 0 or offset > len(data) - 4:
        raise Error("Unexpected end of dataset file.")
    return data.unsafe_ptr().unsafe_offset(offset).unsafe_bitcast[Float32]()[unsafe_offset=0]


def _validate_binary_shape(
    byte_count: Int,
    dimension: Int,
    max_rows: Int,
) raises -> Int:
    if dimension <= 0 or dimension > _MAX_VECTOR_DIMENSION:
        raise Error("Dataset vector dimension is outside the supported range.")
    if max_rows <= 0 or max_rows > _MAX_DATASET_ROWS:
        raise Error("max_rows is outside the supported range.")
    if dimension > (Int.MAX // 4) - 1:
        raise Error("Dataset record size overflows Int.")
    var record_bytes = (dimension + 1) * 4
    if byte_count % record_bytes != 0:
        raise Error("Dataset file ends in a partial vector record.")
    var rows = byte_count // record_bytes
    if rows > max_rows:
        raise Error("Dataset row count exceeds max_rows.")
    return rows


def read_fvecs(
    path: String,
    *,
    id_start: Int = 0,
    max_rows: Int = _MAX_DATASET_ROWS,
) raises -> VectorDataset:
    """Reads the little-endian fvecs format into managed Mojo Lists."""

    var data = _read_dataset_file(path)
    if len(data) == 0:
        return VectorDataset(
            ids=List[Int](),
            embeddings=List[Float32](),
            dimension=0,
        )

    var dimension = _read_i32(data, 0)
    var rows = _validate_binary_shape(len(data), dimension, max_rows)
    if rows > 0 and id_start > Int.MAX - (rows - 1):
        raise Error("Generated dataset IDs overflow Int.")

    var ids = List[Int](capacity=rows)
    var embeddings = List[Float32](capacity=rows * dimension)
    var record_bytes = (dimension + 1) * 4
    for row in range(rows):
        var offset = row * record_bytes
        if _read_i32(data, offset) != dimension:
            raise Error("fvecs records have inconsistent dimensions.")
        ids.append(id_start + row)
        for column in range(dimension):
            embeddings.append(_read_f32(data, offset + 4 + column * 4))
    _ = len(data)
    return VectorDataset(
        ids=ids^,
        embeddings=embeddings^,
        dimension=dimension,
    )


def read_ivecs(
    path: String,
    *,
    max_rows: Int = _MAX_DATASET_ROWS,
) raises -> IntVectorDataset:
    """Reads the little-endian ivecs format into a managed integer matrix."""

    var data = _read_dataset_file(path)
    if len(data) == 0:
        return IntVectorDataset(values=List[Int](), dimension=0)

    var dimension = _read_i32(data, 0)
    var rows = _validate_binary_shape(len(data), dimension, max_rows)
    var values = List[Int](capacity=rows * dimension)
    var record_bytes = (dimension + 1) * 4
    for row in range(rows):
        var offset = row * record_bytes
        if _read_i32(data, offset) != dimension:
            raise Error("ivecs records have inconsistent dimensions.")
        for column in range(dimension):
            values.append(_read_i32(data, offset + 4 + column * 4))
    _ = len(data)
    return IntVectorDataset(values=values^, dimension=dimension)


@always_inline
def _matches_ascii(
    data: List[UInt8],
    offset: Int,
    end: Int,
    pattern: String,
) -> Bool:
    var pattern_bytes = pattern.as_bytes()
    if offset < 0 or offset + len(pattern_bytes) > end:
        return False
    for index in range(len(pattern_bytes)):
        if data[offset + index] != pattern_bytes[index]:
            return False
    return True


def _find_ascii(
    data: List[UInt8],
    start: Int,
    end: Int,
    pattern: String,
) -> Int:
    var pattern_size = pattern.byte_length()
    if pattern_size == 0 or start < 0 or end > len(data):
        return -1
    for offset in range(start, end - pattern_size + 1):
        if _matches_ascii(data, offset, end, pattern):
            return offset
    return -1


@always_inline
def _is_ascii_space(value: UInt8) -> Bool:
    return value == 32 or value == 9 or value == 10 or value == 13


def _parse_decimal(
    data: List[UInt8], start: Int, end: Int
) raises -> Tuple[Int, Int]:
    var offset = start
    while offset < end and _is_ascii_space(data[offset]):
        offset += 1
    var value = 0
    var digits = 0
    while offset < end and data[offset] >= 48 and data[offset] <= 57:
        var digit = Int(data[offset] - 48)
        if value > (Int.MAX - digit) // 10:
            raise Error("NPY shape value overflows Int.")
        value = value * 10 + digit
        digits += 1
        offset += 1
    if digits == 0:
        raise Error("Invalid NPY shape.")
    return (value, offset)


def _parse_npy_shape(
    data: List[UInt8],
    header_start: Int,
    header_end: Int,
) raises -> Tuple[Int, Int]:
    var shape_key = _find_ascii(data, header_start, header_end, "shape")
    if shape_key < 0:
        raise Error("NPY header has no shape.")
    var open_paren = shape_key
    while open_paren < header_end and data[open_paren] != 40:
        open_paren += 1
    if open_paren == header_end:
        raise Error("Invalid NPY shape.")
    var first = _parse_decimal(data, open_paren + 1, header_end)
    var offset = first[1]
    while offset < header_end and _is_ascii_space(data[offset]):
        offset += 1
    if offset == header_end or data[offset] != 44:
        raise Error("NPY input must be a two-dimensional array.")
    var second = _parse_decimal(data, offset + 1, header_end)
    offset = second[1]
    while offset < header_end and _is_ascii_space(data[offset]):
        offset += 1
    if offset == header_end or data[offset] != 41:
        raise Error("NPY input must be a two-dimensional array.")
    return (first[0], second[0])


def read_npy_float32(
    path: String,
    *,
    id_start: Int = 0,
    max_rows: Int = _MAX_DATASET_ROWS,
) raises -> VectorDataset:
    """Reads a C-contiguous, little-endian, two-dimensional float32 NPY file."""

    var data = _read_dataset_file(path)
    if len(data) < 10:
        raise Error("NPY file is too short.")
    if (
        data[0] != 0x93
        or data[1] != 78
        or data[2] != 85
        or data[3] != 77
        or data[4] != 80
        or data[5] != 89
    ):
        raise Error("Invalid NPY magic.")

    var major = data[6]
    var header_prefix: Int
    var header_size: Int
    if major == 1:
        header_prefix = 10
        header_size = Int(data[8]) | (Int(data[9]) << 8)
    elif major == 2 or major == 3:
        if len(data) < 12:
            raise Error("NPY file is too short.")
        header_prefix = 12
        header_size = (
            Int(data[8])
            | (Int(data[9]) << 8)
            | (Int(data[10]) << 16)
            | (Int(data[11]) << 24)
        )
    else:
        raise Error("Unsupported NPY format version.")

    if header_size < 0 or header_prefix > len(data) - header_size:
        raise Error("Invalid NPY header length.")
    var header_end = header_prefix + header_size
    var has_little_f32 = (
        _find_ascii(data, header_prefix, header_end, "'<f4'") >= 0
        or _find_ascii(data, header_prefix, header_end, '"<f4"') >= 0
        or _find_ascii(data, header_prefix, header_end, "'|f4'") >= 0
        or _find_ascii(data, header_prefix, header_end, '"|f4"') >= 0
    )
    if not has_little_f32:
        raise Error("NPY dtype must be little-endian float32.")
    if (
        _find_ascii(data, header_prefix, header_end, "fortran_order") < 0
        or _find_ascii(data, header_prefix, header_end, "False") < 0
    ):
        raise Error("NPY array must be C-contiguous.")

    var shape = _parse_npy_shape(data, header_prefix, header_end)
    var rows = shape[0]
    var dimension = shape[1]
    if (
        rows < 0
        or rows > max_rows
        or max_rows <= 0
        or max_rows > _MAX_DATASET_ROWS
    ):
        raise Error("NPY row count exceeds max_rows.")
    if dimension <= 0 or dimension > _MAX_VECTOR_DIMENSION:
        raise Error("NPY vector dimension is outside the supported range.")
    if rows > 0 and dimension > Int.MAX // rows:
        raise Error("NPY element count overflows Int.")
    var count = rows * dimension
    if header_end % 4 != 0:
        raise Error("NPY payload is not correctly aligned.")
    if count > (len(data) - header_end) // 4 or header_end + count * 4 != len(
        data
    ):
        raise Error("NPY payload size does not match its shape.")
    if rows > 0 and id_start > Int.MAX - (rows - 1):
        raise Error("Generated dataset IDs overflow Int.")

    var ids = List[Int](capacity=rows)
    var embeddings = List[Float32](capacity=count)
    for row in range(rows):
        ids.append(id_start + row)
    for index in range(count):
        embeddings.append(_read_f32(data, header_end + index * 4))
    _ = len(data)
    return VectorDataset(
        ids=ids^,
        embeddings=embeddings^,
        dimension=dimension,
    )


def _read_delimited_vectors(
    path: String,
    dimension: Int,
    delimiter: String,
    has_header: Bool,
    id_start: Int,
    max_rows: Int,
) raises -> VectorDataset:
    if dimension <= 0 or dimension > _MAX_VECTOR_DIMENSION:
        raise Error("Dataset vector dimension is outside the supported range.")
    if delimiter.byte_length() != 1:
        raise Error("Delimiter must be exactly one byte.")
    if max_rows <= 0 or max_rows > _MAX_DATASET_ROWS:
        raise Error("max_rows is outside the supported range.")

    var bytes = _read_dataset_file(path)
    var contents = String(from_utf8=bytes)
    var ids = List[Int]()
    var embeddings = List[Float32]()
    var line_index = 0
    for line_slice in contents.split("\n"):
        if has_header and line_index == 0:
            line_index += 1
            continue
        line_index += 1
        var line = String(line_slice)
        if line.byte_length() == 0 or line == "\r":
            continue
        if len(ids) >= max_rows:
            raise Error("Dataset row count exceeds max_rows.")
        var field_count = 0
        for field_slice in line.split(delimiter):
            if field_count >= dimension:
                raise Error("Delimited row has too many vector components.")
            embeddings.append(Float32(Float64(String(field_slice))))
            field_count += 1
        if field_count != dimension:
            raise Error("Delimited row has the wrong vector dimension.")
        if id_start > Int.MAX - len(ids):
            raise Error("Generated dataset IDs overflow Int.")
        ids.append(id_start + len(ids))
    return VectorDataset(
        ids=ids^,
        embeddings=embeddings^,
        dimension=dimension,
    )


def read_csv(
    path: String,
    dimension: Int,
    *,
    has_header: Bool = True,
    id_start: Int = 0,
    max_rows: Int = _MAX_DATASET_ROWS,
) raises -> VectorDataset:
    """Reads numeric-only CSV rows without quoted-field support."""

    return _read_delimited_vectors(
        path, dimension, ",", has_header, id_start, max_rows
    )


def read_tsv(
    path: String,
    dimension: Int,
    *,
    has_header: Bool = True,
    id_start: Int = 0,
    max_rows: Int = _MAX_DATASET_ROWS,
) raises -> VectorDataset:
    """Reads numeric-only TSV rows and assigns sequential user IDs."""

    return _read_delimited_vectors(
        path, dimension, "\t", has_header, id_start, max_rows
    )
