from std.collections import List
from std.io.file import FileHandle
from std.collections.span import Span
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from mojovec import (
    Collection,
    read_csv,
    read_fvecs,
    read_ivecs,
    read_npy_float32,
    read_tsv,
)


def _write_i32(mut file: FileHandle, value: Int32) raises:
    var storage = InlineArray[Int32, 1](uninitialized=True)
    storage[0] = value
    file.write_all(
        Span[UInt8](unsafe_ptr=storage.unsafe_ptr().unsafe_bitcast[UInt8](), length=4)
    )


def _write_f32(mut file: FileHandle, value: Float32) raises:
    var storage = InlineArray[Float32, 1](uninitialized=True)
    storage[0] = value
    file.write_all(
        Span[UInt8](unsafe_ptr=storage.unsafe_ptr().unsafe_bitcast[UInt8](), length=4)
    )


def _write_u16(mut file: FileHandle, value: Int) raises:
    var storage = InlineArray[UInt8, 2](uninitialized=True)
    storage[0] = UInt8(value & 0xFF)
    storage[1] = UInt8((value >> 8) & 0xFF)
    file.write_all(storage)


def test_fvecs_loads_into_collection() raises:
    var path = "/tmp/mojovec_dataset_test.fvecs"
    var writer = open(path, "w")
    for row in range(2):
        _write_i32(writer, 3)
        for column in range(3):
            _write_f32(writer, Float32(row * 10 + column))
    writer.close()

    var dataset = read_fvecs(path, id_start=100)
    assert_equal(dataset.count(), 2)
    assert_equal(dataset.dimension, 3)
    assert_equal(dataset.ids[0], 100)
    assert_equal(dataset.ids[1], 101)
    assert_almost_equal(dataset.embeddings[5], 12.0)

    var collection = Collection(3, quantized=False)
    collection.add(dataset.ids, dataset.embeddings)
    var query = List[Float32](capacity=3)
    query.append(1.0)
    query.append(2.0)
    query.append(3.0)
    var result = collection.query(query, n_results=1)
    assert_equal(result.ids[0][0], 100)


def test_ivecs_reads_integer_matrix() raises:
    var path = "/tmp/mojovec_dataset_test.ivecs"
    var writer = open(path, "w")
    for row in range(2):
        _write_i32(writer, 2)
        _write_i32(writer, Int32(row * 2 + 7))
        _write_i32(writer, Int32(row * 2 + 8))
    writer.close()

    var dataset = read_ivecs(path)
    assert_equal(dataset.count(), 2)
    assert_equal(dataset.dimension, 2)
    assert_equal(dataset.values[0], 7)
    assert_equal(dataset.values[3], 10)


def test_npy_reads_float32_matrix() raises:
    var path = "/tmp/mojovec_dataset_test.npy"
    # NPY pads the complete preamble and header to a 16-byte boundary.
    var header = (
        "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }       "
        "   \n"
    )
    var writer = open(path, "w")
    var magic = InlineArray[UInt8, 8](uninitialized=True)
    magic[0] = 0x93
    magic[1] = 78
    magic[2] = 85
    magic[3] = 77
    magic[4] = 80
    magic[5] = 89
    magic[6] = 1
    magic[7] = 0
    writer.write_all(magic)
    _write_u16(writer, header.byte_length())
    writer.write_all(header.as_bytes())
    for index in range(6):
        _write_f32(writer, Float32(index) * 0.5)
    writer.close()

    var dataset = read_npy_float32(path, id_start=5)
    assert_equal(dataset.count(), 2)
    assert_equal(dataset.dimension, 3)
    assert_equal(dataset.ids[1], 6)
    assert_almost_equal(dataset.embeddings[5], 2.5)


def test_csv_and_tsv_numeric_matrices() raises:
    var csv_path = "/tmp/mojovec_dataset_test.csv"
    var csv_writer = open(csv_path, "w")
    csv_writer.write("x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
    csv_writer.close()
    var csv = read_csv(csv_path, 3, id_start=20)
    assert_equal(csv.count(), 2)
    assert_equal(csv.ids[0], 20)
    assert_almost_equal(csv.embeddings[4], 5.0)

    var tsv_path = "/tmp/mojovec_dataset_test.tsv"
    var tsv_writer = open(tsv_path, "w")
    tsv_writer.write("1.5\t2.5\r\n3.5\t4.5\r\n")
    tsv_writer.close()
    var tsv = read_tsv(tsv_path, 2, has_header=False)
    assert_equal(tsv.count(), 2)
    assert_almost_equal(tsv.embeddings[0], 1.5)
    assert_almost_equal(tsv.embeddings[3], 4.5)


def test_malformed_binary_dataset_is_rejected() raises:
    var path = "/tmp/mojovec_dataset_bad.fvecs"
    var writer = open(path, "w")
    _write_i32(writer, 3)
    _write_f32(writer, 1.0)
    writer.close()

    var failed = False
    try:
        _ = read_fvecs(path)
    except:
        failed = True
    assert_true(failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
