from std.python import Python
from std.testing import TestSuite, assert_equal, assert_true

from mojovec import (
    Collection,
    DatasetImportOptions,
    read_json,
    read_jsonl,
    read_npz,
)


def test_json_bridge_preserves_payloads_and_batches() raises:
    var path = "/tmp/mojovec_python_dataset.json"
    var writer = open(path, "w")
    writer.write(
        '[{"id":10,"embedding":[1,0],"text":"first",'
        '"year":2026},{"id":20,"embedding":[0,1],'
        '"text":"second","year":2025}]'
    )
    writer.close()

    var options = DatasetImportOptions()
    options.id_column = "id"
    options.document_column = "text"
    options.metadata_columns.append("year")
    options.batch_size = 1

    var reader = read_json(path, 2, options)
    var collection = Collection(2, quantized=False)
    assert_equal(reader.add_to(collection), 2)
    assert_equal(collection.count(), 2)
    assert_equal(collection.get_document(10), "first")
    assert_equal(collection.get_metadata(10).get_int("year"), 2026)

    var consumed_failed = False
    try:
        _ = reader.add_to(collection)
    except:
        consumed_failed = True
    assert_true(consumed_failed)


def test_jsonl_bridge_generates_ids_and_upserts() raises:
    var path = "/tmp/mojovec_python_dataset.jsonl"
    var writer = open(path, "w")
    writer.write('{"embedding":[1,0]}\n{"embedding":[0,1]}\n')
    writer.close()

    var options = DatasetImportOptions()
    options.id_start = 100
    var reader = read_jsonl(path, 2, options)
    var collection = Collection(2, quantized=False)
    assert_equal(reader.upsert_to(collection), 2)
    assert_equal(collection.count(), 2)


def test_npz_bridge_uses_numpy_decoder() raises:
    var path = "/tmp/mojovec_python_dataset.npz"
    var numpy = Python.import_module("numpy")
    var values = Python.list()
    values.append(1.0)
    values.append(0.0)
    values.append(0.0)
    values.append(1.0)
    var embeddings = numpy.asarray(values, dtype=numpy.float32)
    numpy.savez(path, embeddings=embeddings)

    var options = DatasetImportOptions()
    options.id_start = 200
    var reader = read_npz(path, 2, options)
    var collection = Collection(2, quantized=False)
    assert_equal(reader.add_to(collection), 2)
    assert_equal(collection.count(), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
