"""Read JSON, Parquet, Arrow IPC, and Hugging Face data from Mojo.

The entry point and every collection operation are Mojo. Python ecosystem
packages only decode external formats into validated batches; add_to() then
commits those batches through the native managed Collection API.

Run from the repository root:

    pixi run mojo run -I . examples/mojo/api_12_python_dataset_io.mojo
"""

from std.collections import List
from std.python import Python, PythonObject

from mojovec import (
    Collection,
    DatasetImportOptions,
    QueryResults,
    read_arrow,
    read_huggingface,
    read_json,
    read_parquet,
)


comptime LOCAL_DIMENSION = 3
comptime HF_DIMENSION = 1024
comptime HF_DATASET = "christophsonntag/gte_embedded_movies"
comptime HF_CACHE = "scratch/hf_cache"


def local_options() -> DatasetImportOptions:
    """Maps the shared columns used by the local example files."""

    var options = DatasetImportOptions()
    options.id_column = "id"
    options.embedding_column = "embedding"
    options.document_column = "text"
    options.metadata_columns.append("topic")
    options.batch_size = 1
    return options^


def python_vector(x0: Float64, x1: Float64, x2: Float64) raises -> PythonObject:
    var result = Python.list()
    result.append(x0)
    result.append(x1)
    result.append(x2)
    return result


def create_columnar_examples(
    parquet_path: String,
    arrow_path: String,
) raises:
    """Creates tiny files only so this tutorial is self-contained."""

    var pyarrow = Python.import_module("pyarrow")
    var parquet = Python.import_module("pyarrow.parquet")
    var ipc = Python.import_module("pyarrow.ipc")

    var ids = Python.list()
    ids.append(101)
    ids.append(202)

    var embeddings = Python.list()
    embeddings.append(python_vector(1.0, 0.0, 0.0))
    embeddings.append(python_vector(0.8, 0.2, 0.0))

    var documents = Python.list()
    documents.append("Vector search with Mojo")
    documents.append("Reading Arrow and Parquet datasets")

    var topics = Python.list()
    topics.append("search")
    topics.append("data")

    var columns = Python.dict()
    columns["id"] = ids
    columns["embedding"] = embeddings
    columns["text"] = documents
    columns["topic"] = topics
    var table = pyarrow.table(columns)

    parquet.write_table(table, parquet_path)

    var sink = pyarrow.OSFile(arrow_path, "wb")
    # Arrow IPC streams are sequential and avoid the random-access Future
    # machinery that conflicts with Mojo's embedded Python on macOS.
    var ipc_writer = ipc.new_stream(sink, table.schema)
    ipc_writer.write_table(table)
    ipc_writer.close()
    sink.close()


def excerpt(text: String, max_bytes: Int = 180) raises -> String:
    """Returns a short UTF-8-safe Mojo String for readable output."""

    if text.byte_length() <= max_bytes:
        return text.copy()
    var end = 0
    for codepoint in text.codepoint_slices():
        var width = codepoint.byte_length()
        if end + width > max_bytes:
            break
        end += width
    var source = text.as_bytes()
    var prefix = List[UInt8](unsafe_uninit_length=end)
    for index in range(end):
        prefix[index] = source[index]
    return String(from_utf8=prefix)


def print_bm25_results(results: QueryResults) raises:
    for rank in range(len(results.ids[0])):
        if results.ids[0][rank] < 0:
            continue
        print(
            " ",
            rank + 1,
            ". ",
            results.metadatas[0][rank].get_string("title"),
            " | score=",
            results.scores[0][rank],
            sep="",
        )
        print("    ", excerpt(results.documents[0][rank]), "...")


def run_json_example() raises:
    var path = "/tmp/mojovec_dataset_example.json"
    var writer = open(path, "w")
    writer.write(
        '[{"id":10,"embedding":[1,0,0],"text":"Vector search'
        ' with Mojo","topic":"search"},{"id":20,'
        '"embedding":[0,1,0],"text":"Reading external datasets",'
        '"topic":"data"}]'
    )
    writer.close()

    var reader = read_json(path, LOCAL_DIMENSION, local_options())
    var collection = Collection(
        LOCAL_DIMENSION, quantized=True, metric="cosine"
    )
    print("\nJSON imported: ", reader.add_to(collection), sep="")
    print(" document 10: ", collection.get_document(10), sep="")


def run_parquet_example(path: String) raises:
    # In application code `path` simply points to an existing .parquet file.
    var reader = read_parquet(path, LOCAL_DIMENSION, local_options())
    var collection = Collection(
        LOCAL_DIMENSION, quantized=True, metric="cosine"
    )
    print("\nParquet imported: ", reader.add_to(collection), sep="")
    print(" document 101: ", collection.get_document(101), sep="")
    print(
        " topic 101: ",
        collection.get_metadata(101).get_string("topic"),
        sep="",
    )


def run_arrow_example(path: String) raises:
    # This self-contained example uses the Arrow IPC stream format.
    var reader = read_arrow(path, LOCAL_DIMENSION, local_options())
    var collection = Collection(
        LOCAL_DIMENSION, quantized=False, metric="cosine"
    )
    print("\nArrow IPC imported: ", reader.add_to(collection), sep="")
    var neighbors = collection.query([1.0, 0.0, 0.0], n_results=2)
    print(" nearest IDs: ", neighbors.ids[0], sep="")
    print(" nearest documents: ", neighbors.documents[0], sep="")


def run_huggingface_example() raises:
    var options = DatasetImportOptions()
    options.embedding_column = "plot_embedding"
    options.document_column = "fullplot"
    options.metadata_columns.append("title")
    options.metadata_columns.append("runtime")
    options.batch_size = 256
    options.id_start = 0
    options.cache_dir = HF_CACHE

    # Embedded Python and PyArrow streaming scanners can conflict on macOS.
    # The selected dataset is only about 10 MB, so cache it locally while
    # MojoVec still validates and commits it in batches of 256 records.
    options.streaming = False

    var reader = read_huggingface(HF_DATASET, HF_DIMENSION, options)
    var collection = Collection(
        HF_DIMENSION,
        M=16,
        ef_construction=100,
        ef_search=64,
        quantized=True,
        metric="cosine",
        name="hf_movies",
    )
    print("\nHugging Face dataset: ", HF_DATASET, sep="")
    print(" imported movies: ", reader.add_to(collection), sep="")

    # Documents imported from HF are immediately available to native BM25.
    var results = collection.query(
        [String("pirate ship treasure adventure")],
        n_results=5,
    )
    print(" BM25 results:")
    print_bm25_results(results)


def main() raises:
    # Installed packages expose mojovec_dataset_io automatically. Adding the
    # source directory also makes this example work directly from a checkout.
    var sys = Python.import_module("sys")
    sys.path.insert(0, "python")

    var parquet_path = String("/tmp/mojovec_dataset_example.parquet")
    var arrow_path = String("/tmp/mojovec_dataset_example.arrow")
    create_columnar_examples(parquet_path, arrow_path)

    run_json_example()
    run_parquet_example(parquet_path)
    run_arrow_example(arrow_path)
    run_huggingface_example()
