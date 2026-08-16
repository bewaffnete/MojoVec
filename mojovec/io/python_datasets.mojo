"""Python-backed dataset adapters for ecosystem file formats.

The bridge keeps Python objects inside a one-shot reader and converts one
validated batch at a time into managed Mojo Lists. Collection mutations remain
native Mojo operations; Python is used only for decoding external formats.
"""

from std.collections import List
from std.python import Python, PythonObject

from mojovec.api.collection import Collection
from mojovec.api.metadata import Metadata


struct DatasetImportOptions(Movable):
    """Column, batching, NPZ, and Hugging Face reader configuration."""

    var id_column: String
    var embedding_column: String
    var embedding_columns: List[String]
    var document_column: String
    var metadata_columns: List[String]
    var batch_size: Int
    var id_start: Int
    var embeddings_key: String
    var ids_key: String
    var documents_key: String
    var split: String
    var config: String
    var streaming: Bool
    var revision: String
    var cache_dir: String
    var token: String
    var data_dir: String

    def __init__(out self):
        self.id_column = ""
        self.embedding_column = "embedding"
        self.embedding_columns = List[String]()
        self.document_column = ""
        self.metadata_columns = List[String]()
        self.batch_size = 8192
        self.id_start = 0
        self.embeddings_key = "embeddings"
        self.ids_key = "ids"
        self.documents_key = "documents"
        self.split = "train"
        self.config = ""
        self.streaming = True
        self.revision = ""
        self.cache_dir = ""
        self.token = ""
        self.data_dir = ""

    def __init__(out self, *, deinit move: Self):
        self.id_column = move.id_column^
        self.embedding_column = move.embedding_column^
        self.embedding_columns = move.embedding_columns^
        self.document_column = move.document_column^
        self.metadata_columns = move.metadata_columns^
        self.batch_size = move.batch_size
        self.id_start = move.id_start
        self.embeddings_key = move.embeddings_key^
        self.ids_key = move.ids_key^
        self.documents_key = move.documents_key^
        self.split = move.split^
        self.config = move.config^
        self.streaming = move.streaming
        self.revision = move.revision^
        self.cache_dir = move.cache_dir^
        self.token = move.token^
        self.data_dir = move.data_dir^


def _optional_python_string(value: String) raises -> PythonObject:
    if value.byte_length() == 0:
        return Python.none()
    return PythonObject(value)


def _python_string_list_or_none(values: List[String]) raises -> PythonObject:
    if len(values) == 0:
        return Python.none()
    var result = Python.list()
    for value in values:
        result.append(value)
    return result


def _metadata_from_python(value: PythonObject) raises -> Metadata:
    var builtins = Python.import_module("builtins")
    if not Bool(py=builtins.isinstance(value, builtins.dict)):
        raise Error("Each imported metadata value must be a Python dict.")

    var metadata = Metadata()
    for py_key in value:
        if not Bool(py=builtins.isinstance(py_key, builtins.str)):
            raise Error("Imported metadata keys must be strings.")
        var key = String(py=py_key)
        var py_value = value[py_key]
        if Bool(py=builtins.isinstance(py_value, builtins.bool)):
            metadata.set(key, Bool(py=py_value))
        elif Bool(py=builtins.isinstance(py_value, builtins.int)):
            metadata.set(key, Int(py=py_value))
        elif Bool(py=builtins.isinstance(py_value, builtins.float)):
            metadata.set(key, Float64(py=py_value))
        elif Bool(py=builtins.isinstance(py_value, builtins.str)):
            metadata.set(key, String(py=py_value))
        else:
            raise Error(
                "Imported metadata values must be str, int, float, or bool."
            )
    return metadata^


def _ids_from_python(values: PythonObject) raises -> List[Int]:
    var result = List[Int]()
    for value in values:
        result.append(Int(py=value))
    return result^


def _embeddings_from_python(values: PythonObject) raises -> List[Float32]:
    var result = List[Float32]()
    for row in values:
        for value in row:
            result.append(Float32(py=value))
    return result^


def _metadatas_from_python(values: PythonObject) raises -> List[Metadata]:
    var result = List[Metadata]()
    for value in values:
        result.append(_metadata_from_python(value))
    return result^


def _documents_from_python(values: PythonObject) raises -> List[String]:
    var result = List[String]()
    for value in values:
        result.append(String(py=value))
    return result^


struct PythonDatasetReader(Movable):
    """A one-shot Python iterator whose batches are committed by Mojo."""

    var _batches: PythonObject
    var _dimension: Int
    var _consumed: Bool

    def __init__(
        out self,
        batches: PythonObject,
        dimension: Int,
    ):
        self._batches = batches
        self._dimension = dimension
        self._consumed = False

    def __init__(out self, *, deinit move: Self):
        self._batches = move._batches^
        self._dimension = move._dimension
        self._consumed = move._consumed

    def dimension(self) -> Int:
        return self._dimension

    def _ingest(
        mut self,
        mut collection: Collection,
        upsert: Bool,
    ) raises -> Int:
        if self._consumed:
            raise Error("A PythonDatasetReader can only be consumed once.")
        if collection.dimension() != self._dimension:
            raise Error("Dataset and collection dimensions do not match.")
        self._consumed = True

        var builtins = Python.import_module("builtins")
        var imported = 0
        # Mojo's implicit PythonObject iteration can treat an exception raised
        # while advancing a Python generator as normal exhaustion. Pulling
        # batches through builtins.next preserves streaming while propagating
        # decoder and missing-dependency errors to the caller.
        var iterator = builtins.iter(self._batches)
        var iterator_id = Int(py=builtins.id(iterator))
        while True:
            var batch = builtins.next(iterator, iterator)
            if Int(py=builtins.id(batch)) == iterator_id:
                break
            var ids = _ids_from_python(batch.ids)
            var embeddings = _embeddings_from_python(batch.embeddings)
            var has_metadatas = Bool(
                py=builtins.isinstance(batch.metadatas, builtins.list)
            )
            var has_documents = Bool(
                py=builtins.isinstance(batch.documents, builtins.list)
            )

            if has_metadatas and has_documents:
                var metadatas = _metadatas_from_python(batch.metadatas)
                var documents = _documents_from_python(batch.documents)
                if upsert:
                    collection.upsert(ids, embeddings, metadatas, documents)
                else:
                    collection.add(ids, embeddings, metadatas, documents)
            elif has_metadatas:
                var metadatas = _metadatas_from_python(batch.metadatas)
                if upsert:
                    collection.upsert(ids, embeddings, metadatas)
                else:
                    collection.add(ids, embeddings, metadatas)
            elif has_documents:
                var documents = _documents_from_python(batch.documents)
                if upsert:
                    collection.upsert(ids, embeddings, documents)
                else:
                    collection.add(ids, embeddings, documents)
            elif upsert:
                collection.upsert(ids, embeddings)
            else:
                collection.add(ids, embeddings)
            imported += len(ids)
        return imported

    def add_to(
        mut self,
        mut collection: Collection,
    ) raises -> Int:
        """Consumes all batches with insert-only Collection.add semantics."""

        return self._ingest(collection, upsert=False)

    def upsert_to(
        mut self,
        mut collection: Collection,
    ) raises -> Int:
        """Consumes all batches with Collection.upsert semantics."""

        return self._ingest(collection, upsert=True)


def _read_python_file(
    path: String,
    file_format: String,
    dimension: Int,
    options: DatasetImportOptions,
) raises -> PythonDatasetReader:
    var io = Python.import_module("mojovec_dataset_io")
    var batches = io.iter_file_batches(
        path,
        file_format=file_format,
        dimension=dimension,
        id_column=_optional_python_string(options.id_column),
        embedding_column=options.embedding_column,
        embedding_columns=_python_string_list_or_none(
            options.embedding_columns
        ),
        document_column=_optional_python_string(options.document_column),
        metadata_columns=_python_string_list_or_none(options.metadata_columns),
        batch_size=options.batch_size,
        id_start=options.id_start,
        embeddings_key=options.embeddings_key,
        ids_key=options.ids_key,
        documents_key=options.documents_key,
        _embedded_python=True,
    )
    return PythonDatasetReader(batches, dimension)


def read_json(
    path: String,
    dimension: Int,
    options: DatasetImportOptions,
) raises -> PythonDatasetReader:
    """Creates a one-shot reader for a JSON object, list, or data envelope."""

    return _read_python_file(path, "json", dimension, options)


def read_json(path: String, dimension: Int) raises -> PythonDatasetReader:
    return read_json(path, dimension, DatasetImportOptions())


def read_jsonl(
    path: String,
    dimension: Int,
    options: DatasetImportOptions,
) raises -> PythonDatasetReader:
    """Creates a streaming JSON Lines reader."""

    return _read_python_file(path, "jsonl", dimension, options)


def read_jsonl(path: String, dimension: Int) raises -> PythonDatasetReader:
    return read_jsonl(path, dimension, DatasetImportOptions())


def read_parquet(
    path: String,
    dimension: Int,
    options: DatasetImportOptions,
) raises -> PythonDatasetReader:
    """Creates a batched PyArrow Parquet reader."""

    return _read_python_file(path, "parquet", dimension, options)


def read_parquet(path: String, dimension: Int) raises -> PythonDatasetReader:
    return read_parquet(path, dimension, DatasetImportOptions())


def read_arrow(
    path: String,
    dimension: Int,
    options: DatasetImportOptions,
) raises -> PythonDatasetReader:
    """Creates a reader for Arrow IPC file or stream data."""

    return _read_python_file(path, "arrow", dimension, options)


def read_arrow(path: String, dimension: Int) raises -> PythonDatasetReader:
    return read_arrow(path, dimension, DatasetImportOptions())


def read_npz(
    path: String,
    dimension: Int,
    options: DatasetImportOptions,
) raises -> PythonDatasetReader:
    """Creates a reader for compressed or uncompressed NumPy NPZ archives."""

    return _read_python_file(path, "npz", dimension, options)


def read_npz(path: String, dimension: Int) raises -> PythonDatasetReader:
    return read_npz(path, dimension, DatasetImportOptions())


def read_huggingface(
    dataset: String,
    dimension: Int,
    options: DatasetImportOptions,
) raises -> PythonDatasetReader:
    """Creates a batched reader for one Hugging Face dataset split."""

    var io = Python.import_module("mojovec_dataset_io")
    var load_kwargs = Python.dict()
    if options.revision.byte_length() > 0:
        load_kwargs["revision"] = options.revision
    if options.cache_dir.byte_length() > 0:
        load_kwargs["cache_dir"] = options.cache_dir
    if options.token.byte_length() > 0:
        load_kwargs["token"] = options.token
    if options.data_dir.byte_length() > 0:
        load_kwargs["data_dir"] = options.data_dir
    var batches = io.iter_huggingface_batches(
        dataset,
        split=options.split,
        config=_optional_python_string(options.config),
        dimension=dimension,
        id_column=_optional_python_string(options.id_column),
        embedding_column=options.embedding_column,
        embedding_columns=_python_string_list_or_none(
            options.embedding_columns
        ),
        document_column=_optional_python_string(options.document_column),
        metadata_columns=_python_string_list_or_none(options.metadata_columns),
        batch_size=options.batch_size,
        id_start=options.id_start,
        streaming=options.streaming,
        load_kwargs=load_kwargs,
    )
    return PythonDatasetReader(batches, dimension)


def read_huggingface(
    dataset: String,
    dimension: Int,
) raises -> PythonDatasetReader:
    return read_huggingface(dataset, dimension, DatasetImportOptions())
