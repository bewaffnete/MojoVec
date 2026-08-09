from std.os import abort
from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.ffi import external_call
from std.memory import alloc
from std.collections import List
from mojovec.api.collection import Collection
from mojovec.api.metadata import (
    METADATA_BOOL,
    METADATA_FLOAT,
    METADATA_INT,
    METADATA_STRING,
    Metadata,
)
from mojovec.api.results import CollectionStats, CompactReport, QueryResults
from mojovec.api.where import Where


struct _ReleasedPythonThreadState(Movable):
    """Detaches CPython state while one native read-only query runs."""

    var state: Int

    def __init__(out self):
        self.state = external_call["PyEval_SaveThread", Int]()

    def __init__(out self, *, deinit move: Self):
        self.state = move.state

    def __del__(deinit self):
        if self.state != 0:
            external_call["PyEval_RestoreThread", NoneType](self.state)

    def restore(mut self):
        if self.state != 0:
            external_call["PyEval_RestoreThread", NoneType](self.state)
            self.state = 0


def _stats_to_python(stats: CollectionStats) raises -> PythonObject:
    var result = Python.dict()
    result["active_count"] = stats.active_count
    result["deleted_count"] = stats.deleted_count
    result["total_count"] = stats.total_count
    result["deleted_ratio"] = stats.deleted_ratio
    result["dimension"] = stats.dimension
    result["quantized"] = stats.quantized
    result["M"] = stats.M
    result["ef_construction"] = stats.ef_construction
    result["ef_search"] = stats.ef_search
    return result


def _report_to_python(report: CompactReport) raises -> PythonObject:
    var result = Python.dict()
    result["performed"] = report.performed
    result["before"] = _stats_to_python(report.before.copy())
    result["after"] = _stats_to_python(report.after.copy())
    result["reclaimed_records"] = report.reclaimed_records
    result["elapsed_seconds"] = report.elapsed_seconds
    return result


def _metadata_from_python(py_metadata: PythonObject) raises -> Metadata:
    var builtins = Python.import_module("builtins")
    if not Bool(py=builtins.isinstance(py_metadata, builtins.dict)):
        raise Error("Each metadata item must be a Python dict.")

    var metadata = Metadata()
    for py_key in py_metadata:
        if not Bool(py=builtins.isinstance(py_key, builtins.str)):
            raise Error("Metadata keys must be strings.")
        var key = String(py=py_key)
        var value = py_metadata[py_key]
        if Bool(py=builtins.isinstance(value, builtins.bool)):
            metadata.set(key, Bool(py=value))
        elif Bool(py=builtins.isinstance(value, builtins.int)):
            metadata.set(key, Int(py=value))
        elif Bool(py=builtins.isinstance(value, builtins.float)):
            metadata.set(key, Float64(py=value))
        elif Bool(py=builtins.isinstance(value, builtins.str)):
            metadata.set(key, String(py=value))
        else:
            raise Error(
                "Metadata values must be str, int, float, or bool."
            )
    return metadata^


def _metadatas_from_python(py_metadatas: PythonObject) raises -> List[Metadata]:
    var metadatas = List[Metadata]()
    for py_metadata in py_metadatas:
        metadatas.append(_metadata_from_python(py_metadata))
    return metadatas^


def _ids_from_python(py_ids: PythonObject) raises -> List[Int]:
    var ids = List[Int]()
    for py_id in py_ids:
        ids.append(Int(py=py_id))
    return ids^


def _floats_from_python(py_values: PythonObject) raises -> List[Float32]:
    var values = List[Float32]()
    for py_value in py_values:
        values.append(Float32(py=py_value))
    return values^


def _strings_from_python(py_values: PythonObject) raises -> List[String]:
    var values = List[String]()
    for py_value in py_values:
        values.append(String(py=py_value))
    return values^


def _metadata_to_python(metadata: Metadata) raises -> PythonObject:
    var result = Python.dict()
    for index in range(metadata.count()):
        var key = metadata._key_at(index)
        var value = metadata._value_at(index)
        if value.kind() == METADATA_STRING:
            result[key] = value.as_string()
        elif value.kind() == METADATA_INT:
            result[key] = value.as_int()
        elif value.kind() == METADATA_FLOAT:
            result[key] = value.as_float()
        elif value.kind() == METADATA_BOOL:
            result[key] = value.as_bool()
    return result


def _query_results_to_python(results: QueryResults) raises -> PythonObject:
    var out_ids = Python.list()
    for i in range(len(results.ids)):
        var row = Python.list()
        for j in range(len(results.ids[i])):
            row.append(results.ids[i][j])
        out_ids.append(row)

    var out_distances = Python.list()
    for i in range(len(results.distances)):
        var row = Python.list()
        for j in range(len(results.distances[i])):
            row.append(results.distances[i][j])
        out_distances.append(row)

    var out_metadatas = Python.list()
    for i in range(len(results.metadatas)):
        var row = Python.list()
        for j in range(len(results.metadatas[i])):
            row.append(_metadata_to_python(results.metadatas[i][j].copy()))
        out_metadatas.append(row)

    var out_documents = Python.list()
    for i in range(len(results.documents)):
        var row = Python.list()
        for j in range(len(results.documents[i])):
            row.append(results.documents[i][j])
        out_documents.append(row)

    var out_scores = Python.list()
    for i in range(len(results.scores)):
        var row = Python.list()
        for j in range(len(results.scores[i])):
            row.append(results.scores[i][j])
        out_scores.append(row)

    var result = Python.dict()
    result["ids"] = out_ids
    result["distances"] = out_distances
    result["metadatas"] = out_metadatas
    result["documents"] = out_documents
    result["scores"] = out_scores
    return result


struct PyWhere(Movable, Writable):
    var ptr: UnsafePointer[Where, MutAnyOrigin]

    def __init__(out self, value: Where):
        self.ptr = rebind[UnsafePointer[Where, MutAnyOrigin]](
            alloc[Where](1)
        )
        self.ptr.init_pointee_move(value.copy())

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    def __del__(deinit self):
        self.ptr.destroy_pointee()
        self.ptr.free()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Where()")


def _where_predicate(
    operation: String,
    key: String,
    py_value: PythonObject,
) raises -> Where:
    var builtins = Python.import_module("builtins")
    if Bool(py=builtins.isinstance(py_value, builtins.bool)):
        var value = Bool(py=py_value)
        if operation == "eq":
            return Where.eq(key, value)
        if operation == "ne":
            return Where.ne(key, value)
        raise Error("Ordered where predicates require int or float values.")
    if Bool(py=builtins.isinstance(py_value, builtins.int)):
        var value = Int(py=py_value)
        if operation == "eq":
            return Where.eq(key, value)
        if operation == "ne":
            return Where.ne(key, value)
        if operation == "gt":
            return Where.gt(key, value)
        if operation == "gte":
            return Where.gte(key, value)
        if operation == "lt":
            return Where.lt(key, value)
        if operation == "lte":
            return Where.lte(key, value)
    elif Bool(py=builtins.isinstance(py_value, builtins.float)):
        var value = Float64(py=py_value)
        if operation == "eq":
            return Where.eq(key, value)
        if operation == "ne":
            return Where.ne(key, value)
        if operation == "gt":
            return Where.gt(key, value)
        if operation == "gte":
            return Where.gte(key, value)
        if operation == "lt":
            return Where.lt(key, value)
        if operation == "lte":
            return Where.lte(key, value)
    elif Bool(py=builtins.isinstance(py_value, builtins.str)):
        var value = String(py=py_value)
        if operation == "eq":
            return Where.eq(key, value)
        if operation == "ne":
            return Where.ne(key, value)
        raise Error("Ordered where predicates require int or float values.")
    raise Error("Unsupported where predicate or scalar value.")


def py_where_predicate(
    operation: PythonObject,
    key: PythonObject,
    value: PythonObject,
) raises -> PythonObject:
    var where = _where_predicate(
        String(py=operation),
        String(py=key),
        value,
    )
    return PythonObject(alloc=PyWhere(where^))


def py_where_combine(
    operation: PythonObject,
    py_conditions: PythonObject,
) raises -> PythonObject:
    var conditions = List[Where]()
    for py_condition in py_conditions:
        var condition = py_condition.downcast_value_ptr[PyWhere]()
        conditions.append(condition[].ptr[].copy())

    var op = String(py=operation)
    var where: Where
    if op == "all":
        where = Where.all(conditions)
    elif op == "any":
        where = Where.any(conditions)
    elif op == "not":
        if len(conditions) != 1:
            raise Error("Where.not requires exactly one condition.")
        where = Where.not_(conditions[0])
    else:
        raise Error("Unsupported where logical operator.")
    return PythonObject(alloc=PyWhere(where^))


struct PyCollection(Movable, Writable):
    var ptr: UnsafePointer[Collection, MutAnyOrigin]

    def __init__(out self, ptr: UnsafePointer[Collection, MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    def __del__(deinit self):
        self.ptr.destroy_pointee()
        self.ptr.free()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Collection()")

    @staticmethod
    def py_init(
        out self: PyCollection, args: PythonObject, kwargs: PythonObject
    ) raises:
        var d = Int(py=args[0])
        var M = 32
        var ef_c = 40
        var ef_s = 16
        var quantized = True
        var name = String("")
        var metric = String("l2")
        if len(args) > 1:
            M = Int(py=args[1])
        if len(args) > 2:
            ef_c = Int(py=args[2])
        if len(args) > 3:
            ef_s = Int(py=args[3])
        if len(args) > 4:
            quantized = Bool(py=args[4])
        if len(args) > 5:
            name = String(py=args[5])
        if len(args) > 6:
            metric = String(py=args[6])

        var col_ptr = rebind[UnsafePointer[Collection, MutAnyOrigin]](
            alloc[Collection](1)
        )
        col_ptr.init_pointee_move(
            Collection(
                d,
                M,
                ef_c,
                ef_s,
                quantized,
                name,
                metric=metric,
            )
        )
        self = Self(col_ptr)

    @staticmethod
    def py_add(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        self_ptr[].ptr[].add(mojo_ids, mojo_embeddings)
        return Python.none()

    @staticmethod
    def py_add_with_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_metadatas: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var metadatas = _metadatas_from_python(py_metadatas)
        self_ptr[].ptr[].add(mojo_ids, mojo_embeddings, metadatas)
        return Python.none()

    @staticmethod
    def py_add_with_documents(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_documents: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var documents = _strings_from_python(py_documents)
        self_ptr[].ptr[].add(mojo_ids, mojo_embeddings, documents)
        return Python.none()

    @staticmethod
    def py_add_with_payloads(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_metadatas: PythonObject,
        py_documents: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var metadatas = _metadatas_from_python(py_metadatas)
        var documents = _strings_from_python(py_documents)
        self_ptr[].ptr[].add(
            mojo_ids,
            mojo_embeddings,
            metadatas,
            documents,
        )
        return Python.none()

    @staticmethod
    def py_upsert(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        self_ptr[].ptr[].upsert(mojo_ids, mojo_embeddings)
        return Python.none()

    @staticmethod
    def py_upsert_with_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_metadatas: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var metadatas = _metadatas_from_python(py_metadatas)
        self_ptr[].ptr[].upsert(mojo_ids, mojo_embeddings, metadatas)
        return Python.none()

    @staticmethod
    def py_upsert_with_documents(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_documents: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var documents = _strings_from_python(py_documents)
        self_ptr[].ptr[].upsert(mojo_ids, mojo_embeddings, documents)
        return Python.none()

    @staticmethod
    def py_upsert_with_payloads(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_metadatas: PythonObject,
        py_documents: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var metadatas = _metadatas_from_python(py_metadatas)
        var documents = _strings_from_python(py_documents)
        self_ptr[].ptr[].upsert(
            mojo_ids,
            mojo_embeddings,
            metadatas,
            documents,
        )
        return Python.none()

    @staticmethod
    def py_update(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        self_ptr[].ptr[].update(mojo_ids, mojo_embeddings)
        return Python.none()

    @staticmethod
    def py_update_with_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_metadatas: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var metadatas = _metadatas_from_python(py_metadatas)
        self_ptr[].ptr[].update(mojo_ids, mojo_embeddings, metadatas)
        return Python.none()

    @staticmethod
    def py_update_with_documents(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_documents: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var documents = _strings_from_python(py_documents)
        self_ptr[].ptr[].update(mojo_ids, mojo_embeddings, documents)
        return Python.none()

    @staticmethod
    def py_update_with_payloads(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
        py_metadatas: PythonObject,
        py_documents: PythonObject,
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var metadatas = _metadatas_from_python(py_metadatas)
        var documents = _strings_from_python(py_documents)
        self_ptr[].ptr[].update(
            mojo_ids,
            mojo_embeddings,
            metadatas,
            documents,
        )
        return Python.none()

    @staticmethod
    def py_delete(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], py_ids: PythonObject
    ) raises -> PythonObject:
        var mojo_ids = _ids_from_python(py_ids)
        self_ptr[].ptr[].delete(mojo_ids)
        return Python.none()

    @staticmethod
    def py_count(self_ptr: UnsafePointer[Self, MutAnyOrigin]) -> PythonObject:
        return PythonObject(self_ptr[].ptr[].count())

    @staticmethod
    def py_count_deleted(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) -> PythonObject:
        return PythonObject(self_ptr[].ptr[].count_deleted())

    @staticmethod
    def py_name(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].ptr[].name())

    @staticmethod
    def py_dimension(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) -> PythonObject:
        return PythonObject(self_ptr[].ptr[].dimension())

    @staticmethod
    def py_storage_kind(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) -> PythonObject:
        return PythonObject(Int(self_ptr[].ptr[].storage_kind()))

    @staticmethod
    def py_metric(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].ptr[].metric())

    @staticmethod
    def py_is_quantized(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) -> PythonObject:
        return PythonObject(self_ptr[].ptr[].is_quantized())

    @staticmethod
    def py_is_memory_mapped(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) -> PythonObject:
        return PythonObject(self_ptr[].ptr[].is_memory_mapped())

    @staticmethod
    def py_get_metadata(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_record_id: PythonObject,
    ) raises -> PythonObject:
        var metadata = self_ptr[].ptr[].get_metadata(
            Int(py=py_record_id)
        )
        return _metadata_to_python(metadata^)

    @staticmethod
    def py_get_document(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_record_id: PythonObject,
    ) raises -> PythonObject:
        return PythonObject(
            self_ptr[].ptr[].get_document(Int(py=py_record_id))
        )

    @staticmethod
    def py_set_ef_search(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ef_search: PythonObject,
    ) raises -> PythonObject:
        self_ptr[].ptr[].set_ef_search(Int(py=py_ef_search))
        return Python.none()

    @staticmethod
    def py_stats(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var stats = self_ptr[].ptr[].stats()
        return _stats_to_python(stats^)

    @staticmethod
    def py_compact(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var report = self_ptr[].ptr[].compact()
        return _report_to_python(report^)

    @staticmethod
    def py_compact_if_needed(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_deleted_ratio: PythonObject,
    ) raises -> PythonObject:
        var deleted_ratio = Float64(py=py_deleted_ratio)
        var report = self_ptr[].ptr[].compact_if_needed(deleted_ratio)
        return _report_to_python(report^)

    @staticmethod
    def py_query(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_embeddings: PythonObject,
        n_results: PythonObject,
    ) raises -> PythonObject:
        var num_res = Int(py=n_results)
        var mojo_embeddings = _floats_from_python(py_embeddings)
        var released = _ReleasedPythonThreadState()
        var res = self_ptr[].ptr[].query(mojo_embeddings, num_res)
        released.restore()
        return _query_results_to_python(res^)

    @staticmethod
    def py_query_where(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_embeddings: PythonObject,
        n_results: PythonObject,
        py_where: PythonObject,
    ) raises -> PythonObject:
        var where = py_where.downcast_value_ptr[PyWhere]()
        var embeddings = _floats_from_python(py_embeddings)
        var compiled_where = where[].ptr[].copy()
        var num_res = Int(py=n_results)
        var released = _ReleasedPythonThreadState()
        var results = self_ptr[].ptr[].query(
            embeddings,
            compiled_where,
            num_res,
        )
        released.restore()
        return _query_results_to_python(results^)

    @staticmethod
    def py_query_text(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_query_texts: PythonObject,
        n_results: PythonObject,
    ) raises -> PythonObject:
        var query_texts = _strings_from_python(py_query_texts)
        var num_res = Int(py=n_results)
        var released = _ReleasedPythonThreadState()
        var results = self_ptr[].ptr[].query(
            query_texts,
            num_res,
        )
        released.restore()
        return _query_results_to_python(results^)

    @staticmethod
    def py_query_text_where(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_query_texts: PythonObject,
        n_results: PythonObject,
        py_where: PythonObject,
    ) raises -> PythonObject:
        var where = py_where.downcast_value_ptr[PyWhere]()
        var query_texts = _strings_from_python(py_query_texts)
        var compiled_where = where[].ptr[].copy()
        var num_res = Int(py=n_results)
        var released = _ReleasedPythonThreadState()
        var results = self_ptr[].ptr[].query(
            query_texts,
            compiled_where,
            num_res,
        )
        released.restore()
        return _query_results_to_python(results^)

    @staticmethod
    def py_query_hybrid(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_embeddings: PythonObject,
        py_query_texts: PythonObject,
        n_results: PythonObject,
        rrf_k: PythonObject,
        candidate_multiplier: PythonObject,
    ) raises -> PythonObject:
        var embeddings = _floats_from_python(py_embeddings)
        var query_texts = _strings_from_python(py_query_texts)
        var num_res = Int(py=n_results)
        var fusion_k = Int(py=rrf_k)
        var multiplier = Int(py=candidate_multiplier)
        var released = _ReleasedPythonThreadState()
        var results = self_ptr[].ptr[].query_hybrid(
            embeddings,
            query_texts,
            num_res,
            fusion_k,
            multiplier,
        )
        released.restore()
        return _query_results_to_python(results^)

    @staticmethod
    def py_query_hybrid_where(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_embeddings: PythonObject,
        py_query_texts: PythonObject,
        n_results: PythonObject,
        rrf_k: PythonObject,
        candidate_multiplier: PythonObject,
        py_where: PythonObject,
    ) raises -> PythonObject:
        var where = py_where.downcast_value_ptr[PyWhere]()
        var embeddings = _floats_from_python(py_embeddings)
        var query_texts = _strings_from_python(py_query_texts)
        var compiled_where = where[].ptr[].copy()
        var num_res = Int(py=n_results)
        var fusion_k = Int(py=rrf_k)
        var multiplier = Int(py=candidate_multiplier)
        var released = _ReleasedPythonThreadState()
        var results = self_ptr[].ptr[].query_hybrid(
            embeddings,
            query_texts,
            compiled_where,
            num_res,
            fusion_k,
            multiplier,
        )
        released.restore()
        return _query_results_to_python(results^)

    @staticmethod
    def py_add_numpy(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
    ) raises -> PythonObject:
        var num_vectors = Int(py=py_ids.__len__())
        var ids_ptr_int = Int(py=py_ids.__array_interface__["data"][0])
        var emb_ptr_int = Int(py=py_embeddings.__array_interface__["data"][0])

        var ids_ptr = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=ids_ptr_int
        )
        var emb_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=emb_ptr_int
        )

        var ids = Span[Int, MutAnyOrigin](ptr=ids_ptr, length=num_vectors)
        var embeddings = Span[Float32, MutAnyOrigin](
            ptr=emb_ptr,
            length=num_vectors * self_ptr[].ptr[].dimension(),
        )
        self_ptr[].ptr[]._add_from_spans(ids, embeddings)
        return Python.none()

    @staticmethod
    def py_upsert_numpy(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_ids: PythonObject,
        py_embeddings: PythonObject,
    ) raises -> PythonObject:
        var num_vectors = Int(py=py_ids.__len__())
        var ids_ptr_int = Int(py=py_ids.__array_interface__["data"][0])
        var emb_ptr_int = Int(py=py_embeddings.__array_interface__["data"][0])

        var ids_ptr = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=ids_ptr_int
        )
        var emb_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=emb_ptr_int
        )

        var ids = Span[Int, MutAnyOrigin](ptr=ids_ptr, length=num_vectors)
        var embeddings = Span[Float32, MutAnyOrigin](
            ptr=emb_ptr,
            length=num_vectors * self_ptr[].ptr[].dimension(),
        )
        self_ptr[].ptr[]._upsert_from_spans(ids, embeddings)
        return Python.none()

    @staticmethod
    def py_query_numpy(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        py_embeddings: PythonObject,
        n_results: PythonObject,
    ) raises -> PythonObject:
        var num_queries = Int(py=py_embeddings.shape[0])
        var k = Int(py=n_results)
        var emb_ptr_int = Int(py=py_embeddings.__array_interface__["data"][0])
        var emb_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=emb_ptr_int
        )

        var np = Python.import_module("numpy")
        var out_ids = np.empty(Python.tuple(num_queries, k), dtype=np.int64)
        var out_dists = np.empty(Python.tuple(num_queries, k), dtype=np.float32)

        var out_ids_int = Int(py=out_ids.__array_interface__["data"][0])
        var out_dists_int = Int(py=out_dists.__array_interface__["data"][0])

        var out_ids_ptr = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=out_ids_int
        )
        var out_dists_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=out_dists_int
        )

        var queries = Span[Float32, MutAnyOrigin](
            ptr=emb_ptr,
            length=num_queries * self_ptr[].ptr[].dimension(),
        )
        var ids = Span[mut=True, Int, MutAnyOrigin](
            ptr=out_ids_ptr, length=num_queries * k
        )
        var distances = Span[mut=True, Float32, MutAnyOrigin](
            ptr=out_dists_ptr, length=num_queries * k
        )
        var released = _ReleasedPythonThreadState()
        self_ptr[].ptr[]._query_into(queries, k, ids, distances)
        released.restore()

        var result = Python.dict()
        result["ids"] = out_ids
        result["distances"] = out_dists
        return result

    @staticmethod
    def py_save(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], path: PythonObject
    ) raises -> PythonObject:
        self_ptr[].ptr[].save(String(py=path))
        return Python.none()

    @staticmethod
    def py_wal_enabled(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) -> PythonObject:
        return PythonObject(self_ptr[].ptr[].wal_enabled())

    @staticmethod
    def py_wal_sequence(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) -> PythonObject:
        return PythonObject(self_ptr[].ptr[].wal_sequence())

    @staticmethod
    def py_enable_wal(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        path: PythonObject,
        durability: PythonObject,
    ) raises -> PythonObject:
        self_ptr[].ptr[].enable_wal(
            String(py=path),
            Int(py=durability),
        )
        return Python.none()

    @staticmethod
    def py_disable_wal(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) -> PythonObject:
        self_ptr[].ptr[].disable_wal()
        return Python.none()

    @staticmethod
    def py_flush_wal(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        self_ptr[].ptr[].flush_wal()
        return Python.none()

    @staticmethod
    def py_checkpoint(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        path: PythonObject,
    ) raises -> PythonObject:
        self_ptr[].ptr[].checkpoint(String(py=path))
        return Python.none()

    @staticmethod
    def py_snapshot(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        path: PythonObject,
        memory_mapped: PythonObject,
        mmap_threshold_bytes: PythonObject,
    ) raises -> PythonObject:
        var col = self_ptr[].ptr[].snapshot(
            String(py=path),
            Bool(py=memory_mapped),
            Int(py=mmap_threshold_bytes),
        )
        return Self._wrap_collection(col^)

    @staticmethod
    def _wrap_collection(var col: Collection) raises -> PythonObject:
        var col_ptr = rebind[UnsafePointer[Collection, MutAnyOrigin]](
            alloc[Collection](1)
        )
        col_ptr.init_pointee_move(col^)
        var py_col = PyCollection(col_ptr)
        return PythonObject(alloc=py_col^)


def py_load(
    path: PythonObject,
    memory_mapped: PythonObject,
    mmap_threshold_bytes: PythonObject,
) raises -> PythonObject:
    var col = Collection.load(
        String(py=path),
        Bool(py=memory_mapped),
        Int(py=mmap_threshold_bytes),
    )
    return PyCollection._wrap_collection(col^)


def py_recover(
    snapshot_path: PythonObject,
    wal_path: PythonObject,
    durability: PythonObject,
    memory_mapped: PythonObject,
    mmap_threshold_bytes: PythonObject,
) raises -> PythonObject:
    var col = Collection.recover(
        String(py=snapshot_path),
        String(py=wal_path),
        Int(py=durability),
        Bool(py=memory_mapped),
        Int(py=mmap_threshold_bytes),
    )
    return PyCollection._wrap_collection(col^)


@export
def PyInit__native() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_native")
        _ = (
            m.add_type[PyCollection]("_Collection")
            .def_py_init[PyCollection.py_init]()
            .def_method[PyCollection.py_add]("add")
            .def_method[PyCollection.py_add_with_metadata](
                "add_with_metadata"
            )
            .def_method[PyCollection.py_add_with_documents](
                "add_with_documents"
            )
            .def_method[PyCollection.py_add_with_payloads](
                "add_with_payloads"
            )
            .def_method[PyCollection.py_upsert]("upsert")
            .def_method[PyCollection.py_upsert_with_metadata](
                "upsert_with_metadata"
            )
            .def_method[PyCollection.py_upsert_with_documents](
                "upsert_with_documents"
            )
            .def_method[PyCollection.py_upsert_with_payloads](
                "upsert_with_payloads"
            )
            .def_method[PyCollection.py_update]("update")
            .def_method[PyCollection.py_update_with_metadata](
                "update_with_metadata"
            )
            .def_method[PyCollection.py_update_with_documents](
                "update_with_documents"
            )
            .def_method[PyCollection.py_update_with_payloads](
                "update_with_payloads"
            )
            .def_method[PyCollection.py_delete]("delete")
            .def_method[PyCollection.py_count]("count")
            .def_method[PyCollection.py_count_deleted]("count_deleted")
            .def_method[PyCollection.py_name]("name")
            .def_method[PyCollection.py_dimension]("dimension")
            .def_method[PyCollection.py_storage_kind]("storage_kind")
            .def_method[PyCollection.py_metric]("metric")
            .def_method[PyCollection.py_is_quantized]("is_quantized")
            .def_method[PyCollection.py_is_memory_mapped](
                "is_memory_mapped"
            )
            .def_method[PyCollection.py_get_metadata]("get_metadata")
            .def_method[PyCollection.py_get_document]("get_document")
            .def_method[PyCollection.py_set_ef_search]("set_ef_search")
            .def_method[PyCollection.py_stats]("stats")
            .def_method[PyCollection.py_compact]("compact")
            .def_method[PyCollection.py_compact_if_needed](
                "compact_if_needed"
            )
            .def_method[PyCollection.py_query]("query_vector")
            .def_method[PyCollection.py_query_where]("query_vector_where")
            .def_method[PyCollection.py_query_text]("query_text")
            .def_method[PyCollection.py_query_text_where](
                "query_text_where"
            )
            .def_method[PyCollection.py_query_hybrid]("query_hybrid")
            .def_method[PyCollection.py_query_hybrid_where](
                "query_hybrid_where"
            )
            .def_method[PyCollection.py_upsert]("upsert_batch")
            .def_method[PyCollection.py_add_numpy]("add_numpy")
            .def_method[PyCollection.py_upsert_numpy]("upsert_numpy")
            .def_method[PyCollection.py_query_numpy]("query_numpy")
            .def_method[PyCollection.py_upsert_numpy]("upsert_batch_numpy")
            .def_method[PyCollection.py_query_numpy]("query_batch_numpy")
            .def_method[PyCollection.py_save]("save")
            .def_method[PyCollection.py_wal_enabled]("wal_enabled")
            .def_method[PyCollection.py_wal_sequence]("wal_sequence")
            .def_method[PyCollection.py_enable_wal]("enable_wal")
            .def_method[PyCollection.py_disable_wal]("disable_wal")
            .def_method[PyCollection.py_flush_wal]("flush_wal")
            .def_method[PyCollection.py_checkpoint]("checkpoint")
            .def_method[PyCollection.py_snapshot]("snapshot")
        )
        _ = m.add_type[PyWhere]("_Where")
        m.def_function[py_where_predicate]("_where_predicate")
        m.def_function[py_where_combine]("_where_combine")
        m.def_function[py_load]("_load")
        m.def_function[py_recover]("_recover")
        return m.finalize()
    except e:
        print("Mojo Exception:", e)
        abort(String("Failed: ", e))
