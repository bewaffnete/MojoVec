from std.os import abort
from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.memory import alloc
from std.collections import List
from mojovec.api.collection import Collection

struct PyCollection(Movable, Writable):
    var ptr: UnsafePointer[Collection, MutAnyOrigin]
    
    def __init__(out self, ptr: UnsafePointer[Collection, MutAnyOrigin]):
        self.ptr = ptr
        
    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Collection()")

    @staticmethod
    def py_init(out self: PyCollection, args: PythonObject, kwargs: PythonObject) raises:
        var d = Int(py=args[0])
        var M = 32
        var ef_c = 40
        var ef_s = 16
        if len(args) > 1: M = Int(py=args[1])
        if len(args) > 2: ef_c = Int(py=args[2])
        if len(args) > 3: ef_s = Int(py=args[3])
        
        var col_ptr = rebind[UnsafePointer[Collection, MutAnyOrigin]](alloc[Collection](1))
        col_ptr.init_pointee_move(Collection(d, M, ef_c, ef_s))
        self = Self(col_ptr)

    @staticmethod
    def py_upsert(self_ptr: UnsafePointer[Self, MutAnyOrigin], py_ids: PythonObject, py_embeddings: PythonObject) raises -> PythonObject:
        return Self.py_upsert_batch(self_ptr, py_ids, py_embeddings)

    @staticmethod
    def py_upsert_batch(self_ptr: UnsafePointer[Self, MutAnyOrigin], py_ids: PythonObject, py_embeddings: PythonObject) raises -> PythonObject:
        var mojo_ids = List[Int]()
        for py_id in py_ids:
            mojo_ids.append(Int(py=py_id))
        var mojo_embeddings = List[Float32]()
        for emb in py_embeddings:
            mojo_embeddings.append(Float32(py=emb))
        self_ptr[].ptr[].add(mojo_ids, mojo_embeddings)
        return Python.none()

    @staticmethod
    def py_query(self_ptr: UnsafePointer[Self, MutAnyOrigin], py_embeddings: PythonObject, n_results: PythonObject) raises -> PythonObject:
        return Self.py_query_batch(self_ptr, py_embeddings, n_results)

    @staticmethod
    def py_query_batch(self_ptr: UnsafePointer[Self, MutAnyOrigin], py_embeddings: PythonObject, n_results: PythonObject) raises -> PythonObject:
        var num_res = Int(py=n_results)
        var mojo_embeddings = List[Float32]()
        for emb in py_embeddings:
            mojo_embeddings.append(Float32(py=emb))
        
        var res = self_ptr[].ptr[].query(mojo_embeddings, num_res)
        
        var out_ids = Python.list()
        var out_dists = Python.list()
        for i in range(len(res.ids)):
            var row_ids = Python.list()
            var row_dists = Python.list()
            for j in range(len(res.ids[i])):
                row_ids.append(res.ids[i][j])
                row_dists.append(res.distances[i][j])
            out_ids.append(row_ids)
            out_dists.append(row_dists)
            
        var dict = Python.dict()
        dict["ids"] = out_ids
        dict["distances"] = out_dists
        return dict
        
    @staticmethod
    def py_upsert_batch_numpy(self_ptr: UnsafePointer[Self, MutAnyOrigin], py_ids: PythonObject, py_embeddings: PythonObject) raises -> PythonObject:
        var num_vectors = Int(py=py_ids.__len__())
        var ids_ptr_int = Int(py=py_ids.__array_interface__["data"][0])
        var emb_ptr_int = Int(py=py_embeddings.__array_interface__["data"][0])
        
        var ids_ptr = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=ids_ptr_int)
        var emb_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=emb_ptr_int)
        
        self_ptr[].ptr[].add_from_pointers(num_vectors, ids_ptr, emb_ptr)
        return Python.none()

    @staticmethod
    def py_query_batch_numpy(self_ptr: UnsafePointer[Self, MutAnyOrigin], py_embeddings: PythonObject, n_results: PythonObject) raises -> PythonObject:
        var num_queries = Int(py=py_embeddings.shape[0])
        var k = Int(py=n_results)
        var emb_ptr_int = Int(py=py_embeddings.__array_interface__["data"][0])
        var emb_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=emb_ptr_int)
        
        var np = Python.import_module("numpy")
        var out_ids = np.empty(Python.tuple(num_queries, k), dtype=np.int64)
        var out_dists = np.empty(Python.tuple(num_queries, k), dtype=np.float32)
        
        var out_ids_int = Int(py=out_ids.__array_interface__["data"][0])
        var out_dists_int = Int(py=out_dists.__array_interface__["data"][0])
        
        var out_ids_ptr = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=out_ids_int)
        var out_dists_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=out_dists_int)
        
        self_ptr[].ptr[].query_from_pointers(num_queries, emb_ptr, k, out_ids_ptr, out_dists_ptr)
        
        return Python.tuple(out_ids, out_dists)
        
    @staticmethod
    def py_compact(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return Python.none()

    @staticmethod
    def py_save(self_ptr: UnsafePointer[Self, MutAnyOrigin], path: PythonObject) raises -> PythonObject:
        self_ptr[].ptr[].save(String(py=path))
        return Python.none()

    @staticmethod
    def py_load(path: PythonObject) raises -> PythonObject:
        var col = Collection.load(String(py=path))
        var col_ptr = rebind[UnsafePointer[Collection, MutAnyOrigin]](alloc[Collection](1))
        col_ptr.init_pointee_move(col^)
        
        var py_col = PyCollection(col_ptr)
        return PythonObject(alloc=py_col^)


@export
def PyInit_mojovec() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojovec")
        _ = (
            m.add_type[PyCollection]("Collection")
            .def_py_init[PyCollection.py_init]()
            .def_method[PyCollection.py_upsert]("upsert")
            .def_method[PyCollection.py_upsert_batch]("upsert_batch")
            .def_method[PyCollection.py_query]("query")
            .def_method[PyCollection.py_query_batch]("query_batch")
            .def_method[PyCollection.py_upsert_batch_numpy]("upsert_batch_numpy")
            .def_method[PyCollection.py_query_batch_numpy]("query_batch_numpy")
            .def_method[PyCollection.py_compact]("compact")
            .def_method[PyCollection.py_save]("save")
        )
        m.def_function[PyCollection.py_load]("load")
        return m.finalize()
    except e:
        print("Mojo Exception:", e)
        abort(String("Failed: ", e))
        return Python.none()
