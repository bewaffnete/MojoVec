from mojovec.index.index_hnsw import IndexHNSW
from mojovec.index.index_flat_sq8 import IndexFlatSQ8
from mojovec.core.types import METRIC_L2
from std.memory import alloc
from std.collections import List
from std.memory.span import Span
from std.python import PythonObject, Python
from .results import QueryResults

struct Collection(Movable, Writable):
    """
    A vector collection that uses HNSW with SQ8 quantization for efficient nearest neighbor search.
    """
    var _dimension: Int
    var _hnsw: IndexHNSW[IndexFlatSQ8]
    var _user_ids: List[Int]
    var _is_deleted: List[UInt8]

    def __init__(out self, dimension: Int, M: Int = 32, ef_construction: Int = 40, ef_search: Int = 16):
        """
        Initializes a new collection with the given parameters.
        """
        self._dimension = dimension
        var storage = IndexFlatSQ8(dimension, METRIC_L2)
        self._hnsw = IndexHNSW[IndexFlatSQ8](storage^, dimension, METRIC_L2, M=M)
        self._hnsw.hnsw.efConstruction = ef_construction
        self._hnsw.hnsw.efSearch = ef_search
        self._user_ids = List[Int]()
        self._is_deleted = List[UInt8]()
        
    def __init__(out self, *, deinit take: Self):
        """
        Takes ownership of an existing collection.
        """
        self._dimension = take._dimension
        self._hnsw = take._hnsw^
        self._user_ids = take._user_ids^
        self._is_deleted = take._is_deleted^

    def write_to[W: Writer](self, mut writer: W):
        """
        Writes a string representation of the collection to the given writer.
        """
        writer.write("Collection(dimension=", self._dimension, ", vectors=", len(self._user_ids), ")")
        
    def __str__(self) -> String:
        """
        Returns a string representation of the collection.
        """
        return String.write(self)

    def save(self, path: String) raises:
        """
        Saves the collection to the specified file path.
        """
        var f = open(path, "w")
        # Save signature: 'COLL' (1129270348)
        from mojovec.io.serialization import write_int, write_index_hnsw_sq8
        write_int(f, 1129270348)
        write_int(f, self._dimension)
        
        var num_ids = len(self._user_ids)
        write_int(f, num_ids)
        if num_ids > 0:
            var ids_ptr = self._user_ids.unsafe_ptr()
            var cast_ptr = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](ids_ptr)
            f.write_bytes(Span[UInt8](ptr=cast_ptr, length=num_ids * 8))
            
            var deleted_ptr = self._is_deleted.unsafe_ptr()
            f.write_bytes(Span[UInt8](ptr=deleted_ptr, length=num_ids))
            
        write_index_hnsw_sq8(f, self._hnsw)
        f.close()

    @staticmethod
    def load(path: String) raises -> Collection:
        """
        Loads a collection from the specified file path.
        """
        var f = open(path, "r")
        from mojovec.io.serialization import read_int, read_index_hnsw_sq8
        var magic = read_int(f)
        if magic != 1129270348:
            raise Error("Invalid magic for Collection")
            
        var dimension = read_int(f)
        var num_ids = read_int(f)
        
        var col = Collection(dimension)
        if num_ids > 0:
            var read_data = f.read_bytes(num_ids * 8)
            var src = read_data.unsafe_ptr().bitcast[Int]()
            for i in range(num_ids):
                col._user_ids.append(src[i])
            _ = len(read_data)
            
            var read_deleted = f.read_bytes(num_ids)
            var del_src = read_deleted.unsafe_ptr()
            for i in range(num_ids):
                col._is_deleted.append(del_src[i])
            _ = len(read_deleted)
                
        var loaded_hnsw = read_index_hnsw_sq8(f)
        col._hnsw = loaded_hnsw^
        f.close()
        return col^

    def add(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        """
        Adds multiple vectors with associated IDs to the collection.
        """
        var num_vectors = len(ids)
        if len(embeddings) != num_vectors * self._dimension:
            raise Error("Embeddings list length must be equal to len(ids) * dimension.")
            
        if num_vectors == 0:
            return

        for id in ids:
            self._user_ids.append(id)
            self._is_deleted.append(0)
            
        var span = Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings))
        self._hnsw.add(span)

    def add_from_pointers(mut self, num_vectors: Int, ids_ptr: UnsafePointer[Int, MutAnyOrigin], embeddings_ptr: UnsafePointer[Float32, MutAnyOrigin]) raises:
        """
        Adds multiple vectors directly from memory pointers (Zero-Copy Buffer Protocol).
        """
        if num_vectors == 0:
            return
            
        for i in range(num_vectors):
            self._user_ids.append(ids_ptr[i])
            self._is_deleted.append(0)
            
        var span = Span[Float32](ptr=embeddings_ptr, length=num_vectors * self._dimension)
        self._hnsw.add(span)

    def set_ef_search(mut self, ef: Int):
        """
        Updates the efSearch parameter for the HNSW index.
        """
        self._hnsw.hnsw.efSearch = ef

    def delete(mut self, ids: List[Int]) raises:
        """
        Soft-deletes vectors by their user IDs.
        """
        # TODO: Optimize with a Dictionary mapping user_id -> internal_id
        for target_id in ids:
            for i in range(len(self._user_ids)):
                if self._user_ids[i] == target_id:
                    self._is_deleted[i] = 1

    def count_deleted(self) -> Int:
        """
        Returns the number of deleted vectors. Useful to monitor degradation.
        """
        var count = 0
        for i in range(len(self._is_deleted)):
            if self._is_deleted[i] > 0:
                count += 1
        return count

    def query(self, query_embeddings: List[Float32], n_results: Int = 10) raises -> QueryResults:
        """
        Queries the collection to find the nearest neighbors for the given embeddings.
        """
        var num_queries = len(query_embeddings) // self._dimension
        if len(query_embeddings) != num_queries * self._dimension:
            raise Error("Query embeddings length must be a multiple of dimension.")
            
        if num_queries == 0:
            return QueryResults(List[List[Int]](), List[List[Float32]]())

        var distances_ptr = alloc[Float32](num_queries * n_results)
        var labels_ptr = alloc[Int](num_queries * n_results)

        var q_span = Span[Float32](ptr=query_embeddings.unsafe_ptr(), length=len(query_embeddings))
        var d_span = Span[mut=True, Float32, _](ptr=distances_ptr, length=num_queries * n_results)
        var l_span = Span[mut=True, Int, _](ptr=labels_ptr, length=num_queries * n_results)
        var f_span = Span[UInt8](ptr=self._is_deleted.unsafe_ptr(), length=len(self._is_deleted))

        self._hnsw.search(q_span, n_results, d_span, l_span, f_span)

        var all_ids = List[List[Int]](capacity=num_queries)
        var all_distances = List[List[Float32]](capacity=num_queries)

        for i in range(num_queries):
            var q_ids = List[Int](capacity=n_results)
            var q_dists = List[Float32](capacity=n_results)
            for j in range(n_results):
                var internal_label = labels_ptr[i * n_results + j]
                var dist = distances_ptr[i * n_results + j]
                if internal_label >= 0 and internal_label < len(self._user_ids):
                    q_ids.append(self._user_ids[internal_label])
                    q_dists.append(dist)
                else:
                    q_ids.append(-1)
                    q_dists.append(dist)
            all_ids.append(q_ids^)
            all_distances.append(q_dists^)

        distances_ptr.free()
        labels_ptr.free()

        return QueryResults(all_ids^, all_distances^)

    def query_from_pointers(self, num_queries: Int, query_ptr: UnsafePointer[Float32, MutAnyOrigin], n_results: Int, out_ids_ptr: UnsafePointer[Int, MutAnyOrigin], out_dists_ptr: UnsafePointer[Float32, MutAnyOrigin]) raises:
        """
        Queries directly from memory pointers and writes results directly to output pointers (Zero-Copy).
        """
        if num_queries == 0:
            return

        var q_span = Span[Float32](ptr=query_ptr, length=num_queries * self._dimension)
        var d_span = Span[mut=True, Float32, _](ptr=out_dists_ptr, length=num_queries * n_results)
        var l_span = Span[mut=True, Int, _](ptr=out_ids_ptr, length=num_queries * n_results)
        var f_span = Span[UInt8](ptr=self._is_deleted.unsafe_ptr(), length=len(self._is_deleted))
        self._hnsw.search(q_span, n_results, d_span, l_span, f_span)

        # Map internal labels to user IDs in place!
        for i in range(num_queries):
            for j in range(n_results):
                var internal_label = out_ids_ptr[i * n_results + j]
                if internal_label >= 0 and internal_label < len(self._user_ids):
                    out_ids_ptr[i * n_results + j] = self._user_ids[internal_label]
                else:
                    out_ids_ptr[i * n_results + j] = -1

    @staticmethod
    def py_init(out self: Collection, args: PythonObject, kwargs: PythonObject) raises:
        """
        Initializes a Collection instance from Python arguments.
        """
        var d = Int(py=args[0])
        var M = 32
        var ef_c = 40
        var ef_s = 16
        if len(args) > 1: M = Int(py=args[1])
        if len(args) > 2: ef_c = Int(py=args[2])
        if len(args) > 3: ef_s = Int(py=args[3])
        self = Self(d, M, ef_c, ef_s)

    @staticmethod
    def py_add(self_ptr: UnsafePointer[Self, MutAnyOrigin], args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
        """
        Adds vectors to the collection using arguments provided from Python.
        """
        var py_ids = args[0]
        var py_embeddings = args[1]
        var mojo_ids = List[Int]()
        for py_id in py_ids:
            mojo_ids.append(Int(py=py_id))
        var mojo_embeddings = List[Float32]()
        for emb in py_embeddings:
            mojo_embeddings.append(Float32(py=emb))
        self_ptr[].add(mojo_ids, mojo_embeddings)
        return Python.none()

    @staticmethod
    def py_query(self_ptr: UnsafePointer[Self, MutAnyOrigin], args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
        """
        Queries the collection using arguments provided from Python.
        """
        var py_embeddings = args[0]
        var n_results = Int(py=args[1])
        var mojo_embeddings = List[Float32]()
        for emb in py_embeddings:
            mojo_embeddings.append(Float32(py=emb))
        
        var res = self_ptr[].query(mojo_embeddings, n_results)
        
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
    def py_save(self_ptr: UnsafePointer[Self, MutAnyOrigin], args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
        """
        Saves the collection using arguments provided from Python.
        """
        var path = String(py=args[0])
        self_ptr[].save(path)
        return Python.none()

    @staticmethod
    def py_load(args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
        """
        Loads a collection using arguments provided from Python.
        """
        var path = String(py=args[0])
        var col = Collection.load(path)
        return PythonObject(alloc=col^)

    @staticmethod
    def py_set_ef_search(self_ptr: UnsafePointer[Self, MutAnyOrigin], args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
        """
        Updates the efSearch parameter using arguments provided from Python.
        """
        var ef = Int(py=args[0])
        self_ptr[].set_ef_search(ef)
        return Python.none()

