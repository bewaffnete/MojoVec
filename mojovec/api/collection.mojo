from mojovec.index.index_hnsw import IndexHNSW
from mojovec.index.index_flat import IndexFlat
from mojovec.core.types import METRIC_L2
from std.memory import alloc
from std.collections import List
from std.memory.span import Span
from .results import QueryResults

struct Collection(Movable):
    var _dimension: Int
    var _hnsw: IndexHNSW[IndexFlat]
    var _user_ids: List[Int]

    def __init__(out self, dimension: Int):
        self._dimension = dimension
        var storage = IndexFlat(dimension, METRIC_L2)
        self._hnsw = IndexHNSW[IndexFlat](storage^, dimension, METRIC_L2, M=32)
        self._hnsw.hnsw.efConstruction = 200
        self._hnsw.hnsw.efSearch = 40
        self._user_ids = List[Int]()
        
    def __init__(out self, *, deinit take: Self):
        self._dimension = take._dimension
        self._hnsw = take._hnsw^
        self._user_ids = take._user_ids^

    def save(self, path: String) raises:
        var f = open(path, "w")
        # Save signature: 'COLL' (1129270348)
        from mojovec.io.serialization import write_int, write_index_hnsw
        write_int(f, 1129270348)
        write_int(f, self._dimension)
        
        var num_ids = len(self._user_ids)
        write_int(f, num_ids)
        if num_ids > 0:
            var ids_ptr = self._user_ids.unsafe_ptr()
            var cast_ptr = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](ids_ptr)
            f.write_bytes(Span[UInt8, MutUntrackedOrigin](ptr=cast_ptr, length=num_ids * 8))
            
        write_index_hnsw(f, self._hnsw)
        f.close()

    @staticmethod
    def load(path: String) raises -> Collection:
        var f = open(path, "r")
        from mojovec.io.serialization import read_int, read_index_hnsw
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
                
        var loaded_hnsw = read_index_hnsw(f)
        col._hnsw = loaded_hnsw^
        f.close()
        return col^

    def add(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        var num_vectors = len(ids)
        if len(embeddings) != num_vectors * self._dimension:
            raise Error("Embeddings list length must be equal to len(ids) * dimension.")
            
        if num_vectors == 0:
            return

        for id in ids:
            self._user_ids.append(id)
            
        var ptr = rebind[UnsafePointer[Float32, MutUntrackedOrigin]](embeddings.unsafe_ptr())
        self._hnsw.add(num_vectors, ptr)

    def query(self, query_embeddings: List[Float32], n_results: Int = 10) raises -> QueryResults:
        var num_queries = len(query_embeddings) // self._dimension
        if len(query_embeddings) != num_queries * self._dimension:
            raise Error("Query embeddings length must be a multiple of dimension.")
            
        if num_queries == 0:
            return QueryResults(List[List[Int]](), List[List[Float32]]())

        var distances_ptr = alloc[Float32](num_queries * n_results)
        var labels_ptr = alloc[Int](num_queries * n_results)

        var ptr = rebind[UnsafePointer[Float32, MutUntrackedOrigin]](query_embeddings.unsafe_ptr())
        self._hnsw.search(num_queries, ptr, n_results, distances_ptr, labels_ptr)

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
