from mojovec.index.index_ivf_pq import IndexIVFPQ
from mojovec.index.index_flat import IndexFlat
from mojovec.core.types import METRIC_L2
from std.memory import alloc
from std.collections import List
from std.memory.span import Span
from .results import QueryResults

struct CollectionIVFPQ(Movable):
    """
    A vector collection using IVF-PQ index for extreme compression and fast search.
    """
    var _dimension: Int
    var _ivfpq_ptr: UnsafePointer[IndexIVFPQ[IndexFlat], MutUntrackedOrigin]
    var _user_ids: List[Int]
    var _flat_quantizer: UnsafePointer[IndexFlat, MutUntrackedOrigin]

    def __init__(out self, dimension: Int, nlist: Int = 100, M: Int = 16):
        """
        Initializes a new IVF-PQ collection with the given parameters.
        """
        self._dimension = dimension
        
        self._flat_quantizer = alloc[IndexFlat](1)
        self._flat_quantizer.init_pointee_move(IndexFlat(dimension, METRIC_L2))
        
        self._ivfpq_ptr = alloc[IndexIVFPQ[IndexFlat]](1)
        self._ivfpq_ptr.init_pointee_move(IndexIVFPQ[IndexFlat](self._flat_quantizer, dimension, nlist, M))
        self._ivfpq_ptr[].nprobe = 10
        self._user_ids = List[Int]()
        
    def __init__(out self, *, deinit take: Self):
        """
        Takes ownership of an existing IVF-PQ collection.
        """
        self._dimension = take._dimension
        self._ivfpq_ptr = take._ivfpq_ptr
        self._flat_quantizer = take._flat_quantizer
        self._user_ids = take._user_ids^

    def __del__(deinit self):
        """
        Frees the allocated memory for the quantizer and index.
        """
        self._ivfpq_ptr.destroy_pointee()
        self._ivfpq_ptr.free()
        self._flat_quantizer.destroy_pointee()
        self._flat_quantizer.free()

    def save(self, path: String) raises:
        """
        Saves the collection to the specified file path.
        """
        var f = open(path, "w")
        # Save signature: 'CIVF' (1128879686)
        from mojovec.io.serialization import write_int, write_index_ivf_pq
        write_int(f, 1128879686)
        write_int(f, self._dimension)
        
        var num_ids = len(self._user_ids)
        write_int(f, num_ids)
        if num_ids > 0:
            var ids_ptr = self._user_ids.unsafe_ptr()
            var cast_ptr = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](ids_ptr)
            f.write_bytes(Span[UInt8, MutUntrackedOrigin](ptr=cast_ptr, length=num_ids * 8))
            
        write_index_ivf_pq(f, self._ivfpq_ptr[])
        f.close()

    @staticmethod
    def load(path: String) raises -> CollectionIVFPQ:
        """
        Loads a collection from the specified file path.
        """
        var f = open(path, "r")
        from mojovec.io.serialization import read_int, read_index_ivf_pq
        var magic = read_int(f)
        if magic != 1128879686:
            raise Error("Invalid magic for CollectionIVFPQ")
            
        var dimension = read_int(f)
        var num_ids = read_int(f)
        
        # Create a dummy collection to be overwritten.
        # read_index_ivf_pq returns an initialized IndexIVFPQ.
        var col = CollectionIVFPQ(dimension, 1, 1) # dummy values
        
        if num_ids > 0:
            var read_data = f.read_bytes(num_ids * 8)
            var src = read_data.unsafe_ptr().bitcast[Int]()
            for i in range(num_ids):
                col._user_ids.append(src[i])
            _ = len(read_data)
                
        var loaded_ivfpq = read_index_ivf_pq(f)
        col._ivfpq_ptr.destroy_pointee()
        col._ivfpq_ptr.init_pointee_move(loaded_ivfpq^)
        f.close()
        return col^

    def train(mut self, embeddings: List[Float32]) raises:
        """
        Trains the IVF-PQ index using the provided embeddings.
        """
        var num_vectors = len(embeddings) // self._dimension
        if len(embeddings) != num_vectors * self._dimension:
            raise Error("Embeddings list length must be a multiple of dimension.")
            
        if num_vectors == 0:
            return
            
        var ptr = rebind[UnsafePointer[Float32, MutUntrackedOrigin]](embeddings.unsafe_ptr())
        self._ivfpq_ptr[].train(num_vectors, ptr)

    def add(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        """
        Adds vectors to the collection, training the index automatically if necessary.
        """
        var num_vectors = len(ids)
        if len(embeddings) != num_vectors * self._dimension:
            raise Error("Embeddings list length must be equal to len(ids) * dimension.")
            
        if num_vectors == 0:
            return
            
        if not self._ivfpq_ptr[].is_trained:
            # Auto-train if not trained
            self.train(embeddings)

        for id in ids:
            self._user_ids.append(id)
            
        var ptr = rebind[UnsafePointer[Float32, MutUntrackedOrigin]](embeddings.unsafe_ptr())
        self._ivfpq_ptr[].add(num_vectors, ptr)

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

        var ptr = rebind[UnsafePointer[Float32, MutUntrackedOrigin]](query_embeddings.unsafe_ptr())
        self._ivfpq_ptr[].search(num_queries, ptr, n_results, distances_ptr, labels_ptr)

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
