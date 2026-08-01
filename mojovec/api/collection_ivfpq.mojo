from mojovec.index.index_ivf_pq import IndexIVFPQ
from mojovec.index.index_flat import IndexFlat
from mojovec.core.types import METRIC_L2
from std.memory import OwnedPointer
from std.collections import List
from std.memory.span import Span
from mojovec.api.metadata import Metadata
from .results import QueryResults

struct CollectionIVFPQ(Movable):
    """
    A vector collection using IVF-PQ index for extreme compression and fast search.
    """
    var _dimension: Int
    var _ivfpq: OwnedPointer[IndexIVFPQ[IndexFlat]]
    var _user_ids: List[Int]
    var _flat_quantizer: OwnedPointer[IndexFlat]

    def __init__(
        out self, dimension: Int, nlist: Int = 100, M: Int = 16
    ) raises:
        """
        Initializes a new IVF-PQ collection with the given parameters.
        """
        self._dimension = dimension
        
        self._flat_quantizer = OwnedPointer(
            IndexFlat(dimension, METRIC_L2)
        )
        self._ivfpq = OwnedPointer(
            IndexIVFPQ[IndexFlat](
                rebind[
                    UnsafePointer[IndexFlat, MutUntrackedOrigin]
                ](self._flat_quantizer.unsafe_ptr()),
                dimension,
                nlist,
                M,
            )
        )
        self._ivfpq[].nprobe = 10
        self._user_ids = List[Int]()
        
    def __init__(out self, *, deinit take: Self):
        """
        Takes ownership of an existing IVF-PQ collection.
        """
        self._dimension = take._dimension
        self._ivfpq = take._ivfpq^
        self._flat_quantizer = take._flat_quantizer^
        self._user_ids = take._user_ids^

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
            f.write_bytes(
                Span[UInt8, _](
                    ptr=ids_ptr.bitcast[UInt8](),
                    length=num_ids * 8,
                )
            )
            
        write_index_ivf_pq(f, self._ivfpq[])
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
        var loaded_quantizer = loaded_ivfpq.quantizer.take_pointee()
        loaded_ivfpq.quantizer.free()
        col._flat_quantizer[] = loaded_quantizer^
        loaded_ivfpq.quantizer = rebind[
            UnsafePointer[IndexFlat, MutUntrackedOrigin]
        ](col._flat_quantizer.unsafe_ptr())
        col._ivfpq = OwnedPointer(loaded_ivfpq^)
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
            
        self._ivfpq[].train(
            Span[Float32](
                ptr=embeddings.unsafe_ptr(),
                length=len(embeddings),
            )
        )

    def add(mut self, ids: List[Int], embeddings: List[Float32]) raises:
        """
        Adds vectors to the collection, training the index automatically if necessary.
        """
        var num_vectors = len(ids)
        if len(embeddings) != num_vectors * self._dimension:
            raise Error("Embeddings list length must be equal to len(ids) * dimension.")
            
        if num_vectors == 0:
            return
            
        if not self._ivfpq[].is_trained:
            # Auto-train if not trained
            self.train(embeddings)

        for id in ids:
            self._user_ids.append(id)
            
        var span = Span[Float32](ptr=embeddings.unsafe_ptr(), length=len(embeddings))
        self._ivfpq[].add(span)

    def query(self, query_embeddings: List[Float32], n_results: Int = 10) raises -> QueryResults:
        """
        Queries the collection to find the nearest neighbors for the given embeddings.
        """
        var num_queries = len(query_embeddings) // self._dimension
        if len(query_embeddings) != num_queries * self._dimension:
            raise Error("Query embeddings length must be a multiple of dimension.")
            
        if num_queries == 0:
            return QueryResults(
                List[List[Int]](),
                List[List[Float32]](),
                List[List[Metadata]](),
                List[List[String]](),
                List[List[Float32]](),
            )

        var output_size = num_queries * n_results
        var distances_storage = List[Float32](
            unsafe_uninit_length=output_size
        )
        var labels_storage = List[Int](unsafe_uninit_length=output_size)

        var q_span = Span[Float32](ptr=query_embeddings.unsafe_ptr(), length=len(query_embeddings))
        var d_span = Span[mut=True, Float32](distances_storage)
        var l_span = Span[mut=True, Int](labels_storage)
        self._ivfpq[].search(q_span, n_results, d_span, l_span)

        var all_ids = List[List[Int]](capacity=num_queries)
        var all_distances = List[List[Float32]](capacity=num_queries)

        for i in range(num_queries):
            var q_ids = List[Int](capacity=n_results)
            var q_dists = List[Float32](capacity=n_results)
            for j in range(n_results):
                var internal_label = labels_storage[i * n_results + j]
                var dist = distances_storage[i * n_results + j]
                if internal_label >= 0 and internal_label < len(self._user_ids):
                    q_ids.append(self._user_ids[internal_label])
                    q_dists.append(dist)
                else:
                    q_ids.append(-1)
                    q_dists.append(dist)
            all_ids.append(q_ids^)
            all_distances.append(q_dists^)

        return QueryResults(
            all_ids^,
            all_distances^,
            List[List[Metadata]](),
            List[List[String]](),
            List[List[Float32]](),
        )
