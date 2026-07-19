from ..core.index import Index
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT, QuantizerType, QT_8bit, QT_fp16
from ..core.quantizer import ScalarQuantizer
from ..utils.distances import l2_distance_simd, inner_product_simd
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from std.sys.intrinsics import prefetch, PrefetchOptions
from std.memory.span import Span

struct SQDistanceComputer(DistanceComputerTrait):
    """Computes distances between a query vector and scalar quantized database vectors."""
    var d: Int
    var code_size: Int
    var metric_type: MetricType
    var sq: ScalarQuantizer
    var codes: UnsafePointer[UInt8, MutUntrackedOrigin]
    var query: UnsafePointer[Float32, MutUntrackedOrigin]
    var scratch_x: UnsafePointer[Float32, MutUntrackedOrigin]
    # Second scratch buffer for symmetric_distance(i, j): decodes j while scratch_x holds i.
    var scratch_y: UnsafePointer[Float32, MutUntrackedOrigin]

    def __init__(out self, d: Int, code_size: Int, metric_type: MetricType, sq: ScalarQuantizer, codes: UnsafePointer[UInt8, MutUntrackedOrigin], query: UnsafePointer[Float32, MutUntrackedOrigin]):
        """Initializes the distance computer.
        
        Args:
            d: The dimensionality of the vectors.
            code_size: The byte size of a quantized code.
            metric_type: The metric type used for distance computation.
            sq: The scalar quantizer instance used for decoding.
            codes: A pointer to the quantized database codes.
            query: A pointer to the uncompressed query vector.
        """
        self.d = d
        self.code_size = code_size
        self.metric_type = metric_type
        self.sq = sq.copy()
        self.codes = codes
        self.query = query
        self.scratch_x = alloc[Float32](self.d)
        self.scratch_y = alloc[Float32](self.d)

    def __init__(out self, *, deinit move: Self):
        """Moves the distance computer from another instance.
        
        Args:
            move: The instance to move from.
        """
        self.d = move.d
        self.code_size = move.code_size
        self.metric_type = move.metric_type
        self.sq = move.sq^
        self.codes = move.codes
        self.query = move.query
        self.scratch_x = move.scratch_x
        self.scratch_y = move.scratch_y

    def __del__(deinit self):
        """Frees the allocated memory for the scratch buffers."""
        # In Mojo, we can't check pointer truthiness, so we just free if address is not 0
        if Int(self.scratch_x) != 0:
            self.scratch_x.free()
        if Int(self.scratch_y) != 0:
            self.scratch_y.free()
        
    @always_inline
    def distance(self, id: Int, threshold: Float32 = Float32.MAX) -> Float32:
        """Computes the distance between the query and a specified database vector.
        
        Args:
            id: The index of the database vector.
            threshold: An optional threshold for early termination.
            
        Returns:
            The computed approximate distance.
        """
        var db_ptr = self.codes + (id * self.code_size)
        self.sq.decode(db_ptr, self.scratch_x)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[4](self.query, self.scratch_x, self.d)
        else:
            return -inner_product_simd[4](self.query, self.scratch_x, self.d)

    @always_inline
    def symmetric_distance(self, i: Int, j: Int) -> Float32:
        """Computes the distance between two database vectors.
        
        Args:
            i: The index of the first database vector.
            j: The index of the second database vector.
            
        Returns:
            The computed symmetric distance.
        """
        # Decode both codes into separate scratch buffers — scratch_x holds i,
        # scratch_y holds j. Sharing one buffer would overwrite i before the
        # distance is computed. Mirrors FlatDistanceComputer's metric handling.
        self.sq.decode(self.codes + (i * self.code_size), self.scratch_x)
        self.sq.decode(self.codes + (j * self.code_size), self.scratch_y)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[4](self.scratch_x, self.scratch_y, self.d)
        else:
            return -inner_product_simd[4](self.scratch_x, self.scratch_y, self.d)

    @always_inline
    def prefetch_vector(self, id: Int):
        """Prefetch quantized codes for `id` into CPU cache.
        
        For SQ, prefetching the encoded bytes ahead of decode+distance
        hides memory latency during HNSW neighbor traversal.
        
        Args:
            id: The index of the vector to prefetch.
        """
        var ptr = self.codes + (id * self.code_size)
        comptime opts = PrefetchOptions().for_read().medium_locality().to_data_cache()
        prefetch[opts](ptr)

    @always_inline
    def is_exact(self) -> Bool:
        """Indicates whether this computer provides exact distances.
        
        Returns:
            False, since scalar quantization provides approximate distances.
        """
        return False

struct IndexScalarQuantizer(Index, StorageTrait):
    """An index that uses a scalar quantizer to compress vectors and accelerate search."""
    comptime ComputerType = SQDistanceComputer
    var d: Int
    var ntotal: Int
    var metric_type: MetricType
    var is_trained: Bool
    
    var sq: ScalarQuantizer
    var code_size: Int
    
    # Pointer to the raw byte data
    var codes: UnsafePointer[UInt8, MutUntrackedOrigin]
    # Capacity allocated for codes (in vectors)
    var capacity: Int
    # Reusable scratch buffer for get_vector() — decodes a code into Float32 space.
    # Persistent allocation avoids per-call alloc/free in HNSW build paths.
    var scratch_x: UnsafePointer[Float32, MutUntrackedOrigin]

    def __init__(out self, d: Int, qtype: QuantizerType, metric: MetricType = METRIC_L2):
        """Initializes the scalar quantizer index.
        
        Args:
            d: The dimensionality of the vectors.
            qtype: The type of scalar quantizer (e.g., QT_8bit or QT_fp16).
            metric: The metric type used for distance computation.
        """
        self.d = d
        self.ntotal = 0
        self.metric_type = metric

        self.sq = ScalarQuantizer(d, qtype)
        self.code_size = self.sq.code_size()
        self.is_trained = self.sq.is_trained

        self.capacity = 1024  # Initial capacity for 1024 vectors
        self.codes = alloc[UInt8](self.capacity * self.code_size)
        self.scratch_x = alloc[Float32](self.d)

    def __del__(deinit self):
        """Frees the allocated memory for the index."""
        self.scratch_x.free()
        if Int(self.codes) != 0:
            self.codes.free()
        
    def train(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        """Trains the scalar quantizer on a representative dataset.
        
        Args:
            n: The number of training vectors.
            x: A pointer to the training vectors.
        """
        self.sq.train(n, x)
        self.is_trained = self.sq.is_trained

    def add(mut self, x: Span[Float32, _]):
        """Quantizes and adds new vectors to the index.
        
        Args:
            x: A safe Span pointing to the uncompressed vectors to add.
        """
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        if not self.is_trained:
            return # Should raise error
            
        var new_ntotal = self.ntotal + n
        if new_ntotal > self.capacity:
            var new_capacity = max(self.capacity * 2, new_ntotal)
            var new_codes = alloc[UInt8](new_capacity * self.code_size)
            if self.ntotal > 0:
                for i in range(self.ntotal * self.code_size):
                    new_codes[i] = self.codes[i]
            if Int(self.codes) != 0:
                self.codes.free()
            self.codes = new_codes
            self.capacity = new_capacity
            
        var offset_vectors = self.ntotal
        for i in range(n):
            var x_curr = x_ptr + (i * self.d)
            var code_ptr = self.codes + ((offset_vectors + i) * self.code_size)
            self.sq.encode(x_curr, code_ptr)
            
        self.ntotal = new_ntotal
        
    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        """Decodes and retrieves a specific vector in the index.
        
        Args:
            id: The index of the vector to retrieve.
            
        Returns:
            A pointer to the decoded, uncompressed vector.
        """
        self.sq.decode(self.codes + (id * self.code_size), self.scratch_x)
        return self.scratch_x

    def search(
        self,
        x: Span[Float32, _],
        k: Int,
        mut distances: Span[mut=True, Float32, _],
        mut labels: Span[mut=True, Int, _],
    ):
        var empty_filter = Span[UInt8, _](ptr=alloc[UInt8](0), length=0)
        self.search(x, k, distances, labels, empty_filter)

    def search(
        self,
        x: Span[Float32, _],
        k: Int,
        mut distances: Span[mut=True, Float32, _],
        mut labels: Span[mut=True, Int, _],
        filter: Span[UInt8, _],
    ):
        """Searches for the k-nearest neighbors of the given query vectors.
        
        Args:
            x: A safe Span pointing to the uncompressed query vectors.
            k: The number of nearest neighbors to retrieve.
            distances: An output Span for distances.
            labels: An output Span for labels.
            filter: A bitset indicating allowed vectors.
        """
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        var distances_ptr = distances.unsafe_ptr()
        var labels_ptr = labels.unsafe_ptr()
        var scratch_x = alloc[Float32](self.d)
        
        for i in range(n):
            var query_ptr = x_ptr + (i * self.d)
            var res_dist_ptr = distances_ptr + (i * k)
            var res_labels_ptr = labels_ptr + (i * k)
            var heap_size = 0
            
            for j in range(self.ntotal):
                var db_ptr = self.codes + (j * self.code_size)
                self.sq.decode(db_ptr, scratch_x)
                
                var dist: Float32
                if self.metric_type == METRIC_L2:
                    dist = l2_distance_simd[4](query_ptr, scratch_x, self.d)
                else:
                    dist = -inner_product_simd[4](query_ptr, scratch_x, self.d)
                    
                if heap_size < k:
                    max_heap_push(res_dist_ptr, res_labels_ptr, heap_size, dist, j)
                    heap_size += 1
                elif dist < res_dist_ptr[0]:
                    max_heap_replace_top(res_dist_ptr, res_labels_ptr, k, dist, j)
                    
            var current_k = heap_size
            for j in range(current_k):
                var popped = max_heap_pop(res_dist_ptr, res_labels_ptr, heap_size)
                heap_size -= 1
                var idx = current_k - 1 - j
                res_dist_ptr[idx] = popped.dist
                res_labels_ptr[idx] = popped.label
                    
            if self.metric_type == METRIC_INNER_PRODUCT:
                for j in range(k):
                    res_dist_ptr[j] = -res_dist_ptr[j]
                    
        scratch_x.free()
        
    def get_distance_computer(self, query: UnsafePointer[Float32, _]) -> Self.ComputerType:
        """Creates a distance computer for the given query vector.
        
        Args:
            query: A pointer to the query vector.
            
        Returns:
            An instance of the associated distance computer.
        """
        var q_ptr = rebind[UnsafePointer[Float32, MutUntrackedOrigin]](query)
        return SQDistanceComputer(self.d, self.code_size, self.metric_type, self.sq.copy(), self.codes, q_ptr)
