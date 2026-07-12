from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT

# Hardware-optimized for Apple Silicon (ARM NEON)
# While NEON physical width is 4 (128-bit), we unroll by a larger multiple 
# for maximum instruction-level parallelism.
comptime SIMD_WIDTH = 64

from ..utils.distances import l2_distance_simd, inner_product_simd
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from std.sys.intrinsics import prefetch, PrefetchOptions

struct FlatDistanceComputer(DistanceComputerTrait):
    """Computes distances between a query vector and flattened database vectors.
    
    This distance computer operates on uncompressed `Float32` vectors.
    """
    var d: Int
    var metric_type: MetricType
    var codes: UnsafePointer[Float32, MutUntrackedOrigin]
    var query: UnsafePointer[Float32, MutUntrackedOrigin]
    
    def __init__(out self, d: Int, metric_type: MetricType, codes: UnsafePointer[Float32, MutUntrackedOrigin], query: UnsafePointer[Float32, MutUntrackedOrigin]):
        """Initializes the distance computer.
        
        Args:
            d: The dimensionality of the vectors.
            metric_type: The metric type used for distance computation (e.g., L2 or Inner Product).
            codes: A pointer to the flattened database vectors.
            query: A pointer to the query vector.
        """
        self.d = d
        self.metric_type = metric_type
        self.codes = codes
        self.query = query
        
    @always_inline
    def distance(self, id: Int, threshold: Float32 = Float32.MAX) -> Float32:
        """Computes the distance between the query and a specified database vector.
        
        Args:
            id: The index of the database vector.
            threshold: An optional threshold for early termination (not used in flat index).
            
        Returns:
            The computed distance.
        """
        var db_ptr = self.codes + (id * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[SIMD_WIDTH](self.query, db_ptr, self.d)
        else:
            return -inner_product_simd[SIMD_WIDTH](self.query, db_ptr, self.d)
            
    @always_inline
    def symmetric_distance(self, i: Int, j: Int) -> Float32:
        """Computes the distance between two database vectors.
        
        Args:
            i: The index of the first database vector.
            j: The index of the second database vector.
            
        Returns:
            The computed symmetric distance.
        """
        var ptr_i = self.codes + (i * self.d)
        var ptr_j = self.codes + (j * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[SIMD_WIDTH](ptr_i, ptr_j, self.d)
        else:
            return -inner_product_simd[SIMD_WIDTH](ptr_i, ptr_j, self.d)

    @always_inline
    def prefetch_vector(self, id: Int):
        """Prefetch vector data for `id` into CPU cache (L1, read intent).
        
        This is called ahead of distance() to hide memory latency:
        while the CPU computes distance for the current neighbor,
        the next neighbor's vector data is being loaded into cache.
        
        Args:
            id: The index of the vector to prefetch.
        """
        var ptr = self.codes + (id * self.d)
        comptime opts = PrefetchOptions().for_read().low_locality().to_data_cache()
        prefetch[opts](ptr)

    @always_inline
    def is_exact(self) -> Bool:
        """Indicates whether this computer provides exact distances.
        
        Returns:
            True, since flat index computes exact distances.
        """
        return True

struct IndexFlat(Index, StorageTrait, QuantizerTrait, Movable):
    """An exact search index that stores raw, uncompressed vectors."""
    comptime ComputerType = FlatDistanceComputer
    var d: Int
    var ntotal: Int
    var metric_type: MetricType
    # Pointer to the raw flattened data
    var codes: UnsafePointer[Float32, MutUntrackedOrigin]
    # Capacity allocated for codes
    var capacity: Int

    def __init__(out self, d: Int, metric: MetricType = METRIC_L2):
        """Initializes the flat index.
        
        Args:
            d: The dimensionality of the vectors.
            metric: The metric type used for distance computation.
        """
        self.d = d
        self.ntotal = 0
        self.metric_type = metric
        self.capacity = 1024 * d  # Initial capacity for 1024 vectors
        self.capacity = 1024  # Initial capacity for 1024 vectors
        self.codes = alloc[Float32](self.capacity * d)

    def __del__(deinit self):
        """Frees the allocated memory for the index."""
        if Int(self.codes) != 0:
            self.codes.free()

    def __init__(out self, *, deinit move: Self):
        """Moves the index from another instance.
        
        Args:
            move: The instance to move from.
        """
        self.d = move.d
        self.ntotal = move.ntotal
        self.metric_type = move.metric_type
        self.capacity = move.capacity
        self.codes = move.codes

    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        """Adds new vectors to the index.
        
        Args:
            n: The number of vectors to add.
            x: A pointer to the flattened vectors to add.
        """
        if n == 0:
            return
            
        var new_ntotal = self.ntotal + n
        if new_ntotal > self.capacity:
            var new_capacity = max(self.capacity * 2, new_ntotal)
            var new_codes = alloc[Float32](new_capacity * self.d)
            if self.ntotal > 0:
                for i in range(self.ntotal * self.d):
                    new_codes[i] = self.codes[i]
            self.codes.free()
            self.codes = new_codes
            self.capacity = new_capacity
            
        var offset = self.ntotal * self.d
        for i in range(n * self.d):
            self.codes[offset + i] = x[i]
        self.ntotal = new_ntotal
        
    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        """Retrieves a pointer to a specific vector in the index.
        
        Args:
            id: The index of the vector to retrieve.
            
        Returns:
            A pointer to the requested vector.
        """
        return self.codes + (id * self.d)

    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]):
        """Searches for the k-nearest neighbors of the given query vectors.
        
        Args:
            n: The number of query vectors.
            x: A pointer to the flattened query vectors.
            k: The number of nearest neighbors to retrieve.
            distances: A pointer to the output distances array.
            labels: A pointer to the output labels array.
        """
        var self_codes = self.codes
        var self_d = self.d
        var self_ntotal = self.ntotal
        var self_metric_type = self.metric_type
        
        from std.algorithm import parallelize
        
        def process_query(i: Int) {self_d, self_codes, self_ntotal, self_metric_type, x, k, distances, labels}:
            var query_offset = i * self_d
            var query_ptr = x + query_offset
            
            var res_dist_ptr = distances + (i * k)
            var res_labels_ptr = labels + (i * k)
            var heap_size = 0
            
            # Iterate over all database vectors
            for j in range(self_ntotal):
                var db_ptr = self_codes + (j * self_d)
                var dist: Float32
                
                # Compute distance based on metric
                if self_metric_type == METRIC_L2:
                    dist = l2_distance_simd[SIMD_WIDTH](query_ptr, db_ptr, self_d)
                else:
                    dist = -inner_product_simd[SIMD_WIDTH](query_ptr, db_ptr, self_d)
                    
                # Add to heap
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
            
            # Un-negate inner product distances
            if self_metric_type == METRIC_INNER_PRODUCT:
                for j in range(k):
                    res_dist_ptr[j] = -res_dist_ptr[j]
                    
        parallelize(process_query, n, n)
                    
    def get_distance_computer(self, query: UnsafePointer[Float32, MutUntrackedOrigin]) -> Self.ComputerType:
        """Creates a distance computer for the given query vector.
        
        Args:
            query: A pointer to the query vector.
            
        Returns:
            An instance of the associated distance computer.
        """
        return FlatDistanceComputer(self.d, self.metric_type, self.codes, query)
