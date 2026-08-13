from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from std.memory import memcpy, alloc
from std.memory.span import Span

# Hardware-optimized for Apple Silicon (ARM NEON)
# While NEON physical width is 4 (128-bit), we unroll by a larger multiple 
# for maximum instruction-level parallelism.
comptime SIMD_WIDTH = 8

from ..utils.distances import (
    l2_distance_simd,
    inner_product_simd,
    l2_distance_simd_batch4,
    inner_product_simd_batch4,
)
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from std.sys.intrinsics import prefetch, PrefetchOptions
from std.ffi import external_call
from mojovec.io.memory_map import FileMemoryMap
from mojovec.core.validation import _validate_vector_dimension

@always_inline
def _alloc_aligned(count: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
    var bytes = count * 4
    var aligned_bytes = (bytes + 63) // 64 * 64
    var ptr_int = external_call["aligned_alloc", Int](64, aligned_bytes)
    return UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=ptr_int)

@always_inline
def _free_aligned(ptr: UnsafePointer[Float32, MutUntrackedOrigin]):
    if Int(ptr) != 0:
        external_call["free", NoneType](Int(ptr))

struct FlatDistanceComputer(DistanceComputerTrait):
    """Computes distances between a query vector and flattened database vectors.
    
    This distance computer operates on uncompressed `Float32` vectors.
    """
    var d: Int
    var metric_type: MetricType
    var codes: UnsafePointer[Float32, MutUntrackedOrigin]
    var query: UnsafePointer[Float32, MutUntrackedOrigin]
    
    def __init__(
        out self,
        d: Int,
        metric_type: MetricType,
        codes: UnsafePointer[Float32, MutUntrackedOrigin],
        query: UnsafePointer[Float32, MutUntrackedOrigin],
    ):
        """Initializes the distance computer."""
        self.d = d
        self.metric_type = metric_type
        self.codes = codes
        self.query = query
        
    @always_inline
    def distance(self, id: Int, threshold: Float32 = Float32.MAX) -> Float32:
        """Computes the distance between the query and a specified database vector."""
        var db_ptr = self.codes + (id * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[SIMD_WIDTH](
                self.query, db_ptr, self.d, threshold
            )
        else:
            return -inner_product_simd[SIMD_WIDTH](self.query, db_ptr, self.d)
            
    @always_inline
    def distance_batch4(self, id0: Int, id1: Int, id2: Int, id3: Int) -> InlineArray[Float32, 4]:
        """Computes distances between query and 4 database vectors simultaneously."""
        var ptr0 = self.codes + (id0 * self.d)
        var ptr1 = self.codes + (id1 * self.d)
        var ptr2 = self.codes + (id2 * self.d)
        var ptr3 = self.codes + (id3 * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd_batch4[SIMD_WIDTH](
                ptr0, ptr1, ptr2, ptr3, self.query, self.d
            )
        else:
            var res = inner_product_simd_batch4[SIMD_WIDTH](
                ptr0, ptr1, ptr2, ptr3, self.query, self.d
            )
            res[0] = -res[0]
            res[1] = -res[1]
            res[2] = -res[2]
            res[3] = -res[3]
            return res

    @always_inline
    def symmetric_distance(self, i: Int, j: Int) -> Float32:
        """Computes the distance between two database vectors."""
        var ptr_i = self.codes + (i * self.d)
        var ptr_j = self.codes + (j * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[SIMD_WIDTH](ptr_i, ptr_j, self.d)
        else:
            return -inner_product_simd[SIMD_WIDTH](ptr_i, ptr_j, self.d)

    @always_inline
    def prefetch_vector(self, id: Int):
        """Prefetch the first cache line without flooding the cache."""
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
    comptime HNSW_PTHREAD_STORAGE_KIND = 0
    var d: Int
    var ntotal: Int
    var metric_type: MetricType
    # Pointer to the raw flattened data
    var codes: UnsafePointer[Float32, MutUntrackedOrigin]
    # Capacity allocated for codes
    var capacity: Int
    var _mapping: FileMemoryMap

    def __init__(
        out self,
        d: Int,
        metric: MetricType = METRIC_L2,
        initial_capacity: Int = 1024,
    ) raises:
        """Initializes the flat index.

        Args:
            d: The dimensionality of the vectors.
            metric: The metric type used for distance computation.
            initial_capacity: Initial owned allocation; loaders use zero.
        """
        _validate_vector_dimension(d)
        if initial_capacity < 0 or initial_capacity > 1024:
            raise Error("initial_capacity must be between 0 and 1024.")
        self.d = d
        self.ntotal = 0
        self.metric_type = metric
        self.capacity = initial_capacity
        self.codes = _alloc_aligned(max(self.capacity * d, 1))
        self._mapping = FileMemoryMap()

    def __del__(deinit self):
        """Frees the allocated memory for the index."""
        if not self._mapping.is_active():
            _free_aligned(self.codes)
        self._mapping.close()

    def __init__(out self, *, deinit move: Self):
        """Moves the index from another instance."""
        self.d = move.d
        self.ntotal = move.ntotal
        self.metric_type = move.metric_type
        self.capacity = move.capacity
        self.codes = move.codes
        self._mapping = move._mapping^

    @always_inline
    def is_memory_mapped(self) -> Bool:
        return self._mapping.is_active()

    def _detach_mapped(mut self):
        if not self._mapping.is_active():
            return
        var new_codes = _alloc_aligned(max(self.capacity * self.d, 1))
        if self.ntotal > 0:
            memcpy(
                dest=new_codes,
                src=self.codes,
                count=self.ntotal * self.d,
            )
        self._mapping.close()
        self.codes = new_codes

    def add(mut self, x: Span[Float32, _]):
        """Adds new vectors to the index.
        
        Args:
            x: A safe Span pointing to the flattened vectors to add.
        """
        var n = len(x) // self.d
        if n == 0:
            return
        self._detach_mapped()
            
        var x_ptr = x.unsafe_ptr()
            
        var new_ntotal = self.ntotal + n
        if new_ntotal > self.capacity:
            var new_capacity = max(self.capacity * 2, new_ntotal)
            var new_codes = _alloc_aligned(new_capacity * self.d)
            if self.ntotal > 0:
                memcpy(dest=new_codes, src=self.codes, count=self.ntotal * self.d)
            _free_aligned(self.codes)
            self.codes = new_codes
            self.capacity = new_capacity
            
        var offset = self.ntotal * self.d
        memcpy(dest=self.codes + offset, src=x_ptr, count=n * self.d)
        self.ntotal = new_ntotal
        
    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        """Returns a raw vector pointer for the internal HNSW hot path."""
        return self.codes + (id * self.d)

    def get_vector_span(self, id: Int) -> Span[Float32, MutUntrackedOrigin]:
        """Retrieves a borrowed view of a specific vector in the index.
        
        Args:
            id: The index of the vector to retrieve.
            
        Returns:
            A view valid while the index storage remains unchanged.
        """
        return Span[Float32, MutUntrackedOrigin](
            ptr=self.codes + (id * self.d), length=self.d
        )

    def search(self, x: Span[Float32, _], k: Int, mut distances: Span[mut=True, Float32, _], mut labels: Span[mut=True, Int, _]):
        """Searches for the k-nearest neighbors of the given query vectors."""
        var n = len(x) // self.d
        if n == 0:
            var labels_ptr = labels.unsafe_ptr()
            var distances_ptr = distances.unsafe_ptr()
            for i in range(n * k):
                labels_ptr[i] = -1
                distances_ptr[i] = 0.0
            return
            
        # Empty filter case
        var empty_filter = Span[UInt8, MutUntrackedOrigin]()
        self._search_impl[False](x, k, distances, labels, empty_filter)

    def search(self, x: Span[Float32, _], k: Int, mut distances: Span[mut=True, Float32, _], mut labels: Span[mut=True, Int, _], filter: Span[UInt8, _]):
        """Searches for the k-nearest neighbors of the given query vectors with a filter mask."""
        var n = len(x) // self.d
        if n == 0:
            var labels_ptr = labels.unsafe_ptr()
            var distances_ptr = distances.unsafe_ptr()
            for i in range(n * k):
                labels_ptr[i] = -1
                distances_ptr[i] = 0.0
            return

        if len(filter) > 0:
            self._search_impl[True](x, k, distances, labels, filter)
        else:
            self._search_impl[False](x, k, distances, labels, filter)
            
    def _search_impl[HAS_FILTER: Bool](self, x: Span[Float32, _], k: Int, mut distances: Span[mut=True, Float32, _], mut labels: Span[mut=True, Int, _], filter: Span[UInt8, _]):
        var n = len(x) // self.d
            
        var self_codes = self.codes
        var self_d = self.d
        var self_ntotal = self.ntotal
        var self_metric_type = self.metric_type
        
        var x_ptr = x.unsafe_ptr()
        var dist_ptr = distances.unsafe_ptr()
        var labels_ptr = labels.unsafe_ptr()
        var filter_ptr = filter.unsafe_ptr()
        
        from std.algorithm import parallelize
        
        @parameter
        def process_query(i: Int):
            var query_offset = i * self_d
            var query_ptr = x_ptr + query_offset
            
            var res_dist_ptr = dist_ptr + (i * k)
            var res_labels_ptr = labels_ptr + (i * k)
            var heap_size = 0
            
            # Iterate over all database vectors
            for j in range(self_ntotal):
                comptime if HAS_FILTER:
                    if filter_ptr[j] > 0:
                        continue
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
                    
        # Let the runtime size the worker pool. Passing ``n`` as both the
        # number of work items and workers attempted to create one worker per
        # query. Large IVF training batches therefore requested thousands of
        # workers while assigning vectors to their coarse centroids, which is
        # both slower and unstable on Linux x86 runtimes. Every query remains
        # parallel; only the amount of physical concurrency is bounded by the
        # runtime.
        parallelize[process_query](n)
                    
    def get_distance_computer(self, query: UnsafePointer[Float32, _]) -> Self.ComputerType:
        """Creates a distance computer for the given query vector."""
        var q_ptr = rebind[UnsafePointer[Float32, MutUntrackedOrigin]](query)
        return FlatDistanceComputer(self.d, self.metric_type, self.codes, q_ptr)
