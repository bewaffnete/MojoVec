from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from ..utils.distances import l2_distance_simd, inner_product_simd
from ..utils.heap import max_heap_push, max_heap_replace_top
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait

struct FlatDistanceComputer(DistanceComputerTrait):
    var d: Int
    var metric_type: MetricType
    var codes: UnsafePointer[Float32, MutUntrackedOrigin]
    var query: UnsafePointer[Float32, MutUntrackedOrigin]
    
    def __init__(out self, d: Int, metric_type: MetricType, codes: UnsafePointer[Float32, MutUntrackedOrigin], query: UnsafePointer[Float32, MutUntrackedOrigin]):
        self.d = d
        self.metric_type = metric_type
        self.codes = codes
        self.query = query
        
    @always_inline
    def distance(self, id: Int) -> Float32:
        var db_ptr = self.codes + (id * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[4](self.query, db_ptr, self.d)
        else:
            return -inner_product_simd[4](self.query, db_ptr, self.d)
            
    @always_inline
    def symmetric_distance(self, i: Int, j: Int) -> Float32:
        var ptr_i = self.codes + (i * self.d)
        var ptr_j = self.codes + (j * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[4](ptr_i, ptr_j, self.d)
        else:
            return -inner_product_simd[4](ptr_i, ptr_j, self.d)

struct IndexFlat(Index, StorageTrait, QuantizerTrait, Movable):
    comptime ComputerType = FlatDistanceComputer
    var d: Int
    var ntotal: Int
    var metric_type: MetricType
    # Pointer to the raw flattened data
    var codes: UnsafePointer[Float32, MutUntrackedOrigin]
    # Capacity allocated for codes
    var capacity: Int

    def __init__(out self, d: Int, metric: MetricType = METRIC_L2):
        self.d = d
        self.ntotal = 0
        self.metric_type = metric
        self.capacity = 1024 * d  # Initial capacity for 1024 vectors
        self.capacity = 1024  # Initial capacity for 1024 vectors
        self.codes = alloc[Float32](self.capacity * d)

    def __del__(deinit self):
        if Int(self.codes) != 0:
            self.codes.free()

    def __init__(out self, *, deinit move: Self):
        self.d = move.d
        self.ntotal = move.ntotal
        self.metric_type = move.metric_type
        self.capacity = move.capacity
        self.codes = move.codes

    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
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
        return self.codes + (id * self.d)

    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]):
        # Iterate over all queries
        for i in range(n):
            var query_offset = i * self.d
            var query_ptr = x + query_offset
            
            var res_dist_ptr = distances + (i * k)
            var res_labels_ptr = labels + (i * k)
            var heap_size = 0
            
            # Iterate over all database vectors
            for j in range(self.ntotal):
                var db_ptr = self.codes + (j * self.d)
                var dist: Float32 = 0.0
                
                # Compute distance based on metric
                if self.metric_type == METRIC_L2:
                    dist = l2_distance_simd[4](query_ptr, db_ptr, self.d)
                else:
                    dist = -inner_product_simd[4](query_ptr, db_ptr, self.d)
                    
                # Add to heap
                if heap_size < k:
                    max_heap_push(res_dist_ptr, res_labels_ptr, heap_size, dist, j)
                    heap_size += 1
                elif dist < res_dist_ptr[0]:
                    max_heap_replace_top(res_dist_ptr, res_labels_ptr, k, dist, j)
            
            # Un-negate inner product distances
            if self.metric_type == METRIC_INNER_PRODUCT:
                for j in range(k):
                    res_dist_ptr[j] = -res_dist_ptr[j]
                    
    def get_distance_computer(self, query: UnsafePointer[Float32, MutUntrackedOrigin]) -> Self.ComputerType:
        return FlatDistanceComputer(self.d, self.metric_type, self.codes, query)
