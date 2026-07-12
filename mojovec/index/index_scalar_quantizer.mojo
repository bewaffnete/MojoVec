from ..core.index import Index
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT, QuantizerType, QT_8bit, QT_fp16
from ..core.quantizer import ScalarQuantizer
from ..utils.distances import l2_distance_simd, inner_product_simd
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from std.sys.intrinsics import prefetch, PrefetchOptions

struct SQDistanceComputer(DistanceComputerTrait):
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
        self.d = d
        self.code_size = code_size
        self.metric_type = metric_type
        self.sq = sq.copy()
        self.codes = codes
        self.query = query
        self.scratch_x = alloc[Float32](self.d)
        self.scratch_y = alloc[Float32](self.d)

    def __init__(out self, *, deinit move: Self):
        self.d = move.d
        self.code_size = move.code_size
        self.metric_type = move.metric_type
        self.sq = move.sq^
        self.codes = move.codes
        self.query = move.query
        self.scratch_x = move.scratch_x
        self.scratch_y = move.scratch_y

    def __del__(deinit self):
        # In Mojo, we can't check pointer truthiness, so we just free if address is not 0
        if Int(self.scratch_x) != 0:
            self.scratch_x.free()
        if Int(self.scratch_y) != 0:
            self.scratch_y.free()
        
    @always_inline
    def distance(self, id: Int, threshold: Float32 = Float32.MAX) -> Float32:
        var db_ptr = self.codes + (id * self.code_size)
        self.sq.decode(db_ptr, self.scratch_x)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[4](self.query, self.scratch_x, self.d)
        else:
            return -inner_product_simd[4](self.query, self.scratch_x, self.d)

    @always_inline
    def symmetric_distance(self, i: Int, j: Int) -> Float32:
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
        """
        var ptr = self.codes + (id * self.code_size)
        comptime opts = PrefetchOptions().for_read().medium_locality().to_data_cache()
        prefetch[opts](ptr)

    @always_inline
    def is_exact(self) -> Bool:
        return False

struct IndexScalarQuantizer(Index, StorageTrait):
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
        self.scratch_x.free()
        if Int(self.codes) != 0:
            self.codes.free()
        
    def train(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        self.sq.train(n, x)
        self.is_trained = self.sq.is_trained

    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
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
            var x_ptr = x + (i * self.d)
            var code_ptr = self.codes + ((offset_vectors + i) * self.code_size)
            self.sq.encode(x_ptr, code_ptr)
            
        self.ntotal = new_ntotal
        
    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        self.sq.decode(self.codes + (id * self.code_size), self.scratch_x)
        return self.scratch_x

    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]):
        var scratch_x = alloc[Float32](self.d)
        
        for i in range(n):
            var query_ptr = x + (i * self.d)
            var res_dist_ptr = distances + (i * k)
            var res_labels_ptr = labels + (i * k)
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
        
    def get_distance_computer(self, query: UnsafePointer[Float32, MutUntrackedOrigin]) -> Self.ComputerType:
        return SQDistanceComputer(self.d, self.code_size, self.metric_type, self.sq.copy(), self.codes, query)
