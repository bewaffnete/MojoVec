from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT

comptime SIMD_WIDTH = 64

from ..utils.distances import l2_distance_simd, sq8_dot_product_simd, sq8_l2_from_dot, inner_product_simd
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from std.sys.intrinsics import prefetch, PrefetchOptions
from std.memory import memcpy
from std.memory.span import Span
import std.math as math

struct SQ8DistanceComputer(DistanceComputerTrait):
    """Computes distances between a query vector and SQ8 quantized database vectors."""
    var d: Int
    var metric_type: MetricType
    var codes_f32: UnsafePointer[Float32, MutUntrackedOrigin]
    var codes_u8: UnsafePointer[UInt8, MutUntrackedOrigin]
    var norms_u32: UnsafePointer[UInt32, MutUntrackedOrigin]
    var query_f32: UnsafePointer[Float32, MutUntrackedOrigin]
    var query_u8: UnsafePointer[UInt8, MutUntrackedOrigin]
    var query_norm_u32: UInt32
    var scale_sq: Float32
    
    def __init__(out self, d: Int, metric_type: MetricType, codes_f32: UnsafePointer[Float32, MutUntrackedOrigin], codes_u8: UnsafePointer[UInt8, MutUntrackedOrigin], norms_u32: UnsafePointer[UInt32, MutUntrackedOrigin], query_f32: UnsafePointer[Float32, MutUntrackedOrigin], query_u8: UnsafePointer[UInt8, MutUntrackedOrigin], query_norm_u32: UInt32, scale_sq: Float32):
        """Initializes the SQ8 distance computer.
        
        Args:
            d: The dimensionality of the vectors.
            metric_type: The metric type used for distance computation.
            codes_f32: A pointer to the uncompressed database vectors (used for fallback).
            codes_u8: A pointer to the quantized database vectors.
            norms_u32: A pointer to the norms of the quantized database vectors.
            query_f32: A pointer to the uncompressed query vector.
            query_u8: A pointer to the quantized query vector.
            query_norm_u32: The norm of the quantized query vector.
            scale_sq: The squared scaling factor used for quantization.
        """
        self.d = d
        self.metric_type = metric_type
        self.codes_f32 = codes_f32
        self.codes_u8 = codes_u8
        self.norms_u32 = norms_u32
        self.query_f32 = query_f32
        self.query_u8 = query_u8
        self.query_norm_u32 = query_norm_u32
        self.scale_sq = scale_sq
        
    def __del__(deinit self):
        """Frees the allocated memory for the quantized query vector."""
        if Int(self.query_u8) != 0:
            self.query_u8.free()
            
    def __init__(out self, *, deinit move: Self):
        """Moves the SQ8 distance computer from another instance.
        
        Args:
            move: The instance to move from.
        """
        self.d = move.d
        self.metric_type = move.metric_type
        self.codes_f32 = move.codes_f32
        self.codes_u8 = move.codes_u8
        self.norms_u32 = move.norms_u32
        self.query_f32 = move.query_f32
        self.query_u8 = move.query_u8
        self.query_norm_u32 = move.query_norm_u32
        self.scale_sq = move.scale_sq

    @always_inline
    def distance(self, id: Int, threshold: Float32 = Float32.MAX) -> Float32:
        """Computes the distance between the query and a specified database vector.
        
        Args:
            id: The index of the database vector.
            threshold: An optional threshold for early termination.
            
        Returns:
            The computed approximate distance.
        """
        if self.metric_type == METRIC_L2:
            var db_u8_ptr = self.codes_u8 + (id * self.d)
            
            # UDOT dot product
            var dot = sq8_dot_product_simd(self.query_u8, db_u8_ptr, self.d)
            
            # Norm decomposition for L2
            var db_norm = self.norms_u32[id]
            var l2_sq8 = sq8_l2_from_dot(self.query_norm_u32, db_norm, dot)
            return Float32(l2_sq8) * self.scale_sq
        else:
            # Fallback for inner product
            var db_f32_ptr = self.codes_f32 + (id * self.d)
            return -inner_product_simd[SIMD_WIDTH](self.query_f32, db_f32_ptr, self.d)
            
    @always_inline
    def symmetric_distance(self, i: Int, j: Int) -> Float32:
        """Computes the distance between two database vectors.
        
        Args:
            i: The index of the first database vector.
            j: The index of the second database vector.
            
        Returns:
            The computed symmetric distance.
        """
        var ptr_i = self.codes_f32 + (i * self.d)
        var ptr_j = self.codes_f32 + (j * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[SIMD_WIDTH](ptr_i, ptr_j, self.d)
        else:
            return -inner_product_simd[SIMD_WIDTH](ptr_i, ptr_j, self.d)

    @always_inline
    def prefetch_vector(self, id: Int):
        """Prefetch quantized vector data for `id` into CPU cache (L1, read intent).
        
        Args:
            id: The index of the vector to prefetch.
        """
        var ptr = self.codes_u8 + (id * self.d)
        comptime opts = PrefetchOptions().for_read().low_locality().to_data_cache()
        prefetch[opts](ptr)

    @always_inline
    def is_exact(self) -> Bool:
        """Indicates whether this computer provides exact distances.
        
        Returns:
            False, since SQ8 quantization provides approximate distances.
        """
        return False

struct IndexFlatSQ8(Index, StorageTrait, QuantizerTrait, Movable):
    """An index that quantizes vectors to 8-bit integers (SQ8) to accelerate search and reduce memory footprint."""
    comptime ComputerType = SQ8DistanceComputer
    var d: Int
    var ntotal: Int
    var metric_type: MetricType
    var codes_f32: UnsafePointer[Float32, MutUntrackedOrigin]
    var codes_u8: UnsafePointer[UInt8, MutUntrackedOrigin]
    var norms_u32: UnsafePointer[UInt32, MutUntrackedOrigin]
    var capacity: Int
    
    var global_min: Float32
    var global_max: Float32
    var scale: Float32
    
    def __init__(out self, d: Int, metric: MetricType = METRIC_L2):
        """Initializes the SQ8 index.
        
        Args:
            d: The dimensionality of the vectors.
            metric: The metric type used for distance computation.
        """
        self.d = d
        self.ntotal = 0
        self.metric_type = metric
        self.capacity = 1000
        self.codes_f32 = alloc[Float32](self.capacity * self.d)
        self.codes_u8 = alloc[UInt8](self.capacity * self.d)
        self.norms_u32 = alloc[UInt32](self.capacity)
        self.global_min = Float32.MAX
        self.global_max = -Float32.MAX
        self.scale = 1.0

    def __del__(deinit self):
        """Frees the allocated memory for the index."""
        if Int(self.codes_f32) != 0:
            self.codes_f32.free()
        if Int(self.codes_u8) != 0:
            self.codes_u8.free()
        if Int(self.norms_u32) != 0:
            self.norms_u32.free()
            
    def __init__(out self, *, deinit move: Self):
        """Moves the index from another instance.
        
        Args:
            move: The instance to move from.
        """
        self.d = move.d
        self.ntotal = move.ntotal
        self.metric_type = move.metric_type
        self.capacity = move.capacity
        self.codes_f32 = move.codes_f32
        self.codes_u8 = move.codes_u8
        self.norms_u32 = move.norms_u32
        self.global_min = move.global_min
        self.global_max = move.global_max
        self.scale = move.scale

    def add(mut self, x: Span[Float32, _]):
        """Adds new vectors to the index, maintaining dynamic quantization bounds.
        
        Args:
            x: A safe Span pointing to the uncompressed vectors to add.
        """
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        if self.ntotal + n > self.capacity:
            var new_cap = math.max(self.capacity * 2, self.ntotal + n)
            var new_f32 = alloc[Float32](new_cap * self.d)
            var new_u8 = alloc[UInt8](new_cap * self.d)
            var new_norms = alloc[UInt32](new_cap)
            
            for i in range(self.ntotal * self.d):
                new_f32[i] = self.codes_f32[i]
                new_u8[i] = self.codes_u8[i]
            for i in range(self.ntotal):
                new_norms[i] = self.norms_u32[i]
                
            self.codes_f32.free()
            self.codes_u8.free()
            self.norms_u32.free()
            self.codes_f32 = new_f32
            self.codes_u8 = new_u8
            self.norms_u32 = new_norms
            self.capacity = new_cap
            
        # 1. Update global min/max
        var batch_min = Float32.MAX
        var batch_max = -Float32.MAX
        for i in range(n * self.d):
            var val = x_ptr[i]
            if val < batch_min: batch_min = val
            if val > batch_max: batch_max = val
            
        var needs_requantize = False
        if batch_min < self.global_min:
            self.global_min = batch_min
            needs_requantize = True
        if batch_max > self.global_max:
            self.global_max = batch_max
            needs_requantize = True
            
        if self.global_max == self.global_min:
            self.scale = 1.0
        else:
            self.scale = (self.global_max - self.global_min) / 255.0
            
        # Re-quantize existing data if scale changed
        if needs_requantize and self.ntotal > 0:
            var inv_scale = 1.0 / self.scale
            for i in range(self.ntotal):
                var norm: UInt32 = 0
                for j in range(self.d):
                    var val = self.codes_f32[i * self.d + j]
                    var q = (val - self.global_min) * inv_scale
                    var u8_val = UInt8(math.clamp(math.round(q), 0, 255))
                    self.codes_u8[i * self.d + j] = u8_val
                    norm += UInt32(u8_val) * UInt32(u8_val)
                self.norms_u32[i] = norm
                
        # Insert new data
        var inv_scale = 1.0 / self.scale
        var offset_f32 = self.ntotal * self.d
        for i in range(n):
            var norm: UInt32 = 0
            for j in range(self.d):
                var val = x_ptr[i * self.d + j]
                self.codes_f32[offset_f32 + i * self.d + j] = val
                var q = (val - self.global_min) * inv_scale
                var u8_val = UInt8(math.clamp(math.round(q), 0, 255))
                self.codes_u8[offset_f32 + i * self.d + j] = u8_val
                norm += UInt32(u8_val) * UInt32(u8_val)
            self.norms_u32[self.ntotal + i] = norm
            
        self.ntotal += n

    def search(self, x: Span[Float32, _], k: Int, mut distances: Span[mut=True, Float32, _], mut labels: Span[mut=True, Int, _]):
        """Searches for the k-nearest neighbors of the given query vectors using SQ8 acceleration.
        
        Args:
            x: A safe Span pointing to the uncompressed query vectors.
            k: The number of nearest neighbors to retrieve.
            distances: An output Span for distances.
            labels: An output Span for labels.
        """
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        var distances_ptr = distances.unsafe_ptr()
        var labels_ptr = labels.unsafe_ptr()
        
        var self_d = self.d
        var self_ntotal = self.ntotal
        var self_metric_type = self.metric_type
        var self_codes_f32 = self.codes_f32
        var self_codes_u8 = self.codes_u8
        var self_norms_u32 = self.norms_u32
        var self_global_min = self.global_min
        var self_scale = self.scale
        
        from std.algorithm import parallelize
        
        def process_query(i: Int) {self_d, self_codes_f32, self_codes_u8, self_norms_u32, self_global_min, self_scale, self_ntotal, self_metric_type, x_ptr, k, distances_ptr, labels_ptr}:
            var query_ptr = x_ptr + (i * self_d)
            var res_dist_ptr = distances_ptr + (i * k)
            var res_labels_ptr = labels_ptr + (i * k)
            
            var query_u8 = alloc[UInt8](self_d)
            var query_norm: UInt32 = 0
            var inv_scale = 1.0 / self_scale
            for dim in range(self_d):
                var q = (query_ptr[dim] - self_global_min) * inv_scale
                var u8_val = UInt8(math.clamp(math.round(q), 0, 255))
                query_u8[dim] = u8_val
                query_norm += UInt32(u8_val) * UInt32(u8_val)
                
            var heap_size = 0
            var scale_sq = self_scale * self_scale
            
            for j in range(self_ntotal):
                var dist: Float32
                if self_metric_type == METRIC_L2:
                    var db_u8_ptr = self_codes_u8 + (j * self_d)
                    var dot = sq8_dot_product_simd(query_u8, db_u8_ptr, self_d)
                    var db_norm = self_norms_u32[j]
                    var l2_sq8 = sq8_l2_from_dot(query_norm, db_norm, dot)
                    dist = Float32(l2_sq8) * scale_sq
                else:
                    var db_f32_ptr = self_codes_f32 + (j * self_d)
                    dist = -inner_product_simd[SIMD_WIDTH](query_ptr, db_f32_ptr, self_d)
                    
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
                
            if self_metric_type == METRIC_INNER_PRODUCT:
                for j in range(k):
                    res_dist_ptr[j] = -res_dist_ptr[j]
                    
            query_u8.free()
            
        parallelize(process_query, n, n)

    def get_distance_computer(self, query: UnsafePointer[Float32, _]) -> Self.ComputerType:
        """Creates a distance computer for the given query vector.
        
        Args:
            query: A pointer to the query vector.
            
        Returns:
            An instance of the associated distance computer.
        """
        # Quantize the query ONCE for all comparisons
        var query_u8 = alloc[UInt8](self.d)
        var query_norm: UInt32 = 0
        var inv_scale = 1.0 / self.scale
        for i in range(self.d):
            var q = (query[i] - self.global_min) * inv_scale
            var u8_val = UInt8(math.clamp(math.round(q), 0, 255))
            query_u8[i] = u8_val
            query_norm += UInt32(u8_val) * UInt32(u8_val)
            
        return SQ8DistanceComputer(
            self.d,
            self.metric_type,
            self.codes_f32,
            self.codes_u8,
            self.norms_u32,
            rebind[UnsafePointer[Float32, MutUntrackedOrigin]](query),
            query_u8,
            query_norm,
            self.scale * self.scale
        )

    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        """Retrieves a pointer to the uncompressed vector in the index.
        
        Args:
            id: The index of the vector to retrieve.
            
        Returns:
            A pointer to the requested uncompressed vector.
        """
        return self.codes_f32 + (id * self.d)
