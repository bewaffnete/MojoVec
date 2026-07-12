from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT

comptime SIMD_WIDTH = 64

from ..utils.distances import l2_distance_simd, sq8_dot_product_simd, sq8_l2_from_dot, inner_product_simd
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from std.sys.intrinsics import prefetch, PrefetchOptions
from std.memory import alloc
import std.math as math

struct SQ8DistanceComputer(DistanceComputerTrait):
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
        if Int(self.query_u8) != 0:
            self.query_u8.free()
            
    def __init__(out self, *, deinit move: Self):
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
        var ptr_i = self.codes_f32 + (i * self.d)
        var ptr_j = self.codes_f32 + (j * self.d)
        if self.metric_type == METRIC_L2:
            return l2_distance_simd[SIMD_WIDTH](ptr_i, ptr_j, self.d)
        else:
            return -inner_product_simd[SIMD_WIDTH](ptr_i, ptr_j, self.d)

    @always_inline
    def prefetch_vector(self, id: Int):
        var ptr = self.codes_u8 + (id * self.d)
        comptime opts = PrefetchOptions().for_read().low_locality().to_data_cache()
        prefetch[opts](ptr)

    @always_inline
    def is_exact(self) -> Bool:
        return False

struct IndexFlatSQ8(Index, StorageTrait, QuantizerTrait, Movable):
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
        if Int(self.codes_f32) != 0:
            self.codes_f32.free()
        if Int(self.codes_u8) != 0:
            self.codes_u8.free()
        if Int(self.norms_u32) != 0:
            self.norms_u32.free()
            
    def __init__(out self, *, deinit move: Self):
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

    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
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
            var val = x[i]
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
                var val = x[i * self.d + j]
                self.codes_f32[offset_f32 + i * self.d + j] = val
                var q = (val - self.global_min) * inv_scale
                var u8_val = UInt8(math.clamp(math.round(q), 0, 255))
                self.codes_u8[offset_f32 + i * self.d + j] = u8_val
                norm += UInt32(u8_val) * UInt32(u8_val)
            self.norms_u32[self.ntotal + i] = norm
            
        self.ntotal += n

    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]):
        print("Pure search on IndexFlatSQ8 not implemented.")

    def get_distance_computer(self, query: UnsafePointer[Float32, MutUntrackedOrigin]) -> Self.ComputerType:
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
            query,
            query_u8,
            query_norm,
            self.scale * self.scale
        )

    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        return self.codes_f32 + (id * self.d)
