from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from std.memory import alloc

from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from ..utils.distances import sq8_signed_dot_product_simd_batch4
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from std.sys.intrinsics import prefetch, PrefetchOptions
from std.memory import memcpy
from std.memory.span import Span
import std.math as math
from mojovec.io.memory_map import FileMemoryMap
from mojovec.core.validation import _validate_vector_dimension
from mojovec.quantization.sq8 import (
    _encode_sq8_batch,
    _encode_sq8_vector,
    _sq8_code_distance,
)


struct SQ8DistanceComputer(DistanceComputerTrait):
    """Metric-dispatched byte distance computer shared by Flat and HNSW."""
    var d: Int
    var metric_type: MetricType
    var codes_u8: UnsafePointer[UInt8, MutUntrackedOrigin]
    var norms_u32: UnsafePointer[UInt32, MutUntrackedOrigin]
    var query_u8: UnsafePointer[UInt8, MutUntrackedOrigin]
    var query_norm_u32: UInt32
    var scale_sq: Float32
    
    def __init__(
        out self,
        d: Int,
        metric_type: MetricType,
        codes_u8: UnsafePointer[UInt8, MutUntrackedOrigin],
        norms_u32: UnsafePointer[UInt32, MutUntrackedOrigin],
        query_u8: UnsafePointer[UInt8, MutUntrackedOrigin],
        query_norm_u32: UInt32,
        scale_sq: Float32,
    ):
        """Initializes the SQ8 distance computer.
        
        Args:
            d: The dimensionality of the vectors.
            metric_type: The metric type used for distance computation.
            codes_u8: A pointer to the quantized database vectors.
            norms_u32: A pointer to the norms of the quantized database vectors.
            query_u8: A pointer to the quantized query vector.
            query_norm_u32: The norm of the quantized query vector.
            scale_sq: The squared scaling factor used for quantization.
        """
        self.d = d
        self.metric_type = metric_type
        self.codes_u8 = codes_u8
        self.norms_u32 = norms_u32
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
        self.codes_u8 = move.codes_u8
        self.norms_u32 = move.norms_u32
        self.query_u8 = move.query_u8
        self.query_norm_u32 = move.query_norm_u32
        self.scale_sq = move.scale_sq

    @always_inline
    def distance_batch4(self, id0: Int, id1: Int, id2: Int, id3: Int) -> InlineArray[Float32, 4]:
        var res = InlineArray[Float32, 4](uninitialized=True)
        if self.metric_type == METRIC_L2:
            res[0] = self._distance_l2(id0)
            res[1] = self._distance_l2(id1)
            res[2] = self._distance_l2(id2)
            res[3] = self._distance_l2(id3)
        else:
            var dots = sq8_signed_dot_product_simd_batch4(
                (self.codes_u8 + id0 * self.d).bitcast[Int8](),
                (self.codes_u8 + id1 * self.d).bitcast[Int8](),
                (self.codes_u8 + id2 * self.d).bitcast[Int8](),
                (self.codes_u8 + id3 * self.d).bitcast[Int8](),
                self.query_u8.bitcast[Int8](),
                self.d,
            )
            res[0] = -Float32(dots[0]) * self.scale_sq
            res[1] = -Float32(dots[1]) * self.scale_sq
            res[2] = -Float32(dots[2]) * self.scale_sq
            res[3] = -Float32(dots[3]) * self.scale_sq
        return res

    @always_inline
    def _distance_l2(self, id: Int) -> Float32:
        return _sq8_code_distance[False](
            self.query_u8,
            self.codes_u8 + (id * self.d),
            self.query_norm_u32,
            self.norms_u32[id],
            self.d,
            self.scale_sq,
        )

    @always_inline
    def _distance_ip(self, id: Int) -> Float32:
        return _sq8_code_distance[True](
            self.query_u8,
            self.codes_u8 + (id * self.d),
            self.query_norm_u32,
            self.norms_u32[id],
            self.d,
            self.scale_sq,
        )

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
            return self._distance_l2(id)
        else:
            return self._distance_ip(id)
            
    @always_inline
    def symmetric_distance(self, i: Int, j: Int) -> Float32:
        """Computes the distance between two database vectors.
        
        Args:
            i: The index of the first database vector.
            j: The index of the second database vector.
            
        Returns:
            The computed symmetric distance.
        """
        var ptr_i = self.codes_u8 + (i * self.d)
        var ptr_j = self.codes_u8 + (j * self.d)
        if self.metric_type == METRIC_L2:
            return _sq8_code_distance[False](
                ptr_i,
                ptr_j,
                self.norms_u32[i],
                self.norms_u32[j],
                self.d,
                self.scale_sq,
            )
        else:
            return _sq8_code_distance[True](
                ptr_i,
                ptr_j,
                self.norms_u32[i],
                self.norms_u32[j],
                self.d,
                self.scale_sq,
            )

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
    """Affine UInt8 L2 and symmetric Int8 IP storage with exact rerank data."""
    comptime ComputerType = SQ8DistanceComputer
    comptime HNSW_PTHREAD_STORAGE_KIND = 1
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
    var _mapping: FileMemoryMap
    
    def __init__(
        out self,
        d: Int,
        metric: MetricType = METRIC_L2,
        initial_capacity: Int = 1000,
    ) raises:
        """Initializes the SQ8 index.
        
        Args:
            d: The dimensionality of the vectors.
            metric: The metric type used for distance computation.
            initial_capacity: Initial owned allocation; loaders use zero.
        """
        _validate_vector_dimension(d)
        if initial_capacity < 0 or initial_capacity > 1000:
            raise Error("initial_capacity must be between 0 and 1000.")
        self.d = d
        self.ntotal = 0
        self.metric_type = metric
        self.capacity = initial_capacity
        self.codes_f32 = alloc[Float32](max(self.capacity * self.d, 1))
        self.codes_u8 = alloc[UInt8](max(self.capacity * self.d, 1))
        self.norms_u32 = alloc[UInt32](max(self.capacity, 1))
        self.global_min = Float32.MAX
        self.global_max = -Float32.MAX
        self.scale = 1.0
        self._mapping = FileMemoryMap()

    def __del__(deinit self):
        """Frees the allocated memory for the index."""
        if not self._mapping.is_active():
            if Int(self.codes_f32) != 0:
                self.codes_f32.free()
            if Int(self.codes_u8) != 0:
                self.codes_u8.free()
            if Int(self.norms_u32) != 0:
                self.norms_u32.free()
        self._mapping.close()
            
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
        self._mapping = move._mapping^

    @always_inline
    def is_memory_mapped(self) -> Bool:
        return self._mapping.is_active()

    def _detach_mapped(mut self):
        if not self._mapping.is_active():
            return
        var new_f32 = alloc[Float32](max(self.capacity * self.d, 1))
        var new_u8 = alloc[UInt8](max(self.capacity * self.d, 1))
        var new_norms = alloc[UInt32](max(self.capacity, 1))
        for i in range(self.ntotal * self.d):
            new_f32[i] = self.codes_f32[i]
            new_u8[i] = self.codes_u8[i]
        for i in range(self.ntotal):
            new_norms[i] = self.norms_u32[i]
        self._mapping.close()
        self.codes_f32 = new_f32
        self.codes_u8 = new_u8
        self.norms_u32 = new_norms

    def add(mut self, x: Span[Float32, _]):
        """Adds new vectors to the index, maintaining dynamic quantization bounds.
        
        Args:
            x: A safe Span pointing to the uncompressed vectors to add.
        """
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        if n == 0:
            return
        self._detach_mapped()
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
            
        if self.metric_type == METRIC_L2:
            if self.global_max == self.global_min:
                self.scale = 1.0
            else:
                self.scale = (self.global_max - self.global_min) / 255.0
        else:
            var max_abs = math.max(
                math.abs(self.global_min), math.abs(self.global_max)
            )
            self.scale = max_abs / 127.0 if max_abs > 0.0 else 1.0

        var inverse_scale = 1.0 / self.scale
            
        # Re-quantize existing data if scale changed
        if needs_requantize and self.ntotal > 0:
            if self.metric_type == METRIC_L2:
                _encode_sq8_batch[False](
                    self.codes_f32,
                    self.codes_u8,
                    self.norms_u32,
                    self.ntotal,
                    self.d,
                    self.global_min,
                    inverse_scale,
                )
            else:
                _encode_sq8_batch[True](
                    self.codes_f32,
                    self.codes_u8,
                    self.norms_u32,
                    self.ntotal,
                    self.d,
                    0.0,
                    inverse_scale,
                )
                
        # Insert new data
        var offset_f32 = self.ntotal * self.d
        memcpy(
            dest=self.codes_f32 + offset_f32,
            src=x_ptr,
            count=n * self.d,
        )
        if self.metric_type == METRIC_L2:
            _encode_sq8_batch[False](
                x_ptr,
                self.codes_u8 + offset_f32,
                self.norms_u32 + self.ntotal,
                n,
                self.d,
                self.global_min,
                inverse_scale,
            )
        else:
            _encode_sq8_batch[True](
                x_ptr,
                self.codes_u8 + offset_f32,
                self.norms_u32 + self.ntotal,
                n,
                self.d,
                0.0,
                inverse_scale,
            )
            
        self.ntotal += n

    def search(self, x: Span[Float32, _], k: Int, mut distances: Span[mut=True, Float32, _], mut labels: Span[mut=True, Int, _]):
        """Searches for the k-nearest neighbors using SQ8 acceleration."""
        var n = len(x) // self.d
        if n == 0:
            var labels_ptr = labels.unsafe_ptr()
            var distances_ptr = distances.unsafe_ptr()
            for i in range(n * k):
                labels_ptr[i] = -1
                distances_ptr[i] = 0.0
            return
        var empty_filter = Span[UInt8, MutUntrackedOrigin]()
        self._search_impl[False](x, k, distances, labels, empty_filter)

    def search(self, x: Span[Float32, _], k: Int, mut distances: Span[mut=True, Float32, _], mut labels: Span[mut=True, Int, _], filter: Span[UInt8, _]):
        """Searches for the k-nearest neighbors using SQ8 acceleration with a filter mask."""
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
        var x_ptr = x.unsafe_ptr()
        var distances_ptr = distances.unsafe_ptr()
        var labels_ptr = labels.unsafe_ptr()
        var filter_ptr = filter.unsafe_ptr()
        
        var self_d = self.d
        var self_ntotal = self.ntotal
        var self_metric_type = self.metric_type
        var self_codes_u8 = self.codes_u8
        var self_norms_u32 = self.norms_u32
        var self_global_min = self.global_min
        var self_scale = self.scale
        
        from std.algorithm import parallelize
        from std.sys.info import num_performance_cores
        
        @parameter
        def process_query(i: Int):
            var query_ptr = x_ptr + (i * self_d)
            var res_dist_ptr = distances_ptr + (i * k)
            var res_labels_ptr = labels_ptr + (i * k)
            
            var query_u8 = alloc[UInt8](self_d)
            var inverse_scale = 1.0 / self_scale
            var query_norm: UInt32
            if self_metric_type == METRIC_L2:
                query_norm = _encode_sq8_vector[False](
                    query_ptr,
                    query_u8,
                    self_d,
                    self_global_min,
                    inverse_scale,
                )
            else:
                query_norm = _encode_sq8_vector[True](
                    query_ptr,
                    query_u8,
                    self_d,
                    0.0,
                    inverse_scale,
                )
                
            var heap_size = 0
            var scale_sq = self_scale * self_scale
            
            for j in range(self_ntotal):
                comptime if HAS_FILTER:
                    if filter_ptr[j] > 0:
                        continue
                var dist: Float32
                if self_metric_type == METRIC_L2:
                    dist = _sq8_code_distance[False](
                        query_u8,
                        self_codes_u8 + (j * self_d),
                        query_norm,
                        self_norms_u32[j],
                        self_d,
                        scale_sq,
                    )
                else:
                    dist = _sq8_code_distance[True](
                        query_u8,
                        self_codes_u8 + (j * self_d),
                        query_norm,
                        self_norms_u32[j],
                        self_d,
                        scale_sq,
                    )
                    
                if heap_size < k:
                    max_heap_push(res_dist_ptr, res_labels_ptr, heap_size, dist, j)
                    heap_size += 1
                elif dist < res_dist_ptr[0]:
                    max_heap_replace_top(res_dist_ptr, res_labels_ptr, heap_size, dist, j)
                    
            var sorted_dist_ptr = distances_ptr + (i * k)
            var sorted_labels_ptr = labels_ptr + (i * k)
            
            var result_count = heap_size
            while heap_size > 0:
                var popped = max_heap_pop(res_dist_ptr, res_labels_ptr, heap_size)
                heap_size -= 1
                var idx = heap_size
                sorted_dist_ptr[idx] = popped.dist
                sorted_labels_ptr[idx] = Int(popped.label)
                
            for j in range(result_count, k):
                sorted_dist_ptr[j] = 0.0
                sorted_labels_ptr[j] = -1
                
            if self_metric_type == METRIC_INNER_PRODUCT:
                for j in range(result_count):
                    if res_labels_ptr[j] >= 0:
                        res_dist_ptr[j] = -res_dist_ptr[j]
                    
            query_u8.free()
        parallelize[process_query](n, n)
    def get_distance_computer(self, query: UnsafePointer[Float32, _]) -> Self.ComputerType:
        """Creates a distance computer for the given query vector.
        
        Args:
            query: A pointer to the query vector.
            
        Returns:
            An instance of the associated distance computer.
        """
        # Quantize the query once with the codec selected by the metric.
        var query_u8 = alloc[UInt8](self.d)
        var inverse_scale = 1.0 / self.scale
        var query_norm: UInt32
        if self.metric_type == METRIC_L2:
            query_norm = _encode_sq8_vector[False](
                query,
                query_u8,
                self.d,
                self.global_min,
                inverse_scale,
            )
        else:
            query_norm = _encode_sq8_vector[True](
                query,
                query_u8,
                self.d,
                0.0,
                inverse_scale,
            )
            
        return SQ8DistanceComputer(
            self.d,
            self.metric_type,
            self.codes_u8,
            self.norms_u32,
            query_u8,
            query_norm,
            self.scale * self.scale,
        )

    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        """Retrieves a pointer to the uncompressed vector in the index.
        
        Args:
            id: The index of the vector to retrieve.
            
        Returns:
            A pointer to the requested uncompressed vector.
        """
        return self.codes_f32 + (id * self.d)
