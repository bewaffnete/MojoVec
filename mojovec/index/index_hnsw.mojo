from ..core.index import Index
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from ..utils.distances import l2_distance_simd, inner_product_simd
from .hnsw_graph import HNSWGraph
from .hnsw_visited import VisitedTable, VisitedTablePool
from .index_flat import IndexFlat
from .index_flat_sq8 import IndexFlatSQ8
from max.algorithm import parallelize
from std.ffi import external_call
from std.collections.span import Span
from std.memory.alloc import unsafe_alloc
from std.runtime.asyncrt import parallelism_level
from std.sys.info import num_logical_cores
from mojovec.core.validation import (
    _validate_hnsw_parameters,
    _validate_vector_dimension,
)


struct HNSWPThreadContext(Movable):
    """Type-erased arguments passed to a native pthread search worker."""

    var index_address: Int
    var queries_address: Int
    var distances_address: Int
    var labels_address: Int
    var filter_address: Int
    var query_count: Int
    var k: Int
    var worker_id: Int
    var worker_count: Int
    var storage_kind: Int
    var has_filter: Bool

    def __init__(
        out self,
        index_address: Int,
        queries_address: Int,
        distances_address: Int,
        labels_address: Int,
        filter_address: Int,
        query_count: Int,
        k: Int,
        worker_id: Int,
        worker_count: Int,
        storage_kind: Int,
        has_filter: Bool,
    ):
        self.index_address = index_address
        self.queries_address = queries_address
        self.distances_address = distances_address
        self.labels_address = labels_address
        self.filter_address = filter_address
        self.query_count = query_count
        self.k = k
        self.worker_id = worker_id
        self.worker_count = worker_count
        self.storage_kind = storage_kind
        self.has_filter = has_filter


struct IndexHNSW[StorageType: StorageTrait](Index, Movable):
    """An index structure that uses the HNSW graph for fast approximate nearest neighbor search."""
    var d: Int
    var ntotal: Int
    var metric_type: MetricType
    var is_trained: Bool
    var storage: Self.StorageType
    var hnsw: HNSWGraph
    var vt_pool: VisitedTablePool

    def __init__(
        out self,
        var storage: Self.StorageType,
        d: Int,
        metric_type: MetricType,
        M: Int = 32,
        ef_construction: Int = 40,
        ef_search: Int = 16,
    ) raises:
        """Initializes the HNSW index with the provided storage, dimension, metric type, and M parameter."""
        _validate_vector_dimension(d)
        _validate_hnsw_parameters(M, ef_construction, ef_search)
        self.d = d
        self.ntotal = 0
        self.metric_type = metric_type
        self.is_trained = True
        self.storage = storage^
        self.hnsw = HNSWGraph(
            M=M,
            efConstruction=ef_construction,
            efSearch=ef_search,
        )
        self.vt_pool = VisitedTablePool(self.hnsw.capacity)

    def __init__(out self, *, deinit move: Self):
        """Moves the HNSW index."""
        self.d = move.d
        self.ntotal = move.ntotal
        self.metric_type = move.metric_type
        self.is_trained = move.is_trained
        self.storage = move.storage^
        self.hnsw = move.hnsw^
        self.vt_pool = move.vt_pool^

    @always_inline
    def _dispatch_search_layer[
        ComputerType: DistanceComputerTrait,
        origin1: MutOrigin,
        origin2: MutOrigin,
        HAS_FILTER: Bool = False
    ](
        self,
        mut comp: ComputerType,
        ep_id: Int,
        ep_dist: Float32,
        ef: Int,
        level: Int,
        vt: Pointer[VisitedTable, MutUntrackedOrigin],
        mut res_dist: Pointer[Float32, origin1],
        mut res_labels: Pointer[Int32, origin2],
        filter: Pointer[UInt8, _],
    ) -> Int:
        var max_links = self.hnsw.M * 2 if level == 0 else self.hnsw.M
        
        if max_links == 64:
            return self.hnsw.search_layer[MAX_LINKS=64, HAS_FILTER=HAS_FILTER](comp, ep_id, ep_dist, ef, level, vt, res_dist, res_labels, filter)
        elif max_links == 32:
            return self.hnsw.search_layer[MAX_LINKS=32, HAS_FILTER=HAS_FILTER](comp, ep_id, ep_dist, ef, level, vt, res_dist, res_labels, filter)
        elif max_links == 128:
            return self.hnsw.search_layer[MAX_LINKS=128, HAS_FILTER=HAS_FILTER](comp, ep_id, ep_dist, ef, level, vt, res_dist, res_labels, filter)
        elif max_links == 16:
            return self.hnsw.search_layer[MAX_LINKS=16, HAS_FILTER=HAS_FILTER](comp, ep_id, ep_dist, ef, level, vt, res_dist, res_labels, filter)
        else:
            return self.hnsw.search_layer[MAX_LINKS=0, HAS_FILTER=HAS_FILTER](comp, ep_id, ep_dist, ef, level, vt, res_dist, res_labels, filter)

    def add(mut self, x: Span[Float32, _]):
        """Adds vectors to the HNSW index from the given Span x."""
        var n = len(x) // self.d
        if n == 0:
            return

        self.hnsw._detach_mapped()
        self.storage.add(x)
        var x_ptr = x.unsafe_ptr()

        var old_ntotal = self.ntotal
        var pt_levels = unsafe_alloc[Int](n)

        # Preallocate topology
        for i in range(n):
            var pt_id = old_ntotal + i
            var pt_level = self.hnsw.random_level()
            pt_levels[unsafe_offset=i] = pt_level

            while pt_id >= self.hnsw.capacity:
                self.hnsw._grow()
                self.vt_pool.grow(self.hnsw.capacity)

            self.hnsw.levels[unsafe_offset=pt_id] = pt_level

            var size_needed = self.hnsw.cum_nneighbor_per_level[unsafe_offset=pt_level + 1]
            var current_offset = 0
            if pt_id > 0:
                current_offset = (
                    self.hnsw.offsets[unsafe_offset=pt_id - 1]
                    + self.hnsw.cum_nneighbor_per_level[unsafe_offset=
                        self.hnsw.levels[unsafe_offset=pt_id - 1] + 1
                    ]
                )

            if current_offset + size_needed > self.hnsw.neighbors_capacity:
                self.hnsw.grow_neighbors(
                    current_offset + size_needed, current_offset
                )

            self.hnsw.offsets[unsafe_offset=pt_id] = current_offset
            self.hnsw.ntotal = pt_id + 1

        var start_idx = 0
        if old_ntotal == 0:
            self.hnsw.max_level = pt_levels[unsafe_offset=0]
            self.hnsw.entry_point = 0
            start_idx = 1

        # Initialize neighbor lists sequentially to avoid race conditions
        # where thread A sees thread B's node before thread B initializes it.
        for i in range(n):
            var pt_id = old_ntotal + i
            for l in range(pt_levels[unsafe_offset=i] + 1):
                self.hnsw.set_neighbors_len(pt_id, l, 0)

        @parameter
        def add_point(i: Int):
            var actual_i = i + start_idx
            var pt_id = old_ntotal + actual_i
            var pt_level = pt_levels[unsafe_offset=actual_i]

            var q_ptr = x_ptr.unsafe_offset(actual_i * self.d)
            var comp = self.storage.get_distance_computer(q_ptr)

            var ep_id = self.hnsw.entry_point
            var ep_dist = comp.distance(ep_id)

            for level in range(self.hnsw.max_level, pt_level, -1):
                var changed = True
                while changed:
                    changed = False
                    var neighbors_info = self.hnsw.get_neighbors(ep_id, level)
                    var neighbors = neighbors_info.ptr
                    var max_links = neighbors_info.max_links
                    for j in range(max_links):
                        var neigh = neighbors[unsafe_offset=j]
                        if neigh < 0:
                            break

                        # Pipeline prefetch the next neighbor's vector
                        if j + 1 < max_links:
                            var next_neigh = neighbors[unsafe_offset=j + 1]
                            if next_neigh >= 0:
                                comp.prefetch_vector(Int(next_neigh))

                        var d = comp.distance(Int(neigh))
                        if d < ep_dist:
                            ep_dist = d
                            ep_id = Int(neigh)
                            changed = True

            var vt_id = self.vt_pool.acquire()
            var vt = self.vt_pool.get(vt_id)
            var w_dist_array = InlineArray[Float32, 2048](uninitialized=True)
            var w_labels_array = InlineArray[Int32, 2048](uninitialized=True)
            var W_dist = w_dist_array.unsafe_ptr()
            var W_labels = w_labels_array.unsafe_ptr()

            for level in range(min(pt_level, self.hnsw.max_level), -1, -1):
                var W_size = self._dispatch_search_layer[HAS_FILTER=False](
                    comp,
                    ep_id,
                    ep_dist,
                    self.hnsw.efConstruction,
                    level,
                    vt,
                    W_dist,
                    W_labels,
                    Span[UInt8, MutUntrackedOrigin]().unsafe_ptr()
                )

                var w_sorted_dist_array = InlineArray[Float32, 2048](
                    uninitialized=True
                )
                var w_sorted_labels_array = InlineArray[Int32, 2048](
                    uninitialized=True
                )
                var W_sorted_dist = w_sorted_dist_array.unsafe_ptr()
                var W_sorted_labels = w_sorted_labels_array.unsafe_ptr()
                var w_sz = W_size
                var total_w = W_size
                for j in range(total_w):
                    var popped = max_heap_pop(W_dist, W_labels, w_sz)
                    w_sz -= 1
                    var idx = total_w - 1 - j
                    W_sorted_dist[unsafe_offset=idx] = popped.dist
                    W_sorted_labels[unsafe_offset=idx] = popped.label

                var M_l = self.hnsw.M
                if level == 0:
                    M_l = self.hnsw.M * 2

                # Apply extended heuristic to select M_l diverse neighbors from the candidates
                var return_size = 0
                for j in range(total_w):
                    if return_size >= M_l:
                        break
                    var c_id = Int(W_sorted_labels[unsafe_offset=j])
                    var c_dist = W_sorted_dist[unsafe_offset=j]
                    var keep = True
                    for r in range(return_size):
                        var e_id = Int(W_sorted_labels[unsafe_offset=r])
                        var e_c_dist = comp.symmetric_distance(c_id, e_id)
                        if e_c_dist < c_dist:
                            keep = False
                            break
                    if keep:
                        W_sorted_labels[unsafe_offset=return_size] = Int32(c_id)
                        W_sorted_dist[unsafe_offset=return_size] = c_dist
                        return_size += 1

                # Add links from pt_id to neighbors
                for j in range(return_size):
                    var n_id = Int(W_sorted_labels[unsafe_offset=j])
                    self.hnsw.add_link(comp, pt_id, n_id, level, vt)

                    var n_comp = self.storage.get_distance_computer(
                        self.storage.get_vector(n_id)
                    )
                    self.hnsw.add_link(n_comp, n_id, pt_id, level, vt)

                # The entry point for the next level is the closest in W (which is index 0)

                if total_w > 0:
                    ep_id = Int(W_sorted_labels[unsafe_offset=0])
                    ep_dist = W_sorted_dist[unsafe_offset=0]

            self.vt_pool.release(vt_id)

        parallelize[add_point](n - start_idx)

        # Update entry point globally
        for i in range(n):
            if pt_levels[unsafe_offset=i] > self.hnsw.max_level:
                self.hnsw.max_level = pt_levels[unsafe_offset=i]
                self.hnsw.entry_point = old_ntotal + i

        pt_levels.unsafe_free()
        self.ntotal += n

    def search(
        self,
        x: Span[Float32, _],
        k: Int,
        mut distances: Span[mut=True, Float32, _],
        mut labels: Span[mut=True, Int, _],
    ):
        """Searches the HNSW index for the k nearest neighbors for each query vector in x."""
        var n = len(x) // self.d
        if n == 0 or self.ntotal == 0:
            var labels_ptr = labels.unsafe_ptr()
            var distances_ptr = distances.unsafe_ptr()
            for i in range(n * k):
                labels_ptr[unsafe_offset=i] = -1
                distances_ptr[unsafe_offset=i] = 0.0
            return

        var empty_filter = Span[UInt8, MutUntrackedOrigin]()
        self._search_impl[False](x, k, distances, labels, empty_filter)

    def search(
        self,
        x: Span[Float32, _],
        k: Int,
        mut distances: Span[mut=True, Float32, _],
        mut labels: Span[mut=True, Int, _],
        filter: Span[UInt8, _],
    ):
        """Searches the HNSW index for the k nearest neighbors for each query vector in x."""
        var n = len(x) // self.d
        if n == 0 or self.ntotal == 0:
            var labels_ptr = labels.unsafe_ptr()
            var distances_ptr = distances.unsafe_ptr()
            for i in range(n * k):
                labels_ptr[unsafe_offset=i] = -1
                distances_ptr[unsafe_offset=i] = 0.0
            return

        if len(filter) > 0:
            self._search_impl[True](x, k, distances, labels, filter)
        else:
            self._search_impl[False](x, k, distances, labels, filter)

    def _search_range[
        HAS_FILTER: Bool,
        x_origin: Origin,
        distances_origin: MutOrigin,
        labels_origin: MutOrigin,
        filter_origin: Origin,
    ](
        self,
        start: Int,
        stride: Int,
        n: Int,
        k: Int,
        x_ptr: Pointer[Float32, x_origin],
        distances_ptr: Pointer[Float32, distances_origin],
        labels_ptr: Pointer[Int, labels_origin],
        filter_ptr: Pointer[UInt8, filter_origin],
    ):
        var ef = self.hnsw.efSearch
        if ef < k:
            ef = k

        var vt_id = self.vt_pool.acquire()
        var vt = self.vt_pool.get(vt_id)
        var i = start
        while i < n:
            var w_dist_array = InlineArray[Float32, 2048](uninitialized=True)
            var w_labels_array = InlineArray[Int32, 2048](uninitialized=True)
            var W_dist = w_dist_array.unsafe_ptr()
            var W_labels = w_labels_array.unsafe_ptr()

            for j in range(k):
                distances_ptr[unsafe_offset=i * k + j] = -1.0
                labels_ptr[unsafe_offset=i * k + j] = -1

            var q_ptr = x_ptr.unsafe_offset(i * self.d)
            var comp = self.storage.get_distance_computer(q_ptr)

            var ep_id = self.hnsw.entry_point
            var ep_dist = comp.distance(ep_id)

            # Greedy search down to level 1
            for level in range(self.hnsw.max_level, 0, -1):
                var changed = True
                while changed:
                    changed = False
                    var neighbors_info = self.hnsw.get_neighbors(ep_id, level)
                    var neighbors = neighbors_info.ptr
                    var max_links = neighbors_info.max_links
                    for j in range(max_links):
                        var neigh = neighbors[unsafe_offset=j]
                        if neigh < 0:
                            break

                        # Pipeline prefetch the next neighbor's vector
                        if j + 1 < max_links:
                            var next_neigh = neighbors[unsafe_offset=j + 1]
                            if next_neigh >= 0:
                                comp.prefetch_vector(Int(next_neigh))

                        var d = comp.distance(Int(neigh))
                        if d < ep_dist:
                            ep_dist = d
                            ep_id = Int(neigh)
                            changed = True

            # Beam search on level 0
            var W_size = self._dispatch_search_layer[HAS_FILTER=HAS_FILTER](
                comp, ep_id, ep_dist, ef, 0, vt, W_dist, W_labels, filter_ptr
            )

            # W is a max-heap of nearest neighbors. Approximate storage keeps
            # twice as many candidates and reranks them with the original
            # Float32 vectors. This is equivalent to FAISS IndexRefineFlat
            # with k_factor=2, but is automatic because SQ8 already owns the
            # original vectors.
            var res_dist_ptr = distances_ptr.unsafe_offset(i * k)
            var res_labels_ptr = labels_ptr.unsafe_offset(i * k)

            var candidate_count = k
            if not comp.is_exact():
                candidate_count = min(ef, k * 2)

            while W_size > candidate_count:
                _ = max_heap_pop(W_dist, W_labels, W_size)
                W_size -= 1

            if not comp.is_exact():
                # Keep the exact top-k in the caller's result buffers as a
                # max-heap, then pop it backwards to produce ascending order.
                var result_count = 0
                for j in range(W_size):
                    var label = Int(W_labels[unsafe_offset=j])
                    var db_f32 = self.storage.get_vector(label)
                    var exact_dist: Float32
                    if self.metric_type == METRIC_L2:
                        exact_dist = l2_distance_simd[64](
                            q_ptr, db_f32, self.d
                        )
                    else:
                        exact_dist = -inner_product_simd[64](
                            q_ptr, db_f32, self.d
                        )

                    if result_count < k:
                        max_heap_push(
                            res_dist_ptr,
                            res_labels_ptr,
                            result_count,
                            exact_dist,
                            label,
                        )
                        result_count += 1
                    elif exact_dist < res_dist_ptr[unsafe_offset=0]:
                        max_heap_replace_top(
                            res_dist_ptr,
                            res_labels_ptr,
                            k,
                            exact_dist,
                            label,
                        )

                var heap_size = result_count
                for j in range(result_count):
                    var popped = max_heap_pop(
                        res_dist_ptr, res_labels_ptr, heap_size
                    )
                    heap_size -= 1
                    var idx = result_count - 1 - j
                    res_dist_ptr[unsafe_offset=idx] = popped.dist
                    res_labels_ptr[unsafe_offset=idx] = popped.label

                for j in range(result_count, k):
                    res_dist_ptr[unsafe_offset=j] = 0.0
                    res_labels_ptr[unsafe_offset=j] = -1
            else:
                var result_count = W_size
                for j in range(result_count):
                    var popped = max_heap_pop(
                        W_dist, W_labels, W_size
                    )
                    W_size -= 1
                    var idx = result_count - 1 - j
                    res_dist_ptr[unsafe_offset=idx] = popped.dist
                    res_labels_ptr[unsafe_offset=idx] = Int(popped.label)

                for j in range(result_count, k):
                    res_dist_ptr[unsafe_offset=j] = 0.0
                    res_labels_ptr[unsafe_offset=j] = -1

            if self.metric_type == METRIC_INNER_PRODUCT:
                for j in range(k):
                    if res_labels_ptr[unsafe_offset=j] != -1:
                        res_dist_ptr[unsafe_offset=j] = -res_dist_ptr[unsafe_offset=j]

            i += stride

        self.vt_pool.release(vt_id)

    def _search_impl[HAS_FILTER: Bool](
        self,
        x: Span[Float32, _],
        k: Int,
        mut distances: Span[mut=True, Float32, _],
        mut labels: Span[mut=True, Int, _],
        filter: Span[UInt8, _],
    ):
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        var distances_ptr = distances.unsafe_ptr()
        var labels_ptr = labels.unsafe_ptr()
        var filter_ptr = filter.unsafe_ptr()
        comptime storage_kind = Self.StorageType.HNSW_PTHREAD_STORAGE_KIND
        var native_workers = min(n, num_logical_cores())
        var runtime_workers = parallelism_level()
        if runtime_workers <= 0:
            runtime_workers = 1

        # AsyncRT exposes only the performance-core pool on Apple Silicon.
        # For large batches, native pthreads let the OS schedule work across
        # every logical core. Small batches keep the lower-overhead std path.
        if (
            n >= 256
            and native_workers > runtime_workers
            and (storage_kind == 0 or storage_kind == 1)
        ):
            var threads = unsafe_alloc[UInt](native_workers)
            var contexts = unsafe_alloc[HNSWPThreadContext](native_workers)
            var self_address = Int(Pointer(to=self))

            for worker_id in range(native_workers):
                var context = HNSWPThreadContext(
                    self_address,
                    Int(x_ptr),
                    Int(distances_ptr),
                    Int(labels_ptr),
                    Int(filter_ptr),
                    n,
                    k,
                    worker_id,
                    native_workers,
                    storage_kind,
                    HAS_FILTER,
                )
                contexts.unsafe_offset(worker_id).unsafe_write(context^)
                threads[unsafe_offset=worker_id] = 0

            for worker_id in range(1, native_workers):
                var status = external_call["pthread_create", Int](
                    threads.unsafe_offset(worker_id),
                    Int(0),
                    mojovec_hnsw_pthread_worker,
                    Int(contexts.unsafe_offset(worker_id)),
                )
                if status != 0:
                    threads[unsafe_offset=worker_id] = 0

            self._search_range[HAS_FILTER](
                0,
                native_workers,
                n,
                k,
                x_ptr,
                distances_ptr,
                labels_ptr,
                filter_ptr,
            )

            for worker_id in range(1, native_workers):
                if threads[unsafe_offset=worker_id] == 0:
                    self._search_range[HAS_FILTER](
                        worker_id,
                        native_workers,
                        n,
                        k,
                        x_ptr,
                        distances_ptr,
                        labels_ptr,
                        filter_ptr,
                    )
                else:
                    _ = external_call["pthread_join", Int](
                        threads[unsafe_offset=worker_id], Int(0)
                    )

            contexts.unsafe_free()
            threads.unsafe_free()
            return

        # Give each parallel task exclusive ownership of one visited table for
        # its whole query chunk. This removes pool scans and atomics from every
        # query while retaining a compact table count tied to the CPU.
        var num_workers = min(n, runtime_workers)

        @parameter
        def search_worker(worker_id: Int):
            self._search_range[HAS_FILTER](
                worker_id,
                num_workers,
                n,
                k,
                x_ptr,
                distances_ptr,
                labels_ptr,
                filter_ptr,
            )

        parallelize[search_worker](num_workers, num_workers)


def mojovec_hnsw_pthread_worker(context_address: Int) abi("C") -> Int:
    """Runs one strided HNSW query range on a native CPU thread."""
    var context = Pointer[
        HNSWPThreadContext, MutUntrackedOrigin
    ](unsafe_from_address=context_address)
    var queries = Pointer[
        Float32, MutUntrackedOrigin
    ](unsafe_from_address=context[].queries_address)
    var distances = Pointer[
        Float32, MutUntrackedOrigin
    ](unsafe_from_address=context[].distances_address)
    var labels = Pointer[
        Int, MutUntrackedOrigin
    ](unsafe_from_address=context[].labels_address)
    var filter = Pointer[
        UInt8, MutUntrackedOrigin
    ](unsafe_from_address=context[].filter_address)

    if context[].storage_kind == 0:
        var index = Pointer[
            IndexHNSW[IndexFlat], MutUntrackedOrigin
        ](unsafe_from_address=context[].index_address)
        if context[].has_filter:
            index[]._search_range[True](
                context[].worker_id,
                context[].worker_count,
                context[].query_count,
                context[].k,
                queries,
                distances,
                labels,
                filter,
            )
        else:
            index[]._search_range[False](
                context[].worker_id,
                context[].worker_count,
                context[].query_count,
                context[].k,
                queries,
                distances,
                labels,
                filter,
            )
    else:
        var index = Pointer[
            IndexHNSW[IndexFlatSQ8], MutUntrackedOrigin
        ](unsafe_from_address=context[].index_address)
        if context[].has_filter:
            index[]._search_range[True](
                context[].worker_id,
                context[].worker_count,
                context[].query_count,
                context[].k,
                queries,
                distances,
                labels,
                filter,
            )
        else:
            index[]._search_range[False](
                context[].worker_id,
                context[].worker_count,
                context[].query_count,
                context[].k,
                queries,
                distances,
                labels,
                filter,
            )
    return 0
