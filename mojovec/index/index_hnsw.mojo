from ..core.index import Index
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..utils.distance_computer import StorageTrait, DistanceComputerTrait
from ..utils.distances import l2_distance_simd, inner_product_simd
from .hnsw_graph import HNSWGraph
from .hnsw_visited import VisitedTable, VisitedTablePool
from std.algorithm import parallelize
from std.memory.span import Span
from std.memory import alloc


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
    ):
        """Initializes the HNSW index with the provided storage, dimension, metric type, and M parameter."""
        self.d = d
        self.ntotal = 0
        self.metric_type = metric_type
        self.is_trained = True
        self.storage = storage^
        self.hnsw = HNSWGraph(M=M)
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
        vt: UnsafePointer[VisitedTable, MutUntrackedOrigin],
        mut res_dist: UnsafePointer[Float32, origin1],
        mut res_labels: UnsafePointer[Int32, origin2],
        filter: UnsafePointer[UInt8, _],
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

        self.storage.add(x)
        var x_ptr = x.unsafe_ptr()

        var old_ntotal = self.ntotal
        var pt_levels = alloc[Int](n)

        # Preallocate topology
        for i in range(n):
            var pt_id = old_ntotal + i
            var pt_level = self.hnsw.random_level()
            pt_levels[i] = pt_level

            while pt_id >= self.hnsw.capacity:
                self.hnsw._grow()
                self.vt_pool.grow(self.hnsw.capacity)

            self.hnsw.levels[pt_id] = pt_level

            var size_needed = self.hnsw.cum_nneighbor_per_level[pt_level + 1]
            var current_offset = 0
            if pt_id > 0:
                current_offset = (
                    self.hnsw.offsets[pt_id - 1]
                    + self.hnsw.cum_nneighbor_per_level[
                        self.hnsw.levels[pt_id - 1] + 1
                    ]
                )

            if current_offset + size_needed > self.hnsw.neighbors_capacity:
                self.hnsw.grow_neighbors(
                    current_offset + size_needed, current_offset
                )

            self.hnsw.offsets[pt_id] = current_offset
            self.hnsw.ntotal = pt_id + 1

        var start_idx = 0
        if old_ntotal == 0:
            self.hnsw.max_level = pt_levels[0]
            self.hnsw.entry_point = 0
            start_idx = 1

        # Initialize neighbor lists sequentially to avoid race conditions
        # where thread A sees thread B's node before thread B initializes it.
        for i in range(n):
            var pt_id = old_ntotal + i
            for l in range(pt_levels[i] + 1):
                self.hnsw.set_neighbors_len(pt_id, l, 0)

        @parameter
        def add_point(i: Int):
            var actual_i = i + start_idx
            var pt_id = old_ntotal + actual_i
            var pt_level = pt_levels[actual_i]

            var q_ptr = x_ptr + (actual_i * self.d)
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
                        var neigh = neighbors[j]
                        if neigh < 0:
                            break
                            
                        # Pipeline prefetch the next neighbor's vector
                        if j + 1 < max_links:
                            var next_neigh = neighbors[j + 1]
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
                    alloc[UInt8](0)
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
                    W_sorted_dist[idx] = popped.dist
                    W_sorted_labels[idx] = popped.label

                var M_l = self.hnsw.M
                if level == 0:
                    M_l = self.hnsw.M * 2

                # Apply extended heuristic to select M_l diverse neighbors from the candidates
                var return_size = 0
                for j in range(total_w):
                    if return_size >= M_l:
                        break
                    var c_id = Int(W_sorted_labels[j])
                    var c_dist = W_sorted_dist[j]
                    var keep = True
                    for r in range(return_size):
                        var e_id = Int(W_sorted_labels[r])
                        var e_c_dist = comp.symmetric_distance(c_id, e_id)
                        if e_c_dist < c_dist:
                            keep = False
                            break
                    if keep:
                        W_sorted_labels[return_size] = Int32(c_id)
                        W_sorted_dist[return_size] = c_dist
                        return_size += 1

                # Add links from pt_id to neighbors
                for j in range(return_size):
                    var n_id = Int(W_sorted_labels[j])
                    self.hnsw.add_link(comp, pt_id, n_id, level, vt)

                    var n_comp = self.storage.get_distance_computer(
                        self.storage.get_vector(n_id)
                    )
                    self.hnsw.add_link(n_comp, n_id, pt_id, level, vt)

                # The entry point for the next level is the closest in W (which is index 0)

                if total_w > 0:
                    ep_id = Int(W_sorted_labels[0])
                    ep_dist = W_sorted_dist[0]

            self.vt_pool.release(vt_id)

        parallelize[add_point](n - start_idx)

        # Update entry point globally
        for i in range(n):
            if pt_levels[i] > self.hnsw.max_level:
                self.hnsw.max_level = pt_levels[i]
                self.hnsw.entry_point = old_ntotal + i

        pt_levels.free()
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
                labels_ptr[i] = -1
                distances_ptr[i] = 0.0
            return

        var empty_filter = Span[UInt8, _](ptr=alloc[UInt8](0), length=0)
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
                labels_ptr[i] = -1
                distances_ptr[i] = 0.0
            return

        if len(filter) > 0:
            self._search_impl[True](x, k, distances, labels, filter)
        else:
            self._search_impl[False](x, k, distances, labels, filter)

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

        var ef = self.hnsw.efSearch
        if ef < k:
            ef = k

        @parameter
        def search_point(i: Int):
            var vt_id = self.vt_pool.acquire()
            var vt = self.vt_pool.get(vt_id)

            var w_dist_array = InlineArray[Float32, 2048](uninitialized=True)
            var w_labels_array = InlineArray[Int32, 2048](uninitialized=True)
            var W_dist = w_dist_array.unsafe_ptr()
            var W_labels = w_labels_array.unsafe_ptr()

            for j in range(k):
                distances_ptr[i * k + j] = -1.0
                labels_ptr[i * k + j] = -1

            var q_ptr = x_ptr + (i * self.d)
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
                        var neigh = neighbors[j]
                        if neigh < 0:
                            break
                            
                        # Pipeline prefetch the next neighbor's vector
                        if j + 1 < max_links:
                            var next_neigh = neighbors[j + 1]
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

            # W is a max-heap of nearest neighbors. We need to pop them and reverse to get sorted order.
            var res_dist_ptr = distances_ptr + (i * k)
            var res_labels_ptr = labels_ptr + (i * k)

            # Pop to get sorted results (since it's a max heap, popping gives largest first)
            while W_size > k:
                _ = max_heap_pop(W_dist, W_labels, W_size)
                W_size -= 1

            var result_count = W_size
            for j in range(result_count):
                var popped = max_heap_pop(W_dist, W_labels, W_size)
                W_size -= 1
                var idx = result_count - 1 - j
                res_dist_ptr[idx] = popped.dist
                res_labels_ptr[idx] = Int(popped.label)

            for j in range(result_count, k):
                res_dist_ptr[j] = 0.0
                res_labels_ptr[j] = -1

            if not comp.is_exact():
                for j in range(result_count):
                    var l = res_labels_ptr[j]
                    if l >= 0:
                        var db_f32 = self.storage.get_vector(l)
                        var exact_dist: Float32 = 0.0
                        if self.metric_type == METRIC_L2:
                            exact_dist = l2_distance_simd[64](q_ptr, db_f32, self.d)
                        else:
                            exact_dist = -inner_product_simd[64](q_ptr, db_f32, self.d)
                        res_dist_ptr[j] = exact_dist
                        
                # Insertion sort to re-sort the top-K array
                for j in range(1, result_count):
                    var key_dist = res_dist_ptr[j]
                    var key_label = res_labels_ptr[j]
                    var m = j - 1
                    while m >= 0 and res_dist_ptr[m] > key_dist:
                        res_dist_ptr[m + 1] = res_dist_ptr[m]
                        res_labels_ptr[m + 1] = res_labels_ptr[m]
                        m -= 1
                    res_dist_ptr[m + 1] = key_dist
                    res_labels_ptr[m + 1] = key_label

            if self.metric_type == METRIC_INNER_PRODUCT:
                for j in range(k):
                    if res_labels_ptr[j] != -1:
                        res_dist_ptr[j] = -res_dist_ptr[j]

            self.vt_pool.release(vt_id)

        parallelize[search_point](n)
