from std.memory.alloc import unsafe_alloc
from std.random import rand
from std.math import log
from std.atomic import Atomic
from std.collections import InlineArray
from std.sys.intrinsics import prefetch, PrefetchOptions
from mojovec.utils.heap import (
    max_heap_push,
    max_heap_replace_top,
    min_heap_push,
    min_heap_pop,
)
from ..utils.distance_computer import DistanceComputerTrait
from .hnsw_visited import VisitedTable
from mojovec.io.memory_map import FileMemoryMap
from mojovec.core.validation import _validate_hnsw_parameters


struct NeighborsInfo:
    """Contains information about a node's neighbors in the HNSW graph."""
    var ptr: Pointer[Int32, MutUntrackedOrigin]
    var max_links: Int

    def __init__(
        out self, ptr: Pointer[Int32, MutUntrackedOrigin], max_links: Int
    ):
        """Initializes neighbor information with a pointer to the links and the maximum allowed links."""
        self.ptr = ptr
        self.max_links = max_links


struct HNSWGraph(Movable):
    """Represents the Hierarchical Navigable Small World (HNSW) graph structure."""
    var M: Int
    var efConstruction: Int
    var efSearch: Int

    var max_level: Int
    var entry_point: Int
    var ntotal: Int

    var levels: Pointer[Int, MutUntrackedOrigin]
    var offsets: Pointer[Int, MutUntrackedOrigin]
    var neighbors: Pointer[Int32, MutUntrackedOrigin]
    var cum_nneighbor_per_level: Pointer[Int, MutUntrackedOrigin]

    var capacity: Int
    var neighbors_capacity: Int

    var next_tickets: Pointer[UInt32, MutUntrackedOrigin]
    var now_serving: Pointer[UInt32, MutUntrackedOrigin]
    var num_locks: Int
    var _mapping: FileMemoryMap

    def __init__(
        out self, M: Int = 32, efConstruction: Int = 40, efSearch: Int = 16
    ) raises:
        """Initializes the HNSW graph with the given hyperparameters."""
        _validate_hnsw_parameters(M, efConstruction, efSearch)
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.max_level = -1
        self.entry_point = -1
        self.ntotal = 0

        self.capacity = 1024
        self.cum_nneighbor_per_level = unsafe_alloc[Int](33)
        self.cum_nneighbor_per_level[unsafe_offset=0] = 0
        self.cum_nneighbor_per_level[unsafe_offset=1] = self.M * 2
        for i in range(2, 33):
            self.cum_nneighbor_per_level[unsafe_offset=i] = (
                self.cum_nneighbor_per_level[unsafe_offset=i - 1] + self.M
            )

        self.neighbors_capacity = (
            self.capacity * self.cum_nneighbor_per_level[unsafe_offset=4]
        )

        self.levels = unsafe_alloc[Int](self.capacity)
        self.offsets = unsafe_alloc[Int](self.capacity + 1)
        self.offsets[unsafe_offset=0] = 0
        self.neighbors = unsafe_alloc[Int32](self.neighbors_capacity)
        for i in range(self.neighbors_capacity):
            self.neighbors[unsafe_offset=i] = -1

        self.num_locks = 65536
        self.next_tickets = unsafe_alloc[UInt32](self.num_locks)
        self.now_serving = unsafe_alloc[UInt32](self.num_locks)
        for i in range(self.num_locks):
            self.next_tickets[unsafe_offset=i] = 0
            self.now_serving[unsafe_offset=i] = 0
        self._mapping = FileMemoryMap()

    def __deinit__(deinit self):
        """Frees the underlying memory of the HNSW graph."""
        if not self._mapping.is_active():
            if Int(self.levels) != 0:
                self.levels.unsafe_free()
            if Int(self.offsets) != 0:
                self.offsets.unsafe_free()
            if Int(self.neighbors) != 0:
                self.neighbors.unsafe_free()
            if Int(self.cum_nneighbor_per_level) != 0:
                self.cum_nneighbor_per_level.unsafe_free()
        self._mapping.close()
        if Int(self.next_tickets) != 0:
            self.next_tickets.unsafe_free()
        if Int(self.now_serving) != 0:
            self.now_serving.unsafe_free()

    def __init__(out self, *, deinit move: Self):
        """Moves the HNSW graph."""
        self.M = move.M
        self.efConstruction = move.efConstruction
        self.efSearch = move.efSearch
        self.max_level = move.max_level
        self.entry_point = move.entry_point
        self.ntotal = move.ntotal
        self.capacity = move.capacity
        self.neighbors_capacity = move.neighbors_capacity
        self.levels = move.levels
        self.offsets = move.offsets
        self.neighbors = move.neighbors
        self.cum_nneighbor_per_level = move.cum_nneighbor_per_level
        self.next_tickets = move.next_tickets
        self.now_serving = move.now_serving
        self.num_locks = move.num_locks
        self._mapping = move._mapping^

    @always_inline
    def is_memory_mapped(self) -> Bool:
        return self._mapping.is_active()

    def _detach_mapped(mut self):
        if not self._mapping.is_active():
            return
        var new_levels = unsafe_alloc[Int](max(self.capacity, 1))
        var new_offsets = unsafe_alloc[Int](max(self.capacity + 1, 1))
        var new_neighbors = unsafe_alloc[Int32](max(self.neighbors_capacity, 1))
        var new_cumulative = unsafe_alloc[Int](33)
        for i in range(self.capacity):
            new_levels[unsafe_offset=i] = self.levels[unsafe_offset=i]
        for i in range(self.capacity + 1):
            new_offsets[unsafe_offset=i] = self.offsets[unsafe_offset=i]
        for i in range(self.neighbors_capacity):
            new_neighbors[unsafe_offset=i] = self.neighbors[unsafe_offset=i]
        for i in range(33):
            new_cumulative[unsafe_offset=i] = self.cum_nneighbor_per_level[unsafe_offset=i]
        self._mapping.close()
        self.levels = new_levels
        self.offsets = new_offsets
        self.neighbors = new_neighbors
        self.cum_nneighbor_per_level = new_cumulative

    def random_level(self) -> Int:
        """Generates a random level for a new node based on the graph's M parameter."""
        var arr = InlineArray[Float64, 1](uninitialized=True)
        var ptr = arr.unsafe_ptr()
        rand(ptr, 1)
        var f = ptr[unsafe_offset=0]
        if f < 1e-10:
            f = 1e-10
        var mult = 1.0 / log(Float64(self.M))
        var level = Int(-log(f) * mult)
        if level > 32:
            level = 32
        return level

    @always_inline
    def get_neighbors(self, node: Int, level: Int) -> NeighborsInfo:
        """Retrieves the neighbor information for a given node at a specific level."""
        var base_offset = self.offsets[unsafe_offset=node]
        var level_offset = self.cum_nneighbor_per_level[unsafe_offset=level]
        var max_size = self.cum_nneighbor_per_level[unsafe_offset=level + 1] - level_offset
        return NeighborsInfo(
            self.neighbors.unsafe_offset(base_offset + level_offset), max_size
        )

    @always_inline
    def set_neighbors_len(self, node: Int, level: Int, new_len: Int):
        """Sets the effective length of a node's neighbor list at a given level by appending a sentinel."""
        var info = self.get_neighbors(node, level)
        # We store the sentinel -1 to mark the end of the neighbor list
        if new_len < info.max_links:
            info.ptr[unsafe_offset=new_len] = -1

    @always_inline
    def lock_node(self, node: Int):
        """Acquires a ticket lock for a specific node to allow thread-safe updates."""
        var lock_idx = node % self.num_locks
        var ticket = Atomic.fetch_add(self.next_tickets.unsafe_offset(lock_idx), 1)
        while Atomic.load(self.now_serving.unsafe_offset(lock_idx)) != ticket:
            pass

    @always_inline
    def unlock_node(self, node: Int):
        """Releases the ticket lock for a specific node."""
        var lock_idx = node % self.num_locks
        _ = Atomic.fetch_add(self.now_serving.unsafe_offset(lock_idx), 1)

    def _grow(mut self):
        """Doubles the capacity of the node level and offset arrays."""
        self._detach_mapped()
        var new_capacity = max(self.capacity * 2, 1)

        var new_levels = unsafe_alloc[Int](new_capacity)
        for i in range(self.capacity):
            new_levels[unsafe_offset=i] = self.levels[unsafe_offset=i]
        self.levels.unsafe_free()
        self.levels = new_levels

        var new_offsets = unsafe_alloc[Int](new_capacity + 1)
        for i in range(self.capacity + 1):
            new_offsets[unsafe_offset=i] = self.offsets[unsafe_offset=i]
        self.offsets.unsafe_free()
        self.offsets = new_offsets

        self.capacity = new_capacity

    def grow_neighbors(mut self, required_capacity: Int, current_offset: Int):
        """Grows the neighbor links array to accommodate more links."""
        self._detach_mapped()
        var new_capacity = max(self.neighbors_capacity * 2, required_capacity)
        var new_neighbors = unsafe_alloc[Int32](new_capacity)
        for i in range(new_capacity):
            new_neighbors[unsafe_offset=i] = -1

        var old_total_size = current_offset

        if old_total_size > 0:
            for i in range(old_total_size):
                new_neighbors[unsafe_offset=i] = self.neighbors[unsafe_offset=i]

        if Int(self.neighbors) != 0:
            self.neighbors.unsafe_free()
        self.neighbors = new_neighbors
        self.neighbors_capacity = new_capacity

    def search_layer[
        ComputerType: DistanceComputerTrait,
        origin1: MutOrigin,
        origin2: MutOrigin,
        MAX_LINKS: Int = 0,
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
        """Performs a greedy beam search on a specific layer of the HNSW graph."""
        var safe_ef = ef
        if safe_ef > 2048:
            safe_ef = 2048

        var c_dist_array = InlineArray[Float32, 2048](uninitialized=True)
        var c_labels_array = InlineArray[Int32, 2048](uninitialized=True)
        var C_dist = c_dist_array.unsafe_ptr()
        var C_labels = c_labels_array.unsafe_ptr()
        var C_cap = 2048  # Max pre-allocated capacity
        var C_size = 0

        var W_dist = res_dist
        var W_labels = res_labels
        var W_size = 0

        vt[].advance()
        vt[].set_visited(ep_id)

        min_heap_push(C_dist, C_labels, C_size, ep_dist, Int32(ep_id))
        C_size += 1

        comptime if HAS_FILTER:
            if filter[unsafe_offset=ep_id] == 0:
                max_heap_push(W_dist, W_labels, W_size, ep_dist, Int32(ep_id))
                W_size += 1
        else:
            max_heap_push(W_dist, W_labels, W_size, ep_dist, Int32(ep_id))
            W_size += 1

        while C_size > 0:
            var popped = min_heap_pop(C_dist, C_labels, C_size)
            var c_dist = popped.dist
            var c_id = popped.label
            C_size -= 1

            var worst_w_dist = W_dist[unsafe_offset=0]
            if W_size == safe_ef and c_dist > worst_w_dist:
                break

            # Prefetch the next node's neighbor list if there are still candidates left
            if C_size > 0:
                var next_c_id = C_labels[unsafe_offset=0]
                var next_info = self.get_neighbors(Int(next_c_id), level)
                comptime opts_list = PrefetchOptions().for_read().low_locality().to_data_cache()
                prefetch[opts_list](next_info.ptr.unsafe_bitcast[UInt8]())

            var neighbors_info = self.get_neighbors(Int(c_id), level)
            var neighbors = neighbors_info.ptr
            var max_links = neighbors_info.max_links

            var n_buffered = 0
            var buffered_ids = InlineArray[Int32, 4](uninitialized=True)
            var buffered_e = buffered_ids.unsafe_ptr()

            @parameter
            def process_buffered_ids():
                if n_buffered == 4:
                    var d0123 = comp.distance_batch4(Int(buffered_e[unsafe_offset=0]), Int(buffered_e[unsafe_offset=1]), Int(buffered_e[unsafe_offset=2]), Int(buffered_e[unsafe_offset=3]))
                    for j in range(4):
                        var e_id = buffered_e[unsafe_offset=j]
                        var e_dist = d0123[j]
                        var worst_w_dist = W_dist[unsafe_offset=0]
                        if W_size < safe_ef or e_dist < worst_w_dist:
                            if C_size >= C_cap:
                                pass
                            else:
                                min_heap_push(C_dist, C_labels, C_size, e_dist, e_id)
                                C_size += 1

                            if W_size < safe_ef:
                                comptime if HAS_FILTER:
                                    if filter[unsafe_offset=Int(e_id)] > 0:
                                        continue
                                max_heap_push(W_dist, W_labels, W_size, e_dist, e_id)
                                W_size += 1
                            else:
                                comptime if HAS_FILTER:
                                    if filter[unsafe_offset=Int(e_id)] > 0:
                                        continue
                                max_heap_replace_top(W_dist, W_labels, safe_ef, e_dist, e_id)
                    n_buffered = 0

            @parameter
            def process_remainder():
                for j in range(n_buffered):
                    var e_id = buffered_e[unsafe_offset=j]
                    var threshold: Float32 = Float32.MAX
                    var worst_w_dist = W_dist[unsafe_offset=0]
                    if W_size >= safe_ef:
                        threshold = worst_w_dist

                    var e_dist = comp.distance(Int(e_id), threshold)
                    if W_size < safe_ef or e_dist < worst_w_dist:
                        if C_size >= C_cap:
                            pass
                        else:
                            min_heap_push(C_dist, C_labels, C_size, e_dist, e_id)
                            C_size += 1

                        if W_size < safe_ef:
                            comptime if HAS_FILTER:
                                if filter[unsafe_offset=Int(e_id)] > 0:
                                    continue
                            max_heap_push(W_dist, W_labels, W_size, e_dist, e_id)
                            W_size += 1
                        else:
                            comptime if HAS_FILTER:
                                if filter[unsafe_offset=Int(e_id)] > 0:
                                    continue
                            max_heap_replace_top(W_dist, W_labels, safe_ef, e_dist, e_id)

            comptime if MAX_LINKS > 0:
                var look_ahead_idx = 0
                var prefetched_count = 0
                while look_ahead_idx < MAX_LINKS and prefetched_count < 4:
                    var next_e = neighbors[unsafe_offset=look_ahead_idx]
                    if next_e < 0:
                        break
                    if not vt[].is_visited(Int(next_e)):
                        comp.prefetch_vector(Int(next_e))
                        prefetched_count += 1
                    look_ahead_idx += 1

                for i in range(MAX_LINKS):
                    var e = neighbors[unsafe_offset=i]
                    if e < 0:
                        break

                    # Prefetch the visited table flag for the next neighbor
                    if i + 1 < MAX_LINKS:
                        var next_e = neighbors[unsafe_offset=i + 1]
                        if next_e >= 0:
                            vt[].prefetch(Int(next_e))

                    if not vt[].is_visited(Int(e)):
                        vt[].set_visited(Int(e))

                        # Advance look_ahead_idx to prefetch 1 new unvisited neighbor
                        while look_ahead_idx < MAX_LINKS:
                            var next_e = neighbors[unsafe_offset=look_ahead_idx]
                            look_ahead_idx += 1
                            if next_e < 0:
                                break
                            if not vt[].is_visited(Int(next_e)):
                                comp.prefetch_vector(Int(next_e))
                                break

                        buffered_e[unsafe_offset=n_buffered] = e
                        n_buffered += 1
                        if n_buffered == 4:
                            process_buffered_ids()

                process_remainder()
            else:
                var look_ahead_idx = 0
                var prefetched_count = 0
                while look_ahead_idx < max_links and prefetched_count < 4:
                    var next_e = neighbors[unsafe_offset=look_ahead_idx]
                    if next_e < 0:
                        break
                    if not vt[].is_visited(Int(next_e)):
                        comp.prefetch_vector(Int(next_e))
                        prefetched_count += 1
                    look_ahead_idx += 1

                for i in range(max_links):
                    var e = neighbors[unsafe_offset=i]
                    if e < 0:
                        break

                    # Prefetch the visited table flag for the next neighbor
                    if i + 1 < max_links:
                        var next_e = neighbors[unsafe_offset=i + 1]
                        if next_e >= 0:
                            vt[].prefetch(Int(next_e))

                    if not vt[].is_visited(Int(e)):
                        vt[].set_visited(Int(e))

                        # Advance look_ahead_idx to prefetch 1 new unvisited neighbor
                        while look_ahead_idx < max_links:
                            var next_e = neighbors[unsafe_offset=look_ahead_idx]
                            look_ahead_idx += 1
                            if next_e < 0:
                                break
                            if not vt[].is_visited(Int(next_e)):
                                comp.prefetch_vector(Int(next_e))
                                break

                        buffered_e[unsafe_offset=n_buffered] = e
                        n_buffered += 1
                        if n_buffered == 4:
                            process_buffered_ids()

                process_remainder()

        return W_size

    def shrink_neighbor_list[
        ComputerType: DistanceComputerTrait
    ](self, mut comp: ComputerType, node: Int, level: Int, max_links: Int):
        """Shrinks a node's neighbor list by keeping only the closest max_links neighbors."""
        var info = self.get_neighbors(node, level)
        var neighbors = info.ptr

        # Count current links
        var current_links = 0
        while current_links < info.max_links and neighbors[unsafe_offset=current_links] != -1:
            current_links += 1

        if current_links <= max_links:
            return

        # Simple heuristic: keep the closest max_links neighbors
        # Sort by distance
        var dists_array = InlineArray[Float32, 2048](uninitialized=True)
        var labels_array = InlineArray[Int32, 2048](uninitialized=True)
        var dists = dists_array.unsafe_ptr()
        var labels = labels_array.unsafe_ptr()

        for i in range(current_links):
            labels[unsafe_offset=i] = neighbors[unsafe_offset=i]
            dists[unsafe_offset=i] = comp.distance(Int(neighbors[unsafe_offset=i]))

        # We can just sort them manually (selection sort since M is small)
        for i in range(current_links):
            var best_idx = i
            var best_d = dists[unsafe_offset=i]
            for j in range(i + 1, current_links):
                if dists[unsafe_offset=j] < best_d:
                    best_d = dists[unsafe_offset=j]
                    best_idx = j
            if best_idx != i:
                var t_d = dists[unsafe_offset=i]
                dists[unsafe_offset=i] = dists[unsafe_offset=best_idx]
                dists[unsafe_offset=best_idx] = t_d
                var t_l = labels[unsafe_offset=i]
                labels[unsafe_offset=i] = labels[unsafe_offset=best_idx]
                labels[unsafe_offset=best_idx] = t_l

        # Write back
        for i in range(max_links):
            neighbors[unsafe_offset=i] = labels[unsafe_offset=i]
        self.set_neighbors_len(node, level, max_links)

    def add_link[
        ComputerType: DistanceComputerTrait
    ](
        self,
        mut comp: ComputerType,
        src: Int,
        dest: Int,
        level: Int,
        vt: Pointer[VisitedTable, MutUntrackedOrigin],
    ):
        """Adds a bidirectional link between two nodes, applying heuristics to maintain graph quality."""
        self.lock_node(src)
        var info = self.get_neighbors(src, level)
        var neighbors = info.ptr

        var i = 0
        while i < info.max_links and neighbors[unsafe_offset=i] != -1:
            if neighbors[unsafe_offset=i] == Int32(dest):
                self.unlock_node(src)
                return  # Already linked
            i += 1

        if i < info.max_links:
            neighbors[unsafe_offset=i] = Int32(dest)
            if i + 1 < info.max_links:
                neighbors[unsafe_offset=i + 1] = -1
            self.unlock_node(src)
        else:
            var C_size = info.max_links + 1
            if C_size > 2048:
                C_size = 2048
            var c_nodes_array = InlineArray[Int32, 2048](uninitialized=True)
            var c_dists_array = InlineArray[Float32, 2048](uninitialized=True)
            var C_nodes = c_nodes_array.unsafe_ptr()
            var C_dists = c_dists_array.unsafe_ptr()

            for j in range(info.max_links):
                var node_dist = comp.symmetric_distance(src, Int(neighbors[unsafe_offset=j]))
                min_heap_push(C_dists, C_nodes, j, node_dist, neighbors[unsafe_offset=j])

            var dest_dist = comp.symmetric_distance(src, dest)
            min_heap_push(C_dists, C_nodes, info.max_links, dest_dist, Int32(dest))

            var return_list_array = InlineArray[Int32, 2048](uninitialized=True)
            var return_list = return_list_array.unsafe_ptr()
            var return_size = 0

            var current_heap_size = C_size

            while current_heap_size > 0:
                if return_size >= info.max_links:
                    break

                var popped = min_heap_pop(C_dists, C_nodes, current_heap_size)
                current_heap_size -= 1
                var c = popped.label
                var c_dist = popped.dist
                var keep = True

                # Prefetch 'c' vector since it will be compared against multiple 'e' vectors
                comp.prefetch_vector(Int(c))

                for r in range(return_size):
                    var e = return_list[unsafe_offset=r]

                    # Prefetch the next 'e' vector to hide memory latency
                    if r + 1 < return_size:
                        comp.prefetch_vector(Int(return_list[unsafe_offset=r + 1]))

                    var e_c_dist = comp.symmetric_distance(Int(c), Int(e))
                    if e_c_dist < c_dist:
                        keep = False
                        break

                if keep:
                    return_list[unsafe_offset=return_size] = c
                    return_size += 1

            for j in range(return_size):
                neighbors[unsafe_offset=j] = return_list[unsafe_offset=j]

            if return_size < info.max_links:
                neighbors[unsafe_offset=return_size] = -1

            self.unlock_node(src)
