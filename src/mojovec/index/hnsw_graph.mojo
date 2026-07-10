from std.random import rand
from std.math import log
from std.atomic import Atomic
from src.mojovec.utils.heap import max_heap_push, max_heap_replace_top, min_heap_push, min_heap_pop
from src.mojovec.utils.distance_computer import DistanceComputerTrait
from .hnsw_visited import VisitedTable

struct NeighborsInfo:
    var ptr: UnsafePointer[Int, MutUntrackedOrigin]
    var max_links: Int
    
    def __init__(out self, ptr: UnsafePointer[Int, MutUntrackedOrigin], max_links: Int):
        self.ptr = ptr
        self.max_links = max_links

struct HNSWGraph(Movable):
    var M: Int
    var efConstruction: Int
    var efSearch: Int
    
    var max_level: Int
    var entry_point: Int
    var ntotal: Int
    
    var levels: UnsafePointer[Int, MutUntrackedOrigin]
    var offsets: UnsafePointer[Int, MutUntrackedOrigin]
    var neighbors: UnsafePointer[Int, MutUntrackedOrigin]
    var cum_nneighbor_per_level: UnsafePointer[Int, MutUntrackedOrigin]
    
    var capacity: Int
    var neighbors_capacity: Int
    
    var next_tickets: UnsafePointer[UInt32, MutUntrackedOrigin]
    var now_serving: UnsafePointer[UInt32, MutUntrackedOrigin]
    var num_locks: Int
    
    def __init__(out self, M: Int = 32, efConstruction: Int = 40, efSearch: Int = 16):
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.max_level = -1
        self.entry_point = -1
        self.ntotal = 0
        
        self.capacity = 1024
        self.cum_nneighbor_per_level = alloc[Int](33)
        self.cum_nneighbor_per_level[0] = 0
        self.cum_nneighbor_per_level[1] = M * 2
        for i in range(2, 33):
            self.cum_nneighbor_per_level[i] = self.cum_nneighbor_per_level[i-1] + M
            
        self.neighbors_capacity = self.capacity * self.cum_nneighbor_per_level[4]
        
        self.levels = alloc[Int](self.capacity)
        self.offsets = alloc[Int](self.capacity + 1)
        self.offsets[0] = 0
        self.neighbors = alloc[Int](self.neighbors_capacity)
        for i in range(self.neighbors_capacity):
            self.neighbors[i] = -1
        
        self.num_locks = 65536
        self.next_tickets = alloc[UInt32](self.num_locks)
        self.now_serving = alloc[UInt32](self.num_locks)
        for i in range(self.num_locks):
            self.next_tickets[i] = 0
            self.now_serving[i] = 0
            
    def __del__(deinit self):
        if Int(self.levels) != 0: self.levels.free()
        if Int(self.offsets) != 0: self.offsets.free()
        if Int(self.neighbors) != 0: self.neighbors.free()
        if Int(self.cum_nneighbor_per_level) != 0: self.cum_nneighbor_per_level.free()
        if Int(self.next_tickets) != 0: self.next_tickets.free()
        if Int(self.now_serving) != 0: self.now_serving.free()
        
    def __init__(out self, *, deinit move: Self):
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
        
    def random_level(self) -> Int:
        var ptr = alloc[Float64](1)
        rand(ptr, 1)
        var f = ptr[0]
        ptr.free()
        if f < 1e-10:
            f = 1e-10
        var mult = 1.0 / log(Float64(self.M))
        var level = Int(-log(f) * mult)
        if level > 32:
            level = 32
        return level

    @always_inline
    def get_neighbors(self, node: Int, level: Int) -> NeighborsInfo:
        var base_offset = self.offsets[node]
        var level_offset = self.cum_nneighbor_per_level[level]
        var max_size = self.cum_nneighbor_per_level[level + 1] - level_offset
        return NeighborsInfo(self.neighbors + (base_offset + level_offset), max_size)
        
    @always_inline
    def set_neighbors_len(self, node: Int, level: Int, new_len: Int):
        var info = self.get_neighbors(node, level)
        # We store the sentinel -1 to mark the end of the neighbor list
        if new_len < info.max_links:
            info.ptr[new_len] = -1
            
    @always_inline
    def lock_node(self, node: Int):
        var lock_idx = node % self.num_locks
        var ticket = Atomic.fetch_add(self.next_tickets + lock_idx, 1)
        while Atomic.load(self.now_serving + lock_idx) != ticket:
            pass

    @always_inline
    def unlock_node(self, node: Int):
        var lock_idx = node % self.num_locks
        Atomic.fetch_add(self.now_serving + lock_idx, 1)
        
    def _grow(mut self):
        var new_capacity = self.capacity * 2
        
        var new_levels = alloc[Int](new_capacity)
        for i in range(self.capacity):
            new_levels[i] = self.levels[i]
        self.levels.free()
        self.levels = new_levels
        
        var new_offsets = alloc[Int](new_capacity + 1)
        for i in range(self.capacity + 1):
            new_offsets[i] = self.offsets[i]
        self.offsets.free()
        self.offsets = new_offsets
        
        
        self.capacity = new_capacity

    def grow_neighbors(mut self, required_capacity: Int, current_offset: Int):
        var new_capacity = max(self.neighbors_capacity * 2, required_capacity)
        var new_neighbors = alloc[Int](new_capacity)
        for i in range(new_capacity):
            new_neighbors[i] = -1
        
        var old_total_size = current_offset
            
        if old_total_size > 0:
            for i in range(old_total_size):
                new_neighbors[i] = self.neighbors[i]
                
        if Int(self.neighbors) != 0:
            self.neighbors.free()
        self.neighbors = new_neighbors
        self.neighbors_capacity = new_capacity
        

            
    def search_layer[ComputerType: DistanceComputerTrait](
        self, 
        mut comp: ComputerType,
        ep_id: Int, 
        ep_dist: Float32, 
        ef: Int, 
        level: Int, 
        vt: UnsafePointer[VisitedTable, MutUntrackedOrigin],
        mut res_dist: UnsafePointer[Float32, MutUntrackedOrigin], 
        mut res_labels: UnsafePointer[Int, MutUntrackedOrigin]
    ) -> Int:
        var C_cap = ef * 2
        if C_cap < 256: C_cap = 256
        var C_dist = alloc[Float32](C_cap)
        var C_labels = alloc[Int](C_cap)
        var C_size = 0
        
        var W_dist = res_dist
        var W_labels = res_labels
        var W_size = 0
        
        vt[].advance()
        vt[].set_visited(ep_id)
        
        min_heap_push(C_dist, C_labels, C_size, ep_dist, ep_id)
        C_size += 1
        
        max_heap_push(W_dist, W_labels, W_size, ep_dist, ep_id)
        W_size += 1
        
        while C_size > 0:
            var c_dist: Float32 = 0.0
            var c_id: Int = 0
            var popped = min_heap_pop(C_dist, C_labels, C_size)
            c_dist = popped.dist
            c_id = popped.label
            C_size -= 1
            
            var worst_w_dist = W_dist[0]
            if W_size == ef and c_dist > worst_w_dist:
                break
                
            var neighbors_info = self.get_neighbors(c_id, level)
            var neighbors = neighbors_info.ptr
            var max_links = neighbors_info.max_links
            for i in range(max_links):
                var e = neighbors[i]
                if e < 0:
                    break
                if not vt[].is_visited(e):
                    vt[].set_visited(e)
                    
                    # Prefetch next unvisited neighbor's vector while computing distance for current one.
                    # This hides memory latency: while CPU does SIMD math for vector `e`,
                    # the hardware prefetcher loads the next neighbor's data into L1 cache.
                    for look_ahead in range(i + 1, max_links):
                        var next_e = neighbors[look_ahead]
                        if next_e < 0:
                            break
                        if not vt[].is_visited(next_e):
                            comp.prefetch_vector(next_e)
                            break
                    
                    var e_dist = comp.distance(e)
                    
                    worst_w_dist = W_dist[0]
                    if W_size < ef or e_dist < worst_w_dist:
                        if C_size >= C_cap:
                            var new_cap = C_cap * 2
                            var new_C_dist = alloc[Float32](new_cap)
                            var new_C_labels = alloc[Int](new_cap)
                            for i in range(C_size):
                                new_C_dist[i] = C_dist[i]
                                new_C_labels[i] = C_labels[i]
                            C_dist.free()
                            C_labels.free()
                            C_dist = new_C_dist
                            C_labels = new_C_labels
                            C_cap = new_cap
                            
                        min_heap_push(C_dist, C_labels, C_size, e_dist, e)
                        C_size += 1
                        
                        if W_size < ef:
                            max_heap_push(W_dist, W_labels, W_size, e_dist, e)
                            W_size += 1
                        else:
                            max_heap_replace_top(W_dist, W_labels, ef, e_dist, e)
                            
        C_dist.free()
        C_labels.free()
        return W_size
        
    def shrink_neighbor_list[ComputerType: DistanceComputerTrait](
        self,
        mut comp: ComputerType,
        node: Int,
        level: Int,
        max_links: Int
    ):
        var info = self.get_neighbors(node, level)
        var neighbors = info.ptr
        
        # Count current links
        var current_links = 0
        while current_links < info.max_links and neighbors[current_links] != -1:
            current_links += 1
            
        if current_links <= max_links:
            return
            
        # Simple heuristic: keep the closest max_links neighbors
        # Sort by distance
        var dists = alloc[Float32](current_links)
        var labels = alloc[Int](current_links)
        
        for i in range(current_links):
            labels[i] = neighbors[i]
            dists[i] = comp.distance(neighbors[i])
            
        # We can just sort them manually (selection sort since M is small)
        for i in range(current_links):
            var best_idx = i
            var best_d = dists[i]
            for j in range(i + 1, current_links):
                if dists[j] < best_d:
                    best_d = dists[j]
                    best_idx = j
            if best_idx != i:
                var t_d = dists[i]
                dists[i] = dists[best_idx]
                dists[best_idx] = t_d
                var t_l = labels[i]
                labels[i] = labels[best_idx]
                labels[best_idx] = t_l
                
        # Write back
        for i in range(max_links):
            neighbors[i] = labels[i]
        self.set_neighbors_len(node, level, max_links)
        
        dists.free()
        labels.free()

    def add_link[ComputerType: DistanceComputerTrait](
        self,
        mut comp: ComputerType,
        src: Int,
        dest: Int,
        level: Int
    ):
        self.lock_node(src)
        var info = self.get_neighbors(src, level)
        var neighbors = info.ptr
        
        var max_links = self.M
        if level == 0:
            max_links = self.M * 2
            
        var i = 0
        while i < info.max_links and neighbors[i] != -1:
            if neighbors[i] == dest:
                self.unlock_node(src)
                return # Already linked
            i += 1
            
        if i < info.max_links:
            neighbors[i] = dest
            if i + 1 < info.max_links:
                neighbors[i + 1] = -1
            self.unlock_node(src)
        else:
            var C_size = info.max_links + 1
            var C_nodes = alloc[Int](C_size)
            var C_dists = alloc[Float32](C_size)
            
            for j in range(info.max_links):
                C_nodes[j] = neighbors[j]
                C_dists[j] = comp.distance(neighbors[j])
                
            C_nodes[info.max_links] = dest
            C_dists[info.max_links] = comp.distance(dest)
            
            for j in range(1, C_size):
                var key_node = C_nodes[j]
                var key_dist = C_dists[j]
                var p = j - 1
                while p >= 0 and C_dists[p] > key_dist:
                    C_nodes[p + 1] = C_nodes[p]
                    C_dists[p + 1] = C_dists[p]
                    p -= 1
                C_nodes[p + 1] = key_node
                C_dists[p + 1] = key_dist
                
            var return_list = alloc[Int](info.max_links)
            var return_size = 0
            
            for j in range(C_size):
                if return_size >= info.max_links:
                    break
                
                var c = C_nodes[j]
                var c_dist = C_dists[j]
                var keep = True
                
                for r in range(return_size):
                    var e = return_list[r]
                    var e_c_dist = comp.symmetric_distance(c, e)
                    if e_c_dist < c_dist:
                        keep = False
                        break
                        
                if keep:
                    return_list[return_size] = c
                    return_size += 1
                    
            for j in range(return_size):
                neighbors[j] = return_list[j]
                
            if return_size < info.max_links:
                neighbors[return_size] = -1
                
            self.unlock_node(src)
            return_list.free()
            C_nodes.free()
            C_dists.free()
