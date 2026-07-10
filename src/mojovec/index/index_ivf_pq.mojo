from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..storage.inverted_lists import ArrayInvertedLists
from ..clustering.kmeans import KMeans
from ..quantization.pq import ProductQuantizer
from std.memory import alloc

struct IndexIVFPQ[QuantizerType: QuantizerTrait](Index, Movable):
    var d: Int
    var nlist: Int
    var M: Int
    var nprobe: Int
    var ntotal: Int
    var metric_type: MetricType
    var is_trained: Bool
    
    var quantizer: UnsafePointer[Self.QuantizerType, MutUntrackedOrigin]
    var invlists: ArrayInvertedLists
    var pq: ProductQuantizer
    var by_residual: Bool
    
    def __init__(out self, quantizer: UnsafePointer[Self.QuantizerType, MutUntrackedOrigin], d: Int, nlist: Int, M: Int, metric: MetricType = METRIC_L2):
        self.d = d
        self.nlist = nlist
        self.M = M
        self.nprobe = 1
        self.ntotal = 0
        self.metric_type = metric
        self.is_trained = False
        
        self.quantizer = quantizer
        self.invlists = ArrayInvertedLists(nlist, M)  # Bucket element size is M bytes!
        self.pq = ProductQuantizer(d, M, 256)
        self.by_residual = True

    def __init__(out self, *, deinit move: Self):
        self.d = move.d
        self.nlist = move.nlist
        self.M = move.M
        self.nprobe = move.nprobe
        self.ntotal = move.ntotal
        self.metric_type = move.metric_type
        self.is_trained = move.is_trained
        self.quantizer = move.quantizer
        self.invlists = move.invlists^
        self.pq = move.pq^
        self.by_residual = move.by_residual
        
    def train(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        if self.is_trained: return
        
        # 1. Train Coarse Quantizer (K-Means)
        var kmeans = KMeans(self.d, self.nlist, 15)
        kmeans.train(n, x)
        self.quantizer[0].add(self.nlist, kmeans.centroids)
        
        # 2. Train PQ 
        if self.by_residual:
            var residuals = alloc[Float32](n * self.d)
            var assign_distances = alloc[Float32](n)
            var assign_labels = alloc[Int](n)
            
            self.quantizer[0].search(n, x, 1, assign_distances, assign_labels)
            

            # Wait, quantizer is IndexFlat, we can use its .codes directly!
            # But wait, IndexFlat has `get_vector(id)`
            for i in range(n):
                var list_no = assign_labels[i]
                if list_no < 0 or list_no >= self.nlist:
                    list_no = 0
                var c_ptr = self.quantizer[0].get_vector(list_no)
                for j in range(self.d):
                    residuals[i * self.d + j] = x[i * self.d + j] - c_ptr[j]
                    
            self.pq.train(n, residuals)
            
            residuals.free()
            assign_distances.free()
            assign_labels.free()
        else:
            self.pq.train(n, x)
        
        self.is_trained = True

    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        var ids = alloc[Int](n)
        for i in range(n):
            ids[i] = self.ntotal + i
        self.add_with_ids(n, x, ids)
        ids.free()

    def add_with_ids(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], ids: UnsafePointer[Int, MutUntrackedOrigin]):
        if not self.is_trained: return
            
        var assign_distances = alloc[Float32](n)
        var assign_labels = alloc[Int](n)
        self.quantizer[0].search(n, x, 1, assign_distances, assign_labels)
        
        var pq_codes = alloc[UInt8](n * self.M)
        
        if self.by_residual:
            var residuals = alloc[Float32](n * self.d)
            for i in range(n):
                var list_no = assign_labels[i]
                if list_no < 0 or list_no >= self.nlist:
                    list_no = 0
                var c_ptr = self.quantizer[0].get_vector(list_no)
                for j in range(self.d):
                    residuals[i * self.d + j] = x[i * self.d + j] - c_ptr[j]
            self.pq.compute_codes(n, residuals, pq_codes)
            residuals.free()
        else:
            self.pq.compute_codes(n, x, pq_codes)
        
        for i in range(n):
            var list_no = assign_labels[i]
            if list_no < 0 or list_no >= self.nlist: continue
            
            var single_id = ids + i
            var single_code = pq_codes + (i * self.M)
            self.invlists.add_entries(list_no, 1, single_id, single_code)
            
        self.ntotal += n
        
        assign_distances.free()
        assign_labels.free()
        pq_codes.free()

    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]):
        if not self.is_trained or self.ntotal == 0:
            for i in range(n * k):
                distances[i] = 1e38
                labels[i] = -1
            return
            
        var nprobe = self.nprobe
        if nprobe > self.nlist: nprobe = self.nlist
        
        var q_distances = alloc[Float32](n * nprobe)
        var q_labels = alloc[Int](n * nprobe)
        self.quantizer[0].search(n, x, nprobe, q_distances, q_labels)
        
        var dis_table = alloc[Float32](self.M * self.pq.ksub)
        var q_residual = alloc[Float32](self.d)
        
        for i in range(n):
            var q_ptr = x + i * self.d
            var res_dist_ptr = distances + i * k
            var res_labels_ptr = labels + i * k
            var heap_size = 0
            
            if not self.by_residual:
                self.pq.compute_distance_table(q_ptr, dis_table, self.metric_type)
            
            for p in range(nprobe):
                var list_no = q_labels[i * nprobe + p]
                if list_no < 0 or list_no >= self.nlist: continue
                
                var list_size = self.invlists.list_size(list_no)
                if list_size == 0: continue
                
                var list_codes = self.invlists.get_codes(list_no)
                var list_ids = self.invlists.get_ids(list_no)
                
                if self.by_residual:
                    var c_ptr = self.quantizer[0].get_vector(list_no)
                    for j in range(self.d):
                        q_residual[j] = q_ptr[j] - c_ptr[j]
                    self.pq.compute_distance_table(q_residual, dis_table, self.metric_type)
                
                for j in range(list_size):
                    var c_ptr = list_codes + j * self.M
                    var dist: Float32 = 0.0
                    
                    # O(M) distance computation!
                    for m in range(self.M):
                        var sub_k = Int(c_ptr[m])
                        dist += dis_table[m * self.pq.ksub + sub_k]
                        
                    if heap_size < k:
                        max_heap_push(res_dist_ptr, res_labels_ptr, heap_size, dist, list_ids[j])
                        heap_size += 1
                    elif dist < res_dist_ptr[0]:
                        max_heap_replace_top(res_dist_ptr, res_labels_ptr, k, dist, list_ids[j])
                        
            var current_k = heap_size
            for j in range(current_k):
                var popped = max_heap_pop(res_dist_ptr, res_labels_ptr, heap_size)
                heap_size -= 1
                var idx = current_k - 1 - j
                res_dist_ptr[idx] = popped.dist
                res_labels_ptr[idx] = popped.label
                        
            # Un-negate inner product distances
            if self.metric_type == METRIC_INNER_PRODUCT:
                for j in range(k):
                    res_dist_ptr[j] = -res_dist_ptr[j]
                        
        q_distances.free()
        q_labels.free()
        dis_table.free()
        q_residual.free()
