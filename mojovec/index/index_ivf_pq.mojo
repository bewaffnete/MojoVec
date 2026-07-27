from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..storage.inverted_lists import ArrayInvertedLists
from ..clustering.kmeans import KMeans
from ..quantization.pq import ProductQuantizer
from std.collections import List
from std.math import max, min, log2
from std.memory.span import Span

struct IndexIVFPQ[QuantizerType: QuantizerTrait](Index, Movable):
    """An Inverted File (IVF) index with Product Quantization (PQ) compression.
    
    This index uses a coarse quantizer to partition vectors into cells and a Product
    Quantizer to compress the vectors (or their residuals) within each cell, enabling
    highly memory-efficient approximate nearest neighbor search.
    """
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
        """Initializes an IVF-PQ index.
        
        Args:
            quantizer: Pointer to the coarse quantizer used for assigning vectors to lists.
            d: Dimensionality of the original vectors.
            nlist: Number of inverted lists (cells/clusters).
            M: Number of sub-vector spaces for product quantization.
            metric: The distance metric to use.
        """
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
        """Move constructor for the index."""
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
        
    def train(mut self, x: Span[Float32, _]):
        """Trains the coarse quantizer and the product quantizer.
        
        Args:
            x: Contiguous flattened training vectors.
        """
        if self.is_trained: return
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        
        # 1. Train Coarse Quantizer (K-Means)
        var kmeans = KMeans(self.d, self.nlist, 15)
        kmeans.train(x)
        self.quantizer[0].add(Span[Float32, MutUntrackedOrigin](ptr=kmeans.centroids, length=self.nlist * self.d))
        
        # 2. Train PQ 
        if self.by_residual:
            var residuals = List[Float32](
                unsafe_uninit_length=n * self.d
            )
            var assign_distances = List[Float32](unsafe_uninit_length=n)
            var assign_labels = List[Int](unsafe_uninit_length=n)
            
            var d_span = Span[mut=True, Float32](assign_distances)
            var l_span = Span[mut=True, Int](assign_labels)
            self.quantizer[0].search(x, 1, d_span, l_span)
            

            # Use IndexFlat's get_vector method to compute residuals against coarse centroids.
            for i in range(n):
                var list_no = assign_labels[i]
                if list_no < 0 or list_no >= self.nlist:
                    list_no = 0
                var c_ptr = self.quantizer[0].get_vector(list_no)
                for j in range(self.d):
                    residuals[i * self.d + j] = (
                        x_ptr[i * self.d + j] - c_ptr[j]
                    )
                    
            self.pq.train(
                Span[mut=True, Float32](residuals)
            )
        else:
            self.pq.train(x)
        
        self.is_trained = True

    def add(mut self, x: Span[Float32, _]):
        """Adds vectors to the index, automatically assigning sequential IDs.
        
        Args:
            x: A safe Span pointing to the contiguous array of vectors.
        """
        var n = len(x) // self.d
        var ids = List[Int](unsafe_uninit_length=n)
        for i in range(n):
            ids[i] = self.ntotal + i
        self.add_with_ids(x, Span[mut=True, Int](ids))

    def add_with_ids(
        mut self,
        x: Span[Float32, _],
        ids: Span[Int, _],
    ):
        """Compresses and adds vectors to the index with explicitly provided IDs.
        
        Args:
            x: A safe Span pointing to the contiguous array of vectors.
            ids: Vector IDs, one per input vector.
        """
        if not self.is_trained: return
            
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        var ids_ptr = ids.unsafe_ptr()
        
        var assign_distances = List[Float32](unsafe_uninit_length=n)
        var assign_labels = List[Int](unsafe_uninit_length=n)
        
        var d_span = Span[mut=True, Float32](assign_distances)
        var l_span = Span[mut=True, Int](assign_labels)
        self.quantizer[0].search(x, 1, d_span, l_span)
        
        var pq_codes = List[UInt8](unsafe_uninit_length=n * self.M)
        var pq_codes_span = Span[mut=True, UInt8](pq_codes)
        
        if self.by_residual:
            var residuals = List[Float32](
                unsafe_uninit_length=n * self.d
            )
            for i in range(n):
                var list_no = assign_labels[i]
                if list_no < 0 or list_no >= self.nlist:
                    list_no = 0
                var c_ptr = self.quantizer[0].get_vector(list_no)
                for j in range(self.d):
                    residuals[i * self.d + j] = x_ptr[i * self.d + j] - c_ptr[j]
            self.pq.compute_codes(
                Span[mut=True, Float32](residuals),
                pq_codes_span,
            )
        else:
            self.pq.compute_codes(
                x,
                pq_codes_span,
            )
        
        for i in range(n):
            var list_no = assign_labels[i]
            if list_no < 0 or list_no >= self.nlist: continue
            
            var single_id = ids_ptr + i
            var single_code = pq_codes.unsafe_ptr() + (i * self.M)
            self.invlists.add_entries(
                list_no,
                Span[Int, _](ptr=single_id, length=1),
                Span[UInt8, _](
                    ptr=single_code, length=self.M
                ),
            )
            
        self.ntotal += n

    def search(
        self,
        x: Span[Float32, _],
        k: Int,
        mut distances: Span[mut=True, Float32, _],
        mut labels: Span[mut=True, Int, _],
    ):
        var empty_filter = Span[UInt8, MutUntrackedOrigin]()
        self.search(x, k, distances, labels, empty_filter)

    def search(
        self,
        x: Span[Float32, _],
        k: Int,
        mut distances: Span[mut=True, Float32, _],
        mut labels: Span[mut=True, Int, _],
        filter: Span[UInt8, _],
    ):
        """Searches the index for the k nearest neighbors using asymmetric distance computation (ADC).
        
        Args:
            x: A safe Span pointing to the contiguous array of query vectors.
            k: The number of nearest neighbors to retrieve for each query.
            distances: An output Span for storing distances.
            labels: An output Span for storing the IDs.
            filter: A bitset/mask (optional) for filtering candidates.
        """
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        var distances_ptr = distances.unsafe_ptr()
        var labels_ptr = labels.unsafe_ptr()
        
        if not self.is_trained or self.ntotal == 0:
            for i in range(n * k):
                distances_ptr[i] = 1e38
                labels_ptr[i] = -1
            return
            
        var nprobe = self.nprobe
        if nprobe > self.nlist: nprobe = self.nlist
        
        var probe_count = n * nprobe
        var q_distances = List[Float32](
            unsafe_uninit_length=probe_count
        )
        var q_labels = List[Int](unsafe_uninit_length=probe_count)
        var qd_span = Span[mut=True, Float32](q_distances)
        var ql_span = Span[mut=True, Int](q_labels)
        
        self.quantizer[0].search(x, nprobe, qd_span, ql_span)
        
        var dis_table = List[Float32](
            unsafe_uninit_length=self.M * self.pq.ksub
        )
        var dis_table_span = Span[mut=True, Float32](dis_table)
        var dis_table_ptr = dis_table.unsafe_ptr()
        var q_residual = List[Float32](
            unsafe_uninit_length=self.d
        )
        var q_residual_ptr = q_residual.unsafe_ptr()
        
        for i in range(n):
            var q_ptr = x_ptr + i * self.d
            var res_dist_ptr = distances_ptr + i * k
            var res_labels_ptr = labels_ptr + i * k
            var heap_size = 0
            
            if not self.by_residual:
                self.pq.compute_distance_table(
                    Span[Float32, _](ptr=q_ptr, length=self.d),
                    dis_table_span,
                    self.metric_type,
                )
            
            for p in range(nprobe):
                var list_no = q_labels[i * nprobe + p]
                if list_no < 0 or list_no >= self.nlist: continue
                
                var list_size = self.invlists.list_size(list_no)
                if list_size == 0: continue
                
                var list_codes = self.invlists.get_codes(list_no).unsafe_ptr()
                var list_ids = self.invlists.get_ids(list_no).unsafe_ptr()
                
                if self.by_residual:
                    var c_ptr = self.quantizer[0].get_vector(list_no)
                    for j in range(self.d):
                        q_residual_ptr[j] = q_ptr[j] - c_ptr[j]
                    self.pq.compute_distance_table(
                        Span[mut=True, Float32](q_residual),
                        dis_table_span,
                        self.metric_type,
                    )
                
                for j in range(list_size):
                    var c_ptr = list_codes + j * self.M
                    var dist: Float32 = 0.0
                    
                    # O(M) distance computation!
                    for m in range(self.M):
                        var sub_k = Int(c_ptr[m])
                        dist += dis_table_ptr[m * self.pq.ksub + sub_k]
                        
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
                        
