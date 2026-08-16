from ..core.index import Index, QuantizerTrait
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from ..utils.distances import l2_distance_simd, inner_product_simd
from ..utils.heap import max_heap_push, max_heap_replace_top, max_heap_pop
from ..storage.inverted_lists import ArrayInvertedLists
from ..clustering.kmeans import KMeans
from std.collections import List
from std.collections.span import Span

struct IndexIVFFlat[QuantizerType: QuantizerTrait](Index, Movable):
    """An Inverted File (IVF) index with exact flat storage for vectors.

    This index uses a coarse quantizer to partition the vector space into cells (inverted lists)
    and stores the exact vectors in these lists. During search, it only compares the query
    with vectors in the most promising cells.
    """
    var d: Int
    var nlist: Int
    var nprobe: Int
    var ntotal: Int
    var metric_type: MetricType
    var is_trained: Bool
    
    # The coarse quantizer has exactly one owner and the same lifetime as the
    # IVF index. Store it inline instead of exposing manual heap ownership.
    var quantizer: Self.QuantizerType
    var invlists: ArrayInvertedLists
    
    def __init__(
        out self,
        var quantizer: Self.QuantizerType,
        d: Int,
        nlist: Int,
        metric: MetricType = METRIC_L2,
    ):
        """Initializes an IVF-Flat index.

        Args:
            quantizer: Owned coarse quantizer used to assign vectors to lists.
            d: Dimensionality of the vectors.
            nlist: Number of inverted lists (cells/clusters).
            metric: The distance metric to use (e.g., L2 or Inner Product).
        """
        self.d = d
        self.nlist = nlist
        self.nprobe = 1
        self.ntotal = 0
        self.metric_type = metric
        self.is_trained = False
        
        self.quantizer = quantizer^
        self.invlists = ArrayInvertedLists(nlist, self.d * 4)

    def __init__(out self, *, deinit move: Self):
        """Move constructor for the index."""
        self.d = move.d
        self.nlist = move.nlist
        self.nprobe = move.nprobe
        self.ntotal = move.ntotal
        self.metric_type = move.metric_type
        self.is_trained = move.is_trained
        self.quantizer = move.quantizer^
        self.invlists = move.invlists^

    def train(mut self, x: Span[Float32, _]):
        """Trains the coarse quantizer using a set of training vectors.
        
        Args:
            x: Contiguous flattened training vectors.
        """
        if self.is_trained: return

        var kmeans = KMeans(self.d, self.nlist, 15)
        kmeans.train(x)
        var coarse_centroids = kmeans.take_centroids()
        self.quantizer.add(Span(coarse_centroids))
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
        """Adds vectors to the index with explicitly provided IDs.
        
        Args:
            x: A safe Span pointing to the contiguous array of vectors.
            ids: Vector IDs, one per input vector.
        """
        if not self.is_trained:
            # Cannot add without training
            return

        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        var ids_ptr = ids.unsafe_ptr()

        var assign_distances = List[Float32](unsafe_uninit_length=n)
        var assign_labels = List[Int](unsafe_uninit_length=n)

        var d_span = Span[mut=True, Float32](assign_distances)
        var l_span = Span[mut=True, Int](assign_labels)
        self.quantizer.search(x, 1, d_span, l_span)
        var assign_labels_ptr = assign_labels.unsafe_ptr()

        # In a real scenario we could group vectors by list_no to minimize resize calls.
        # But ArrayInvertedLists already has O(1) amortized add via capacity doubling.
        var code_ptr = x_ptr.unsafe_bitcast[UInt8]()
        var code_size = self.d * 4

        for i in range(n):
            var list_no = assign_labels_ptr[unsafe_offset=i]
            if list_no < 0 or list_no >= self.nlist: continue

            var single_id = ids_ptr.unsafe_offset(i)
            var single_code = code_ptr.unsafe_offset(i * code_size)
            self.invlists.add_entries(
                list_no,
                Span[Int, _](unsafe_ptr=single_id, length=1),
                Span[UInt8, _](
                    unsafe_ptr=single_code, length=code_size
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
        """Searches the index for the k nearest neighbors of the given query vectors.
        
        Args:
            x: A safe Span pointing to the contiguous array of query vectors.
            k: The number of nearest neighbors to retrieve for each query.
            distances: An output Span for storing the distances of the k nearest neighbors.
            labels: An output Span for storing the IDs of the k nearest neighbors.
            filter: An optional bitmask (as Span) to filter candidates by ID.
        """
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        var distances_ptr = distances.unsafe_ptr()
        var labels_ptr = labels.unsafe_ptr()

        if not self.is_trained or self.ntotal == 0:
            for i in range(n * k):
                distances_ptr[unsafe_offset=i] = 1e38
                labels_ptr[unsafe_offset=i] = -1
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

        self.quantizer.search(x, nprobe, qd_span, ql_span)
        var q_labels_ptr = q_labels.unsafe_ptr()

        for i in range(n):
            var q_ptr = x_ptr.unsafe_offset(i * self.d)
            var res_dist_ptr = distances_ptr.unsafe_offset(i * k)
            var res_labels_ptr = labels_ptr.unsafe_offset(i * k)
            var heap_size = 0

            # A probed IVF subset may contain fewer than k candidates. Keep the
            # unused result tail deterministic instead of exposing allocator data.
            for j in range(k):
                res_dist_ptr[unsafe_offset=j] = 1e38
                res_labels_ptr[unsafe_offset=j] = -1

            for p in range(nprobe):
                var list_no = q_labels_ptr[unsafe_offset=i * nprobe + p]
                if list_no < 0 or list_no >= self.nlist: continue

                var list_size = self.invlists.list_size(list_no)
                if list_size == 0: continue

                var list_codes = (
                    self.invlists.get_codes(list_no)
                    .unsafe_ptr()
                    .unsafe_bitcast[Float32]()
                )
                var list_ids = self.invlists.get_ids(list_no).unsafe_ptr()

                for j in range(list_size):
                    var db_ptr = list_codes.unsafe_offset(j * self.d)
                    var dist: Float32

                    if self.metric_type == METRIC_L2:
                        dist = l2_distance_simd[8](q_ptr, db_ptr, self.d)
                    else:
                        dist = -inner_product_simd[4](q_ptr, db_ptr, self.d)

                    if heap_size < k:
                        max_heap_push(res_dist_ptr, res_labels_ptr, heap_size, dist, list_ids[unsafe_offset=j])
                        heap_size += 1
                    elif dist < res_dist_ptr[unsafe_offset=0]:
                        max_heap_replace_top(res_dist_ptr, res_labels_ptr, k, dist, list_ids[unsafe_offset=j])

            var current_k = heap_size
            for j in range(current_k):
                var popped = max_heap_pop(res_dist_ptr, res_labels_ptr, heap_size)
                heap_size -= 1
                var idx = current_k - 1 - j
                res_dist_ptr[unsafe_offset=idx] = popped.dist
                res_labels_ptr[unsafe_offset=idx] = popped.label

            if self.metric_type == METRIC_INNER_PRODUCT:
                for j in range(current_k):
                    res_dist_ptr[unsafe_offset=j] = -res_dist_ptr[unsafe_offset=j]

