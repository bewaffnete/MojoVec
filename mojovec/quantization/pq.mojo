from std.collections import List
from std.ffi import external_call
from std.memory import alloc
from std.memory.span import Span
from ..clustering.kmeans import KMeans
from ..utils.distances import (
    inner_product_short4,
    inner_product_simd,
    l2_distance_short4,
    l2_distance_simd,
)
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT


struct PQTrainContext(Movable):
    """Arguments owned by one native PQ subspace training worker."""

    var input_address: Int
    var output_address: Int
    var n: Int
    var d: Int
    var dsub: Int
    var ksub: Int
    var subspace: Int

    def __init__(
        out self,
        input_address: Int,
        output_address: Int,
        n: Int,
        d: Int,
        dsub: Int,
        ksub: Int,
        subspace: Int,
    ):
        self.input_address = input_address
        self.output_address = output_address
        self.n = n
        self.d = d
        self.dsub = dsub
        self.ksub = ksub
        self.subspace = subspace


def _train_pq_subspace(context: PQTrainContext):
    """Extracts and trains one independent PQ subspace."""
    var x_ptr = UnsafePointer[
        Float32, MutUntrackedOrigin
    ](unsafe_from_address=context.input_address)
    var pq_centroids = UnsafePointer[
        Float32, MutUntrackedOrigin
    ](unsafe_from_address=context.output_address)
    var sub_x = List[Float32](
        unsafe_uninit_length=context.n * context.dsub
    )
    var sub_x_ptr = sub_x.unsafe_ptr()

    for i in range(context.n):
        for j in range(context.dsub):
            sub_x_ptr[i * context.dsub + j] = x_ptr[
                i * context.d + context.subspace * context.dsub + j
            ]

    var kmeans = KMeans(context.dsub, context.ksub, 15)
    kmeans.train_serial(Span[mut=True, Float32](sub_x))

    var offset = context.subspace * context.ksub * context.dsub
    for i in range(context.ksub * context.dsub):
        pq_centroids[offset + i] = kmeans.centroids[i]


def mojovec_pq_pthread_worker(context_address: Int) abi("C") -> Int:
    """Native pthread entry point for one PQ subspace."""
    var context = UnsafePointer[
        PQTrainContext, MutUntrackedOrigin
    ](unsafe_from_address=context_address)
    _train_pq_subspace(context[])
    return 0

struct ProductQuantizer(Movable):
    """Product Quantizer (PQ) for vector compression.
    
    Splits vectors into sub-vectors and quantizes each sub-space independently using K-Means.
    This allows representing high-dimensional vectors compactly as sequences of byte codes.
    """
    var d: Int
    var M: Int
    var ksub: Int
    var dsub: Int
    var centroids: List[Float32]
    var is_trained: Bool

    def __init__(out self, d: Int, M: Int, ksub: Int = 256) raises:
        """Initializes the Product Quantizer.
        
        Args:
            d: Dimensionality of the input vectors.
            M: Number of sub-vector spaces (must divide d).
            ksub: Number of centroids per sub-space (typically 256 for byte-sized codes).
        """
        if d <= 0:
            raise Error("PQ dimension must be positive.")
        if M <= 0 or M > d or d % M != 0:
            raise Error("PQ M must divide dimension and be between 1 and dimension.")
        if ksub <= 0 or ksub > 256:
            raise Error("PQ ksub must be between 1 and 256 for byte codes.")
        self.d = d
        self.M = M
        self.ksub = ksub
        self.dsub = d // M
        self.centroids = List[Float32](
            unsafe_uninit_length=M * ksub * self.dsub
        )
        self.is_trained = False

    def __init__(out self, *, deinit move: Self):
        """Move constructor for the product quantizer."""
        self.d = move.d
        self.M = move.M
        self.ksub = move.ksub
        self.dsub = move.dsub
        self.centroids = move.centroids^
        self.is_trained = move.is_trained

    def train(mut self, x: Span[Float32, _]):
        """Trains the quantizer by finding centroids for each sub-space using K-Means.
        
        Args:
            x: Contiguous flattened training vectors.
        """
        if self.is_trained: return
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()
        var centroids_ptr = self.centroids.unsafe_ptr()

        # Train independent subspaces on native pthreads. This keeps the
        # parallelism at the algorithm level while avoiding AsyncRT's repeated
        # raw-pointer worker-pool corruption on Linux x86. Each context owns a
        # disjoint output slice; the caller joins every worker before publish.
        var threads = alloc[UInt](self.M)
        var contexts = alloc[PQTrainContext](self.M)
        for m in range(self.M):
            var context = PQTrainContext(
                Int(x_ptr),
                Int(centroids_ptr),
                n,
                self.d,
                self.dsub,
                self.ksub,
                m,
            )
            (contexts + m).init_pointee_move(context^)
            threads[m] = 0

        for m in range(1, self.M):
            var status = external_call["pthread_create", Int](
                threads + m,
                Int(0),
                mojovec_pq_pthread_worker,
                Int(contexts + m),
            )
            if status != 0:
                threads[m] = 0

        _train_pq_subspace(contexts[0])

        for m in range(1, self.M):
            if threads[m] == 0:
                _train_pq_subspace(contexts[m])
            else:
                _ = external_call["pthread_join", Int](threads[m], Int(0))

        contexts.free()
        threads.free()

        self.is_trained = True

    def compute_codes(
        self,
        x: Span[Float32, _],
        mut codes: Span[mut=True, UInt8, _],
    ):
        """Encodes vectors into compact byte codes.
        
        Args:
            x: Contiguous flattened vectors.
            codes: Output storage for the encoded byte codes.
        """
        var n = len(x) // self.d
        var x_data = x.unsafe_ptr()
        var codes_data = codes.unsafe_ptr()
        var centroid_data = self.centroids.unsafe_ptr()
        for i in range(n):
            var x_ptr = x_data + i * self.d
            var codes_ptr = codes_data + i * self.M
            
            for m in range(self.M):
                var min_dist: Float32 = 1e38
                var best_k = -1
                var sub_x = x_ptr + m * self.dsub
                var centroids_m = centroid_data + m * self.ksub * self.dsub
                
                for k in range(self.ksub):
                    var c_ptr = centroids_m + k * self.dsub
                    var dist: Float32
                    if self.dsub <= 4:
                        dist = l2_distance_short4(
                            sub_x, c_ptr, self.dsub
                        )
                    else:
                        dist = l2_distance_simd[8](
                            sub_x, c_ptr, self.dsub
                        )
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_k = k
                        
                codes_ptr[m] = UInt8(best_k)

    def decode(
        self,
        codes: Span[UInt8, _],
        mut x: Span[mut=True, Float32, _],
    ):
        """Decodes byte codes back to approximate vectors.
        
        Args:
            codes: Contiguous byte codes.
            x: Output storage for reconstructed vectors.
        """
        var n = len(codes) // self.M
        var codes_data = codes.unsafe_ptr()
        var x_data = x.unsafe_ptr()
        var centroid_data = self.centroids.unsafe_ptr()
        for i in range(n):
            var codes_ptr = codes_data + i * self.M
            var x_ptr = x_data + i * self.d
            
            for m in range(self.M):
                var k = Int(codes_ptr[m])
                var c_ptr = (
                    centroid_data
                    + m * self.ksub * self.dsub
                    + k * self.dsub
                )
                var sub_x = x_ptr + m * self.dsub
                
                for j in range(self.dsub):
                    sub_x[j] = c_ptr[j]

    def compute_distance_table(
        self,
        query: Span[Float32, _],
        mut dis_table: Span[mut=True, Float32, _],
        metric_type: MetricType = METRIC_L2,
    ):
        """Precomputes distances between a query vector and all sub-space centroids.
        
        Args:
            query: A single query vector.
            dis_table: Output distance table.
            metric_type: The distance metric to use.
        """
        var query_ptr = query.unsafe_ptr()
        var table_ptr = dis_table.unsafe_ptr()
        var centroid_data = self.centroids.unsafe_ptr()
        for m in range(self.M):
            var sub_q = query_ptr + m * self.dsub
            var centroids_m = centroid_data + m * self.ksub * self.dsub
            var table_m = table_ptr + m * self.ksub
            
            for k in range(self.ksub):
                var c_ptr = centroids_m + k * self.dsub
                if metric_type == METRIC_L2:
                    if self.dsub <= 4:
                        table_m[k] = l2_distance_short4(
                            sub_q, c_ptr, self.dsub
                        )
                    else:
                        table_m[k] = l2_distance_simd[8](
                            sub_q, c_ptr, self.dsub
                        )
                else:
                    if self.dsub <= 4:
                        table_m[k] = -inner_product_short4(
                            sub_q, c_ptr, self.dsub
                        )
                    else:
                        table_m[k] = -inner_product_simd[4](
                            sub_q, c_ptr, self.dsub
                        )
