from std.collections import List
from std.memory.span import Span
from ..clustering.kmeans import KMeans
from ..utils.distances import l2_distance_simd, inner_product_simd
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT

struct ProductQuantizer(Movable):
    """Product Quantizer (PQ) for vector compression.
    
    Splits vectors into sub-vectors and quantizes each sub-space independently using K-Means.
    This allows representing high-dimensional vectors compactly as sequences of byte codes.
    """
    var d: Int
    var M: Int
    var ksub: Int
    var dsub: Int
    var centroids: UnsafePointer[Float32, MutUntrackedOrigin]
    var is_trained: Bool

    def __init__(out self, d: Int, M: Int, ksub: Int = 256):
        """Initializes the Product Quantizer.
        
        Args:
            d: Dimensionality of the input vectors.
            M: Number of sub-vector spaces (must divide d).
            ksub: Number of centroids per sub-space (typically 256 for byte-sized codes).
        """
        self.d = d
        self.M = M
        self.ksub = ksub
        self.dsub = d // M
        self.centroids = alloc[Float32](M * ksub * self.dsub)
        self.is_trained = False

    def __init__(out self, *, deinit move: Self):
        """Move constructor for the product quantizer."""
        self.d = move.d
        self.M = move.M
        self.ksub = move.ksub
        self.dsub = move.dsub
        self.centroids = move.centroids
        self.is_trained = move.is_trained

    def __del__(deinit self):
        """Deallocates the centroids array."""
        if Int(self.centroids) != 0:
            self.centroids.free()

    def train(mut self, x: Span[Float32, _]):
        """Trains the quantizer by finding centroids for each sub-space using K-Means.
        
        Args:
            x: Contiguous flattened training vectors.
        """
        if self.is_trained: return
        var n = len(x) // self.d
        var x_ptr = x.unsafe_ptr()

        # For each subspace, we need to extract the sub-vectors.
        var sub_x = List[Float32](
            unsafe_uninit_length=n * self.dsub
        )
        var sub_x_ptr = sub_x.unsafe_ptr()
        
        for m in range(self.M):
            # Extract sub-vectors
            for i in range(n):
                for j in range(self.dsub):
                    sub_x_ptr[i * self.dsub + j] = x_ptr[
                        i * self.d + m * self.dsub + j
                    ]
            
            var kmeans = KMeans(self.dsub, self.ksub, 15)
            kmeans.train(
                Span[mut=True, Float32](sub_x)
            )
            
            # Copy learned centroids
            var offset = m * self.ksub * self.dsub
            for i in range(self.ksub * self.dsub):
                self.centroids[offset + i] = kmeans.centroids[i]
                
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
        for i in range(n):
            var x_ptr = x_data + i * self.d
            var codes_ptr = codes_data + i * self.M
            
            for m in range(self.M):
                var min_dist: Float32 = 1e38
                var best_k = -1
                var sub_x = x_ptr + m * self.dsub
                var centroids_m = self.centroids + m * self.ksub * self.dsub
                
                for k in range(self.ksub):
                    var c_ptr = centroids_m + k * self.dsub
                    var dist = l2_distance_simd[4](sub_x, c_ptr, self.dsub)
                    
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
        for i in range(n):
            var codes_ptr = codes_data + i * self.M
            var x_ptr = x_data + i * self.d
            
            for m in range(self.M):
                var k = Int(codes_ptr[m])
                var c_ptr = self.centroids + m * self.ksub * self.dsub + k * self.dsub
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
        for m in range(self.M):
            var sub_q = query_ptr + m * self.dsub
            var centroids_m = self.centroids + m * self.ksub * self.dsub
            var table_m = table_ptr + m * self.ksub
            
            for k in range(self.ksub):
                var c_ptr = centroids_m + k * self.dsub
                if metric_type == METRIC_L2:
                    table_m[k] = l2_distance_simd[4](sub_q, c_ptr, self.dsub)
                else:
                    table_m[k] = -inner_product_simd[4](sub_q, c_ptr, self.dsub)
