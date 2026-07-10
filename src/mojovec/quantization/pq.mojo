from std.memory import alloc
from ..clustering.kmeans import KMeans
from ..utils.distances import l2_distance_simd, inner_product_simd
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT

struct ProductQuantizer(Movable):
    var d: Int
    var M: Int
    var ksub: Int
    var dsub: Int
    var centroids: UnsafePointer[Float32, MutUntrackedOrigin]
    var is_trained: Bool

    def __init__(out self, d: Int, M: Int, ksub: Int = 256):
        self.d = d
        self.M = M
        self.ksub = ksub
        self.dsub = d // M
        self.centroids = alloc[Float32](M * ksub * self.dsub)
        self.is_trained = False

    def __init__(out self, *, deinit move: Self):
        self.d = move.d
        self.M = move.M
        self.ksub = move.ksub
        self.dsub = move.dsub
        self.centroids = move.centroids
        self.is_trained = move.is_trained

    def __del__(deinit self):
        if Int(self.centroids) != 0:
            self.centroids.free()

    def train(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        if self.is_trained: return

        # For each subspace, we need to extract the sub-vectors.
        var sub_x = alloc[Float32](n * self.dsub)
        
        for m in range(self.M):
            # Extract sub-vectors
            for i in range(n):
                for j in range(self.dsub):
                    sub_x[i * self.dsub + j] = x[i * self.d + m * self.dsub + j]
            
            var kmeans = KMeans(self.dsub, self.ksub, 15)
            kmeans.train(n, sub_x)
            
            # Copy learned centroids
            var offset = m * self.ksub * self.dsub
            for i in range(self.ksub * self.dsub):
                self.centroids[offset + i] = kmeans.centroids[i]
                
        sub_x.free()
        self.is_trained = True

    def compute_codes(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], codes: UnsafePointer[UInt8, MutUntrackedOrigin]):
        for i in range(n):
            var x_ptr = x + i * self.d
            var codes_ptr = codes + i * self.M
            
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

    def decode(self, n: Int, codes: UnsafePointer[UInt8, MutUntrackedOrigin], x: UnsafePointer[Float32, MutUntrackedOrigin]):
        for i in range(n):
            var codes_ptr = codes + i * self.M
            var x_ptr = x + i * self.d
            
            for m in range(self.M):
                var k = Int(codes_ptr[m])
                var c_ptr = self.centroids + m * self.ksub * self.dsub + k * self.dsub
                var sub_x = x_ptr + m * self.dsub
                
                for j in range(self.dsub):
                    sub_x[j] = c_ptr[j]

    def compute_distance_table(self, query: UnsafePointer[Float32, MutUntrackedOrigin], dis_table: UnsafePointer[Float32, MutUntrackedOrigin], metric_type: MetricType = METRIC_L2):
        for m in range(self.M):
            var sub_q = query + m * self.dsub
            var centroids_m = self.centroids + m * self.ksub * self.dsub
            var table_m = dis_table + m * self.ksub
            
            for k in range(self.ksub):
                var c_ptr = centroids_m + k * self.dsub
                if metric_type == METRIC_L2:
                    table_m[k] = l2_distance_simd[4](sub_q, c_ptr, self.dsub)
                else:
                    table_m[k] = -inner_product_simd[4](sub_q, c_ptr, self.dsub)
