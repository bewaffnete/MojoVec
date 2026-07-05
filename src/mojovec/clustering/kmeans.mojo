from std.algorithm import parallelize
from std.memory import alloc
from std.random import random_si64
from ..utils.distances import l2_distance_simd

struct KMeans:
    var d: Int
    var k: Int
    var niter: Int
    var centroids: UnsafePointer[Float32, MutUntrackedOrigin]
    var assignments: UnsafePointer[Int, MutUntrackedOrigin]
    var counts: UnsafePointer[Int, MutUntrackedOrigin]
    
    def __init__(out self, d: Int, k: Int, niter: Int = 15):
        self.d = d
        self.k = k
        self.niter = niter
        self.centroids = alloc[Float32](k * d)
        self.assignments = alloc[Int](1)
        self.counts = alloc[Int](k)
        
    def __del__(deinit self):
        if Int(self.centroids) != 0: self.centroids.free()
        if Int(self.assignments) != 0: self.assignments.free()
        if Int(self.counts) != 0: self.counts.free()
            
    def train(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        if n == 0: return
            
        if Int(self.assignments) != 0: self.assignments.free()
        self.assignments = alloc[Int](n)
        
        # 1. Initialize centroids (random subsampling)
        for i in range(self.k):
            var src_idx = Int(random_si64(0, Int64(n - 1)))
            var src_ptr = x + src_idx * self.d
            var dst_ptr = self.centroids + i * self.d
            for j in range(self.d):
                dst_ptr[j] = src_ptr[j]
                
        # Main loop
        for _ in range(self.niter):
            # E-step: Assign points to centroids
            @parameter
            def process_point(i: Int):
                var min_dist: Float32 = 1e38
                var best_c = -1
                var x_ptr = x + i * self.d
                
                for c in range(self.k):
                    var c_ptr = self.centroids + c * self.d
                    var dist = l2_distance_simd[4](x_ptr, c_ptr, self.d)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_c = c
                        
                self.assignments[i] = best_c
                
            parallelize[process_point](n)
            
            # M-step: Update centroids
            for c in range(self.k):
                self.counts[c] = 0
                var c_ptr = self.centroids + c * self.d
                for j in range(self.d):
                    c_ptr[j] = 0.0
                    
            for i in range(n):
                var c = self.assignments[i]
                self.counts[c] += 1
                var c_ptr = self.centroids + c * self.d
                var x_ptr = x + i * self.d
                
                for j in range(self.d):
                    c_ptr[j] += x_ptr[j]
                
            for c in range(self.k):
                var count = self.counts[c]
                if count > 0:
                    var c_ptr = self.centroids + c * self.d
                    var inv_count: Float32 = 1.0 / Float32(count)
                    
                    for j in range(self.d):
                        c_ptr[j] *= inv_count
