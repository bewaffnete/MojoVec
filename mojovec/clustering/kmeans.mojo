from std.algorithm import parallelize
from std.memory import alloc
from std.random import random_si64
from ..utils.distances import l2_distance_simd
from std.math import min

struct KMeans:
    """K-Means clustering algorithm for vector quantization.
    
    Partitions a set of vectors into k clusters, finding the centroids
    that minimize the distance between points and their assigned centroids.
    """
    var d: Int
    var k: Int
    var niter: Int
    var centroids: UnsafePointer[Float32, MutUntrackedOrigin]
    var assignments: UnsafePointer[Int, MutUntrackedOrigin]
    var counts: UnsafePointer[Int, MutUntrackedOrigin]
    
    def __init__(out self, d: Int, k: Int, niter: Int = 15):
        """Initializes the K-Means clustering algorithm.
        
        Args:
            d: Dimensionality of the vectors.
            k: Number of clusters (centroids).
            niter: Number of iterations to perform during training.
        """
        self.d = d
        self.k = k
        self.niter = niter
        self.centroids = alloc[Float32](k * d)
        self.assignments = alloc[Int](1)
        self.counts = alloc[Int](k)
        
    def __del__(deinit self):
        """Deallocates memory used for centroids and assignments."""
        if Int(self.centroids) != 0: self.centroids.free()
        if Int(self.assignments) != 0: self.assignments.free()
        if Int(self.counts) != 0: self.counts.free()
            
    def train(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        """Trains the K-Means model to find cluster centroids.
        
        Args:
            n: Number of training vectors.
            x: Pointer to the contiguous array of training vectors.
        """
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
                
        var num_chunks = 32
        var chunk_size = (n + num_chunks - 1) // num_chunks
        var local_centroids = alloc[Float32](num_chunks * self.k * self.d)
        var local_counts = alloc[Int](num_chunks * self.k)
        
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
            
            # Zero out thread-local accumulators
            for i in range(num_chunks * self.k * self.d):
                local_centroids[i] = 0.0
            for i in range(num_chunks * self.k):
                local_counts[i] = 0
                
            # M-step: Update centroids in parallel
            @parameter
            def process_chunk(chunk_id: Int):
                var start = chunk_id * chunk_size
                var end = min(start + chunk_size, n)
                var my_centroids = local_centroids + chunk_id * self.k * self.d
                var my_counts = local_counts + chunk_id * self.k
                
                for i in range(start, end):
                    var c = self.assignments[i]
                    my_counts[c] += 1
                    var c_ptr = my_centroids + c * self.d
                    var x_ptr = x + i * self.d
                    
                    var j = 0
                    while j <= self.d - 4:
                        var cx = c_ptr.load[width=4](j)
                        var xx = x_ptr.load[width=4](j)
                        c_ptr.store(j, cx + xx)
                        j += 4
                    while j < self.d:
                        c_ptr[j] += x_ptr[j]
                        j += 1
                        
            parallelize[process_chunk](num_chunks)
            
            # Reduce phase
            for c in range(self.k):
                self.counts[c] = 0
                var c_ptr = self.centroids + c * self.d
                for j in range(self.d):
                    c_ptr[j] = 0.0
                    
            for chunk_id in range(num_chunks):
                var my_centroids = local_centroids + chunk_id * self.k * self.d
                var my_counts = local_counts + chunk_id * self.k
                
                for c in range(self.k):
                    self.counts[c] += my_counts[c]
                    var c_ptr = self.centroids + c * self.d
                    var src_ptr = my_centroids + c * self.d
                    
                    var j = 0
                    while j <= self.d - 4:
                        var cx = c_ptr.load[width=4](j)
                        var sx = src_ptr.load[width=4](j)
                        c_ptr.store(j, cx + sx)
                        j += 4
                    while j < self.d:
                        c_ptr[j] += src_ptr[j]
                        j += 1
            
            for c in range(self.k):
                var count = self.counts[c]
                if count > 0:
                    var c_ptr = self.centroids + c * self.d
                    var inv_count: Float32 = 1.0 / Float32(count)
                    
                    var j = 0
                    while j <= self.d - 4:
                        var cx = c_ptr.load[width=4](j)
                        c_ptr.store(j, cx * inv_count)
                        j += 4
                    while j < self.d:
                        c_ptr[j] *= inv_count
                        j += 1

        local_centroids.free()
        local_counts.free()
