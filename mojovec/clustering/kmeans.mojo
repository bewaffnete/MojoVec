from std.algorithm import parallelize
from std.collections import List
from std.memory import alloc
from std.memory.span import Span
from std.random.philox import Random
from ..utils.distances import l2_distance_simd
from std.math import max, min

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
            
    def train(mut self, x: Span[Float32, _]):
        """Trains the K-Means model to find cluster centroids.
        
        Args:
            x: Contiguous flattened training vectors.
        """
        var n = len(x) // self.d
        if n == 0: return
        var data_ptr = x.unsafe_ptr()
            
        if Int(self.assignments) != 0: self.assignments.free()
        self.assignments = alloc[Int](n)
        
        # 1. Initialize centroids from a private deterministic PRNG. The
        # package-level std.random state is shared across threads, so using it
        # here makes concurrent index training race and platform-dependent.
        var generator = Random(seed=UInt64(0x4D4F4A4F564543))
        var random_words = generator.step()
        # Reduce while the PRNG word is unsigned. Casting first can produce a
        # negative signed index on platforms where the high bit is set.
        var sample_offset = Int(random_words[0] % UInt32(n))
        for i in range(self.k):
            if i != 0 and i % 4 == 0:
                random_words = generator.step()
            var src_idx = Int(random_words[i % 4] % UInt32(n))
            var src_ptr = data_ptr + src_idx * self.d
            var dst_ptr = self.centroids + i * self.d
            for j in range(self.d):
                dst_ptr[j] = src_ptr[j]
                
        var num_chunks = 32
        var chunk_size = (n + num_chunks - 1) // num_chunks
        var local_centroids = List[Float32](
            unsafe_uninit_length=num_chunks * self.k * self.d
        )
        var local_counts = List[Int](
            unsafe_uninit_length=num_chunks * self.k
        )
        var local_centroids_ptr = local_centroids.unsafe_ptr()
        var local_counts_ptr = local_counts.unsafe_ptr()
        var centroids_ptr = self.centroids
        var assignments_ptr = self.assignments
        var dimension = self.d
        var cluster_count = self.k
        # Main loop
        for iteration in range(self.niter):
            # E-step: Assign points to centroids
            @parameter
            def process_assignment_chunk(chunk_id: Int):
                var start = chunk_id * chunk_size
                var end = min(start + chunk_size, n)
                for i in range(start, end):
                    var min_dist: Float32 = 1e38
                    var best_c = -1
                    var x_ptr = data_ptr + i * dimension

                    for c in range(cluster_count):
                        var c_ptr = centroids_ptr + c * dimension
                        var dist = l2_distance_simd[8](
                            x_ptr,
                            c_ptr,
                            dimension,
                        )

                        if dist < min_dist:
                            min_dist = dist
                            best_c = c

                    # Keep malformed numeric input from becoming an
                    # out-of-bounds cluster write. Finite input always selects
                    # a real centroid.
                    assignments_ptr[i] = max(best_c, 0)

            parallelize[process_assignment_chunk](num_chunks)
            
            # Zero out thread-local accumulators
            for i in range(num_chunks * self.k * self.d):
                local_centroids_ptr[i] = 0.0
            for i in range(num_chunks * self.k):
                local_counts_ptr[i] = 0
                
            # M-step: Update centroids in parallel
            @parameter
            def process_chunk(chunk_id: Int):
                var start = chunk_id * chunk_size
                var end = min(start + chunk_size, n)
                var my_centroids = (
                    local_centroids_ptr
                    + chunk_id * cluster_count * dimension
                )
                var my_counts = local_counts_ptr + chunk_id * cluster_count
                
                for i in range(start, end):
                    var c = assignments_ptr[i]
                    my_counts[c] += 1
                    var c_ptr = my_centroids + c * dimension
                    var x_ptr = data_ptr + i * dimension
                    
                    var j = 0
                    while j <= dimension - 4:
                        var cx = c_ptr.load[width=4](j)
                        var xx = x_ptr.load[width=4](j)
                        c_ptr.store(j, cx + xx)
                        j += 4
                    while j < dimension:
                        c_ptr[j] += x_ptr[j]
                        j += 1
                        
            # Assignment is the O(n * k * d) hot phase and remains fully
            # parallel. Accumulation is only O(n * d); executing its 32
            # disjoint chunks deterministically avoids a Mojo 1.0.0b2 x86
            # worker-pool corruption observed when many small-subvector
            # K-Means instances are trained back-to-back for PQ.
            for chunk_id in range(num_chunks):
                process_chunk(chunk_id)
            
            # Reduce phase
            for c in range(self.k):
                self.counts[c] = 0
                var c_ptr = self.centroids + c * self.d
                for j in range(self.d):
                    c_ptr[j] = 0.0
                    
            for chunk_id in range(num_chunks):
                var my_centroids = (
                    local_centroids_ptr + chunk_id * self.k * self.d
                )
                var my_counts = local_counts_ptr + chunk_id * self.k
                
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
                else:
                    # A zero centroid is especially destructive for residual
                    # PQ. Deterministically reseed empty clusters from data.
                    var src_idx = (
                        sample_offset + (iteration + 1) * self.k + c
                    ) % n
                    var src_ptr = data_ptr + src_idx * self.d
                    var c_ptr = self.centroids + c * self.d
                    for j in range(self.d):
                        c_ptr[j] = src_ptr[j]
