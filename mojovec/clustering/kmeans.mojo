from std.algorithm import parallelize
from std.collections import List
from std.memory.span import Span
from std.random.philox import Random
from ..utils.distances import l2_distance_short4, l2_distance_simd
from std.math import max, min

struct KMeans:
    """K-Means clustering algorithm for vector quantization.
    
    Partitions a set of vectors into k clusters, finding the centroids
    that minimize the distance between points and their assigned centroids.
    """
    var d: Int
    var k: Int
    var niter: Int
    var centroids: List[Float32]
    var assignments: List[Int]
    var counts: List[Int]
    
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
        self.centroids = List[Float32](unsafe_uninit_length=k * d)
        self.assignments = List[Int]()
        self.counts = List[Int](unsafe_uninit_length=k)
            
    def train(mut self, x: Span[Float32, _]):
        """Trains the K-Means model to find cluster centroids.
        
        Args:
            x: Contiguous flattened training vectors.
        """
        self._train[True](x)

    def train_serial(mut self, x: Span[Float32, _]):
        """Trains without creating a nested worker pool.

        Used when independent K-Means models are already scheduled in
        parallel by a higher-level quantizer.
        """
        self._train[False](x)

    def take_centroids(
        mut self,
    ) -> List[Float32]:
        """Transfers managed ownership of trained centroids to the caller."""
        var result = self.centroids^
        self.centroids = List[Float32]()
        return result^

    def _train[PARALLEL: Bool](mut self, x: Span[Float32, _]):
        var n = len(x) // self.d
        if n == 0: return
        var data_ptr = x.unsafe_ptr()
            
        self.assignments = List[Int](unsafe_uninit_length=n)
        
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
            var dst_ptr = self.centroids.unsafe_ptr() + i * self.d
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
        var centroids_ptr = self.centroids.unsafe_ptr()
        var assignments_ptr = self.assignments.unsafe_ptr()
        var counts_ptr = self.counts.unsafe_ptr()
        var dimension = self.d
        var cluster_count = self.k
        # Main loop
        for iteration in range(self.niter):
            # Clear the per-chunk accumulators before publishing them to the
            # worker pool.
            for i in range(num_chunks * self.k * self.d):
                local_centroids_ptr[i] = 0.0
            for i in range(num_chunks * self.k):
                local_counts_ptr[i] = 0

            # Fuse assignment and accumulation into one parallel region. Each
            # worker owns a disjoint input range and accumulator slice, so no
            # cross-thread assignment buffer hand-off is needed between the E
            # and M steps. The expensive O(n * k * d) distance work remains
            # parallel while reduction order stays deterministic below.
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
                    var min_dist: Float32 = 1e38
                    var best_c = -1
                    var x_ptr = data_ptr + i * dimension

                    for c in range(cluster_count):
                        var c_ptr = centroids_ptr + c * dimension
                        var dist: Float32
                        if dimension <= 4:
                            dist = l2_distance_short4(
                                x_ptr, c_ptr, dimension
                            )
                        else:
                            dist = l2_distance_simd[8](
                                x_ptr, c_ptr, dimension
                            )

                        if dist < min_dist:
                            min_dist = dist
                            best_c = c

                    # Keep malformed numeric input from becoming an
                    # out-of-bounds cluster write. Finite input always selects
                    # a real centroid.
                    var c = max(best_c, 0)
                    assignments_ptr[i] = c
                    my_counts[c] += 1
                    var c_ptr = my_centroids + c * dimension
                    x_ptr = data_ptr + i * dimension
                    
                    var j = 0
                    while j <= dimension - 4:
                        var cx = c_ptr.load[width=4](j)
                        var xx = x_ptr.load[width=4](j)
                        c_ptr.store(j, cx + xx)
                        j += 4
                    while j < dimension:
                        c_ptr[j] += x_ptr[j]
                        j += 1

            comptime if PARALLEL:
                parallelize[process_chunk](num_chunks)
            else:
                for chunk_id in range(num_chunks):
                    process_chunk(chunk_id)
            
            # Reduce phase
            for c in range(self.k):
                counts_ptr[c] = 0
                var c_ptr = centroids_ptr + c * self.d
                for j in range(self.d):
                    c_ptr[j] = 0.0
                    
            for chunk_id in range(num_chunks):
                var my_centroids = (
                    local_centroids_ptr + chunk_id * self.k * self.d
                )
                var my_counts = local_counts_ptr + chunk_id * self.k
                
                for c in range(self.k):
                    counts_ptr[c] += my_counts[c]
                    var c_ptr = centroids_ptr + c * self.d
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
                var count = counts_ptr[c]
                if count > 0:
                    var c_ptr = centroids_ptr + c * self.d
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
                    var c_ptr = centroids_ptr + c * self.d
                    for j in range(self.d):
                        c_ptr[j] = src_ptr[j]
