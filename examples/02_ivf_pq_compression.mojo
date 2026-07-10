"""
MojoVec Example 2: Inverted Files and Product Quantization (IVF-PQ)
------------------------------------------------------------------
This example demonstrates how to use the IndexIVFPQ class for extreme dataset compression.
Product Quantization slices vectors into sub-vectors and replaces them with 8-bit cluster IDs,
drastically reducing memory consumption (e.g., from 4 bytes per dim to ~1 byte per sub-vector).
IVF prevents exhaustive search by routing queries to the most relevant clusters.
"""

from mojovec import IndexIVFPQ, IndexFlat, METRIC_L2
from std.memory import alloc
from std.random import rand

def main() raises:
    # 1. Define dataset parameters
    var d = 128          # Dimensions of the vector
    var num_vectors = 10000
    var num_queries = 5
    var k = 5            # We want to find the top 5 nearest neighbors

    print("Generating random dataset...")
    var xb = alloc[Float32](num_vectors * d)
    var xq = alloc[Float32](num_queries * d)
    rand(xb, num_vectors * d)
    rand(xq, num_queries * d)

    # 2. IVF and PQ Parameters
    var nlist = 100      # Number of clusters for the Inverted File (Voronoi cells)
    var M = 16           # Number of sub-vectors for Product Quantization
                         # Memory per vector goes from 128 * 4 bytes (512B) to M bytes (16B) + IVF overhead!

    # We need a base quantizer to route vectors into IVF buckets. We use Flat exact search.
    var flat_quantizer = alloc[IndexFlat](1)
    flat_quantizer.init_pointee_move(IndexFlat(d))

    # Initialize the IVFPQ Index
    var ivf_pq = IndexIVFPQ[IndexFlat](flat_quantizer, d, nlist, M)

    # 3. Train the Index
    # Unlike HNSW or Flat, IVF and PQ require a training phase to run K-Means and find the centroids.
    print("Training IVF and PQ (running K-Means)...")
    ivf_pq.train(num_vectors, xb)
    print("Training complete!")

    # 4. Add vectors
    print("Adding vectors to the index...")
    ivf_pq.add(num_vectors, xb)

    # 5. Search
    # nprobe controls how many neighboring IVF clusters are searched (higher = more recall, slower)
    ivf_pq.nprobe = 10

    var distances = alloc[Float32](num_queries * k)
    var labels = alloc[Int](num_queries * k)

    print("Searching for the top", k, "nearest neighbors for", num_queries, "queries...")
    ivf_pq.search(num_queries, xq, k, distances, labels)

    # 6. Display Results
    for i in range(num_queries):
        print("--- Query", i, "---")
        for j in range(k):
            var id = labels[i * k + j]
            var dist = distances[i * k + j]
            # Since PQ compresses the vectors, distances are computed via Asymmetric Distance Computation (ADC)
            # and are approximate.
            print("Rank", j+1, "-> ID:", id, "| Approx L2 Dist:", dist)
    
    # Keep ivf_pq alive until we free its dependencies
    _ = ivf_pq.ntotal

    # 7. Cleanup
    flat_quantizer.free()
    xb.free()
    xq.free()
    distances.free()
    labels.free()
