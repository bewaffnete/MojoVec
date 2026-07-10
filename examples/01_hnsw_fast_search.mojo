"""
MojoVec Example 1: HNSW Fast Search
-----------------------------------
This example demonstrates how to build and query an HNSW graph (Hierarchical Navigable Small World).
HNSW provides extremely fast logarithmic O(log N) search times, trading memory overhead for search speed.
It is the gold standard for exact/approximate nearest neighbor search in modern vector databases.
"""

from mojovec import IndexHNSW, IndexFlat, METRIC_L2
from std.memory import alloc
from std.random import rand

def main() raises:
    # 1. Define dataset parameters
    var d = 128          # Dimensions of the vector
    var num_vectors = 10000
    var num_queries = 5
    var k = 10           # We want to find the top 10 nearest neighbors

    print("Generating random dataset...")
    var xb = alloc[Float32](num_vectors * d)
    var xq = alloc[Float32](num_queries * d)
    rand(xb, num_vectors * d)
    rand(xq, num_queries * d)

    # 2. Initialize the HNSW Index
    # HNSW needs a base storage layer to keep the actual vectors. We use IndexFlat for exact storage.
    var storage = IndexFlat(d, METRIC_L2)
    
    # M limits the maximum number of outgoing connections in the graph per node.
    var M = 32
    var hnsw = IndexHNSW[IndexFlat](storage^, d, METRIC_L2, M=M)
    
    # efConstruction controls the index build time/accuracy tradeoff (higher is more accurate but slower to build)
    hnsw.hnsw.efConstruction = 200

    print("Adding vectors to HNSW Index (building the graph)...")
    hnsw.add(num_vectors, xb)
    print("Graph built successfully!")

    # 3. Querying the HNSW Index
    # efSearch controls search accuracy/speed tradeoff (higher is more accurate but slower to search)
    # efSearch MUST be >= k.
    hnsw.hnsw.efSearch = 40

    var distances = alloc[Float32](num_queries * k)
    var labels = alloc[Int](num_queries * k)

    print("Searching for the top", k, "nearest neighbors for", num_queries, "queries...")
    hnsw.search(num_queries, xq, k, distances, labels)

    # 4. Display Results
    for i in range(num_queries):
        print("--- Query", i, "---")
        for j in range(k):
            var id = labels[i * k + j]
            var dist = distances[i * k + j]
            print("Rank", j+1, "-> ID:", id, "| L2 Distance:", dist)
    
    # 5. Manual Memory Management Cleanup
    xb.free()
    xq.free()
    distances.free()
    labels.free()
