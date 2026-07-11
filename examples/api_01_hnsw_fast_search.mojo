"""
MojoVec Example 4: High-Level Chroma-like API
---------------------------------------------
This example demonstrates how to use the high-level `Client` and `Collection` 
API to quickly add and search for vectors without managing memory pointers,
index types, or manual metric configuration.
"""

from mojovec import Client
from std.collections import List
from std.random import rand

def main() raises:
    # 1. Initialize Client and Collection
    var client = Client()
    var d = 128
    print("Creating collection 'my_documents'...")
    var collection = client.create_collection("my_documents", dimension=d)

    # 2. Prepare Data
    var num_vectors = 1000
    print("Generating", num_vectors, "random vectors...")
    
    var ids = List[Int](capacity=num_vectors)
    for i in range(num_vectors):
        # We can use arbitrary integer IDs, not necessarily 0 to N-1
        ids.append(10000 + i)
        
    var embeddings = List[Float32](capacity=num_vectors * d)
    for i in range(num_vectors * d):
        # Random floats 0.0 to 1.0 (just simulating `rand` output for simplicity)
        embeddings.append(Float32(i % 100) / 100.0)

    # 3. Add to Collection
    print("Adding vectors to collection...")
    collection.add(ids, embeddings)
    print("Added successfully!")

    # 4. Query
    var num_queries = 2
    var k = 5
    var query_embeddings = List[Float32](capacity=num_queries * d)
    for i in range(num_queries * d):
        query_embeddings.append(Float32(i % 50) / 50.0)

    print("Searching for the top", k, "results for", num_queries, "queries...")
    var results = collection.query(query_embeddings, n_results=k)

    # 5. Display Results
    for i in range(num_queries):
        print("--- Query", i, "---")
        for j in range(k):
            print("Rank", j + 1, "-> ID:", results.ids[i][j], "| L2 Distance:", results.distances[i][j])
