"""
MojoVec API Example 2: IVF-PQ Compression API
---------------------------------------------
This example demonstrates how to use the high-level `Client` and `CollectionIVFPQ` 
API to compress and search for vectors using Product Quantization. 
The API automatically handles training the index under the hood when `add()` is called.
"""

from mojovec import Client
from std.collections import List
from std.random import rand

def main() raises:
    # 1. Initialize Client and IVFPQ Collection
    var client = Client()
    var d = 128
    
    # nlist = number of clusters (Voronoi cells)
    # M = number of sub-vectors (compression ratio)
    print("Creating IVFPQ collection 'compressed_docs'...")
    var collection = client.create_ivfpq_collection("compressed_docs", dimension=d, nlist=100, M=16)

    # 2. Prepare Data
    var num_vectors = 10000
    print("Generating", num_vectors, "random vectors for training and indexing...")
    
    var ids = List[Int](capacity=num_vectors)
    for i in range(num_vectors):
        ids.append(50000 + i)
        
    var embeddings = List[Float32](capacity=num_vectors * d)
    for i in range(num_vectors * d):
        embeddings.append(Float32(i % 100) / 100.0)

    # 3. Add to Collection
    # Because IVFPQ requires training, the API will automatically take these embeddings
    # and train the K-Means and PQ codebooks before inserting them.
    print("Training and adding vectors to IVFPQ collection... (This might take a moment)")
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
            print("Rank", j + 1, "-> ID:", results.ids[i][j], "| Approx L2 Distance:", results.distances[i][j])
