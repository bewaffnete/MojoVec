"""
MojoVec API Example 3: Serialization API
----------------------------------------
This example demonstrates how to persist your high-level Collection to disk 
and reload it. The `save()` and `load()` methods automatically serialize both 
the internal index structures (like HNSW graphs) and your custom IDs.
"""

from mojovec import Client, Collection
from std.collections import List

def main() raises:
    var client = Client()
    var d = 128
    var file_path = "my_highlevel_collection.bin"

    # 1. Create, Populate, and Save
    print("--- 1. Saving Phase ---")
    var collection = client.create_collection("my_docs", dimension=d)
    
    var num_vectors = 1000
    var ids = List[Int](capacity=num_vectors)
    var embeddings = List[Float32](capacity=num_vectors * d)
    
    for i in range(num_vectors):
        ids.append(90000 + i) # Custom IDs
        
    for i in range(num_vectors * d):
        embeddings.append(Float32(i % 50) / 50.0)

    print("Adding 1000 vectors...")
    collection.add(ids, embeddings)
    
    print("Saving collection to disk ('" + file_path + "')...")
    collection.save(file_path)
    print("Saved successfully!\n")

    # 2. Load and Query
    print("--- 2. Loading Phase ---")
    print("Loading collection from disk...")
    
    # We use the static method to load it from disk
    var loaded_collection = Collection.load(file_path)
    print("Loaded successfully!")
    
    var num_queries = 2
    var k = 3
    var query_embeddings = List[Float32](capacity=num_queries * d)
    for i in range(num_queries * d):
        query_embeddings.append(Float32(i % 50) / 50.0)

    print("Querying the loaded collection...")
    var results = loaded_collection.query(query_embeddings, n_results=k)

    for i in range(num_queries):
        print("Query", i, "results:")
        for j in range(k):
            print("  Rank", j + 1, "-> ID:", results.ids[i][j], "| L2 Distance:", results.distances[i][j])
