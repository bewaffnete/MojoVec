import numpy as np
import os
import faiss

def main():
    np.random.seed(42)
    N = 100_000
    Q = 10_000
    D = 128
    K = 10

    print(f"Generating dataset: N={N}, Q={Q}, D={D}")
    
    # We will generate a clustered dataset to mimic realistic data
    num_clusters = 1024
    cluster_centers = np.random.uniform(-1.0, 1.0, size=(num_clusters, D)).astype(np.float32)
    
    # Assign vectors to clusters and add noise
    db = np.empty((N, D), dtype=np.float32)
    for i in range(N):
        c_id = np.random.randint(0, num_clusters)
        noise = np.random.uniform(-0.2, 0.2, size=D).astype(np.float32)
        db[i] = cluster_centers[c_id] + noise
        
    queries = np.empty((Q, D), dtype=np.float32)
    for i in range(Q):
        c_id = np.random.randint(0, num_clusters)
        noise = np.random.uniform(-0.2, 0.2, size=D).astype(np.float32)
        queries[i] = cluster_centers[c_id] + noise
        
    print("Computing Exact Ground Truth...")
    index_flat = faiss.IndexFlatL2(D)
    index_flat.add(db)
    
    distances, labels = index_flat.search(queries, K)
    
    print("Saving to files...")
    # Save raw bytes
    db.tofile("benchmarks/suite/db.bin")
    queries.tofile("benchmarks/suite/queries.bin")
    labels.astype(np.int32).tofile("benchmarks/suite/groundtruth.bin")
    
    print("Done!")

if __name__ == "__main__":
    main()
