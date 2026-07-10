import numpy as np
import faiss
import time
import os

def load_data():
    db = np.fromfile("benchmarks/suite/db.bin", dtype=np.float32).reshape(-1, 128)
    queries = np.fromfile("benchmarks/suite/queries.bin", dtype=np.float32).reshape(-1, 128)
    gt = np.fromfile("benchmarks/suite/groundtruth.bin", dtype=np.int32).reshape(-1, 10)
    return db, queries, gt

def evaluate(I, gt):
    k = I.shape[1]
    recalls = []
    for i in range(len(I)):
        hits = np.intersect1d(I[i], gt[i])
        recalls.append(len(hits) / k)
    return np.mean(recalls)

def main():
    db, queries, gt = load_data()
    D = 128
    M = 32
    efConstruction = 200
    
    print("--------------------------------------------------")
    print(f"[Faiss] IndexHNSWFlat (M={M}, efConstruction={efConstruction})")
    
    # 1. Build
    index = faiss.IndexHNSWFlat(D, M)
    index.hnsw.efConstruction = efConstruction
    
    t0 = time.time()
    index.add(db)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.4f} s")
    
    # 2. Search
    print("Search:")
    for ef in [10, 40, 50, 100, 200]:
        index.hnsw.efSearch = ef
        
        # warmup
        index.search(queries[:100], 10)
        
        t0 = time.perf_counter()
        distances, I = index.search(queries, 10)
        search_time = time.perf_counter() - t0
        
        qps = len(queries) / search_time
        recall = evaluate(I, gt)
        print(f"  efSearch={ef:<4} | QPS: {qps:8.0f} | Recall@10: {recall:.4f}")

if __name__ == "__main__":
    main()
