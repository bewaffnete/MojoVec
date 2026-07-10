import chromadb
import numpy as np
import time

def load_data():
    db = np.fromfile("benchmarks/suite/db.bin", dtype=np.float32).reshape(-1, 128)
    queries = np.fromfile("benchmarks/suite/queries.bin", dtype=np.float32).reshape(-1, 128)
    gt = np.fromfile("benchmarks/suite/groundtruth.bin", dtype=np.int32).reshape(-1, 10)
    return db, queries, gt

def evaluate(I, gt):
    k = len(I[0])
    recalls = []
    for i in range(len(I)):
        # ChromaDB returns list of ids (strings)
        hits = np.intersect1d(np.array(I[i], dtype=np.int32), gt[i])
        recalls.append(len(hits) / k)
    return np.mean(recalls)

def main():
    db, queries, gt = load_data()
    D = 128
    M = 32
    efConstruction = 200
    
    print("--------------------------------------------------")
    print(f"[ChromaDB] hnsw:space=l2, M={M}, efConstruction={efConstruction}")
    
    client = chromadb.Client()
    
    # Create collection
    collection = client.create_collection(
        name="bench_collection",
        metadata={
            "hnsw:space": "l2",
            "hnsw:construction_ef": efConstruction,
            "hnsw:M": M
        }
    )
    
    ids = [str(i) for i in range(len(db))]
    
    # 1. Build (Add)
    t0 = time.time()
    batch_size = 5000
    for i in range(0, len(db), batch_size):
        collection.add(
            embeddings=db[i:i+batch_size].tolist(),
            ids=ids[i:i+batch_size]
        )
    build_time = time.time() - t0
    print(f"Build time (includes DB overhead): {build_time:.4f} s")
    
    # 2. Search
    print("Search:")
    queries_list = queries.tolist()
    
    # ChromaDB doesn't allow setting efSearch globally easily without recreating collection. 
    # Actually wait, we can pass it in query params? No, unfortunately ChromaDB handles it internally or in metadata.
    # We will test default query speed and manually test different efSearch by changing the underlying index parameter if possible,
    # but actually Chroma doesn't expose efSearch at query time in the stable API easily.
    # We will just do a single query benchmark for ChromaDB.
    
    # Warmup
    collection.query(query_embeddings=queries_list[:100], n_results=10)
    
    t0 = time.perf_counter()
    
    batch_size = 1000
    all_I = []
    for i in range(0, len(queries_list), batch_size):
        results = collection.query(
            query_embeddings=queries_list[i:i+batch_size],
            n_results=10
        )
        all_I.extend(results['ids'])
        
    search_time = time.perf_counter() - t0
    
    qps = len(queries) / search_time
    I = all_I
    recall = evaluate(I, gt)
    print(f"  QPS: {qps:8.0f} | Recall@10: {recall:.4f}")

if __name__ == "__main__":
    main()
