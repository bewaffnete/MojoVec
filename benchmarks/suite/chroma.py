import numpy as np
import chromadb
import time

d = 128
n = 1_000_000
nq = 10_000
k = 10

def read_fvecs(file_path, max_n=None):
    a = np.fromfile(file_path, dtype='int32')
    dim = a[0]
    a = a.reshape(-1, dim + 1)[:, 1:].copy().view('float32')
    return a[:max_n] if max_n else a

def read_ivecs(file_path, max_n=None):
    a = np.fromfile(file_path, dtype='int32')
    dim = a[0]
    a = a.reshape(-1, dim + 1)[:, 1:].copy()
    return a[:max_n] if max_n else a

print("Loading SIFT1M...")
db = read_fvecs("benchmarks/data/sift1m/sift_base.fvecs", max_n=n)
queries = read_fvecs("benchmarks/data/sift1m/sift_query.fvecs", max_n=nq)
gt = read_ivecs(
    "benchmarks/data/sift1m/sift_groundtruth.ivecs", max_n=nq
)

client = chromadb.Client()

# Configuration for HNSW
config = {
    "hnsw": {
        "space": "l2",
        "ef_construction": 200,
        "ef_search": 40,
        "max_neighbors": 32,
        "num_threads": 10,
    }
}

print("Creating collection...")
collection = client.create_collection(
    name="sift_test",
    configuration=config
)

# Add embeddings
print("Adding embeddings...")
ids = [str(i) for i in range(len(db))]

batch_size = 5000
t0 = time.perf_counter()

for i in range(0, len(db), batch_size):
    collection.add(
        embeddings=db[i:i+batch_size].tolist(),
        ids=ids[i:i+batch_size]
    )

print(f"Build time: {time.perf_counter() - t0:.2f} s")

# Search
print("Searching...")
loops = 50
queries_list = queries.tolist()

t0 = time.perf_counter()
all_results = []

for _ in range(loops):
    for j in range(0, len(queries_list), 500):
        results = collection.query(
            query_embeddings=queries_list[j:j+500],
            n_results=k
        )
        all_results.extend(results["ids"])

search_time = time.perf_counter() - t0
qps = (nq * loops) / search_time

print(f"Total search time ({loops} loops): {search_time:.2f} s")
print(f"Avg QPS: {int(qps)}")

# Recall
hits = 0
for i in range(nq):
    retrieved = [int(x) for x in all_results[i]]
    hits += len(np.intersect1d(retrieved, gt[i, :k]))

recall = hits / (nq * k)
print(f"Recall@{k}: {recall:.5f}")
