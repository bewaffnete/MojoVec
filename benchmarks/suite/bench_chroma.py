import numpy as np
import chromadb
import time

d = 128
n = 1000000
nq = 10000
k = 10

print("Loading data for Chroma...")
def read_fvecs(file_path, max_n=None):
    a = np.fromfile(file_path, dtype='int32')
    d = a[0]
    a = a.reshape(-1, d + 1)
    if max_n is not None:
        a = a[:max_n]
    return a[:, 1:].copy().view('float32')

db = read_fvecs("benchmarks/suite/sift1m/sift_base.fvecs", max_n=n)
queries = read_fvecs("benchmarks/suite/sift1m/sift_query.fvecs", max_n=nq)

if db.shape[0] > n: db = db[:n]
if queries.shape[0] > nq: queries = queries[:nq]

client = chromadb.Client()
print("Building Chroma DB (HNSW)...")
collection = client.create_collection(name="sift_test", metadata={"hnsw:space": "l2", "hnsw:construction_ef": 200, "hnsw:M": 32})

ids = [str(i) for i in range(len(db))]
db_list = db.tolist()

batch_size = 5000
t0 = time.perf_counter()
for i in range(0, len(db), batch_size):
    collection.add(
        embeddings=db_list[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )
t_add = time.perf_counter() - t0
print(f"Build time: {t_add:.3f} s")

def read_ivecs(file_path, max_n=None):
    a = np.fromfile(file_path, dtype='int32')
    d = a[0]
    a = a.reshape(-1, d + 1)
    if max_n is not None:
        a = a[:max_n]
    return a[:, 1:].copy()

gt = read_ivecs("benchmarks/suite/sift1m/sift_groundtruth.ivecs", max_n=nq)

print("Searching Chroma...")
loops = 100  # Chroma Python API overhead is high, but testing with 100 loops
queries_list = queries.tolist()

t0 = time.perf_counter()
query_batch_size = 500
all_ids = []
for _ in range(loops):
    all_ids.clear()
    for j in range(0, len(queries_list), query_batch_size):
        results = collection.query(
            query_embeddings=queries_list[j:j+query_batch_size],
            n_results=k
        )
        for batch_res in results["ids"]:
            all_ids.append([int(x) for x in batch_res])
search_t = time.perf_counter() - t0
print(f"Total Search Time for {loops} loops: {search_t:.3f} s")
qps = (nq * loops) / search_t
print(f"Avg QPS: {int(qps)}")

hits = 0
for i in range(nq):
    hits += len(np.intersect1d(all_ids[i], gt[i, :k]))
recall = hits / (nq * k)
print(f"Recall@{k}: {recall:.5f}")
