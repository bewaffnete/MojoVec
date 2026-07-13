import numpy as np
import faiss
import time
import os

d = 128
n = 1000000
nq = 10000
k = 10

print("Loading data for FAISS...")

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

print("Building FAISS HNSW...")
index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.efConstruction = 200
t0 = time.perf_counter()
index.add(db)
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

index.hnsw.efSearch = 40
print("Searching FAISS...")
loops = 100
t0 = time.perf_counter()
for _ in range(loops):
    D, I = index.search(queries, k)
search_t = time.perf_counter() - t0
print(f"Total Search Time for {loops} loops: {search_t:.3f} s")
qps = (nq * loops) / search_t
print(f"Avg QPS: {int(qps)}")

hits = 0
for i in range(nq):
    hits += len(np.intersect1d(I[i], gt[i, :k]))
recall = hits / (nq * k)
print(f"Recall@{k}: {recall:.5f}")
