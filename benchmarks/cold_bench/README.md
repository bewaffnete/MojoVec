# Cold SIFT1M benchmarks

Reproducible cold-search benchmarks for MojoVec and FAISS on SIFT1M.

The two `bench_*_api.mojo` programs exercise only the public managed API:

```mojo
var collection = Collection.load(path)
collection.set_ef_search(96)
var results = collection.query(queries, n_results=10)
```

They do not call `_query_into` or other internal methods and contain no
`UnsafePointer`, `Span`, `alloc`, or `free`. Binary SIFT decoding is isolated in
`sift1m_data.mojo` and is outside the timed region.

## Configuration

- 1,000,000 base vectors;
- 10,000 query vectors;
- 128 dimensions and squared L2 distance;
- HNSW `M=32`, `efConstruction=200`, and `efSearch=96`;
- SQ8 exact reranking with 20 candidates (`k_factor=2`);
- `k=10`;
- 10 samples, each containing 5 searches over all queries.

## Cold protocol

Every MojoVec and FAISS sample follows the same sequence:

1. Wait 30 seconds.
2. Run an untimed 256-query warm-up through `Collection.query()`.
3. Run five full 10,000-query searches.
4. Record QPS without including cooldown or warm-up time.

Recall@10 is computed from the final managed `QueryResults` against the
provided SIFT1M ground truth.

## Required local files

Run commands from the repository root. The SIFT1M files must be available at:

```text
benchmarks/data/sift1m/sift_query.fvecs
benchmarks/data/sift1m/sift_groundtruth.ivecs
```

The search programs load prebuilt collections from:

```text
benchmarks/cold_bench/mojovec_hnsw_flat_1m.mojovec
benchmarks/cold_bench/mojovec_hnsw_sq8_1m.mojovec
```

Saved MojoVec graphs, FAISS indexes, and compiled executables are intentionally
excluded from Git: together the local artifacts are roughly 2.9 GB. The source
files and recorded measurements are versioned.

## Build Mojo benchmarks

```bash
mojo build -I . \
  -I benchmarks/cold_bench \
  benchmarks/cold_bench/bench_flat_1m_api.mojo \
  -o benchmarks/cold_bench/bench_flat_1m_api

mojo build -I . \
  -I benchmarks/cold_bench \
  benchmarks/cold_bench/bench_sq8_1m_api.mojo \
  -o benchmarks/cold_bench/bench_sq8_1m_api
```

## Run MojoVec

```bash
./benchmarks/cold_bench/bench_flat_1m_api
./benchmarks/cold_bench/bench_sq8_1m_api
```

## Build and run FAISS

The FAISS wrapper can recreate its local index files:

```bash
.venv/bin/python \
  benchmarks/cold_bench/faiss_python_hnsw_benchmark.py build-flat

.venv/bin/python \
  benchmarks/cold_bench/faiss_python_hnsw_benchmark.py build-sq8
```

Run the saved indexes with the same per-sample 30-second cooldown:

```bash
.venv/bin/python \
  benchmarks/cold_bench/faiss_python_hnsw_benchmark.py \
  search-flat --cooldown 30

.venv/bin/python \
  benchmarks/cold_bench/faiss_python_hnsw_benchmark.py \
  search-sq8 --cooldown 30
```

See `RESULTS.md` for the recorded Apple Silicon measurements.
