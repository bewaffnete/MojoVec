import argparse
import os
import time

import faiss
import numpy as np


DIMENSION = 128
DATABASE_SIZE = 1_000_000
QUERY_COUNT = 10_000
K = 10
M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 96
RERANK_FACTOR = 2.0
SAMPLES = 10
LOOPS_PER_SAMPLE = 5

BASE_PATH = "benchmarks/data/sift1m/sift_base.fvecs"
LEARN_PATH = "benchmarks/data/sift1m/sift_learn.fvecs"
QUERY_PATH = "benchmarks/data/sift1m/sift_query.fvecs"
GROUND_TRUTH_PATH = "benchmarks/data/sift1m/sift_groundtruth.ivecs"
FLAT_INDEX_PATH = "benchmarks/cold_bench/faiss_hnsw_flat_1m.index"
SQ8_INDEX_PATH = "benchmarks/cold_bench/faiss_hnsw_sq8_1m.index"


def read_fvecs(path: str, limit: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dimension = int(raw[0])
    rows = raw.reshape(-1, dimension + 1)
    if limit is not None:
        rows = rows[:limit]
    return rows[:, 1:].copy().view(np.float32)


def read_ivecs(path: str, limit: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dimension = int(raw[0])
    rows = raw.reshape(-1, dimension + 1)
    if limit is not None:
        rows = rows[:limit]
    return rows[:, 1:].copy()


def build(kind: str) -> None:
    print("Loading SIFT1M base vectors...", flush=True)
    database = read_fvecs(BASE_PATH, DATABASE_SIZE)

    if kind == "flat":
        index = faiss.IndexHNSWFlat(DIMENSION, M)
        training_seconds = 0.0
        output_path = FLAT_INDEX_PATH
    else:
        index = faiss.IndexHNSWSQ(
            DIMENSION,
            faiss.ScalarQuantizer.QT_8bit,
            M,
        )
        print("Loading and training on SIFT1M learn vectors...", flush=True)
        learn = read_fvecs(LEARN_PATH)
        started = time.perf_counter()
        index.train(learn)
        training_seconds = time.perf_counter() - started
        output_path = SQ8_INDEX_PATH

    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH
    started = time.perf_counter()
    index.add(database)
    build_seconds = time.perf_counter() - started
    faiss.write_index(index, output_path)

    print(f"format: {kind}", flush=True)
    print(f"training: {training_seconds:.6f} s", flush=True)
    print(f"build: {build_seconds:.6f} s", flush=True)
    print(f"saved: {output_path}", flush=True)
    print(f"size: {os.path.getsize(output_path)} bytes", flush=True)


def recall_at_10(labels: np.ndarray, ground_truth: np.ndarray) -> float:
    hits = 0
    for query_id in range(labels.shape[0]):
        hits += np.isin(labels[query_id], ground_truth[query_id, :K]).sum()
    return hits / (labels.shape[0] * K)


def search(kind: str, cooldown: int) -> None:
    index_path = FLAT_INDEX_PATH if kind == "flat" else SQ8_INDEX_PATH
    print(f"Loading {kind} index...", flush=True)
    index = faiss.read_index(index_path)
    index.hnsw.efSearch = EF_SEARCH
    queries = read_fvecs(QUERY_PATH, QUERY_COUNT)
    ground_truth = read_ivecs(GROUND_TRUTH_PATH, QUERY_COUNT)
    search_index = index
    if kind == "sq8":
        database = read_fvecs(BASE_PATH, DATABASE_SIZE)
        search_index = faiss.IndexRefineFlat(
            index, faiss.swig_ptr(database)
        )
        search_index.k_factor = RERANK_FACTOR

    print(
        f"threads: {faiss.omp_get_max_threads()}, "
        f"bounded_queue: {index.hnsw.search_bounded_queue}",
        flush=True,
    )
    measured_seconds = 0.0
    labels = None
    for sample in range(SAMPLES):
        print(
            f"cooldown before sample {sample + 1}: {cooldown} s...",
            flush=True,
        )
        time.sleep(cooldown)
        search_index.search(queries[:256], K)

        started = time.perf_counter()
        for _ in range(LOOPS_PER_SAMPLE):
            _, labels = search_index.search(queries, K)
        elapsed = time.perf_counter() - started
        measured_seconds += elapsed
        qps = QUERY_COUNT * LOOPS_PER_SAMPLE / elapsed
        print(f"sample {sample + 1}: {qps:.3f} QPS", flush=True)

    aggregate_qps = (
        QUERY_COUNT * LOOPS_PER_SAMPLE * SAMPLES / measured_seconds
    )
    print(f"aggregate: {aggregate_qps:.3f} QPS", flush=True)
    print(
        f"Recall@10: {recall_at_10(labels, ground_truth):.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("build-flat", "build-sq8", "search-flat", "search-sq8"),
    )
    parser.add_argument("--cooldown", type=int, default=30)
    args = parser.parse_args()

    faiss.omp_set_num_threads(10)
    action, kind = args.mode.split("-", maxsplit=1)
    if action == "build":
        build(kind)
    else:
        search(kind, args.cooldown)


if __name__ == "__main__":
    main()
