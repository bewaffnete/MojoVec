import time

import faiss
import numpy as np


QUERY_PATH = "benchmarks/data/sift1m/sift_query.fvecs"
GROUND_TRUTH_PATH = "benchmarks/data/sift1m/sift_groundtruth.ivecs"
FLAT_INDEX_PATH = "benchmarks/cold_bench/faiss_hnsw_flat_1m.index"
SQ8_INDEX_PATH = "benchmarks/cold_bench/faiss_hnsw_sq8_1m.index"
EF_VALUES = (48, 64, 80, 96, 112, 128, 160, 192, 256, 384, 512)
K = 10


def read_fvecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dimension = int(raw[0])
    return raw.reshape(-1, dimension + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dimension = int(raw[0])
    return raw.reshape(-1, dimension + 1)[:, 1:].copy()


def recall_at_10(labels: np.ndarray, ground_truth: np.ndarray) -> float:
    hits = 0
    for query_id in range(labels.shape[0]):
        hits += np.isin(labels[query_id], ground_truth[query_id, :K]).sum()
    return hits / (labels.shape[0] * K)


def run_sweep(path: str, name: str, queries: np.ndarray, gt: np.ndarray) -> None:
    print(f"\n{name}", flush=True)
    index = faiss.read_index(path)
    for ef_search in EF_VALUES:
        index.hnsw.efSearch = ef_search
        started = time.perf_counter()
        _, labels = index.search(queries, K)
        elapsed = time.perf_counter() - started
        print(
            f"efSearch={ef_search} | QPS={len(queries) / elapsed:.3f} "
            f"| Recall@10={recall_at_10(labels, gt):.6f}",
            flush=True,
        )


def main() -> None:
    faiss.omp_set_num_threads(10)
    queries = read_fvecs(QUERY_PATH)
    ground_truth = read_ivecs(GROUND_TRUTH_PATH)
    run_sweep(FLAT_INDEX_PATH, "FAISS Flat", queries, ground_truth)
    run_sweep(SQ8_INDEX_PATH, "FAISS QT_8bit", queries, ground_truth)


if __name__ == "__main__":
    main()
