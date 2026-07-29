import time

import faiss
import numpy as np


BASE_PATH = "benchmarks/data/sift1m/sift_base.fvecs"
QUERY_PATH = "benchmarks/data/sift1m/sift_query.fvecs"
GROUND_TRUTH_PATH = "benchmarks/data/sift1m/sift_groundtruth.ivecs"
INDEX_PATH = "benchmarks/cold_bench/faiss_hnsw_sq8_1m.index"
K = 10
EF_SEARCH = 96
RERANK_VALUES = (20, 30, 40, 60, 80, 96)


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


def main() -> None:
    faiss.omp_set_num_threads(10)
    database = read_fvecs(BASE_PATH)
    queries = read_fvecs(QUERY_PATH)
    ground_truth = read_ivecs(GROUND_TRUTH_PATH)
    base_index = faiss.read_index(INDEX_PATH)
    base_index.hnsw.efSearch = EF_SEARCH
    refined = faiss.IndexRefineFlat(base_index, faiss.swig_ptr(database))

    for rerank_candidates in RERANK_VALUES:
        refined.k_factor = rerank_candidates / K
        started = time.perf_counter()
        _, labels = refined.search(queries, K)
        elapsed = time.perf_counter() - started
        print(
            f"efSearch={EF_SEARCH} | rerank={rerank_candidates} "
            f"| QPS={len(queries) / elapsed:.3f} "
            f"| Recall@10={recall_at_10(labels, ground_truth):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
