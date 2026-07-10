# MojoVec 🔥

**MojoVec** is a high-performance library for vector search and vector databases, written entirely from scratch in the **Mojo** 🔥 programming language.

The architecture and design philosophy are heavily inspired by Meta's **Faiss**. However, MojoVec leverages the cutting-edge capabilities of Mojo to achieve maximum bare-metal performance, featuring hardware SIMD vectorization, zero-cost abstractions, manual memory management, explicit prefetching, and aggressive parallelization.

## Features & Architecture

### 1. Core Architecture and Utilities
- **Zero-cost Traits**: Designed a strict system of abstractions (`Index`, `StorageTrait`, `DistanceComputerTrait`), allowing composition of different types of storages and indexes without virtual dispatch or runtime overhead.
- **Hardware-Optimized SIMD Distances**: Blazing-fast functions for calculating `L2 Distance` and `Inner Product`. `l2_distance_simd` and `inner_product_simd` unroll directly into hardware SIMD registers. The `SIMD_WIDTH` is globally configured as a `comptime` constant (currently optimized for Apple Silicon ARM NEON, generating massive Instruction-Level Parallelism via compiler unrolling).
- **Generic Heaps**: Custom Max-Heap and Min-Heap structures for maintaining Top-K nearest neighbors dynamically, leveraging `TrivialRegisterPassable` for zero-overhead, register-level value semantics.
- **Memory Management**: All mission-critical data structures operate exclusively using raw pointers (`UnsafePointer` and `alloc`) to completely bypass overhead and maintain strict, manual memory safety.

### 2. Exact Search (Flat Indexes)
- **`IndexFlat`**: The baseline index for exact "one-to-all" brute-force search. It guarantees 100% recall and serves as the Ground Truth for testing the accuracy of all other approximate indexes. Perfectly sorts results using in-place heap operations.

### 3. Graph-Based Search (HNSW)
- **`HNSWGraph`**: A Hierarchical Navigable Small World multi-layer graph structure.
- **Extended HNSW Heuristic**: fully implements the "Diversity Check" heuristic during graph construction, preserving links to neighbors in diverse directions (just like Faiss). This guarantees optimal graph navigability.
- **Int32 Architecture**: Internal node IDs are strictly 32-bit (`Int32`), tightly packing 16 neighbors into a single 64-byte CPU cache line. Combined with `__builtin_prefetch` strategies, this completely eliminates memory bandwidth bottlenecks.
- **`IndexHNSW`**: A composite index (running on top of any `StorageTrait`, such as `IndexFlat`) that provides lightning-fast logarithmic $O(\log N)$ search time. Achieves performance and recall metrics **on par with Faiss C++**.

### 4. Clustering and Inverted Files (IVF)
- **`KMeans`**: A multithreaded (via `parallelize` with thread-local accumulators) and SIMD-vectorized K-Means algorithm for clustering vectors and locating centroids in the vector space.
- **`ArrayInvertedLists`**: An optimized in-memory storage engine for inverted lists. Each bucket independently allocates raw memory for vector IDs and byte codes. Fully thread-safe for concurrent additions using granular ticket-locks per list bucket.
- **`IndexIVFFlat`**: An inverted file index with exact Flat storage in the buckets.

### 5. Advanced Quantization
- **`ScalarQuantizer` / `IndexScalarQuantizer`**: Compresses `Float32` vectors into 8-bit integers (or `Float16`), reducing memory consumption by 4x on the fly.
- **`ProductQuantizer`**: Splits high-dimensional vectors into sub-vectors, quantizing each independently using K-Means centroids. Fully supports both `L2` and `Inner Product` metric types for asymmetric distance computation (ADC).
- **`IndexIVFPQ`**: The pinnacle of memory efficiency. Combines Inverted Files (IVF) for pruning the search space with Product Quantization (PQ) for massively compressing vectors. 

### 6. I/O Serialization
- Fully supports saving and loading indexes directly to/from disk using custom binary formats. Includes `write_index` and `read_index` capabilities for Flat, IVFFlat, and IVFPQ indexes.

## Benchmarks
On an Apple Silicon M-series chip with 100,000 vectors ($d=128$), MojoVec matches Faiss C++ in recall while approaching its heavily hand-tuned assembly speeds on pure CPU:

| Index (efSearch=10) | QPS | Recall@10 |
|---------------------|---------|-----------|
| **Faiss** | ~430k | 0.925 |
| **MojoVec** | ~241k | 0.921 |

*(MojoVec's speed is achieved with pure Mojo SIMD loops without a single line of C++ or assembly).*

## Roadmap / Next Steps
- **Python Wrappers**: Leveraging Mojo's FFI capabilities to package MojoVec into a standard Python wheel. This will allow Python developers to use MojoVec as a drop-in, zero-dependency alternative to `faiss-cpu`.
- **SIMD Refinements**: More aggressive `sys.intrinsics.prefetch` strategies (e.g., prefetching neighbor metadata lists) to fully close the QPS gap with Faiss.

## Running Tests
Every component is covered by correctness and accuracy tests. Run the test suite using the Mojo CLI:

```bash
# Run all tests
for f in tests/test_*.mojo; do mojo run -I . "$f"; done

# Run benchmarking suite
mojo run -I . benchmarks/suite/bench_mojovec.mojo
```
