# MojoVec

**MojoVec** is a high-performance library for vector search and vector databases, written entirely from scratch in the **Mojo** 🔥 programming language.

The architecture and design philosophy are heavily inspired by Meta's **Faiss**. However, MojoVec leverages the cutting-edge capabilities of Mojo to achieve maximum performance, featuring hardware SIMD vectorization, zero-cost abstractions, manual memory management, and aggressive parallelization.

## Features Implemented So Far

The following components have been successfully implemented and covered by tests:

### 1. Core Architecture and Metrics
- **Zero-cost Traits**: Designed a strict system of abstractions (`Index`, `StorageTrait`, `DistanceComputerTrait`), which allows composing different types of storages and indexes without runtime overhead.
- **Vectorized Distances**: Blazing-fast functions for calculating `L2 Distance` and `Inner Product` (`src/mojovec/utils/distances.mojo`). These unroll directly into hardware SIMD registers specific to the CPU's supported width (e.g., using `l2_distance_simd[4]`).
- **Heap Structures**: Implemented Max-Heap and Min-Heap structures for efficiently maintaining the Top-K nearest neighbors during graph traversal or brute-force search.

### 2. Exact Search (Flat Indexes)
- **`IndexFlat`**: The baseline index for exact "one-to-all" brute-force search. It guarantees 100% recall and serves as the Ground Truth for testing the accuracy of all other approximate indexes.

### 3. Quantization
- **`ScalarQuantizer`**: A utility for 8-bit compression (quantization) of `Float32` vectors.
- **`IndexScalarQuantizer`**: An index that reduces memory consumption by 4x by compressing vectors on the fly, while maintaining acceptable search accuracy.

### 4. Graph-Based Search (HNSW)
- **`HNSWGraph`**: A Hierarchical Navigable Small World multi-layer graph structure. It serves as the core of the index, implementing greedy beam search and neighbor selection heuristics identical to the original Faiss/hnswlib.
- **`IndexHNSW`**: A composite index (running on top of any `StorageTrait`, such as `IndexFlat`) that provides lightning-fast logarithmic $O(\log N)$ search time. The implementation includes dynamic list resizing for neighbor connections, carefully avoiding Use-After-Free traps.

### 5. Clustering and IVF (Inverted File) - *In Progress*
- **`KMeans`**: A multithreaded (via `parallelize`) and SIMD-vectorized K-Means algorithm for clustering vectors and locating centroids in the vector space.
- **`ArrayInvertedLists`**: An optimized in-memory storage engine for inverted lists. Each bucket independently allocates raw memory for vector IDs and byte codes, doubling its `capacity` dynamically to ensure $O(1)$ amortized insertion time.

## Technical Details
- All mission-critical data structures (graphs, lists, heaps) operate exclusively using raw pointers (`UnsafePointer` and `alloc`) to bypass overhead and achieve complete control over memory.
- The library successfully navigates Mojo's aggressive ASAP (As Soon As Possible) destruction model, explicitly managing pointer lifetimes and object scopes to ensure absolute memory safety.

## Running Tests
Every component is covered by correctness and accuracy tests (verifying recall against the Flat Index). You can run the tests using the Mojo CLI:

```bash
mojo run -I src tests/test_hnsw.mojo
mojo run -I src tests/test_kmeans.mojo
mojo run -I src tests/test_inverted_lists.mojo
```
