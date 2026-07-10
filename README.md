# MojoVec 🔥

<p align="center">
  <b>A Vector Database (HNSW, IVF, PQ) written entirely in Mojo.</b>
</p>

MojoVec is a high-performance Approximate Nearest Neighbor (ANN) search library built from scratch, featuring **HNSW (Hierarchical Navigable Small World)**, **Inverted Files (IVF)**, and **Product Quantization (PQ)**.

---

## ⚡️ Design Decisions

1. **Pure Mojo Implementation** — zero C++ dependencies, natively compiled.
2. **Flattened Graph Memory Layout** — the entire HNSW graph lives in a flattened list. Node IDs are strictly `Int32`, packing 16 neighbors into a single 64-byte CPU cache line.
3. **Hardware-Optimized SIMD Distances** — L2 and Inner Product distances are deeply vectorized and unroll directly into hardware SIMD registers for massive Instruction-Level Parallelism.
4. **Cache-Line Aligned Ticket Locks** — multi-threaded inverted list insertions use strictly controlled ticket locks to safely handle concurrent indexing.
5. **Zero-cost Traits** — Built on a strict system of abstractions (`StorageTrait`, `DistanceComputerTrait`), allowing composition without virtual dispatch or runtime overhead.
6. **Value Semantics & Manual Memory** — core heaps and lists use `TrivialRegisterPassable` structs and raw `UnsafePointer` memory allocations to bypass all overhead.

---

## 🚀 Performance Snapshot

Apple Silicon (M-series), 100,000 vectors, 128 dimensions, 10,000 queries:

| Index (efSearch=40) | Upsert / Build Time | QPS | Recall@10 |
|---------------------|---------------------|------------|-----------|
| **FAISS** (HNSW, C++ backend) | ~2.4 sec | **~218k** | **0.999** |
| **MojoVec** (Pure Mojo HNSW)| **~3.1 sec** | **~118k** | **0.999** |
| ChromaDB (hnswlib, Python) | ~5.1 sec | ~15k | 0.999 |

> **Note on Performance:** MojoVec matches the exact recall of heavily optimized FAISS while providing QPS that sits squarely between bare-metal C++ and slower Python wrappers. MojoVec does this using pure Mojo SIMD loops without a single line of C/C++ or assembly! *(HNSW Parameters: `M=32`, `efConstruction=200`)*

---

## 🎯 Honest Tradeoffs

MojoVec is a cutting edge project leveraging an evolving language. While it offers extreme performance out of the box, please consider the following:

- **Mojo is Evolving.** MojoVec targets the very latest Mojo nightly builds. As the language matures, some syntaxes may require updates.
- **Python Wrappers.** We are actively working on wrapping MojoVec using Mojo's FFI capabilities into a standard Python wheel (`pip install mojovec`), but currently it is best consumed directly from Mojo.

---

## 🗜️ Advanced Quantization (Memory Savings)

MojoVec supports extreme memory compression while maintaining incredible search speeds through Product Quantization and Scalar Quantization.

- **`IndexScalarQuantizer`**: Compresses `Float32` vectors into 8-bit integers (`SQ8`) or `Float16`, reducing memory by 4x-2x on the fly.
- **`IndexIVFPQ`**: Combines Inverted Files (IVF) and Product Quantization (PQ) for massive dataset compression. Vectors are split into sub-vectors and quantized to centroids, and searched via Asymmetric Distance Computation (ADC) lookup tables.

---

## 📦 Quick Start

### 1. Initialize the Index (Mojo)

```mojo
from src.mojovec.index.index_hnsw import IndexHNSW
from src.mojovec.index.index_flat import IndexFlat
from src.mojovec.core.types import METRIC_L2

def main() raises:
    var d = 128
    
    # Initialize the underlying storage (Flat)
    var storage = IndexFlat(d, METRIC_L2)
    
    # Initialize HNSW index on top of the storage
    var hnsw = IndexHNSW[IndexFlat](storage^, d, METRIC_L2, M=32)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 40
```

### 2. Ingest Vectors

```mojo
from std.memory import alloc

    var num_vectors = 1000
    var xb = alloc[Float32](num_vectors * d)
    
    # ... (fill xb with data) ...
    
    # Add to index
    hnsw.add(num_vectors, xb)
```

### 3. High-Speed Querying

```mojo
    var num_queries = 10
    var xq = alloc[Float32](num_queries * d)
    var k = 10
    
    # ... (fill xq with query vectors) ...
    
    var distances = alloc[Float32](num_queries * k)
    var labels = alloc[Int](num_queries * k)
    
    # Search top_k nearest neighbors
    hnsw.search(num_queries, xq, k, distances, labels)
    
    # Process results
    for i in range(num_queries):
        print("Query", i)
        for j in range(k):
            print("ID:", labels[i * k + j], "Dist:", distances[i * k + j])
```

### 4. Disk Persistence

```mojo
from src.mojovec.io.index_io import write_index, read_index

    # Save to disk
    write_index(hnsw, "my_index.bin")
    
    # Load from disk
    var loaded_index = read_index("my_index.bin")
```

---

## 🧪 Running Tests & Benchmarks

Requires Mojo (via Pixi/Magic).

```bash
# Run all tests
for f in tests/test_*.mojo; do mojo run -I . "$f"; done

# Run benchmarking suite
mojo run -I . benchmarks/suite/bench_mojovec.mojo
```

---

## License

MIT License.
