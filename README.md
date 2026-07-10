# MojoVec 🔥

**A Vector Database (HNSW) written entirely in Mojo.**

MojoVec is an Approximate Nearest Neighbor (ANN) search library built from scratch in pure Mojo — no C++ dependencies. HNSW is implemented and benchmarked below. IVF and Product Quantization are in progress (see Status).

---

## Why

FAISS and hnswlib are C++ with Python bindings. MojoVec exists to answer a narrower question: can a pure-Mojo implementation get close to hand-tuned C++ SIMD performance without dropping into C/C++/assembly, using only what the language and its `SIMD` type give you today. This was prompted by curiosity about Mojo's SIMD codegen and a desire to build a zero-dependency, bare-metal alternative to FAISS for the Mojo ecosystem.

---

## Status

| Component | Status |
|---|---|
| HNSW (build + search) | ✅ Implemented, benchmarked below |
| SQ8 / F16 scalar quantization | ✅ Implemented — memory numbers not yet benchmarked (see below) |
| IVF + PQ (`IndexIVFPQ`) | 🚧 In progress — no working example or benchmark yet |
| Python bindings (`pip install mojovec`) | 🚧 Planned, not published |

---

## Design Decisions

1. **Pure Mojo implementation** — no C++ dependencies, natively compiled.
2. **Flattened graph memory layout** — the HNSW graph lives in a flat array; node IDs are `Int32`, 16 neighbors packed per 64-byte cache line.
3. **SIMD distance kernels** — L2 and Inner Product distances are vectorized to hardware SIMD registers.
4. **Cache-line-aligned ticket locks** — concurrent inverted-list insertion uses ticket locks. *(Not yet exercised by any example below — add a multi-threaded ingest sample once this is validated under load.)*
5. **Static trait-based abstractions** (`StorageTrait`, `DistanceComputerTrait`) instead of virtual dispatch.
6. **Manual memory management** — core structures use `TrivialRegisterPassable` + `UnsafePointer`, no GC overhead.

---

## Performance

100,000 vectors, 128 dimensions, 10,000 queries, `efSearch=40`, `M=32`, `efConstruction=200`.

| Index | Build Time | QPS | Recall@10 |
|---|---|---|---|
| FAISS (HNSW, C++) | ~2.4 s | ~218k | 0.999 |
| MojoVec (pure Mojo HNSW) | ~3.1 s | ~118k | 0.999 |
| ChromaDB (hnswlib, Python) | ~5.1 s | ~15k | 0.999 |

**Methodology:** Apple Silicon M-series (ARM64), multi-threaded index build, single-threaded queries, random synthetic dataset (uniform distribution). Recall computed by directly intersecting with exact 100% ground-truth brute-force Flat index results in the benchmark script (not assumed).

MojoVec's QPS sits between FAISS and a Python-wrapped hnswlib, using SIMD loops without C/C++/assembly. The gap to FAISS (~118k vs ~218k QPS) is the honest cost of a younger compiler and no hand-tuned intrinsics — closing it further is the current focus, not something to gloss over.

---

## Quantization (Memory Compression)

- **`IndexScalarQuantizer`** — compresses `Float32` vectors to 8-bit (`SQ8`) or `Float16`, a 4x/2x reduction on paper.
- **`IndexIVFPQ`** — IVF + PQ for larger compression ratios via Asymmetric Distance Computation. *(Status: in progress, no benchmark yet.)*

*(Note: measured memory footprint before/after quantization, and recall delta vs full-precision benchmarks are coming soon.)*

---

## Quick Start

### 1. Initialize the Index (Mojo)

```mojo
from src.mojovec import IndexHNSW, IndexFlat, METRIC_L2

def main() raises:
    var d = 128

    var storage = IndexFlat(d, METRIC_L2)
    var hnsw = IndexHNSW[IndexFlat](storage^, d, METRIC_L2, M=32)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 40
```

### 2. Ingest Vectors

```mojo
from std.memory import alloc

    var num_vectors = 1000
    var xb = alloc[Float32](num_vectors * d)

    # ... fill xb with data ...

    hnsw.add(num_vectors, xb)
```

### 3. Search

```mojo
    var num_queries = 10
    var xq = alloc[Float32](num_queries * d)
    var k = 10

    # ... fill xq with query vectors ...

    var distances = alloc[Float32](num_queries * k)
    var labels = alloc[Int](num_queries * k)

    hnsw.search(num_queries, xq, k, distances, labels)

    for i in range(num_queries):
        print("Query", i)
        for j in range(k):
            print("ID:", labels[i * k + j], "Dist:", distances[i * k + j])
```

### 4. Disk Persistence

```mojo
from src.mojovec import write_index, read_index

    write_index(hnsw, "my_index.bin")
    var loaded_index = read_index("my_index.bin")
```

---

## Running Tests & Benchmarks

Requires Mojo (via Pixi/Magic).

```bash
# Run all tests
for f in tests/test_*.mojo; do mojo run -I . "$f"; done

# Run benchmark suite
mojo run -I . benchmarks/suite/bench_mojovec.mojo
```

---

## License

MIT License.
