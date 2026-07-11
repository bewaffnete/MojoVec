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
| High-Level API (`Client`, `Collection`) | ✅ Implemented, Chroma-like developer experience |
| IVF + PQ (`IndexIVFPQ`) | ✅ Implemented, extreme compression with automatic training |
| SQ8 / F16 scalar quantization | ✅ Implemented — memory numbers not yet benchmarked |
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
- **`IndexIVFPQ`** — IVF + PQ for extreme compression ratios via Asymmetric Distance Computation. Integrated natively into the high-level API.

*(Note: measured memory footprint before/after quantization, and recall delta vs full-precision benchmarks are coming soon.)*

---

## Quick Start

### 1. Initialize the Client

```mojo
from mojovec.api import Client
from std.collections import List

def main() raises:
    var client = Client()
    # Create an HNSW collection (or use create_ivfpq_collection for compression)
    var collection = client.create_collection("my_docs", dimension=128)
```

### 2. Add Vectors

```mojo
    var ids = List[Int]()
    var embeddings = List[Float32]()
    
    # ... fill ids and embeddings ...
    
    # No pointers, no alloc/free!
    collection.add(ids, embeddings)
```

### 3. Search

```mojo
    var query_embeddings = List[Float32]()
    # ... fill query ...
    
    var results = collection.query(query_embeddings, n_results=5)
    
    for i in range(len(results.ids)):
        print("Query", i)
        for j in range(len(results.ids[i])):
            print("ID:", results.ids[i][j], "Dist:", results.distances[i][j])
```

### 4. Disk Persistence

```mojo
    # Save to disk
    collection.save("my_database.bin")
    
    # Reload anywhere
    from mojovec.api import Collection
    var loaded = Collection.load("my_database.bin")
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
