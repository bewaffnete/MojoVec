# MojoVec 🔥

**A Vector Database (HNSW) written entirely in Mojo.**

MojoVec is an Approximate Nearest Neighbor (ANN) search library built from scratch in pure Mojo — no C++ dependencies. HNSW, IVF, and Product Quantization (PQ) are implemented.

---

## Why

FAISS and hnswlib are C++ with Python bindings. MojoVec exists to answer a narrower question: can a pure-Mojo implementation get close to hand-tuned C++ SIMD performance without dropping into C/C++/assembly, using only what the language and its `SIMD` type give you today. This was prompted by curiosity about Mojo's SIMD codegen and a desire to build a zero-dependency, bare-metal alternative to FAISS for the Mojo ecosystem.



---

## Performance

**Dataset:** SIFT1M (1,000,000 base vectors, 10,000 query vectors, 128 dimensions).
**Parameters:** `M=32`, `efConstruction=200`, `efSearch=40`, `k=10`. L2 Distance.
### Apple Silicon (ARM64)

| Index | Build Time | QPS | Recall@10 |
|---|---|---|---|
| MojoVec Flat | ~97.9 s | **45,126** | 95.88% |
| MojoVec Quantized (SQ8) | **~59.3 s** | **70,161** | 94.66% |
| FAISS Flat | ~100.8 s | 32,103 | 95.85% |
| FAISS Quantized (SQ8) | ~104.9 s | 30,127 | 94.99% |
| Chroma | ~99.15 s | ~6,815 | 95.91% |

### x86_64 (4 Cores VM)

| Index | Build Time | QPS | Recall@10 |
|---|---|---|---|
| MojoVec (Pure Mojo) | **~367.1 s** | **~8,912** | 94.64% |
| FAISS (HNSW, C++ via Python) | ~693.2 s | ~4,773 | 95.88% |
| ChromaDB (hnswlib, Python) | ~658.3 s | ~1,610 | 99.20% |

**Methodology:** FAISS uses 10 OpenMP threads. MojoVec uses the Mojo AsyncRT
worker pool for construction and small query batches; large Flat/SQ8 HNSW
batches use native OS threads so that all logical CPU cores participate. Apple
Silicon QPS and recall are isolated cold search-only measurements over ten
samples; build times come from separate end-to-end runs and are more sensitive
to thermal state. Recall is the exact intersection against SIFT1M's provided
ground truth (`sift_groundtruth.ivecs`).

On Apple Silicon, MojoVec Flat delivers **about 1.4x the QPS of FAISS Flat**,
while MojoVec SQ8 delivers **about 2.3x the QPS of FAISS SQ8**, with no C/C++
dependency or handwritten assembly.



---

## Installation

```bash
# Download the latest mojovec.mojoc
curl -LO https://github.com/bewaffnete/MojoVec/releases/latest/download/mojovec.mojoc
```

Place the `mojovec.mojoc` file in your project directory. You can now import it directly in your code. (If you place it elsewhere, pass the include path to the compiler: `mojo run -I /path/to/dir your_script.mojo`).

---

## Quick Start

### 1. Initialize the Client

```mojo
from mojovec import Client
from std.collections import List

def main() raises:
    var client = Client()
    # quantized=True selects SQ8; False selects exact Flat storage.
    var collection = client.create_collection(
        "my_docs", 
        dimension=128,
        M=32, 
        ef_construction=40, 
        ef_search=16,
        quantized=True,
    )
```

### 2. Add Vectors

```mojo
    var ids = List[Int]()
    var embeddings = List[Float32]()
    
    # ... fill ids and embeddings ...
    
    # No pointers, no alloc/free!
    # add rejects existing IDs; upsert inserts or replaces them.
    collection.upsert(ids, embeddings)
```

### 3. Search

```mojo
    var query_embeddings = List[Float32]()
    # ... fill query ...
    
    # Optionally increase precision before search
    collection.set_ef_search(100)
    var results = collection.query(query_embeddings, n_results=5)
    
    for i in range(len(results.ids)):
        print("Query", i)
        for j in range(len(results.ids[i])):
            print("ID:", results.ids[i][j], "Dist:", results.distances[i][j])
```

`query()` returns managed `QueryResults`. Application code never allocates
output buffers, handles pointers, or calls `free`; result memory is released
automatically when it is no longer used.

### 4. Disk Persistence

```mojo
    # Save to disk
    collection.save("my_database.bin")
    
    # Reload anywhere
    from mojovec import Collection
    var loaded = Collection.load("my_database.bin")
```

---

## Python API

MojoVec now includes high-performance native Python bindings. You can install the package directly from PyPI (Linux x86_64 and macOS Apple Silicon are supported):

```bash
pip install mojovec
```

```python
import mojovec

# dimension, M, ef_construction, ef_search, quantized
collection = mojovec.Collection(128, 32, 200, 40, True)

# Add vectors
ids = [1, 2, 3]
embeddings = [0.1] * (128 * 3) # Flattened 1D list
collection.upsert(ids, embeddings)

# Search
res = collection.query(embeddings[:128], 3)
print("IDs:", res["ids"])             # [[2, 3, 1]]
print("Distances:", res["distances"]) # [[0.0, 0.0, 0.0]]

# Save to disk
collection.save("my_database.bin")

# Load from disk
loaded_collection = mojovec.load("my_database.bin")
```

---

## Running Tests & Benchmarks

Requires Mojo (via Pixi/Magic).

```bash
# Run all tests
for f in tests/test_*.mojo; do mojo run -I . "$f"; done

# Run the SIFT1M HNSW benchmarks
mojo run -I . benchmarks/suite/mojovec_sq8.mojo
mojo run -I . benchmarks/suite/mojovec_flat.mojo
```

---

## License

MIT License.
