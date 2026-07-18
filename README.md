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
**Hardware:** Apple Silicon (ARM64).

| Index | Build Time | QPS | Recall@10 |
|---|---|---|---|
| MojoVec (Pure Mojo) | **~45.9 s** | **~67,700** | 94.67% |
| FAISS (HNSW, C++ via Python) | ~100.8 s | ~25,400 | 95.83% |
| ChromaDB (hnswlib, Python) | ~105.6 s | ~1,990 | 99.22% |

**Methodology:** Apple Silicon M-series (ARM64). FAISS uses OpenMP with 10 threads; MojoVec uses `std.algorithm.parallelize` across logical cores. Recall computed by exact intersection against SIFT1M's provided ground truth (`sift_groundtruth.ivecs`).

MojoVec achieves **over 2.5x the QPS of FAISS** and builds the index **twice as fast** on Apple Silicon, remaining 100% pure Mojo without dropping into C/C++ or assembly.



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
    # Create an HNSW collection (or use create_ivfpq_collection for compression)
    # You can optionally tune HNSW hyperparameters:
    var collection = client.create_collection(
        "my_docs", 
        dimension=128,
        M=32, 
        ef_construction=40, 
        ef_search=16
    )
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
    
    # Optionally increase precision before search
    collection.set_ef_search(100)
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
    from mojovec import Collection
    var loaded = Collection.load("my_database.bin")
```

---

## Python API

MojoVec now includes high-performance native Python bindings. You can install the pre-compiled `.whl` from the GitHub Releases page.

```python
import mojovec

# Initialize collection (dimension=128, M=32, ef_construction=200, ef_search=40)
collection = mojovec.Collection(128, 32, 200, 40)

# Add vectors
ids = [1, 2, 3]
embeddings = [0.1] * (128 * 3) # Flattened 1D list
collection.upsert_batch(ids, embeddings)

# Search
res = collection.query_batch(embeddings[:128], 3)
print("IDs:", res["ids"])             # [[2, 3, 1]]
print("Distances:", res["distances"]) # [[0.0, 0.0, 0.0]]

# Save to disk
collection.save("my_database.bin")

# Load from disk
loaded_collection = mojovec.Collection.load("my_database.bin")
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
