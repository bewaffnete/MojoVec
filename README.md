# MojoVec 🔥

**A Vector Database (HNSW) written entirely in Mojo.**

MojoVec is an Approximate Nearest Neighbor (ANN) search library built from scratch in pure Mojo — no C++ dependencies. HNSW, IVF, and Product Quantization (PQ) are implemented.

---

## Why

FAISS and hnswlib are C++ with Python bindings. MojoVec exists to answer a narrower question: can a pure-Mojo implementation get close to hand-tuned C++ SIMD performance without dropping into C/C++/assembly, using only what the language and its `SIMD` type give you today. This was prompted by curiosity about Mojo's SIMD codegen and a desire to build a zero-dependency, bare-metal alternative to FAISS for the Mojo ecosystem.

## Features

- Flat and SQ8 HNSW collections behind one Chroma-style API;
- managed `add`, `upsert`, `update`, `delete`, and batched `query`;
- typed `String`, `Int`, `Float64`, and `Bool` record metadata;
- record documents returned directly with query results;
- native BM25 full-text search over collection documents;
- hybrid vector + BM25 search with reciprocal rank fusion (RRF);
- typed `where` filtering backed by automatic sparse bitmap indexes;
- compound `and_`, `or_`, `not_`, `in_`, and `not_in` expressions;
- save/load, collection statistics, and graph compaction;
- IVF-Flat and IVF-PQ lower-level indexes implemented in pure Mojo.


---

## Performance

**Dataset:** SIFT1M (1,000,000 base vectors, 10,000 query vectors, 128 dimensions).
**MojoVec/FAISS parameters:** `M=32`, `efConstruction=200`,
`efSearch=96`, `k=10`. L2 Distance. SQ8 results use exact reranking of
20 candidates (`k_factor=2`).

### Apple Silicon (ARM64)

| Index | Build Time | QPS | Recall@10 |
|---|---|---|---|
| MojoVec Flat | ~97.9 s | **23,783** | 99.160% |
| MojoVec Quantized (SQ8) | **~59.3 s** | **36,353** | 99.144% |
| FAISS Flat | ~100.8 s | 17,330 | 99.201% |
| FAISS Quantized (SQ8) | ~104.9 s | 14,712 | 99.185% |
| ChromaDB (hnswlib, Python) | ~105.6 s | ~1,990 | 99.22% |

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

On Apple Silicon, MojoVec Flat delivers **about 1.37x the QPS of FAISS Flat**,
while MojoVec SQ8 delivers **about 2.47x the QPS of FAISS SQ8**, with no C/C++
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
from mojovec import Client, Metadata, Where
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
    # add inserts new records and rejects IDs that already exist.
    collection.add(ids, embeddings)
```

### 3. Insert or Replace with Upsert

Use `upsert` when the same batch may contain both new and existing IDs. New
IDs are inserted; existing active records are replaced:

```mojo
    collection.upsert(ids, embeddings)
```

A single `add` or `upsert` batch must not contain duplicate IDs.

### 4. Add Metadata and Documents

Metadata is an owned, typed record supporting `String`, `Int`, `Float64`, and
`Bool` values. Documents are ordinary owned `String` values:

```mojo
    var document = Metadata()
    document.set("title", "Mojo vector search")
    document.set("year", 2026)
    document.set("score", Float64(0.97))
    document.set("published", True)

    var document_embedding = List[Float32]()
    # ... append exactly collection.dimension() components ...

    var metadatas = List[Metadata]()
    metadatas.append(document.copy())
    var documents = [String("A guide to fast vector search in Mojo.")]
    collection.add([42], document_embedding, metadatas, documents)

    var stored = collection.get_metadata(42)
    print(stored.get_string("title"))
    print(stored.keys())
    print(collection.get_document(42))
```

When supplied, there must be one metadata object and one document per ID.
`add`, `upsert`, and `update` accept vector-only, metadata-only, document-only,
or metadata-plus-document batches. Vector-only operations preserve both
payloads. Supplying one payload kind replaces that kind while preserving the
other. An empty metadata object or document string removes that payload.
Metadata and documents are owned values preserved by save/load and compaction.

### 5. Search

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
            if len(results.metadatas) > 0:
                print("Metadata:", results.metadatas[i][j])
            if len(results.documents) > 0:
                print("Document:", results.documents[i][j])
```

`query()` returns managed `QueryResults`. Application code never allocates
output buffers, handles pointers, or calls `free`; result memory is released
automatically when it is no longer used. Metadata and document rows align with
`ids` by query and rank. Missing per-record values use empty placeholders. If
the collection has no metadata or no documents at all, the corresponding outer
List is empty and no payload matrix is allocated.

### 6. Filter by Metadata

`Where` overloads every predicate by metadata type. Ordered comparisons accept
`Int` or `Float64`:

```mojo
    var conditions = List[Where]()
    conditions.append(Where.eq("published", True))
    conditions.append(Where.gte("year", 2024))

    var filtered = collection.query(
        query_embeddings,
        where=Where.and_(conditions),
        n_results=5,
    )
```

Available predicates are `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in_`, and
`not_in`. Combine them with `and_`, `or_`, and `not_`. MojoVec automatically
maintains typed bitmap indexes for metadata fields and rebuilds them on load or
compaction; no index-management API is required.

| Constructor | Accepted values |
|---|---|
| `eq`, `ne` | `String`, `Int`, `Float64`, or `Bool` |
| `gt`, `gte`, `lt`, `lte` | `Int` or `Float64` |
| `in_`, `not_in` | typed `List[String/Int/Float64/Bool]` |
| `and_`, `or_` | `List[Where]` |
| `not_` | one `Where` expression |

Scalar predicates require the field to exist. Consequently, `ne` and `not_in`
do not match records missing that field. `not_(...)` negates the complete
expression and can therefore include records with missing fields.

### 7. Search Documents with BM25

The same `query` name accepts `List[String]` for native full-text search. No
embedding model or second collection is required:

```mojo
    var text_results = collection.query(
        [String("fast vector search"), String("HNSW graph")],
        n_results=5,
    )

    for query_index in range(len(text_results.ids)):
        for rank in range(len(text_results.ids[query_index])):
            print(
                text_results.ids[query_index][rank],
                text_results.scores[query_index][rank],
                text_results.documents[query_index][rank],
            )
```

BM25 results are ordered by descending `scores`—larger is better—and leave
`distances` empty. Vector results populate `distances` and leave `scores`
empty. Text queries also accept the same typed `where` filter:

```mojo
    var filtered_text = collection.query(
        [String("vector database")],
        where=Where.eq("published", True),
        n_results=5,
    )
```

BM25 uses the same analyzer while indexing documents and parsing queries:

- Unicode-aware lowercase normalization;
- word boundaries at whitespace, ASCII/Unicode punctuation, symbols, and emoji;
- bundled English and Russian stopwords.

ASCII apostrophes and underscores are preserved inside a term, so values such
as `rock'n'roll` and `model_v2` stay intact. Stemming is intentionally not
applied: `run` and `running` remain different terms. A query containing only
stopwords returns the normal padded no-match result.

### 8. Hybrid Search with RRF

`query_hybrid` combines vector and BM25 rankings without comparing their
incompatible raw distance and relevance scales:

```mojo
    var hybrid = collection.query_hybrid(
        query_embeddings,
        [String("fast vector search")],
        n_results=5,
    )
```

The embedding batch and text batch must contain the same number of queries.
For each pair, MojoVec retrieves candidates independently from HNSW and BM25,
then adds an equal-weight reciprocal-rank contribution from each list:

```text
RRF score(document) = Σ 1 / (rrf_k + one_based_rank)
```

Defaults are `rrf_k=60` and `candidate_multiplier=4`, so each source supplies
up to `4 * n_results` candidates before fusion. The candidate pool is capped at
2048. Larger `scores` are better; `distances` is empty because raw vector
distances are not meaningful after rank fusion.

The same typed metadata filter is applied to both candidate sources:

```mojo
    var filtered_hybrid = collection.query_hybrid(
        query_embeddings,
        [String("vector database")],
        where=Where.eq("published", True),
        n_results=5,
        rrf_k=60,
        candidate_multiplier=4,
    )
```

Results that occur in both lists receive both contributions. A text query with
no indexed terms naturally falls back to the vector ranking; an empty vector
candidate list naturally falls back to BM25.

### 9. Disk Persistence

```mojo
    # Save to disk
    collection.save("my_database.bin")
    
    # Files of at least 64 MiB use read-only memory-mapped vector and graph
    # arrays by default. Smaller files are copied to owned heap memory.
    from mojovec import Collection
    var loaded = Collection.load("my_database.bin")
    print("memory mapped:", loaded.is_memory_mapped())

    # Force mmap regardless of file size, or disable it explicitly.
    var forced = Collection.load(
        "my_database.bin",
        mmap_threshold_bytes=0,
    )
    var copied = Collection.load(
        "my_database.bin",
        memory_mapped=False,
    )
```

Saved collections omit unused reserved capacity and store populated vector rows
plus occupied HNSW links in 64-byte-aligned regions. Large Flat and SQ8 indexes
can therefore start without copying those arrays into a second heap allocation.
IDs, metadata, documents, deletion flags, and the BM25 index remain managed by
the collection.

Mappings are read-only and owned by the loaded collection; the caller does not
manage file handles or release memory. Queries, filtering, BM25, hybrid search,
soft deletion, and `set_ef_search()` keep the mapping active. The first
operation that extends or rebuilds the graph (`add`, `update`, `upsert`, or
compaction) transparently materializes owned writable arrays, after which
`is_memory_mapped()` returns `False`.

### 10. Inspect and Compact

`update`, `upsert`, and `delete` use soft deletion so searches can continue
without mutating HNSW links in place. Inspect accumulated historical rows and
rebuild the graph when they occupy a meaningful fraction of the collection:

```mojo
    var snapshot = collection.stats()
    print("active:", snapshot.active_count)
    print("deleted:", snapshot.deleted_count)
    print("deleted ratio:", snapshot.deleted_ratio)

    # Rebuild only when at least 20% of stored rows are deleted.
    var report = collection.compact_if_needed(deleted_ratio=0.20)
    print("compacted:", report.performed)
    print("reclaimed rows:", report.reclaimed_records)

    # Use collection.compact() to rebuild immediately when garbage exists.
```

Compaction builds the replacement independently and installs it only after a
successful rebuild. Collection name, storage kind, `M`, `ef_construction`, and
`ef_search` are preserved.

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

# Add records with scalar metadata.
collection.add_with_metadata(
    [4],
    [0.2] * 128,
    [{"title": "Mojo vector search", "year": 2026, "published": True}],
)
print(collection.get_metadata(4))

# Search
res = collection.query(embeddings[:128], 3)
print("IDs:", res["ids"])             # [[2, 3, 1]]
print("Distances:", res["distances"]) # [[0.0, 0.0, 0.0]]

# Inspect soft-deleted historical rows and compact at a 20% threshold.
print(collection.stats())
report = collection.compact_if_needed(0.20)
print(report)

# Save to disk
collection.save("my_database.bin")

# Load from disk
loaded_collection = mojovec.load("my_database.bin")
```

Python metadata accepts dictionaries with string keys and scalar `str`, `int`,
`float`, or `bool` values. Use `add_with_metadata`,
`upsert_with_metadata`, or `update_with_metadata` when replacing metadata;
vector-only operations retain an existing record's metadata.

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
