# MojoVec examples

The examples are split by language:

- [`quickstart_mojovec.ipynb`](quickstart_mojovec.ipynb) is the complete
  interactive Jupyter/Colab walkthrough;
- [`mojo/`](mojo/) contains twelve executable tutorials for the Mojo API;
- [`python/`](python/) contains nine executable tutorials for the managed
  Python API.

Both tracks use deterministic, locally generated vectors and require no
downloaded datasets.

## Before you start

Run every command from the repository root.

Build and install the Python package once before running the Python track:

```bash
python -m pip install .
```

Then run:

```bash
python examples/python/api_01_collection_crud.py
python examples/python/api_02_metadata_where.py
python examples/python/api_03_bm25_hybrid.py
python examples/python/api_04_persistence_compaction.py
python examples/python/api_05_wal_recovery.py
python examples/python/api_06_numpy_fast_path.py
python examples/python/api_07_distance_metrics.py
python examples/python/api_08_dataset_io.py
python examples/python/api_09_ivfpq.py
```

The standard MojoVec installation includes NumPy, PyArrow, and Hugging Face
`datasets`, so every Python dataset example uses the same base installation.

For the Mojo track, `-I .` resolves the local `mojovec` package:

Example 12 additionally uses Mojo's embedded Python. When running from a source
checkout, expose the source adapter to that interpreter:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

```bash
mojo run -I . examples/mojo/api_01_hnsw_fast_search.mojo
mojo run -I . examples/mojo/api_02_ivfpq_compression.mojo
mojo run -I . examples/mojo/api_03_serialization.mojo
mojo run -I . examples/mojo/api_04_compaction.mojo
mojo run -I . examples/mojo/api_05_metadata.mojo
mojo run -I . examples/mojo/api_06_where_filters.mojo
mojo run -I . examples/mojo/api_07_bm25.mojo
mojo run -I . examples/mojo/api_08_hybrid_rrf.mojo
mojo run -I . examples/mojo/api_09_wal.mojo
mojo run -I . examples/mojo/api_10_distance_metrics.mojo
mojo run -I . examples/mojo/api_11_dataset_io.mojo
mojo run -I . examples/mojo/api_12_python_dataset_io.mojo
```

To compile without executing:

```bash
mojo build -I . examples/mojo/api_01_hnsw_fast_search.mojo
mojo build -I . examples/mojo/api_02_ivfpq_compression.mojo
mojo build -I . examples/mojo/api_03_serialization.mojo
mojo build -I . examples/mojo/api_04_compaction.mojo
mojo build -I . examples/mojo/api_05_metadata.mojo
mojo build -I . examples/mojo/api_06_where_filters.mojo
mojo build -I . examples/mojo/api_07_bm25.mojo
mojo build -I . examples/mojo/api_08_hybrid_rrf.mojo
mojo build -I . examples/mojo/api_09_wal.mojo
mojo build -I . examples/mojo/api_10_distance_metrics.mojo
mojo build -I . examples/mojo/api_11_dataset_io.mojo
mojo build -I . examples/mojo/api_12_python_dataset_io.mojo
```

## Python tutorial track

Run the Python examples in numeric order:

| Example | Topic | Important API |
| --- | --- | --- |
| [`api_01_collection_crud.py`](python/api_01_collection_crud.py) | Flat/SQ8 construction, nested vectors, CRUD, batching, stats | `Collection`, `add`, `upsert`, `update`, `delete`, `query` |
| [`api_02_metadata_where.py`](python/api_02_metadata_where.py) | Batch payloads and nested Chroma-style filters | `metadatas`, `documents`, `where`, `get_metadata`, `get_document` |
| [`api_03_bm25_hybrid.py`](python/api_03_bm25_hybrid.py) | Unicode BM25 and reciprocal-rank fusion | `query_texts`, `query_hybrid`, `scores` |
| [`api_04_persistence_compaction.py`](python/api_04_persistence_compaction.py) | Atomic save, mmap ownership, point-in-time readers, garbage reclamation | `save`, `load`, `snapshot`, `is_memory_mapped`, `compact_if_needed` |
| [`api_05_wal_recovery.py`](python/api_05_wal_recovery.py) | Optional async/sync durability and restart recovery | `enable_wal`, `flush_wal`, `recover`, `checkpoint` |
| [`api_06_numpy_fast_path.py`](python/api_06_numpy_fast_path.py) | Contiguous zero-copy vector buffers | `upsert_numpy`, `query_numpy` |
| [`api_07_distance_metrics.py`](python/api_07_distance_metrics.py) | L2, cosine, and inner-product ordering | `metric`, `storage_kind` |
| [`api_08_dataset_io.py`](python/api_08_dataset_io.py) | Batched local files and Hugging Face streaming | `add_from`, `upsert_from`, `add_huggingface` |
| [`api_09_ivfpq.py`](python/api_09_ivfpq.py) | Product-quantized training, probing, metrics, and persistence | `IVFPQCollection`, `train`, `set_nprobe`, `save`, `load` |
| [`quickstart_mojovec.ipynb`](quickstart_mojovec.ipynb) | Complete interactive Jupyter/Colab quickstart | CRUD, `where`, BM25, hybrid RRF, Flat/SQ8, NumPy, compaction, mmap, WAL |

The managed Python methods return ordinary Python values and own all native
resources. No example opens an index file manually, releases a query buffer, or
calls a native `free`.

## Mojo collection selection

| Collection | Creation | Vector representation | Strength | Main trade-off |
| --- | --- | --- | --- | --- |
| HNSW + SQ8 | `create_collection(..., quantized=True)` | 8-bit scalar-quantized search storage | High QPS and lower search bandwidth | Approximate distances |
| HNSW + Flat | `create_collection(..., quantized=False)` | Original `Float32` vectors | Exact vector distances inside HNSW traversal | More memory bandwidth |
| IVF-PQ | `create_ivfpq_collection(...)` | Compact product-quantized codes | Strong compression for large datasets | Requires training and recall tuning |

All three indexes perform approximate nearest-neighbor search at the index
level. “Flat” here means that HNSW evaluates candidates against unquantized
`Float32` vectors; it does not mean an exhaustive scan of every vector.

## Example 1: complete HNSW lifecycle

File: [`api_01_hnsw_fast_search.mojo`](mojo/api_01_hnsw_fast_search.mojo)

This is the recommended first example. It covers:

- row-major flattened embedding layout;
- all `create_collection` parameters;
- SQ8 and Flat collections through the same API;
- `add`, `upsert`, `update`, and soft `delete`;
- batched `query` results;
- runtime `ef_search` tuning;
- automatic ownership of inputs, index storage, and query results.

The CRUD operations intentionally have different contracts:

| Operation | Existing ID | Missing ID | Duplicate ID in one batch |
| --- | --- | --- | --- |
| `add` | Error | Insert | Error |
| `upsert` | Replace | Insert | Error |
| `update` | Replace | Error | Error |
| `delete` | Soft-delete | Ignore | Safe/idempotent |

Updates and upserts append a new internal vector and mark the previous version
deleted. `count()` reports active application IDs. `count_deleted()` reports
deleted historical internal records, so it can grow after repeated updates.
Use `stats()` to inspect that storage and `compact()` or
`compact_if_needed()` to rebuild from only active records.

## Example 2: IVF-PQ training and compression

File: [`api_02_ivfpq_compression.mojo`](mojo/api_02_ivfpq_compression.mojo)

This example explains:

- what IVF and PQ contribute;
- the meaning of `dimension`, `nlist`, and PQ `M`;
- runtime `nprobe` recall/speed tuning and collection statistics;
- L2, cosine, and inner-product distance semantics;
- the equivalent managed `IVFPQCollection` Python API;
- the requirement that `dimension` be divisible by PQ `M`;
- explicit training on representative data;
- automatic training during the first `add`;
- why approximate distances should not be compared for exact equality.

Training requires at least `max(nlist, 256)` vectors because each eight-bit PQ
subquantizer learns 256 centroids. For production data, use a larger,
representative sample. A biased first ingestion batch usually reduces recall.

## Example 3: serialization and recovery

File: [`api_03_serialization.mojo`](mojo/api_03_serialization.mojo)

This example round-trips both SQ8 and Flat HNSW collections and verifies the
collection state visible after loading:

- collection name;
- dimension and storage kind;
- memory-mapped ownership state;
- user-provided IDs;
- active and soft-deleted state;
- vectors and HNSW graph.

The example writes:

```text
/tmp/mojovec_example_sq8.bin
/tmp/mojovec_example_flat.bin
```

`Collection.load(path)` detects Flat versus SQ8 from the file header. Do not
create a new collection or pass `quantized` when loading. Saved files store the
large vector and HNSW arrays in aligned regions. Files of at least 64 MiB are
mapped read-only by default; the example uses `mmap_threshold_bytes=0` to force
that path for its small fixture. Pass `memory_mapped=False` to force a copied
heap load.

The mapping belongs to `Collection`; users do not keep a file open or release
memory manually. Read-only search operations preserve it. The first
`add`/`update`/`upsert` or compaction transparently detaches the mapped arrays
into writable owned storage.

`save()` publishes through `fsync` and atomic rename, so an existing destination
is never replaced by a partially written collection. `snapshot(path)` performs
that publication and returns an independent point-in-time reader. Existing
mmap readers keep their previous state while one writer publishes the next
state. Each snapshot carries a checksum that `load()` verifies before parsing
or mmap, so truncated and corrupted files are rejected. Changes become
crash-durable after `save()` or `snapshot()` returns.
Applications that need to recover mutations made between snapshots can enable
the optional WAL shown in Example 9.

## Example 4: statistics and compaction

File: [`api_04_compaction.mojo`](mojo/api_04_compaction.mojo)

This example creates historical rows through updates and deletion, then shows:

- the fields returned by `Collection.stats()`;
- threshold-based `compact_if_needed(deleted_ratio=...)`;
- the before/after `CompactReport`;
- preservation of HNSW parameters and search results;
- the no-op behavior when no deleted rows remain.

Compaction does not edit HNSW links in place. It builds a complete replacement
from active records and swaps it into the collection only after construction
succeeds. Peak memory therefore includes both indexes plus a bounded ingestion
batch. Do not mutate or query the same collection concurrently while a
compaction call is running.

## Example 5: typed record metadata

File: [`api_05_metadata.mojo`](mojo/api_05_metadata.mojo)

This example covers:

- `Metadata` values of type `String`, `Int`, `Float64`, and `Bool`;
- adding one metadata object per vector;
- adding one document string per vector in the same batch;
- retrieving an owned metadata copy by application ID;
- metadata inheritance during vector-only `update` and `upsert`;
- complete replacement when metadata is supplied explicitly;
- reading aligned metadata and documents directly from `QueryResults`;
- preservation through save/load and compaction.

Metadata and documents are stored as snapshots alongside every internal vector
version. Calling `get_metadata(id)` or `get_document(id)` for a missing or
deleted ID raises an error. `QueryResults.metadatas` and
`QueryResults.documents` are aligned with result IDs; missing per-record
payloads use empty placeholders.

## Example 6: typed `where` filters and bitmap indexes

File: [`api_06_where_filters.mojo`](mojo/api_06_where_filters.mojo)

This example covers:

- type-safe String, Int, Float64, and Bool predicates;
- equality, ordered comparisons, and typed `in_` membership;
- nested `and_` and `or_` expressions;
- filtered search through `Collection.query(..., where=...)`;
- the distinction between `ne` and logical `not_` for missing fields.

MojoVec maintains sparse packed bitmap postings automatically as metadata is
added or replaced. Deleted historical rows are excluded by the normal
collection deletion mask. Bitmap state is rebuilt from metadata after load and
compaction, so applications do not serialize or manage indexes separately.
Fields exceeding 1024 distinct values use a scan fallback to avoid excessive
per-value bitmap overhead.

## Example 7: native BM25 document search

File: [`api_07_bm25.mojo`](mojo/api_07_bm25.mojo)

This example covers:

- adding documents and metadata in one managed batch;
- single and batched BM25 queries through `Collection.query(List[String])`;
- descending BM25 scores and aligned payloads in `QueryResults`;
- Unicode lowercase and word-boundary tokenization;
- automatic English and Russian stopword removal without stemming;
- combining full-text ranking with typed metadata filters;
- automatic posting updates after `update`, `upsert`, and `delete`;
- rebuilding the text index from documents after save/load and compaction.

The BM25 index is append-only over internal record versions, just like the
collection itself. Replaced and deleted versions are deactivated immediately.
Compaction discards their stale postings, while load deterministically rebuilds
the index from stored documents. There is no separate text-index file or manual
index-maintenance API.

## Example 8: hybrid vector + BM25 search with RRF

File: [`api_08_hybrid_rrf.mojo`](mojo/api_08_hybrid_rrf.mojo)

This example covers:

- aligned embedding and text query batches;
- reciprocal rank fusion of HNSW and BM25 candidates;
- the `rrf_k` and `candidate_multiplier` controls;
- typed metadata filtering applied to both candidate sources;
- hybrid `scores`, padded results, metadata, and documents;
- natural vector-only fallback for a stopword-only text query.

RRF uses ranks rather than mixing raw vector distances with BM25 scores. A
record present in both candidate lists receives a contribution from both.
Hybrid search leaves `distances` empty and returns the fused values through
`QueryResults.scores`.

## Example 9: optional write-ahead log

File: [`api_09_wal.mojo`](mojo/api_09_wal.mojo)

This example demonstrates standalone crash recovery without adding WAL work to
queries:

- creating an atomic snapshot as the recovery base;
- enabling the high-throughput `WAL_ASYNC` mode;
- appending one WAL frame per complete public mutation batch;
- choosing an explicit durability boundary with `flush_wal()`;
- restoring vectors, metadata, documents, updates, and deletions with
  `Collection.recover()`;
- publishing a new durable snapshot and rotating covered frames with
  `checkpoint()`;
- selecting `WAL_SYNC` when every mutation call must perform its own `fsync`.

`WAL_ASYNC` does not mean that a background worker owns the collection. It means
that appends avoid a per-call `fsync`, allowing several API batches to share one
`flush_wal()` barrier. `WAL_SYNC` trades ingestion throughput for the smaller
durability window. Both modes use the same Flat/SQ8 collection API, and neither
changes query execution.

## Embedding layout

Every public high-level method receives one flattened `List[Float32]`. For
three vectors of dimension four:

```text
[
  v0[0], v0[1], v0[2], v0[3],
  v1[0], v1[1], v1[2], v1[3],
  v2[0], v2[1], v2[2], v2[3],
]
```

For ingestion, the required invariant is:

```text
len(embeddings) == len(ids) * collection.dimension()
```

For queries:

```text
len(query_embeddings) % collection.dimension() == 0
```

The number of query rows is inferred from the flattened list length.

## HNSW tuning

The high-level HNSW constructor exposes three important controls:

- `M`: graph connectivity. Increasing it typically improves recall, build time,
  and memory usage.
- `ef_construction`: candidate count during graph construction. Increasing it
  generally builds a better graph more slowly.
- `ef_search`: candidate count during search. Increasing it generally improves
  recall and reduces QPS.

Start with `ef_search >= n_results`, then measure recall and latency on the
actual dataset. Parameters that work well for one embedding model may be poor
for another.

## Managed query results

The public API exposes one managed query operation:

```mojo
var results = collection.query(queries, n_results=10)
```

It returns `QueryResults` containing managed nested `List`s:

```text
results.ids[query_index][rank]
results.distances[query_index][rank]
results.metadatas[query_index][rank]
results.documents[query_index][rank]
results.scores[query_index][rank]
```

MojoVec owns all temporary search buffers and transfers the result Lists to the
caller. User code does not call `alloc`, work with pointers, or call `free`.
Dropping `QueryResults` releases its storage automatically. If the collection
does not store a payload kind, its corresponding outer List is empty. Vector
queries populate `distances`; BM25 and hybrid queries populate `scores`.

## Distance interpretation

`create_collection(..., metric=...)` accepts three Chroma-style metrics:

| `metric` | Stored/query preprocessing | Returned distance |
| --- | --- | --- |
| `"l2"` | Unchanged | Squared Euclidean distance |
| `"cosine"` | Automatic row-wise normalization | `1 - cosine_similarity` |
| `"ip"` | Unchanged | `1 - inner_product` |

Smaller is always closer. Cosine rejects zero and non-finite vectors because
their direction is undefined. IP distances can be negative when the dot product
is greater than one. Metric selection is independent of `quantized=True` SQ8
versus `quantized=False` Flat storage and is preserved by save/load, mmap,
compaction, filtering, hybrid search, and WAL recovery.

See [`api_10_distance_metrics.mojo`](mojo/api_10_distance_metrics.mojo) for a
complete executable comparison. SQ8 and IVF-PQ distances are approximate;
validate production quality with recall@k, not distance equality alone.

## Common errors

### Embedding length mismatch

Check that every inserted ID has exactly `dimension` components and that query
lists contain complete rows.

### Duplicate IDs

Use `add` only for new IDs. Use `upsert` when a batch may contain IDs already
stored in the collection. A single operation must still not contain the same ID
twice.

### Growing deleted count

Repeated updates and upserts retain old internal rows until compaction. Inspect
`collection.stats().deleted_ratio` and call
`compact_if_needed(deleted_ratio=0.20)` at an application-appropriate
maintenance point. A lower threshold reclaims storage more eagerly but rebuilds
the HNSW graph more often.

### Invalid IVF-PQ shape

Choose PQ `M` so `dimension % M == 0`. Also provide enough representative
training rows for the requested number of clusters and codebooks.

### Low recall

- Increase HNSW `ef_search`.
- Improve `ef_construction` or `M` and rebuild.
- For IVF-PQ, improve the training sample and tune IVF probing.
- Compare Flat and quantized variants on the same queries and ground truth.

### Misleading local benchmarks

Compile before timing and separate build benchmarks from search-only
benchmarks. Internal MojoVec benchmarks may use private reusable buffers to
measure the search kernel without result-construction overhead; that mechanism
is deliberately not part of the user API. On laptops, long index construction
can heat the CPU and lower the QPS measured immediately afterward.
