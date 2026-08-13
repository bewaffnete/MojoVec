# MojoVec Guide for AI Coding Agents

This document is a standalone implementation guide. Give it to an AI coding
agent when you want that agent to write application code using MojoVec without
first reading the entire repository.

The public API is the source of truth. Use only the managed APIs documented
here unless the task explicitly asks for MojoVec internals, index development,
or benchmarking.

## Non-negotiable rules

1. Prefer the high-level `Collection` API. It owns all native memory and result
   storage.
2. Never make application code import Python modules such as
   `mojovec._native`, `_runtime`, or CPU-specific binary packages. Use only
   `import mojovec`.
3. Never expose `UnsafePointer`, `alloc`, or `free` in high-level Mojo examples.
   Use `List`, `Metadata`, `Where`, and `QueryResults`.
4. Do not manually normalize cosine vectors. MojoVec normalizes stored vectors
   and queries without modifying caller-owned values.
5. Reject or repair shape errors before calling the library. Every vector must
   have exactly the collection dimension.
6. Do not mutate a collection while another thread queries or mutates that
   same object. Use application locking or independent `snapshot()` readers.
7. Use HNSW `Collection` by default. Use `CollectionIVFPQ` only when the user
   explicitly needs IVF-PQ compression and accepts its smaller feature set.
8. Do not hard-code a MojoVec package version unless the surrounding project
   deliberately pins dependencies.

## Mental model

A MojoVec HNSW collection owns one logical record per active application ID:

```text
record ID
  + fixed-size Float32 embedding
  + optional typed metadata
  + optional document string
```

The vector index, metadata bitmap indexes, and BM25 document index are kept in
sync by collection mutations. The application does not manage these indexes
separately.

`add`, `upsert`, `update`, and `delete` operate on application IDs. HNSW uses
internal row numbers, but those must never be exposed as record IDs.

Updates and deletions use soft deletion. Replaced historical rows remain in
the graph until `compact()` or `compact_if_needed()` rebuilds it.

## Choose storage and metric independently

```text
quantized=False  -> HNSW + exact Flat Float32 candidate traversal
quantized=True   -> HNSW + SQ8 candidate traversal + exact Float32 reranking
```

SQ8 retains original Float32 rows for exact reranking. Choose SQ8 primarily for
faster candidate traversal; do not describe it as a mode that removes every
Float32 copy from memory.

Supported metrics:

| `metric` | Public distance | Notes |
|---|---|---|
| `"l2"` | squared Euclidean distance | Smaller is better. |
| `"cosine"` | `1 - cosine_similarity` | Automatic normalization; zero vectors are rejected. |
| `"ip"` | `1 - inner_product` | Can be negative when the dot product is greater than one. |

All public vector distances use smaller-is-better ordering. NaN and infinite
components are rejected for every metric.

SQ8 uses affine `UInt8` codes for L2 and symmetric `Int8` codes for cosine/IP.
Its approximate HNSW traversal is followed by exact Float32 reranking of the
small final candidate set.

## HNSW parameters

The Python and Mojo high-level constructors use these defaults:

```text
M=32
ef_construction=40
ef_search=16
quantized=True
metric="l2"
```

Validated bounds:

```text
1 <= dimension <= 65_536
2 <= M <= 1_000
1 <= ef_construction <= 2_048
1 <= ef_search <= 2_048
```

- `M` increases graph connectivity, memory, build time, and usually recall.
- `ef_construction` increases build quality and build cost.
- `ef_search` is the main query-time recall/latency control and can be changed
  with `set_ef_search()`.
- Normally use `ef_search >= n_results`. Measure recall and latency on the
  application's real embeddings before choosing production values.

For high-recall workloads, `M=32`, `ef_construction=96` or higher, and
`ef_search=96` are reasonable starting points, not universal guarantees.

## Python API

### Installation and import

```bash
python -m pip install mojovec
```

```python
import mojovec
```

Linux wheels select AVX-512 when the complete supported ISA is available and
otherwise select AVX2. Application code must not import a backend directly.
`mojovec.native_backend()` is available for diagnostics.

### Complete Python example

```python
import mojovec

collection = mojovec.Collection(
    dimension=3,
    M=32,
    ef_construction=96,
    ef_search=96,
    quantized=True,
    metric="cosine",
    name="articles",
)

collection.add(
    ids=[101, 202, 303],
    embeddings=[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0],
    ],
    metadatas=[
        {"kind": "guide", "year": 2026, "published": True},
        {"kind": "reference", "year": 2025, "published": True},
        {"kind": "guide", "year": 2024, "published": False},
    ],
    documents=[
        "MojoVec vector search guide",
        "Python API reference",
        "Hybrid retrieval tutorial",
    ],
)

result = collection.query(
    query_embeddings=[[1.0, 0.0, 0.0]],
    n_results=2,
    where={
        "$and": [
            {"kind": {"$in": ["guide", "reference"]}},
            {"published": True},
        ]
    },
)

for record_id, distance, metadata, document in zip(
    result["ids"][0],
    result["distances"][0],
    result["metadatas"][0],
    result["documents"][0],
):
    if record_id >= 0:
        print(record_id, distance, metadata, document)
```

### Python constructor

```python
mojovec.Collection(
    dimension: int,
    M: int = 32,
    ef_construction: int = 40,
    ef_search: int = 16,
    quantized: bool = True,
    metric: str = "l2",
    name: str = "",
)
```

### Python mutations

```python
collection.add(ids, embeddings, metadatas=None, documents=None)
collection.upsert(ids, embeddings, metadatas=None, documents=None)
collection.update(ids, embeddings, metadatas=None, documents=None)
collection.delete(ids)
```

- `add` requires every ID to be new.
- `upsert` inserts missing IDs and replaces existing IDs.
- `update` requires every ID to exist.
- A mutation batch cannot contain duplicate IDs.
- `delete` is idempotent and ignores unknown IDs.
- Embeddings may be nested rows or one flat row-major sequence.
- `len(flat_embeddings)` must equal `len(ids) * dimension`.
- When supplied, `metadatas` and `documents` must contain one item per ID.
- Vector-only `upsert` and `update` preserve existing metadata and documents.
- Supplying a payload kind replaces that complete payload for each record.
- An empty metadata mapping or empty document removes that payload.
- A public mutation batch is prepared and committed atomically. Validation,
  payload preparation, or WAL failure must not leave a partially applied batch.

Python metadata is a mapping with string keys and scalar `str`, `int`, `float`,
or `bool` values. Nested objects, arrays, and `None` are not metadata values.

### Python vector, BM25, and hybrid queries

```python
# Vector search
vector_result = collection.query(
    query_embeddings=[[1.0, 0.0, 0.0]],
    n_results=10,
    where=None,
)

# BM25 document search
text_result = collection.query(
    query_texts=["vector search"],
    n_results=10,
    where=None,
)

# Equivalent BM25 shorthand
text_result = collection.query(["vector search"], n_results=10)

# Hybrid vector + BM25 search using reciprocal rank fusion
hybrid_result = collection.query_hybrid(
    query_embeddings=[[1.0, 0.0, 0.0]],
    query_texts=["vector search"],
    n_results=10,
    rrf_k=60,
    candidate_multiplier=4,
    where=None,
)
```

Do not pass both `query_embeddings` and `query_texts` to `query()`. Use
`query_hybrid()` for combined retrieval. A hybrid call requires the same number
of embedding and text queries. Its candidate pool is capped at 2048.

Every managed query returns this dictionary shape:

```python
{
    "ids": list[list[int]],
    "distances": list[list[float]],
    "metadatas": list[list[dict[str, str | int | float | bool]]],
    "documents": list[list[str]],
    "scores": list[list[float]],
}
```

- Vector search populates `distances` and leaves `scores` empty.
- BM25 and hybrid search populate `scores` and leave `distances` empty.
- BM25 and RRF scores use larger-is-better ordering.
- Payload rows align with IDs by query and rank.
- Missing payload values are empty mappings or strings.
- If the collection contains no values of a payload kind, that entire outer
  list is empty.
- When fewer than `n_results` records match, padded IDs may be `-1`. Ignore
  entries whose ID is negative.

### Python metadata filters

Supported operators:

```text
$eq  $ne  $gt  $gte  $lt  $lte  $in  $nin  $and  $or  $not
```

Examples:

```python
# Multiple fields in one mapping are implicitly ANDed.
where = {
    "published": True,
    "year": {"$gte": 2024},
}

where = {
    "$or": [
        {"kind": {"$eq": "guide"}},
        {"score": {"$gt": 0.95}},
    ]
}

where = {"kind": {"$nin": ["private", "archived"]}}
where = {"$not": {"published": False}}
```

`$in` and `$nin` values must be non-empty homogeneous scalar lists. Ordered
comparisons accept numeric values. Scalar predicates require the field to
exist, so `$ne` and `$nin` do not match a record missing that field. `$not`
negates the complete expression and can include records with missing fields.

Metadata bitmap indexes are maintained automatically. Do not create or update
them in application code.

### NumPy fast path

NumPy is optional. Use it for large numeric batches:

```python
import numpy as np
import mojovec

collection = mojovec.Collection(dimension=128)

ids = np.arange(1000, 2000, dtype=np.int64)
embeddings = np.ascontiguousarray(data, dtype=np.float32)
collection.upsert_numpy(ids, embeddings)
# add_numpy(ids, embeddings) provides insert-only semantics through the same
# contiguous-buffer path.

queries = np.ascontiguousarray(query_data, dtype=np.float32)
result = collection.query_numpy(queries, n_results=10)
```

Requirements:

- IDs: contiguous one-dimensional `numpy.int64`.
- Embeddings: contiguous `numpy.float32`, flat or `(count, dimension)`.
- Queries: contiguous `numpy.float32` with shape `(count, dimension)`.
- `query_numpy()` is vector-only and returns empty payload lists.
- Supplying Python metadata or documents to `upsert_numpy()` intentionally
  uses the managed conversion path.

### External dataset readers

Use collection-level import methods instead of writing format-specific loops:

```python
collection.add_from(
    "records.parquet",
    id_column="id",
    embedding_column="embedding",
    document_column="text",
    metadata_columns=["category"],
    batch_size=8192,
)
collection.upsert_huggingface(
    "owner/dataset",
    split="train",
    embedding_column="embedding",
    document_column="text",
)
```

`add_from`/`upsert_from` support CSV, TSV, JSON, JSONL, Parquet, Arrow IPC,
NPY, NPZ, fvecs, and ivecs. Use `mojovec[io]` for NumPy-backed formats,
`mojovec[arrow]` for Parquet/Arrow, and `mojovec[huggingface]` for datasets.
Imports are atomic per batch, not across the complete file.

### Python inspection and compaction

```python
print(collection.name())
print(collection.dimension())
print(collection.metric())
print(collection.storage_kind())  # "flat" or "sq8"
print(collection.is_quantized())
print(collection.count())
print(collection.count_deleted())
print(collection.stats())

collection.set_ef_search(128)
report = collection.compact_if_needed(deleted_ratio=0.25)
# Use collection.compact() to rebuild immediately when garbage exists.
```

`stats()` includes `active_count`, `deleted_count`, `total_count`,
`deleted_ratio`, `dimension`, `quantized`, `M`, `ef_construction`, and
`ef_search`.

### Python persistence, mmap, snapshots, and WAL

```python
collection.save("articles.mojovec")

loaded = mojovec.load(
    "articles.mojovec",
    memory_mapped=True,
    mmap_threshold_bytes=64 * 1024 * 1024,
)

# Force mmap for a small test file.
forced = mojovec.Collection.load(
    "articles.mojovec",
    mmap_threshold_bytes=0,
)

reader = collection.snapshot(
    "reader.mojovec",
    mmap_threshold_bytes=0,
)
```

Snapshots are checksummed and atomically published. A mapped collection owns
its mapping; never manage its file handle manually. Read-only queries and
`set_ef_search()` keep mapping active. The first operation that grows or
rebuilds storage transparently materializes writable arrays.

WAL workflow:

```python
collection.save("base.mojovec")
collection.enable_wal("changes.wal", durability=mojovec.WAL_ASYNC)

collection.upsert([1], [[1.0, 0.0, 0.0]])
collection.flush_wal()

recovered = mojovec.recover(
    "base.mojovec",
    "changes.wal",
    durability=mojovec.WAL_ASYNC,
)
recovered.checkpoint("base.mojovec")
```

- `WAL_ASYNC` is the throughput-oriented default. Call `flush_wal()` when the
  application needs an explicit durability boundary.
- `WAL_SYNC` performs a durability sync for every complete public mutation
  batch and lowers ingestion throughput.
- Recovery starts from a complete snapshot and replays committed WAL frames.
- `checkpoint()` publishes the applied state and rotates covered WAL records.
- Do not attach a non-empty WAL from another collection. Snapshot/WAL identity
  validation intentionally rejects it.

## Mojo API

### Installation and import

Place the released `mojovec.mojoc` package in the project or provide its parent
directory through Mojo's include path.

```mojo
from mojovec import Client, Collection, Metadata, QueryResults, Where
from std.collections import List
```

### Complete Mojo example

```mojo
from mojovec import Client, Metadata, QueryResults, Where
from std.collections import List


def main() raises:
    var client = Client()
    var collection = client.create_collection(
        "articles",
        dimension=3,
        M=32,
        ef_construction=96,
        ef_search=96,
        quantized=True,
        metric="cosine",
    )

    var metadata = Metadata()
    metadata.set("kind", "guide")
    metadata.set("year", 2026)
    metadata.set("published", True)

    var metadatas = List[Metadata]()
    metadatas.append(metadata.copy())

    # Mojo embeddings are flattened row-major Float32 values.
    var ids = [101]
    var embeddings = [Float32(1.0), Float32(0.0), Float32(0.0)]
    var documents = [String("MojoVec vector search guide")]
    collection.add(ids, embeddings, metadatas, documents)

    var conditions = List[Where]()
    conditions.append(Where.eq("kind", "guide"))
    conditions.append(Where.gte("year", 2024))

    var results = collection.query(
        embeddings,
        where=Where.and_(conditions),
        n_results=5,
    )

    for query_index in range(len(results.ids)):
        for rank in range(len(results.ids[query_index])):
            var record_id = results.ids[query_index][rank]
            if record_id >= 0:
                print(
                    record_id,
                    results.distances[query_index][rank],
                    results.documents[query_index][rank],
                )
```

Mojo accepts a flattened `List[Float32]`; its length must be
`number_of_rows * dimension`. High-level collection methods own and manage all
result memory.

### Mojo collection creation

```mojo
var client = Client()
var collection = client.create_collection(
    name,
    dimension,
    M=32,
    ef_construction=40,
    ef_search=16,
    quantized=True,
    metric="l2",
)
```

### Mojo payloads and mutations

`Metadata` supports `String`, `Int`, `Float64`, and `Bool`:

```mojo
var metadata = Metadata()
metadata.set("title", "Mojo vector search")
metadata.set("year", 2026)
metadata.set("score", Float64(0.97))
metadata.set("published", True)

var title = metadata.get_string("title")
var year = metadata.get_int("year")
var score = metadata.get_float("score")
var published = metadata.get_bool("published")
```

Batch payloads use one `Metadata` and one `String` per ID:

```mojo
collection.add(ids, embeddings)
collection.add(ids, embeddings, metadatas)
collection.add(ids, embeddings, documents)
collection.add(ids, embeddings, metadatas, documents)

collection.upsert(ids, embeddings, metadatas, documents)
collection.update(ids, embeddings, metadatas, documents)
collection.delete(ids)
```

Copy a metadata value when the original must remain usable:

```mojo
metadatas.append(metadata.copy())
```

The mutation semantics are the same as Python: `add` is insert-only,
`upsert` inserts or replaces, `update` requires existing IDs, and `delete` is
idempotent.

Numeric vector formats can be read directly by Mojo:

```mojo
from mojovec import read_fvecs, read_npy_float32

var dataset = read_fvecs("base.fvecs", id_start=0)
var collection = Collection(dataset.dimension, quantized=True)
collection.add(dataset.ids, dataset.embeddings)
```

Available direct readers are `read_fvecs`, `read_ivecs`,
`read_npy_float32`, `read_csv`, and `read_tsv`. CSV/TSV must
be numeric matrices, and NPY must be a C-contiguous little-endian float32
matrix.

For ecosystem formats, Mojo imports the lightweight Python adapter:

```mojo
from mojovec import DatasetImportOptions, read_parquet

var options = DatasetImportOptions()
options.id_column = "id"
options.document_column = "text"
options.metadata_columns.append("category")
var reader = read_parquet("records.parquet", dimension, options)
var count = reader.add_to(collection)
```

`read_json`, `read_jsonl`, `read_parquet`, `read_arrow`, `read_npz`, and
`read_huggingface` return one-shot `PythonDatasetReader` values. `add_to` is
insert-only and `upsert_to` inserts or replaces. The decoder dependencies must
be installed into Mojo's embedded Python, and source checkouts must include
the repository's `python/` directory in `PYTHONPATH`.

### Mojo typed filters

```mojo
var conditions = List[Where]()
conditions.append(Where.eq("published", True))
conditions.append(Where.gte("year", 2024))
var where = Where.and_(conditions)

var result = collection.query(
    query_embeddings,
    where=where,
    n_results=10,
)
```

Available constructors:

```text
Where.eq / ne       String, Int, Float64, Bool
Where.gt / gte      Int, Float64
Where.lt / lte      Int, Float64
Where.in_ / not_in  non-empty typed List[String/Int/Float64/Bool]
Where.and_ / or_    non-empty List[Where]
Where.not_          one Where expression
```

### Mojo text and hybrid search

```mojo
var text = collection.query(
    [String("vector search")],
    n_results=10,
)

var hybrid = collection.query_hybrid(
    query_embeddings,
    [String("vector search")],
    n_results=10,
    rrf_k=60,
    candidate_multiplier=4,
)
```

BM25 performs Unicode-aware lowercase normalization, splits at whitespace,
punctuation, symbols, and emoji, and includes English and Russian stopwords.
ASCII apostrophes and underscores remain inside terms. Stemming is not
performed.

### Mojo persistence and WAL

```mojo
from mojovec import Collection, WAL_ASYNC

collection.save("articles.mojovec")
var loaded = Collection.load("articles.mojovec")
var forced_mmap = Collection.load(
    "articles.mojovec",
    mmap_threshold_bytes=0,
)

collection.enable_wal("articles.wal", durability=WAL_ASYNC)
collection.flush_wal()
var recovered = Collection.recover(
    "articles.mojovec",
    "articles.wal",
    durability=WAL_ASYNC,
)
recovered.checkpoint("articles.mojovec")
```

Use `stats()`, `compact()`, `compact_if_needed()`, `snapshot()`,
`is_memory_mapped()`, `wal_enabled()`, and `wal_sequence()` exactly as in the
Python API. Mojo returns typed `CollectionStats`, `CompactReport`, and
`QueryResults` values rather than dictionaries.

## BM25 and hybrid-search rules

- Documents are indexed automatically when they are supplied to a mutation.
- Replacing or deleting a document deactivates its previous BM25 postings.
- A stopword-only query returns a normal padded no-match result.
- BM25 scores must not be compared numerically with vector distances.
- Hybrid search uses equal-weight reciprocal rank fusion:

```text
RRF score = sum(1 / (rrf_k + one_based_rank))
```

- Apply the same metadata `where` filter to both hybrid retrieval paths.
- If an application needs weighted fusion, score calibration, stemming, or a
  custom analyzer, that is additional functionality and must not be invented
  as though it already exists.

## Concurrency contract

Supported:

- Concurrent vector, filtered, BM25, and hybrid queries on the same unchanged
  collection.
- One writer publishing independent point-in-time readers with `snapshot()`.
- Python native read-only search releases the GIL during the search itself.

Not supported without external synchronization:

- Querying while mutating the same collection object.
- Two simultaneous mutations on the same object.
- Mutating while saving, checkpointing, compacting, or changing configuration.

For a one-writer/many-reader service, publish a snapshot and swap the reader
reference at the application level. Old reader objects continue seeing their
complete point-in-time state.

## Persistence safety and operational assumptions

- `save()` writes a checksummed temporary file, synchronizes it, and publishes
  it with atomic rename.
- `load()` validates the checksummed payload, counts, byte arithmetic, deletion
  flags, active ID uniqueness, and internal headers before exposing data.
- Snapshots are limited to 10 million records and 256 GiB.
- Mapping is read-only and owned by the collection.
- Snapshot files are an internal binary format. Do not parse or edit them in
  application code, and do not promise compatibility across arbitrary library
  revisions unless the project explicitly establishes that contract.
- Failed save and WAL-rotation paths perform best-effort temporary-file cleanup.

## IVF-PQ is a separate managed collection

The fully managed HNSW `Collection` supports metadata, documents, filters,
BM25, hybrid search, mutations, compaction, mmap, and WAL. `CollectionIVFPQ`
is a separate Mojo and Python API intended for extreme compression. It is not
the default and deliberately has a smaller record-management surface.

Use it only when explicitly requested:

```mojo
var client = Client()
var collection = client.create_ivfpq_collection(
    "compressed",
    dimension=128,
    nlist=256,
    M=16,
    nprobe=16,
    metric="cosine",
)

collection.train(representative_training_embeddings)
collection.add(ids, embeddings)
var result = collection.query(queries, n_results=10)
```

The equivalent Python API is:

```python
collection = mojovec.IVFPQCollection(
    dimension=128,
    nlist=256,
    pq_subvectors=16,
    nprobe=16,
    metric="cosine",
    name="compressed",
)
collection.train(training_embeddings)
collection.add(ids, embeddings)
result = collection.query(query_embeddings, n_results=10)
```

Here IVF-PQ `M` means the number of PQ subvectors, not HNSW graph connectivity,
and `dimension` must be divisible by it. Training requires at least
`max(nlist, 256)` representative vectors. The first `add()` can auto-train,
but explicit training is preferable. Tune `nprobe` between 1 and `nlist` and
always evaluate recall on the real dataset. L2, cosine, inner product,
statistics, unique-ID insertion, and owned save/load are supported. Do not
assume the HNSW collection's metadata, BM25, filters, update/upsert/delete,
compaction, mmap, WAL, or exact reranking exists on `CollectionIVFPQ`.

## Common agent mistakes

Avoid generating code that:

- calls `upsert` when insert-only `add` was requested;
- passes one vector with the wrong dimension;
- mixes nested and flat embedding representations in one Python batch;
- treats cosine/IP output as larger-is-better similarity;
- treats BM25/RRF scores as smaller-is-better distances;
- assumes exact candidate reranking makes the approximate HNSW search globally
  exact;
- manually normalizes cosine vectors and accidentally changes zero-vector
  validation or caller data;
- supplies nested metadata values;
- reads payload rows without checking whether the outer list is empty;
- forgets to skip padded ID `-1` results;
- mutates a collection concurrently with queries;
- enables a WAL before creating a base snapshot for recovery;
- uses `WAL_ASYNC` but never defines a `flush_wal()` or checkpoint policy;
- assumes `delete()` immediately shrinks the graph or snapshot;
- imports internal Python native modules or exposes manual memory management;
- presents IVF-PQ as API-equivalent to the HNSW collection.

## Implementation checklist for an AI agent

Before returning MojoVec application code, verify:

1. The selected language is Python or Mojo as requested.
2. The embedding dimension is known and consistent everywhere.
3. Storage (`quantized`) and metric are chosen independently and explained if
   the user asked for an architectural decision.
4. IDs are stable application IDs and unique inside each batch.
5. Mutation semantics match the intent: `add`, `upsert`, or `update`.
6. Metadata contains only supported scalar types.
7. Vector, BM25, and hybrid result ordering is handled correctly.
8. Filter syntax matches the language: mappings in Python, typed `Where` in
   Mojo.
9. Production code has an explicit save/snapshot/WAL durability strategy when
   persistence is required.
10. Concurrent mutation is synchronized or isolated behind snapshots.
11. No internal binary import, pointer allocation, or manual release appears
    in high-level code.
12. Large external imports use the built-in batched readers and acknowledge
    that atomicity is per batch.
13. Tests cover shape validation, duplicate/existing ID behavior, result IDs,
    filtering, persistence round-trip, and any required recall target.

## Repository references

When the full repository is available, use these maintained examples rather
than inventing API spellings:

- [`README.md`](README.md) — public overview and performance results.
- [`examples/python/`](examples/python/) — complete Python programs.
- [`examples/mojo/`](examples/mojo/) — complete Mojo programs.
- [`python/mojovec/_collection.py`](python/mojovec/_collection.py) — Python
  signatures and docstrings.
- [`mojovec/api/collection.mojo`](mojovec/api/collection.mojo) — managed Mojo
  collection surface.
- [`tests/`](tests/) and [`python/tests/`](python/tests/) — executable API
  contracts.

Useful validation commands from a source checkout:

```bash
bash tests/run_all.sh
python -m pytest python/tests/
mojo run -I . examples/mojo/api_01_hnsw_fast_search.mojo
python examples/python/api_01_collection_crud.py
```
