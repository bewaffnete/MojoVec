# MojoVec examples

These examples are executable tutorials for the current high-level MojoVec API.
They use deterministic, locally generated vectors and require no downloaded
datasets.

## Before you start

Run every command from the repository root so `-I .` can resolve the local
`mojovec` package:

```bash
mojo run -I . examples/api_01_hnsw_fast_search.mojo
mojo run -I . examples/api_02_ivfpq_compression.mojo
mojo run -I . examples/api_03_serialization.mojo
mojo run -I . examples/api_04_compaction.mojo
mojo run -I . examples/api_05_metadata.mojo
mojo run -I . examples/api_06_where_filters.mojo
```

To compile without executing:

```bash
mojo build -I . examples/api_01_hnsw_fast_search.mojo
mojo build -I . examples/api_02_ivfpq_compression.mojo
mojo build -I . examples/api_03_serialization.mojo
mojo build -I . examples/api_04_compaction.mojo
mojo build -I . examples/api_05_metadata.mojo
mojo build -I . examples/api_06_where_filters.mojo
```

## Which collection should I use?

| Collection | Creation | Vector representation | Strength | Main trade-off |
| --- | --- | --- | --- | --- |
| HNSW + SQ8 | `create_collection(..., quantized=True)` | 8-bit scalar-quantized search storage | High QPS and lower search bandwidth | Approximate distances |
| HNSW + Flat | `create_collection(..., quantized=False)` | Original `Float32` vectors | Exact vector distances inside HNSW traversal | More memory bandwidth |
| IVF-PQ | `create_ivfpq_collection(...)` | Compact product-quantized codes | Strong compression for large datasets | Requires training and recall tuning |

All three indexes perform approximate nearest-neighbor search at the index
level. “Flat” here means that HNSW evaluates candidates against unquantized
`Float32` vectors; it does not mean an exhaustive scan of every vector.

## Example 1: complete HNSW lifecycle

File: [`api_01_hnsw_fast_search.mojo`](api_01_hnsw_fast_search.mojo)

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

File: [`api_02_ivfpq_compression.mojo`](api_02_ivfpq_compression.mojo)

This example explains:

- what IVF and PQ contribute;
- the meaning of `dimension`, `nlist`, and PQ `M`;
- the requirement that `dimension` be divisible by PQ `M`;
- explicit training on representative data;
- automatic training during the first `add`;
- why approximate distances should not be compared for exact equality.

For production data, train on a representative sample. Training on a tiny or
biased first ingestion batch usually reduces recall.

## Example 3: serialization and recovery

File: [`api_03_serialization.mojo`](api_03_serialization.mojo)

This example round-trips both SQ8 and Flat HNSW collections and verifies the
collection state visible after loading:

- collection name;
- dimension and storage kind;
- user-provided IDs;
- active and soft-deleted state;
- vectors and HNSW graph.

The example writes:

```text
/tmp/mojovec_example_sq8.bin
/tmp/mojovec_example_flat.bin
```

`Collection.load(path)` detects Flat versus SQ8 from the file header. Do not
create a new collection or pass `quantized` when loading.

## Example 4: statistics and compaction

File: [`api_04_compaction.mojo`](api_04_compaction.mojo)

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

File: [`api_05_metadata.mojo`](api_05_metadata.mojo)

This example covers:

- `Metadata` values of type `String`, `Int`, `Float64`, and `Bool`;
- adding one metadata object per vector;
- retrieving an owned metadata copy by application ID;
- metadata inheritance during vector-only `update` and `upsert`;
- complete replacement when metadata is supplied explicitly;
- reading metadata for IDs returned by `query`;
- preservation through save/load and compaction.

Metadata is stored as a snapshot alongside every internal vector version.
Calling `get_metadata(id)` for a missing or deleted ID raises an error. Metadata
is not embedded into every `QueryResults` object; fetch it only for returned
IDs that the application needs.

## Example 6: typed `where` filters and bitmap indexes

File: [`api_06_where_filters.mojo`](api_06_where_filters.mojo)

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
```

MojoVec owns all temporary search buffers and transfers the result Lists to the
caller. User code does not call `alloc`, work with pointers, or call `free`.
Dropping `QueryResults` releases its storage automatically.

## Distance interpretation

Current high-level examples use squared L2 distance:

- smaller is closer;
- `0.0` means identical vectors;
- the value is squared distance, so MojoVec does not apply a final square root;
- SQ8 and IVF-PQ distances are approximate;
- validate quality with recall@k, not distance equality alone.

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
