# MojoVec Examples

This directory contains examples demonstrating how to use the different modules and APIs in MojoVec.

## 1. Fast Approximate Search (HNSW)
**File**: `01_hnsw_fast_search.mojo`

Demonstrates how to build a Hierarchical Navigable Small World (HNSW) graph for extremely fast logarithmic search times. This is the go-to index for most production vector databases.

## 2. Massive Dataset Compression (IVF-PQ)
**File**: `02_ivf_pq_compression.mojo`

Demonstrates how to combine Inverted Files (IVF) and Product Quantization (PQ) to slash memory requirements by up to 32x while maintaining fast search speeds through Asymmetric Distance Computation (ADC). Requires training via K-Means before inserting data.

## 3. Serialization and I/O
**File**: `03_serialization.mojo`

Demonstrates how to safely write an entire built index (including its graph, centroids, and vectors) to a binary file on disk, and load it back into memory instantly.

## How to Run
Run any of the examples directly using the Mojo compiler from the project root:

```bash
mojo run -I . examples/01_hnsw_fast_search.mojo
mojo run -I . examples/02_ivf_pq_compression.mojo
mojo run -I . examples/03_serialization.mojo
```
