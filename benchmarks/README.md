# Benchmarks

Run commands from the repository root so dataset and package paths resolve
consistently.

## Layout

- `suite/` — comparable SIFT1M HNSW benchmarks for MojoVec, FAISS, and Chroma.
- `tools/` — dataset generation utilities.
- `data/` — local benchmark datasets and generated binary data.
- `cold_bench/` — saved indexes and reproducible cold search-only programs.
- `scratch/` — experimental benchmarks that are not part of the supported suite.

`data/sift1m/`, `cold_bench/`, and `scratch/` are intentionally excluded from
Git because they contain local datasets, multi-gigabyte indexes, or exploratory
programs.

## SIFT1M suite

All HNSW suite programs use one million 128-dimensional base vectors, 10,000
queries, L2 distance, `M=32`, `efConstruction=200`, `efSearch=40`, and `k=10`.

```bash
# MojoVec
mojo run -I . benchmarks/suite/mojovec_flat.mojo
mojo run -I . benchmarks/suite/mojovec_sq8.mojo

# FAISS through its Python wrapper
.venv/bin/python benchmarks/suite/faiss_flat.py
.venv/bin/python benchmarks/suite/faiss_sq8.py

# Chroma
.venv/bin/python benchmarks/suite/chroma.py
```

See `cold_bench/README.md` for the thermal-control protocol and search-only
commands.
