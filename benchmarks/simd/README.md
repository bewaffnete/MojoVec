# FP32 SIMD tuning on Mojo 1.2.0

This is a single-thread, cache-resident distance-kernel microbenchmark, **not**
an end-to-end HNSW QPS benchmark. It does not load or rebuild SIFT indexes.

## Selected policy

| Target | Dimensions below 128 | Dimensions >= 128 | Single-distance unroll |
|---|---:|---:|---:|
| ARM / Apple M4 | 8 FP32 lanes | 8 FP32 lanes | 4 |
| x86-64-v3 / AVX2 | 8 FP32 lanes | 8 FP32 lanes | 4 |
| x86-64-v4 / AVX-512 | 8 FP32 lanes | 16 FP32 lanes | 4 |

The policy is shared by `IndexFlat` (including HNSW's Flat distance computer)
and HNSW exact reranking. Reranking previously forced width 64 on every target.
Batch4 uses the selected width with four independent result accumulators;
its temporal loop is not additionally unrolled.

PQ/K-Means short-vector call sites retain their existing widths and specialized
`short4` kernels. No blanket widening is applied to small subvectors.

The old `d < simd_width` scalar guard remains enabled by default. Bypassing it
passes the new Linux correctness matrix on both wheel targets, but is slower
on some tested dimensions. `short_scalar=False` is an experimental benchmark
option, not the production policy.

## Experiment

Measured on 2026-11-15 with Mojo 1.2.0 (2016d0cd):

- Native Apple M4, 10 logical cores, 16 GiB RAM.
- GitHub Codespaces x86-64 VM, Intel Xeon Platinum 8370C, 2 virtual CPUs.
  AVX2 and the full AVX-512 x86-64-v4 flag set were available. Both x86 targets
  ran on that same VM, sequentially, pinned to virtual CPU 0. No ISA emulation.

The sweep covers widths 4/8/16/32/64 and single-distance unroll 1/2/4/8, L2/IP,
batch4, short4 and the guard-bypass experiment. Dimensions are
1/2/3/4/8/16/32/128/384/768/1536. Each cell has five samples, calibrated to at
least 5 ms. Candidate order reverses on alternate samples. Every computed
result contributes to a printed checksum; each call visits a changing row.

The working set is 256 database rows plus three extra rows for batch4, and one
query vector. Results include a non-inlined kernel-call boundary. Data
generation, compilation and correctness tests are outside timed regions.

Raw final CSVs are in `results/`. Compare per-cell medians and geometric means
of speedup ratios, not absolute ARM-vs-cloud timings. Cloud scheduling, CPU
frequency, cache footprint and the fanless Mac can affect the numbers. Small
single-digit improvements were not used to justify changing default policies.
The data do not establish an end-to-end search speedup or the isolated effect
of upgrading from an older compiler.

Final selected wrappers versus width8 baseline (geometric mean speedup across
dimensions 128/384/768/1536, using each cell's median of five samples):

| Target | L2 | IP | L2 batch4 | IP batch4 |
|---|---:|---:|---:|---:|
| Apple M4 | 0.984x | 1.039x | 1.024x | 0.981x |
| Xeon AVX2 | 1.007x | 1.024x | 1.005x | 0.994x |
| Xeon AVX-512 | 1.223x | 1.229x | 1.482x | 1.386x |

ARM/AVX2 wrappers retain the same width8 kernels as the baseline; their
small fluctuations should be treated as measurement noise, not gains.
The original width64 rerank call sites are a separate baseline: width8 was
faster on ARM/AVX2 in the sweep, while AVX-512 retains wide hardware lanes
through the selected width16 policy.

## Reproduce without changing source

ARM (host target):

```bash
pixi run mojo build -I . tests/test_simd_tuning.mojo -o /tmp/mojovec-simd-test
/tmp/mojovec-simd-test
pixi run mojo build -I . benchmarks/simd/bench_distances.mojo -o /tmp/mojovec-simd-bench
/tmp/mojovec-simd-bench > /tmp/simd-arm.csv
python3 benchmarks/simd/analyze.py /tmp/simd-arm.csv
```

Native Linux x86, repeated separately for `x86-64-v3` and `x86-64-v4`:

```bash
pixi run mojo build -I . --target-cpu x86-64-v3 \
  tests/test_simd_tuning.mojo -o /tmp/mojovec-simd-test
taskset -c 0 /tmp/mojovec-simd-test
pixi run mojo build -I . --target-cpu x86-64-v3 \
  benchmarks/simd/bench_distances.mojo -o /tmp/mojovec-simd-bench
taskset -c 0 /tmp/mojovec-simd-bench > /tmp/simd-avx2.csv
python3 benchmarks/simd/analyze.py /tmp/simd-avx2.csv
```

Never execute an x86-64-v4 binary unless the CPU supports
AVX512F/DQ/CD/BW/VL and the x86-64-v3 prerequisites. CI explicitly checks the
AVX-512 flags before running that target; cross-compilation alone is not a
runtime correctness or performance result.

CSV modes: 0=L2, 1=IP, 2=L2 batch4, 3=IP batch4, 4=L2 short4, 5=IP short4,
6=tuned L2, 7=tuned IP, 8=tuned L2 batch4, 9=tuned IP batch4. Tuned modes use
width/unroll columns `0,0` because their policy comes from the compilation
target and runtime dimension. Reported times are nanoseconds per distance,
including division by four for batch4.

`test_simd_tuning.mojo` checks scalar references across widths/unroll factors,
zero lengths, SIMD boundaries and tails, unaligned addresses, and the
127/128/129 policy boundary. It also explicitly tests the old guard bypass.

## Validation results

- Full Mojo suite: 27 test files passed on both native macOS ARM and Linux x86.
- Tuning matrix: passed separately with explicit `x86-64-v3` and `x86-64-v4`.
- Python API on macOS ARM: 171 passed, 1 skipped.
- Precompiled package build passed. AVX-512 assembly contains ZMM operations.
- CI runs the tuning matrix on both x86 wheel targets when supported, in
  addition to the ordinary host-target test suite.
