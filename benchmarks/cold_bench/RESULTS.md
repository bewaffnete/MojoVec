# Recorded results

Hardware: Apple Silicon M4, fanless Mac.

## Public MojoVec API: tuned SIMD, 2026-11-15

Recorded with `Collection.load()`, `set_ef_search()`, and `query()`. These
measurements include creation and destruction of managed `QueryResults`.

Protocol: 10 samples, five 10,000-query searches per sample, 10-second cooldown
and an untimed 256-query warm-up before each sample. Loading and recall
calculation are not timed. Both executables were rebuilt with Mojo 1.2.0
(2016d0cd) from source revision `d662d77`, after FP32 SIMD tuning. Existing
indexes were loaded without rebuilding their graphs.

| Index | QPS | Recall@10 |
|---|---:|---:|
| MojoVec Flat HNSW | 25,501 | 99.147% |
| MojoVec SQ8 HNSW | 42,212 | 99.163% |

QPS is the arithmetic mean of the ten sample QPS values. Values are rounded to
whole QPS in the table; sample values below are recorded to three decimal
places. The full-precision logs are saved in [results/](results/).
The executables also print a time-weighted `aggregate` (total queries / total
measured time): 25,457.541 QPS for Flat and 42,026.430 QPS for SQ8. This is a
different statistic and is not the mean reported in the README QPS column.

MojoVec Flat public API samples:

```text
25851.497
26524.172
26294.187
25195.339
26404.090
23378.218
25620.910
23804.196
26069.542
25867.104
```

MojoVec SQ8 public API samples:

```text
43624.612
43927.814
38605.060
43119.730
45247.517
40137.269
41757.629
36675.444
44933.238
44093.383
```

## Reference search records

The FAISS figures below remain the reference baseline used in the main README;
FAISS was not rerun during the MojoVec SIMD update. The MojoVec rows in this
table are the earlier low-level search records, not the refreshed public API
measurements above. Index sizes refer to those reference artifacts, not the
currently saved collection files. Comparisons use separately recorded runs,
not a fresh simultaneous head-to-head measurement.

| Index | Aggregate QPS | Recall@10 | Saved size |
|---|---:|---:|---:|
| MojoVec Flat HNSW | 23,783 | 99.160% | 873,322,008 bytes |
| MojoVec SQ8 HNSW | 36,353 | 99.144% | 1,005,322,015 bytes |
| FAISS Flat HNSW | 17,330 | 99.201% | 784,129,506 bytes |
| FAISS SQ8 HNSW | 14,712 | 99.185% | 400,130,566 bytes |

MojoVec Flat samples:

```text
22639.001
23506.724
23634.069
23595.769
23459.893
23722.676
24297.853
24514.995
24513.685
24074.318
```

MojoVec SQ8 samples:

```text
31651.920
34688.088
36361.732
36204.914
38187.470
36915.953
38371.484
37840.403
37387.203
37000.694
```

FAISS Flat samples:

```text
17750.652
18067.326
17597.385
17387.603
17695.148
17036.004
16328.997
17319.297
17192.731
17051.561
```

FAISS SQ8 samples:

```text
16384.040
15466.050
14706.096
15085.918
15147.361
14997.850
14912.795
14852.888
11708.875
14870.599
```
