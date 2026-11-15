# Recorded results

Hardware: Apple Silicon M4, fanless Mac.

## Public MojoVec API

Recorded with `Collection.load()`, `set_ef_search()`, and `query()`. These
measurements include creation and destruction of managed `QueryResults`.

| Index | Aggregate QPS | Recall@10 |
|---|---:|---:|
| MojoVec Flat HNSW | 25,588 | 99.147% |
| MojoVec SQ8 HNSW | 41,808 | 99.163% |

MojoVec Flat public API samples:

```text
26085.414
25800.739
25062.556
25689.335
25321.648
27129.209
26418.209
25230.595
24406.221
24952.441
```

MojoVec SQ8 public API samples:

```text
45900.321
42168.898
40900.565
42391.742
41163.579
42107.249
39729.616
40297.557
41630.795
42375.970
```

## Zero-copy search kernel

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
