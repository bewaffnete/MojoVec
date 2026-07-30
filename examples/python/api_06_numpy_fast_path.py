"""Zero-copy NumPy ingestion and query buffers.

NumPy is an optional dependency. Install it before running this example:

    python -m pip install numpy
"""

from __future__ import annotations

import mojovec

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "This optional example requires NumPy: python -m pip install numpy"
    ) from error


def main() -> None:
    dimension = 4
    collection = mojovec.Collection(
        dimension=dimension,
        quantized=False,
        name="numpy_fast_path",
    )

    # The native fast path requires contiguous int64 IDs and float32 vectors.
    # A two-dimensional (count, dimension) embedding matrix is recommended.
    ids = np.arange(1_000, 1_006, dtype=np.int64)
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
        order="C",
    )
    collection.upsert_numpy(ids, embeddings)

    queries = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
        order="C",
    )
    result = collection.query_numpy(queries, n_results=3)
    print("IDs dtype/shape:", result["ids"].dtype, result["ids"].shape)
    print("distances dtype/shape:", result["distances"].dtype, result["distances"].shape)
    print("IDs:\n", result["ids"])
    print("distances:\n", result["distances"])

    # query_numpy() is the raw vector-only fast path, so payload lists are
    # empty. Use managed query() when aligned metadata/documents are required.
    print("metadata payload:", result["metadatas"])

    # Payload ingestion is still supported through upsert_numpy(), but it
    # intentionally falls back to managed conversion because Python objects
    # cannot be represented by a raw Float32 span.
    collection.upsert_numpy(
        np.asarray([2_000], dtype=np.int64),
        np.asarray([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32),
        metadatas=[{"source": "numpy"}],
        documents=["NumPy record with managed payload"],
    )
    managed = collection.query(
        [[0.5, 0.5, 0.0, 0.0]],
        n_results=1,
    )
    print("managed payload result:", managed)


if __name__ == "__main__":
    main()
