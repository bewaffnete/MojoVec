import pytest

import mojovec


np = pytest.importorskip("numpy")


def test_add_numpy_preserves_insert_only_semantics():
    collection = mojovec.Collection(2, quantized=False)
    ids = np.asarray([10, 20], dtype=np.int64)
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )

    collection.add_numpy(ids, embeddings)
    assert collection.query([[1.0, 0.0]], 1)["ids"] == [[10]]
    with pytest.raises(Exception, match="already exists"):
        collection.add_numpy(ids[:1], embeddings[:1])


def test_numpy_fast_path_accepts_flat_and_matrix_buffers():
    collection = mojovec.Collection(3, quantized=False)
    ids = np.asarray([10, 20], dtype=np.int64)
    embeddings = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
        order="C",
    )
    collection.upsert_numpy(ids, embeddings.reshape(-1))

    flat_query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    result = collection.query_numpy(flat_query, n_results=2)

    assert result["ids"].shape == (1, 2)
    assert result["distances"].shape == (1, 2)
    assert result["ids"].dtype == np.int64
    assert result["distances"].dtype == np.float32
    assert result["ids"][0].tolist() == [10, 20]
    assert result["metadatas"] == []
    assert result["documents"] == []
    assert result["scores"] == []


@pytest.mark.parametrize(
    "ids",
    [
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([[1, 2]], dtype=np.int64),
        np.arange(4, dtype=np.int64)[::2],
    ],
)
def test_upsert_numpy_rejects_invalid_id_buffers(ids):
    collection = mojovec.Collection(2)
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    with pytest.raises(
        TypeError, match="contiguous one-dimensional numpy.int64"
    ):
        collection.upsert_numpy(ids, embeddings)

    assert collection.count() == 0


def test_upsert_numpy_rejects_wrong_embedding_dtype():
    collection = mojovec.Collection(2)
    ids = np.asarray([1, 2], dtype=np.int64)
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    with pytest.raises(TypeError, match="contiguous numpy.float32"):
        collection.upsert_numpy(ids, embeddings)


def test_upsert_numpy_rejects_non_contiguous_embeddings():
    collection = mojovec.Collection(2)
    ids = np.asarray([1, 2], dtype=np.int64)
    base = np.asarray(
        [[1.0, 9.0, 0.0, 9.0], [0.0, 9.0, 1.0, 9.0]],
        dtype=np.float32,
    )
    embeddings = base[:, ::2]
    assert not embeddings.flags.c_contiguous

    with pytest.raises(TypeError, match="contiguous numpy.float32"):
        collection.upsert_numpy(ids, embeddings)


@pytest.mark.parametrize(
    "embeddings",
    [
        np.zeros((1, 2, 2), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    ],
)
def test_upsert_numpy_rejects_invalid_embedding_shape_or_size(embeddings):
    collection = mojovec.Collection(2)
    ids = np.asarray([1, 2], dtype=np.int64)

    with pytest.raises((TypeError, ValueError)):
        collection.upsert_numpy(ids, embeddings)

    assert collection.count() == 0


@pytest.mark.parametrize(
    ("query", "error_type"),
    [
        (np.asarray([1.0, 0.0], dtype=np.float64), TypeError),
        (np.asarray([1.0, 0.0, 0.0], dtype=np.float32), ValueError),
        (np.zeros((2, 3), dtype=np.float32), ValueError),
        (np.zeros((1, 1, 2), dtype=np.float32), ValueError),
    ],
)
def test_query_numpy_rejects_invalid_dtype_and_shape(query, error_type):
    collection = mojovec.Collection(2)
    with pytest.raises(error_type):
        collection.query_numpy(query)


def test_query_numpy_rejects_non_contiguous_matrix():
    collection = mojovec.Collection(2)
    base = np.asarray(
        [[1.0, 9.0, 0.0, 9.0], [0.0, 9.0, 1.0, 9.0]],
        dtype=np.float32,
    )
    queries = base[:, ::2]
    assert queries.shape == (2, 2)
    assert not queries.flags.c_contiguous

    with pytest.raises(ValueError, match="contiguous"):
        collection.query_numpy(queries)


def test_upsert_numpy_with_payloads_uses_managed_fallback():
    collection = mojovec.Collection(2, quantized=False)
    ids = np.asarray([1], dtype=np.int64)
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)

    collection.upsert_numpy(
        ids,
        embeddings,
        metadatas=[{"source": "numpy"}],
        documents=["managed payload"],
    )

    result = collection.query([[1.0, 0.0]], 1)
    assert result["ids"][0][0] == 1
    assert result["metadatas"][0][0] == {"source": "numpy"}
    assert result["documents"][0][0] == "managed payload"


def test_numpy_compatibility_aliases_execute_the_same_fast_path():
    collection = mojovec.Collection(2, quantized=False)
    ids = np.asarray([1], dtype=np.int64)
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)

    collection.upsert_batch_numpy(ids, embeddings)
    result = collection.query_batch_numpy(embeddings, n_results=1)

    assert result["ids"][0, 0] == 1
