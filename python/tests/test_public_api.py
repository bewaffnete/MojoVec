import inspect
import os
import subprocess
import sys

import mojovec


PUBLIC_COLLECTION_METHODS = (
    "load",
    "recover",
    "name",
    "dimension",
    "storage_kind",
    "metric",
    "is_quantized",
    "count",
    "count_deleted",
    "get_metadata",
    "get_document",
    "stats",
    "is_memory_mapped",
    "set_ef_search",
    "add",
    "upsert",
    "update",
    "add_with_metadata",
    "upsert_with_metadata",
    "update_with_metadata",
    "add_from",
    "add_numpy",
    "upsert_from",
    "add_huggingface",
    "upsert_huggingface",
    "delete",
    "query",
    "query_hybrid",
    "compact",
    "compact_if_needed",
    "save",
    "snapshot",
    "wal_enabled",
    "wal_sequence",
    "enable_wal",
    "disable_wal",
    "flush_wal",
    "checkpoint",
    "upsert_numpy",
    "query_numpy",
    "upsert_batch",
    "query_batch",
    "upsert_batch_numpy",
    "query_batch_numpy",
)


def test_root_module_is_a_small_explicit_facade():
    assert mojovec.__all__ == [
        "Client",
        "Collection",
        "IVFPQCollection",
        "IVFPQStats",
        "DEFAULT_MMAP_THRESHOLD_BYTES",
        "Metadata",
        "QueryResult",
        "WAL_ASYNC",
        "WAL_SYNC",
        "Where",
        "__version__",
        "io",
        "load",
        "native_backend",
        "read_fvecs",
        "read_ivecs",
        "recover",
    ]
    assert mojovec.Client.__module__ == "mojovec"
    assert mojovec.Collection.__module__ == "mojovec"
    assert mojovec.IVFPQCollection.__module__ == "mojovec"
    assert mojovec.load.__module__ == "mojovec"
    assert mojovec.recover.__module__ == "mojovec"
    assert not hasattr(mojovec, "ctypes")
    assert not hasattr(mojovec, "site")
    assert not hasattr(mojovec, "_compile_where")


def test_public_callables_have_runtime_docstrings():
    assert inspect.getdoc(mojovec.Client)
    for method_name in (
        "create_collection",
        "create_ivfpq_collection",
        "load",
        "recover",
    ):
        assert inspect.getdoc(getattr(mojovec.Client, method_name)), method_name
    assert inspect.getdoc(mojovec.Collection)
    assert inspect.getdoc(mojovec.load)
    assert inspect.getdoc(mojovec.recover)
    assert inspect.getdoc(mojovec.native_backend)
    for method_name in PUBLIC_COLLECTION_METHODS:
        assert inspect.getdoc(getattr(mojovec.Collection, method_name)), method_name
    assert inspect.getdoc(mojovec.IVFPQCollection)
    for method_name in (
        "load",
        "name",
        "dimension",
        "metric",
        "count",
        "is_trained",
        "nlist",
        "pq_subvectors",
        "nprobe",
        "set_nprobe",
        "stats",
        "train",
        "train_numpy",
        "add",
        "add_numpy",
        "query",
        "query_numpy",
        "save",
    ):
        assert inspect.getdoc(getattr(mojovec.IVFPQCollection, method_name))


def test_public_signatures_are_python_introspectable():
    constructor = inspect.signature(mojovec.Collection)
    assert list(constructor.parameters) == [
        "dimension",
        "M",
        "ef_construction",
        "ef_search",
        "quantized",
        "metric",
        "name",
    ]
    assert constructor.parameters["quantized"].default is True
    assert constructor.parameters["metric"].default == "l2"

    query = inspect.signature(mojovec.Collection.query)
    assert list(query.parameters) == [
        "self",
        "query_embeddings",
        "n_results",
        "query_texts",
        "where",
    ]
    assert query.parameters["query_texts"].kind is inspect.Parameter.KEYWORD_ONLY
    assert query.parameters["where"].kind is inspect.Parameter.KEYWORD_ONLY

    ivfpq_constructor = inspect.signature(mojovec.IVFPQCollection)
    assert list(ivfpq_constructor.parameters) == [
        "dimension",
        "nlist",
        "pq_subvectors",
        "nprobe",
        "metric",
        "name",
    ]
    assert ivfpq_constructor.parameters["nprobe"].default is None
    assert ivfpq_constructor.parameters["metric"].default == "l2"

    ivfpq_query = inspect.signature(mojovec.IVFPQCollection.query)
    assert list(ivfpq_query.parameters) == [
        "self",
        "query_embeddings",
        "n_results",
    ]

    client_create_collection = inspect.signature(mojovec.Client.create_collection)
    assert list(client_create_collection.parameters) == [
        "self",
        "name",
        "dimension",
        "M",
        "ef_construction",
        "ef_search",
        "quantized",
        "metric",
        "kwargs",
    ]
    assert client_create_collection.parameters["M"].default == 32
    assert client_create_collection.parameters["ef_construction"].default == 40
    assert client_create_collection.parameters["ef_search"].default == 16
    assert client_create_collection.parameters["quantized"].default is True
    assert client_create_collection.parameters["metric"].default == "l2"

    client_create_ivfpq = inspect.signature(mojovec.Client.create_ivfpq_collection)
    assert list(client_create_ivfpq.parameters) == [
        "self",
        "name",
        "dimension",
        "nlist",
        "M",
        "nprobe",
        "metric",
    ]
    assert client_create_ivfpq.parameters["nlist"].default == 100
    assert client_create_ivfpq.parameters["M"].default == 16
    assert client_create_ivfpq.parameters["nprobe"].default == 0
    assert client_create_ivfpq.parameters["metric"].default == "l2"

    client_load = inspect.signature(mojovec.Client.load)
    assert list(client_load.parameters) == ["self", "path"]

    client_recover = inspect.signature(mojovec.Client.recover)
    assert list(client_recover.parameters) == ["self", "path", "wal_directory"]


def test_direct_imports_and_reexports():
    from mojovec import Client, io, read_fvecs, read_ivecs

    assert Client is mojovec.Client
    assert io is mojovec.io
    assert read_fvecs is mojovec.read_fvecs is io.read_fvecs
    assert read_ivecs is mojovec.read_ivecs is io.read_ivecs


def test_client_create_collection(tmp_path):
    client = mojovec.Client()
    col = client.create_collection(
        "test_col",
        dimension=4,
        M=16,
        ef_construction=30,
        ef_search=10,
        quantized=False,
        metric="l2",
    )
    assert isinstance(col, mojovec.Collection)
    assert col.name() == "test_col"
    assert col.dimension() == 4
    assert col.metric() == "l2"
    assert not col.is_quantized()

    col.add([1, 2], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    res = col.query([[0.1, 0.2, 0.3, 0.4]], 1)
    assert res["ids"][0] == [1]

    save_path = tmp_path / "col.bin"
    col.save(save_path)
    loaded_col = client.load(save_path)
    assert isinstance(loaded_col, mojovec.Collection)
    assert loaded_col.name() == "test_col"
    assert loaded_col.count() == 2
    loaded_res = loaded_col.query([[0.1, 0.2, 0.3, 0.4]], 1)
    assert loaded_res["ids"][0] == [1]


def test_client_recover(tmp_path):
    client = mojovec.Client()
    snapshot = tmp_path / "base.mojovec"
    wal = tmp_path / "changes.wal"
    col = client.create_collection("wal_col", dimension=2, quantized=False, metric="l2")
    col.save(snapshot)
    col.enable_wal(wal, durability=mojovec.WAL_ASYNC)
    col.add([1], [[1.0, 0.0]], documents=["first doc"])
    col.flush_wal()
    col.disable_wal()

    recovered = client.recover(snapshot, wal)
    assert isinstance(recovered, mojovec.Collection)
    assert recovered.name() == "wal_col"
    assert recovered.count() == 1
    assert recovered.get_document(1) == "first doc"


def test_client_create_ivfpq_collection():
    client = mojovec.Client()
    ivf = client.create_ivfpq_collection(
        "test_ivf",
        dimension=8,
        nlist=4,
        M=2,
        nprobe=2,
        metric="l2",
    )
    assert isinstance(ivf, mojovec.IVFPQCollection)
    assert ivf.name() == "test_ivf"
    assert ivf.dimension() == 8
    assert ivf.nlist() == 4
    assert ivf.pq_subvectors() == 2
    assert ivf.nprobe() == 2
    assert ivf.metric() == "l2"

    rows = [
        [float((i + j) % 10) for j in range(8)]
        for i in range(256)
    ]
    ids = list(range(256))
    ivf.train(rows)
    ivf.add(ids, rows)
    assert ivf.count() == 256
    res = ivf.query(rows[:2], 1)
    assert len(res["ids"]) == 2
    assert res["ids"][0] == [0]
    assert res["ids"][1] == [1]


def test_native_backend_is_publicly_observable():
    assert mojovec.native_backend() in {"native", "avx2", "avx512"}
    assert mojovec.native_backend() == mojovec._native_backend


def test_native_module_exits_cleanly_in_fresh_interpreter():
    """Detect heap corruption during native-module shutdown."""
    environment = os.environ.copy()
    if sys.platform.startswith("linux"):
        environment["MALLOC_CHECK_"] = "3"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import mojovec; "
                "print(mojovec.__version__, mojovec.native_backend())"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
