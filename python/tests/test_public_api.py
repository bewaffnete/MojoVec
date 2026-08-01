import inspect

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
        "Collection",
        "DEFAULT_MMAP_THRESHOLD_BYTES",
        "Metadata",
        "QueryResult",
        "WAL_ASYNC",
        "WAL_SYNC",
        "Where",
        "__version__",
        "load",
        "native_backend",
        "recover",
    ]
    assert mojovec.Collection.__module__ == "mojovec"
    assert mojovec.load.__module__ == "mojovec"
    assert mojovec.recover.__module__ == "mojovec"
    assert not hasattr(mojovec, "ctypes")
    assert not hasattr(mojovec, "site")
    assert not hasattr(mojovec, "_compile_where")


def test_public_callables_have_runtime_docstrings():
    assert inspect.getdoc(mojovec.Collection)
    assert inspect.getdoc(mojovec.load)
    assert inspect.getdoc(mojovec.recover)
    assert inspect.getdoc(mojovec.native_backend)
    for method_name in PUBLIC_COLLECTION_METHODS:
        assert inspect.getdoc(getattr(mojovec.Collection, method_name)), method_name


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


def test_native_backend_is_publicly_observable():
    assert mojovec.native_backend() in {"native", "avx2", "avx512"}
    assert mojovec.native_backend() == mojovec._native_backend
