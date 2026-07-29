from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from mojovec import Collection, Metadata, QueryResults, Where


comptime DIMENSION = 4


def append_vector(mut embeddings: List[Float32], value: Float32):
    for _ in range(DIMENSION):
        embeddings.append(value)


def metadata(
    category: String,
    year: Int,
    score: Float64,
    published: Bool,
) -> Metadata:
    var result = Metadata()
    result.set("category", category)
    result.set("year", year)
    result.set("score", score)
    result.set("published", published)
    return result^


def make_collection(quantized: Bool = False) raises -> Collection:
    var collection = Collection(
        DIMENSION,
        M=8,
        ef_construction=64,
        ef_search=64,
        quantized=quantized,
    )
    var ids = [10, 20, 30, 40, 50]
    var embeddings = List[Float32]()
    append_vector(embeddings, 0.0)
    append_vector(embeddings, 10.0)
    append_vector(embeddings, 20.0)
    append_vector(embeddings, 30.0)
    append_vector(embeddings, 40.0)

    var metadatas = List[Metadata]()
    metadatas.append(metadata("docs", 2022, 0.5, True))
    metadatas.append(metadata("docs", 2024, 0.8, False))
    metadatas.append(metadata("code", 2025, 0.95, True))
    metadatas.append(metadata("code", 2026, 1.0, False))
    var sparse = Metadata()
    sparse.set("owner", "missing-filter-fields")
    metadatas.append(sparse^)

    collection.add(ids, embeddings, metadatas)
    return collection^


def contains(values: List[Int], value: Int) -> Bool:
    for candidate in values:
        if candidate == value:
            return True
    return False


def assert_ids(
    results: QueryResults,
    expected: List[Int],
) raises:
    assert_equal(len(results.ids), 1)
    var matched = 0
    for record_id in results.ids[0]:
        if record_id < 0:
            continue
        assert_true(
            contains(expected, record_id),
            "Filtered query returned an unexpected ID.",
        )
        matched += 1
    assert_equal(matched, len(expected))
    for expected_id in expected:
        assert_true(
            contains(results.ids[0], expected_id),
            "Filtered query omitted an expected ID.",
        )


def run_where(
    mut collection: Collection,
    where: Where,
) raises -> QueryResults:
    var query = List[Float32]()
    append_vector(query, 20.0)
    return collection.query(query, where=where, n_results=5)


def test_typed_scalar_predicates() raises:
    var collection = make_collection()

    assert_ids(
        run_where(collection, Where.eq("category", "docs")),
        [10, 20],
    )
    # Missing fields do not satisfy a typed `ne` predicate.
    assert_ids(
        run_where(collection, Where.ne("category", "docs")),
        [30, 40],
    )
    assert_ids(
        run_where(collection, Where.gt("year", 2024)),
        [30, 40],
    )
    assert_ids(
        run_where(collection, Where.gte("score", Float64(0.8))),
        [20, 30, 40],
    )
    assert_ids(
        run_where(collection, Where.lt("year", 2025)),
        [10, 20],
    )
    assert_ids(
        run_where(collection, Where.lte("score", Float64(0.8))),
        [10, 20],
    )
    assert_ids(
        run_where(collection, Where.eq("published", True)),
        [10, 30],
    )
    var years = List[Int]()
    years.append(2022)
    years.append(2026)
    assert_ids(
        run_where(collection, Where.in_("year", years)),
        [10, 40],
    )
    assert_ids(
        run_where(
            collection,
            Where.not_in("category", ["docs", "missing"]),
        ),
        [30, 40],
    )
    # Ordered predicates compare Int and Float64 metadata numerically.
    assert_ids(
        run_where(collection, Where.gt("year", Float64(2024.5))),
        [30, 40],
    )
    assert_ids(
        run_where(collection, Where.eq("missing", "value")),
        List[Int](),
    )


def test_logical_where_expressions() raises:
    var collection = make_collection()

    var all_conditions = List[Where]()
    all_conditions.append(Where.eq("category", "code"))
    all_conditions.append(Where.eq("published", True))
    assert_ids(
        run_where(collection, Where.all(all_conditions)),
        [30],
    )

    var any_conditions = List[Where]()
    any_conditions.append(Where.lt("year", 2023))
    any_conditions.append(Where.eq("published", False))
    assert_ids(
        run_where(collection, Where.any(any_conditions)),
        [10, 20, 40],
    )

    assert_ids(
        run_where(
            collection,
            Where.not_(Where.eq("category", "docs")),
        ),
        [30, 40, 50],
    )

    var nested_any = List[Where]()
    nested_any.append(Where.eq("year", 2022))
    nested_any.append(Where.eq("year", 2025))
    var nested_all = List[Where]()
    nested_all.append(Where.any(nested_any))
    nested_all.append(Where.eq("published", True))
    assert_ids(
        run_where(collection, Where.all(nested_all)),
        [10, 30],
    )

    var empty_failed = False
    try:
        _ = run_where(collection, Where())
    except:
        empty_failed = True
    assert_true(empty_failed)


def test_bitmap_lifecycle_update_delete_load_compact() raises:
    var collection = make_collection()

    # Vector-only replacement inherits metadata and receives new postings.
    var inherited_vector = List[Float32]()
    append_vector(inherited_vector, 1.0)
    collection.upsert([10], inherited_vector)
    assert_ids(
        run_where(collection, Where.eq("category", "docs")),
        [10, 20],
    )

    # Explicit metadata replacement indexes the new values. Historical
    # postings remain harmless because the deletion mask excludes old rows.
    var replacement = metadata("code", 2030, 1.5, True)
    var replacements = List[Metadata]()
    replacements.append(replacement^)
    var replacement_vector = List[Float32]()
    append_vector(replacement_vector, 11.0)
    collection.update([20], replacement_vector, replacements)
    assert_ids(
        run_where(collection, Where.eq("category", "docs")),
        [10],
    )
    assert_ids(
        run_where(collection, Where.eq("category", "code")),
        [20, 30, 40],
    )

    collection.delete([30])
    assert_ids(
        run_where(collection, Where.eq("category", "code")),
        [20, 40],
    )

    var path = "/tmp/mojovec_where_roundtrip.mojovec"
    collection.save(path)
    var loaded = Collection.load(path)
    assert_ids(
        run_where(loaded, Where.gte("year", 2026)),
        [20, 40],
    )

    var report = loaded.compact()
    assert_true(report.performed)
    assert_ids(
        run_where(loaded, Where.eq("published", True)),
        [10, 20],
    )


def test_sq8_where_roundtrip() raises:
    var collection = make_collection(quantized=True)
    var path = "/tmp/mojovec_where_sq8.mojovec"
    collection.save(path)
    var loaded = Collection.load(path)
    assert_ids(
        run_where(loaded, Where.eq("category", "code")),
        [30, 40],
    )


def test_high_cardinality_scan_fallback() raises:
    comptime count = 1_026
    var collection = Collection(
        DIMENSION,
        M=4,
        ef_construction=32,
        ef_search=64,
        quantized=False,
    )
    var ids = List[Int](capacity=count)
    var embeddings = List[Float32](capacity=count * DIMENSION)
    var metadatas = List[Metadata](capacity=count)
    for index in range(count):
        ids.append(index)
        append_vector(embeddings, Float32(index))
        var record = Metadata()
        record.set("unique", "value-" + String(index))
        metadatas.append(record^)
    collection.add(ids, embeddings, metadatas)

    var query = List[Float32]()
    append_vector(query, Float32(count - 1))
    var results = collection.query(
        query,
        where=Where.eq("unique", "value-" + String(count - 1)),
        n_results=1,
    )
    assert_equal(results.ids[0][0], count - 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
