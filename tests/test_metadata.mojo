from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from mojovec import Collection, Metadata


comptime DIMENSION = 4


def vector(base: Float32) -> List[Float32]:
    return [base, base + 1.0, base + 2.0, base + 3.0]


def full_metadata(label: String, year: Int) -> Metadata:
    var metadata = Metadata()
    metadata.set("label", label)
    metadata.set("year", year)
    metadata.set("score", Float64(0.875))
    metadata.set("published", True)
    return metadata^


def test_metadata_value_types_and_copy() raises:
    var metadata = full_metadata("guide", 2026)
    assert_equal(metadata.count(), 4)
    var keys = metadata.keys()
    assert_equal(len(keys), 4)
    assert_equal(keys[0], "label")
    assert_equal(keys[3], "published")
    assert_true(metadata.contains("label"))
    assert_false(metadata.contains("missing"))
    assert_equal(metadata.get_string("label"), "guide")
    assert_equal(metadata.get_int("year"), 2026)
    assert_almost_equal(metadata.get_float("score"), 0.875, atol=1e-12)
    assert_true(metadata.get_bool("published"))

    var copied = metadata.copy()
    metadata.set("label", "changed")
    metadata.set("year", 2030)
    assert_equal(copied.get_string("label"), "guide")
    assert_equal(copied.get_int("year"), 2026)

    var wrong_type_failed = False
    try:
        _ = copied.get_int("label")
    except:
        wrong_type_failed = True
    assert_true(wrong_type_failed)

    var missing_key_failed = False
    try:
        _ = copied.get("missing")
    except:
        missing_key_failed = True
    assert_true(missing_key_failed)


def test_collection_metadata_crud_semantics() raises:
    var collection = Collection(DIMENSION, quantized=False)
    var original = full_metadata("original", 2024)
    var metadatas = List[Metadata]()
    metadatas.append(original.copy())
    collection.add([10], vector(1.0), metadatas)

    # The collection owns a copy rather than aliasing the caller's object.
    original.set("label", "mutated outside")
    assert_equal(collection.get_metadata(10).get_string("label"), "original")

    # Updating only the vector preserves the active metadata snapshot.
    collection.update([10], vector(10.0))
    var preserved = collection.get_metadata(10)
    assert_equal(preserved.get_string("label"), "original")
    assert_equal(preserved.get_int("year"), 2024)

    # Passing metadata explicitly replaces the complete metadata object.
    var replacement = Metadata()
    replacement.set("label", "replacement")
    replacement.set("version", 2)
    var replacements = List[Metadata]()
    replacements.append(replacement.copy())
    collection.upsert([10], vector(20.0), replacements)
    var replaced = collection.get_metadata(10)
    assert_equal(replaced.count(), 2)
    assert_equal(replaced.get_string("label"), "replacement")
    assert_equal(replaced.get_int("version"), 2)
    assert_false(replaced.contains("year"))

    # A newly upserted record without metadata receives an empty object.
    collection.upsert([20], vector(30.0))
    assert_equal(collection.get_metadata(20).count(), 0)

    collection.delete([20])
    var deleted_failed = False
    try:
        _ = collection.get_metadata(20)
    except:
        deleted_failed = True
    assert_true(deleted_failed)

    var missing_metadata = List[Metadata]()
    var mismatch_failed = False
    try:
        collection.add([30], vector(40.0), missing_metadata)
    except:
        mismatch_failed = True
    assert_true(mismatch_failed)
    assert_equal(collection.count(), 1)


def check_metadata_round_trip(quantized: Bool, path: String) raises:
    var collection = Collection(
        DIMENSION,
        M=8,
        ef_construction=48,
        ef_search=24,
        quantized=quantized,
        name="metadata_records",
    )

    var first = full_metadata("first", 2025)
    var second = Metadata()
    second.set("label", "second")
    second.set("weight", Float64(-12.5))
    second.set("enabled", False)
    var metadatas = List[Metadata]()
    metadatas.append(first.copy())
    metadatas.append(second.copy())

    var embeddings = List[Float32]()
    for value in vector(1.0):
        embeddings.append(value)
    for value in vector(10.0):
        embeddings.append(value)
    collection.add([100, 200], embeddings, metadatas)

    # Creates one historical version whose metadata must also remain aligned
    # in the serialized file. The active replacement inherits metadata.
    collection.upsert([100], vector(2.0))
    collection.delete([200])
    collection.save(path)

    var loaded = Collection.load(path)
    assert_equal(loaded.name(), "metadata_records")
    assert_equal(loaded.count(), 1)
    assert_equal(loaded.count_deleted(), 2)
    var loaded_metadata = loaded.get_metadata(100)
    assert_equal(loaded_metadata.get_string("label"), "first")
    assert_equal(loaded_metadata.get_int("year"), 2025)
    assert_almost_equal(
        loaded_metadata.get_float("score"), 0.875, atol=1e-12
    )
    assert_true(loaded_metadata.get_bool("published"))

    var report = loaded.compact()
    assert_true(report.performed)
    assert_equal(report.reclaimed_records, 2)
    assert_equal(loaded.count_deleted(), 0)
    var compacted_metadata = loaded.get_metadata(100)
    assert_equal(compacted_metadata.get_string("label"), "first")
    assert_equal(compacted_metadata.get_int("year"), 2025)

    loaded.save(path + ".compacted")
    var reloaded = Collection.load(path + ".compacted")
    assert_equal(reloaded.get_metadata(100).get_string("label"), "first")
    assert_equal(reloaded.count_deleted(), 0)


def test_flat_metadata_serialization() raises:
    check_metadata_round_trip(
        False, "/tmp/mojovec_metadata_flat_v3.mojovec"
    )


def test_sq8_metadata_serialization() raises:
    check_metadata_round_trip(
        True, "/tmp/mojovec_metadata_sq8_v3.mojovec"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
