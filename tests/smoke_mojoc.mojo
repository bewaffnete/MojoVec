"""
Smoke test to verify that the compiled mojovec.mojoc package can be imported
and used successfully by a downstream client.
"""

from std.testing import TestSuite, assert_equal
from std.collections import List
from mojovec import Client, CollectionIVFPQ


def test_mojoc_package_smoke() raises:
    var client = Client()
    var collection = client.create_collection(
        "package_smoke",
        dimension=4,
        M=8,
        ef_construction=32,
        ef_search=16,
        quantized=False,
    )
    collection.add(
        [10, 20],
        [
            Float32(1.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(1.0),
            Float32(0.0),
            Float32(0.0),
        ],
    )
    var results = collection.query(
        [
            Float32(1.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
        ],
        n_results=1,
    )
    assert_equal(results.ids[0][0], 10, "Precompiled package returned an unexpected result.")


def test_mojoc_ivfpq_package_smoke() raises:
    var ids = List[Int](capacity=256)
    var vectors = List[Float32](capacity=256 * 8)
    for row in range(256):
        ids.append(row)
        for column in range(8):
            vectors.append(
                Float32((row * 17 + column * 29) % 251) / 251.0
            )
    var collection = CollectionIVFPQ(
        8, nlist=8, M=2, nprobe=8
    )
    collection.add(ids, vectors)
    var query = List[Float32](capacity=8)
    for column in range(8):
        query.append(vectors[column])
    var results = collection.query(query, n_results=1)
    assert_equal(
        results.ids[0][0],
        0,
        "Precompiled IVF-PQ package returned an unexpected result.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
