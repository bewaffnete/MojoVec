"""
Smoke test to verify that the compiled mojovec.mojoc package can be imported
and used successfully by a downstream client.
"""

from std.testing import TestSuite, assert_equal
from mojovec import Client


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
