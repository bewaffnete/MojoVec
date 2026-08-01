from std.testing import TestSuite, assert_equal, assert_true

from mojovec import Collection
from mojovec.core.types import METRIC_L2
from mojovec.index.hnsw_graph import HNSWGraph
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_flat_sq8 import IndexFlatSQ8


def _collection_is_rejected(
    dimension: Int = 8,
    M: Int = 32,
    ef_construction: Int = 40,
    ef_search: Int = 16,
) -> Bool:
    try:
        _ = Collection(
            dimension,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
    except:
        return True
    return False


def test_collection_validates_before_storage_allocation() raises:
    assert_true(_collection_is_rejected(dimension=0))
    assert_true(_collection_is_rejected(dimension=-1))
    assert_true(_collection_is_rejected(dimension=65_537))
    assert_true(_collection_is_rejected(M=1))
    assert_true(_collection_is_rejected(M=1_001))
    assert_true(_collection_is_rejected(ef_construction=0))
    assert_true(_collection_is_rejected(ef_construction=2_049))
    assert_true(_collection_is_rejected(ef_search=0))
    assert_true(_collection_is_rejected(ef_search=2_049))

    var boundary = Collection(
        1,
        M=2,
        ef_construction=1,
        ef_search=1,
        quantized=False,
    )
    assert_equal(boundary.dimension(), 1)


def test_low_level_allocating_constructors_validate_inputs() raises:
    var flat_failed = False
    try:
        _ = IndexFlat(-1, METRIC_L2)
    except:
        flat_failed = True
    assert_true(flat_failed)

    var sq8_failed = False
    try:
        _ = IndexFlatSQ8(0, METRIC_L2)
    except:
        sq8_failed = True
    assert_true(sq8_failed)

    var M_failed = False
    try:
        _ = HNSWGraph(M=1_001)
    except:
        M_failed = True
    assert_true(M_failed)

    var ef_construction_failed = False
    try:
        _ = HNSWGraph(efConstruction=0)
    except:
        ef_construction_failed = True
    assert_true(ef_construction_failed)

    var ef_search_failed = False
    try:
        _ = HNSWGraph(efSearch=2_049)
    except:
        ef_search_failed = True
    assert_true(ef_search_failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
