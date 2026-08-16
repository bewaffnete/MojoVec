from std.collections.span import Span
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite

from mojovec.utils.heap import max_heap_push, min_heap_push, min_heap_pop, max_heap_replace_top
from mojovec.index.hnsw_visited import VisitedTable
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_flat_sq8 import IndexFlatSQ8
from mojovec.index.index_hnsw import IndexHNSW
from mojovec.core.types import METRIC_L2
from mojovec.utils.distance_computer import StorageTrait

def test_heap_popmin() raises:
    # Mimics Faiss's test_popmin
    var k = 10
    var dists = unsafe_alloc[Float32](k)
    var labels = unsafe_alloc[Int](k)
    
    # Initialize empty heap (max heap)
    for i in range(k):
        dists[unsafe_offset=i] = 1000000.0
        labels[unsafe_offset=i] = -1
        
    # Push elements
    var current_size = 0
    max_heap_push(dists, labels, current_size, 0.5, 5)
    current_size += 1
    max_heap_push(dists, labels, current_size, 0.2, 2)
    current_size += 1
    max_heap_push(dists, labels, current_size, 0.8, 8)
    current_size += 1
    max_heap_push(dists, labels, current_size, 0.1, 1)
    current_size += 1
    max_heap_push(dists, labels, current_size, 0.9, 9)
    current_size += 1
    
    assert_equal(current_size, 5)
    
    # In a max heap, the top element is the maximum
    assert_equal(dists[unsafe_offset=0], 0.9)
    assert_equal(labels[unsafe_offset=0], 9)
    
    # Now, test min heap
    var min_dists = unsafe_alloc[Float32](k)
    var min_labels = unsafe_alloc[Int](k)
    var min_size = 0
    
    min_heap_push(min_dists, min_labels, min_size, 0.5, 5)
    min_size += 1
    min_heap_push(min_dists, min_labels, min_size, 0.2, 2)
    min_size += 1
    min_heap_push(min_dists, min_labels, min_size, 0.8, 8)
    min_size += 1
    min_heap_push(min_dists, min_labels, min_size, 0.1, 1)
    min_size += 1
    
    assert_equal(min_size, 4)
    assert_equal(min_dists[unsafe_offset=0], 0.1)
    
    var pop1 = min_heap_pop(min_dists, min_labels, min_size)
    min_size -= 1
    assert_equal(pop1.label, 1)
    
    var pop2 = min_heap_pop(min_dists, min_labels, min_size)
    min_size -= 1
    assert_equal(pop2.label, 2)
    
    var pop3 = min_heap_pop(min_dists, min_labels, min_size)
    min_size -= 1
    assert_equal(pop3.label, 5)
    
    var pop4 = min_heap_pop(min_dists, min_labels, min_size)
    min_size -= 1
    assert_equal(pop4.label, 8)
    
    assert_equal(min_size, 0)
    
    dists.unsafe_free()
    labels.unsafe_free()
    min_dists.unsafe_free()
    min_labels.unsafe_free()

def test_visited_table() raises:
    var vt = VisitedTable(100)
    
    # Initially none visited
    assert_true(not vt.is_visited(10))
    assert_true(not vt.is_visited(20))
    
    vt.set_visited(10)
    assert_true(vt.is_visited(10))
    assert_true(not vt.is_visited(20))
    
    vt.advance()
    # After advance, 10 should not be visited again!
    assert_true(not vt.is_visited(10))
    assert_true(not vt.is_visited(20))
    
    vt.set_visited(20)
    assert_true(vt.is_visited(20))


def test_visited_table_wraparound() raises:
    var vt = VisitedTable(8)
    vt.set_visited(3)

    # UInt8 version marks periodically clear the backing table. Old entries
    # must never become visible again after the version wraps.
    for _ in range(300):
        vt.advance()

    assert_true(not vt.is_visited(3))
    vt.set_visited(3)
    assert_true(vt.is_visited(3))


def check_pthread_matches_serial[StorageType: StorageTrait](
    var index: IndexHNSW[StorageType]
) raises:
    comptime dimension = 8
    comptime database_size = 64
    comptime query_count = 256
    comptime k = 5

    var database = unsafe_alloc[Float32](database_size * dimension)
    for i in range(database_size):
        for j in range(dimension):
            database[unsafe_offset=i * dimension + j] = Float32(i + j) / 64.0
    index.add(
        Span[Float32](unsafe_ptr=database, length=database_size * dimension)
    )
    index.hnsw.efSearch = 16

    var queries = unsafe_alloc[Float32](query_count * dimension)
    for i in range(query_count):
        var source = i % database_size
        for j in range(dimension):
            queries[unsafe_offset=i * dimension + j] = database[unsafe_offset=
                source * dimension + j
            ]

    var threaded_distances = unsafe_alloc[Float32](query_count * k)
    var threaded_labels = unsafe_alloc[Int](query_count * k)
    var serial_distances = unsafe_alloc[Float32](query_count * k)
    var serial_labels = unsafe_alloc[Int](query_count * k)
    var empty_filter_storage = unsafe_alloc[UInt8](1)
    var empty_filter = Span[UInt8](
        unsafe_ptr=empty_filter_storage, length=0
    )
    var query_span = Span[Float32](
        unsafe_ptr=queries, length=query_count * dimension
    )
    var threaded_distance_span = Span[mut=True, Float32](
        unsafe_ptr=threaded_distances, length=query_count * k
    )
    var threaded_label_span = Span[mut=True, Int](
        unsafe_ptr=threaded_labels, length=query_count * k
    )

    index.search(
        query_span,
        k,
        threaded_distance_span,
        threaded_label_span,
        empty_filter,
    )
    index._search_range[False](
        0,
        1,
        query_count,
        k,
        queries,
        serial_distances,
        serial_labels,
        empty_filter_storage,
    )

    for i in range(query_count * k):
        assert_equal(threaded_labels[unsafe_offset=i], serial_labels[unsafe_offset=i])
        assert_equal(threaded_distances[unsafe_offset=i], serial_distances[unsafe_offset=i])

    database.unsafe_free()
    queries.unsafe_free()
    threaded_distances.unsafe_free()
    threaded_labels.unsafe_free()
    serial_distances.unsafe_free()
    serial_labels.unsafe_free()
    empty_filter_storage.unsafe_free()


def test_flat_pthread_matches_serial() raises:
    var storage = IndexFlat(8, METRIC_L2)
    var index = IndexHNSW[IndexFlat](storage^, 8, METRIC_L2, M=8)
    check_pthread_matches_serial(index^)


def test_sq8_pthread_matches_serial() raises:
    var storage = IndexFlatSQ8(8, METRIC_L2)
    var index = IndexHNSW[IndexFlatSQ8](storage^, 8, METRIC_L2, M=8)
    check_pthread_matches_serial(index^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
