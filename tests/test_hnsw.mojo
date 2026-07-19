from std.memory.span import Span
from std.random import rand
from std.memory import alloc
from std.testing import assert_true, assert_equal, TestSuite

from mojovec.utils.heap import max_heap_push, min_heap_push, min_heap_pop, max_heap_replace_top
from mojovec.index.hnsw_visited import VisitedTable

def test_heap_popmin() raises:
    # Mimics Faiss's test_popmin
    var k = 10
    var dists = alloc[Float32](k)
    var labels = alloc[Int](k)
    
    # Initialize empty heap (max heap)
    for i in range(k):
        dists[i] = 1000000.0
        labels[i] = -1
        
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
    assert_equal(dists[0], 0.9)
    assert_equal(labels[0], 9)
    
    # Now, test min heap
    var min_dists = alloc[Float32](k)
    var min_labels = alloc[Int](k)
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
    assert_equal(min_dists[0], 0.1)
    
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
    
    dists.free()
    labels.free()
    min_dists.free()
    min_labels.free()

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

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
