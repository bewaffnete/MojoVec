from std.memory.span import Span
from mojovec.storage.inverted_lists import ArrayInvertedLists
from std.memory import alloc

from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite

def test_inverted_lists_crud() raises:
    var nlist = 10
    var code_size = 4
    
    var invlists = ArrayInvertedLists(nlist, code_size)
    
    # Add initial items
    var ids = alloc[Int](2)
    ids[0] = 100
    ids[1] = 101
    
    var codes = alloc[UInt8](8)
    for i in range(8):
        codes[i] = UInt8(i)
        
    invlists.add_entries(5, 2, ids, codes)
    
    assert_equal(invlists.list_size(5), 2)
    assert_equal(invlists.get_ids(5)[0], 100)
    assert_equal(invlists.get_ids(5)[1], 101)
    
    var ptr = invlists.get_codes(5)
    for i in range(8):
        assert_equal(ptr[i], UInt8(i))
        
    # Test capacity expansion without data loss
    var large_n = 2000
    var large_ids = alloc[Int](large_n)
    var large_codes = alloc[UInt8](large_n * code_size)
    
    for i in range(large_n):
        large_ids[i] = i * 10
        for j in range(code_size):
            large_codes[i * code_size + j] = UInt8(j)
            
    # Add large number of entries to list 5, triggering resize
    invlists.add_entries(5, large_n, large_ids, large_codes)
    
    assert_equal(invlists.list_size(5), 2 + large_n)
    
    # Verify first elements are still intact
    assert_equal(invlists.get_ids(5)[0], 100)
    assert_equal(invlists.get_ids(5)[1], 101)
    ptr = invlists.get_codes(5)
    for i in range(8):
        assert_equal(ptr[i], UInt8(i))
        
    # Verify last elements
    var last_idx = 2 + large_n - 1
    assert_equal(invlists.get_ids(5)[last_idx], (large_n - 1) * 10)
    
    _ = invlists.list_size(0) # Keep invlists alive
    
    ids.free()
    codes.free()
    large_ids.free()
    large_codes.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
