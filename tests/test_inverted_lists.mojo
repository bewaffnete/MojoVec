from std.collections.span import Span
from mojovec.storage.inverted_lists import ArrayInvertedLists
from std.memory.alloc import unsafe_alloc

from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite

def test_inverted_lists_crud() raises:
    var nlist = 10
    var code_size = 4
    
    var invlists = ArrayInvertedLists(nlist, code_size)
    
    # Add initial items
    var ids = unsafe_alloc[Int](2)
    ids[unsafe_offset=0] = 100
    ids[unsafe_offset=1] = 101
    
    var codes = unsafe_alloc[UInt8](8)
    for i in range(8):
        codes[unsafe_offset=i] = UInt8(i)
        
    invlists.add_entries(
        5,
        Span[Int, MutUntrackedOrigin](unsafe_ptr=ids, length=2),
        Span[UInt8, MutUntrackedOrigin](unsafe_ptr=codes, length=8),
    )
    
    assert_equal(invlists.list_size(5), 2)
    assert_equal(invlists.get_ids(5)[0], 100)
    assert_equal(invlists.get_ids(5)[1], 101)
    
    var ptr = invlists.get_codes(5)
    for i in range(8):
        assert_equal(ptr[i], UInt8(i))
        
    # Test capacity expansion without data loss
    var large_n = 2000
    var large_ids = unsafe_alloc[Int](large_n)
    var large_codes = unsafe_alloc[UInt8](large_n * code_size)
    
    for i in range(large_n):
        large_ids[unsafe_offset=i] = i * 10
        for j in range(code_size):
            large_codes[unsafe_offset=i * code_size + j] = UInt8(j)
            
    # Add large number of entries to list 5, triggering resize
    invlists.add_entries(
        5,
        Span[Int, MutUntrackedOrigin](unsafe_ptr=large_ids, length=large_n),
        Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=large_codes, length=large_n * code_size
        ),
    )
    
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
    
    ids.unsafe_free()
    codes.unsafe_free()
    large_ids.unsafe_free()
    large_codes.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
