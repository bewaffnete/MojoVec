from src.mojovec.storage.inverted_lists import ArrayInvertedLists
from std.memory import alloc

def assert_true(cond: Bool, msg: String = "Assertion failed") raises:
    if not cond:
        raise Error(msg)

def main() raises:
    var nlist = 10
    var code_size = 4
    
    var invlists = ArrayInvertedLists(nlist, code_size)
    
    var ids = alloc[Int](2)
    ids[0] = 100
    ids[1] = 101
    
    var codes = alloc[UInt8](8)
    for i in range(8):
        codes[i] = UInt8(i)
        
    invlists.add_entries(5, 2, ids, codes)
    
    assert_true(invlists.list_size(5) == 2, "Size should be 2")
    assert_true(invlists.get_ids(5)[0] == 100, "ID 0 mismatch")
    assert_true(invlists.get_ids(5)[1] == 101, "ID 1 mismatch")
    
    var ptr = invlists.get_codes(5)
    for i in range(8):
        assert_true(ptr[i] == UInt8(i), "Code mismatch")
        
    print("All InvertedLists tests passed!")
    
    _ = invlists.list_size(0) # Keep invlists alive
    
    ids.free()
    codes.free()
