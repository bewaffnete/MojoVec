from std.memory import alloc
from std.math import max

trait InvertedListsTrait(Movable, ImplicitlyDeletable):
    def list_size(self, list_no: Int) -> Int: ...
    def get_codes(self, list_no: Int) -> UnsafePointer[UInt8, MutUntrackedOrigin]: ...
    def get_ids(self, list_no: Int) -> UnsafePointer[Int, MutUntrackedOrigin]: ...
    def add_entries(mut self, list_no: Int, n_entry: Int, ids: UnsafePointer[Int, MutUntrackedOrigin], codes: UnsafePointer[UInt8, MutUntrackedOrigin]): ...
    def resize(mut self, list_no: Int, new_size: Int): ...

struct InvertedListBucket(Movable, Copyable, ImplicitlyCopyable):
    var size: Int
    var capacity: Int
    var ids: UnsafePointer[Int, MutUntrackedOrigin]
    var codes: UnsafePointer[UInt8, MutUntrackedOrigin]
    
    def __init__(out self):
        self.size = 0
        self.capacity = 0
        self.ids = alloc[Int](1)
        self.codes = alloc[UInt8](1)
        
    def free(mut self):
        if Int(self.ids) != 0: self.ids.free()
        if Int(self.codes) != 0: self.codes.free()

    def __copyinit__(out self, existing: Self):
        self.size = existing.size
        self.capacity = existing.capacity
        self.ids = existing.ids
        self.codes = existing.codes

struct ArrayInvertedLists(Movable, InvertedListsTrait):
    var nlist: Int
    var code_size: Int
    var lists: UnsafePointer[InvertedListBucket, MutUntrackedOrigin]
    
    def __init__(out self, nlist: Int, code_size: Int):
        self.nlist = nlist
        self.code_size = code_size
        self.lists = alloc[InvertedListBucket](nlist)
        for i in range(nlist):
            self.lists[i] = InvertedListBucket()
            
    def __init__(out self, *, deinit move: Self):
        self.nlist = move.nlist
        self.code_size = move.code_size
        self.lists = move.lists

    def __del__(deinit self):
        if self.nlist > 0 and Int(self.lists) != 0:
            for i in range(self.nlist):
                self.lists[i].free()
            self.lists.free()
            
    @always_inline
    def list_size(self, list_no: Int) -> Int:
        return self.lists[list_no].size
        
    @always_inline
    def get_codes(self, list_no: Int) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
        return self.lists[list_no].codes
        
    @always_inline
    def get_ids(self, list_no: Int) -> UnsafePointer[Int, MutUntrackedOrigin]:
        return self.lists[list_no].ids
        
    def resize(mut self, list_no: Int, new_size: Int):
        var bucket = self.lists[list_no]
        if new_size <= bucket.capacity:
            bucket.size = new_size
            self.lists[list_no] = bucket
            return
            
        var new_cap = max(bucket.capacity * 2, new_size)
        if bucket.capacity == 0: new_cap = max(16, new_size)
        
        var new_ids = alloc[Int](new_cap)
        var new_codes = alloc[UInt8](new_cap * self.code_size)
        
        if bucket.size > 0:
            for i in range(bucket.size):
                new_ids[i] = bucket.ids[i]
            for i in range(bucket.size * self.code_size):
                new_codes[i] = bucket.codes[i]
                
        bucket.ids.free()
        bucket.codes.free()
        
        bucket.ids = new_ids
        bucket.codes = new_codes
        bucket.capacity = new_cap
        bucket.size = new_size
        self.lists[list_no] = bucket

    def add_entries(mut self, list_no: Int, n_entry: Int, ids: UnsafePointer[Int, MutUntrackedOrigin], codes: UnsafePointer[UInt8, MutUntrackedOrigin]):
        _ = Int(self.lists)  # WORKAROUND: Force memory materialization to avoid MLIR/LLVM alias analysis bug in Mojo 2.4+
        var old_size = self.lists[list_no].size
        self.resize(list_no, old_size + n_entry)
        
        var bucket = self.lists[list_no]
        for i in range(n_entry):
            bucket.ids[old_size + i] = ids[i]
            
        var code_offset = old_size * self.code_size
        for i in range(n_entry * self.code_size):
            bucket.codes[code_offset + i] = codes[i]
            
        self.lists[list_no] = bucket
