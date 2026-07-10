from std.memory import alloc
from std.math import max
from std.atomic import Atomic

trait InvertedListsTrait(Movable, ImplicitlyDeletable):
    def list_size(self, list_no: Int) -> Int: ...
    def get_codes(self, list_no: Int) -> UnsafePointer[UInt8, MutUntrackedOrigin]: ...
    def get_ids(self, list_no: Int) -> UnsafePointer[Int, MutUntrackedOrigin]: ...
    def add_entries(mut self, list_no: Int, n_entry: Int, ids: UnsafePointer[Int, MutUntrackedOrigin], codes: UnsafePointer[UInt8, MutUntrackedOrigin]): ...
    def resize(mut self, list_no: Int, new_size: Int): ...

# Plain data struct — no ownership, no __del__. Purely a record for metadata.
# This avoids all shallow-copy / double-free issues that arose when InvertedListBucket
# owned UnsafePointer fields and was ImplicitlyCopyable.
@fieldwise_init
struct InvertedListBucket(Movable, Copyable, ImplicitlyCopyable):
    var size: Int
    var capacity: Int
    var ids: UnsafePointer[Int, MutUntrackedOrigin]
    var codes: UnsafePointer[UInt8, MutUntrackedOrigin]

# Helpers — direct field access on the flat array avoids any bucket copies.
# Using 'var' avoids conflicting with the 'ref' convention keyword.
struct ArrayInvertedLists(Movable, InvertedListsTrait):
    var nlist: Int
    var code_size: Int
    var lists: UnsafePointer[InvertedListBucket, MutUntrackedOrigin]
    var next_tickets: UnsafePointer[UInt32, MutUntrackedOrigin]
    var now_serving: UnsafePointer[UInt32, MutUntrackedOrigin]

    def __init__(out self, nlist: Int, code_size: Int):
        self.nlist = nlist
        self.code_size = code_size
        self.lists = alloc[InvertedListBucket](nlist)
        self.next_tickets = alloc[UInt32](nlist)
        self.now_serving = alloc[UInt32](nlist)
        for i in range(nlist):
            self.lists[i] = InvertedListBucket(
                size=0, capacity=0,
                ids=alloc[Int](1),
                codes=alloc[UInt8](1),
            )
            self.next_tickets[i] = 0
            self.now_serving[i] = 0

    def __init__(out self, *, deinit move: Self):
        self.nlist = move.nlist
        self.code_size = move.code_size
        self.lists = move.lists
        self.next_tickets = move.next_tickets
        self.now_serving = move.now_serving

    def __del__(deinit self):
        if self.nlist > 0 and Int(self.lists) != 0:
            for i in range(self.nlist):
                var ids_ptr = self.lists[i].ids
                var codes_ptr = self.lists[i].codes
                if Int(ids_ptr) != 0:
                    ids_ptr.free()
                if Int(codes_ptr) != 0:
                    codes_ptr.free()
            self.lists.free()
            if Int(self.next_tickets) != 0:
                self.next_tickets.free()
            if Int(self.now_serving) != 0:
                self.now_serving.free()

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
        if new_size <= self.lists[list_no].capacity:
            self.lists[list_no].size = new_size
            return

        var old_cap = self.lists[list_no].capacity
        var new_cap = max(old_cap * 2, new_size)
        if old_cap == 0:
            new_cap = max(16, new_size)

        var new_ids = alloc[Int](new_cap)
        var new_codes = alloc[UInt8](new_cap * self.code_size)

        var old_size = self.lists[list_no].size
        if old_size > 0:
            for i in range(old_size):
                new_ids[i] = self.lists[list_no].ids[i]
            for i in range(old_size * self.code_size):
                new_codes[i] = self.lists[list_no].codes[i]

        if Int(self.lists[list_no].ids) != 0:
            self.lists[list_no].ids.free()
        if Int(self.lists[list_no].codes) != 0:
            self.lists[list_no].codes.free()

        self.lists[list_no].ids = new_ids
        self.lists[list_no].codes = new_codes
        self.lists[list_no].capacity = new_cap
        self.lists[list_no].size = new_size

    @always_inline
    def lock_list(self, list_no: Int):
        var ticket = Atomic.fetch_add(self.next_tickets + list_no, 1)
        while Atomic.load(self.now_serving + list_no) != ticket:
            pass  # spin

    @always_inline
    def unlock_list(self, list_no: Int):
        _ = Atomic.fetch_add(self.now_serving + list_no, 1)

    def add_entries(mut self, list_no: Int, n_entry: Int, ids: UnsafePointer[Int, MutUntrackedOrigin], codes: UnsafePointer[UInt8, MutUntrackedOrigin]):
        self.lock_list(list_no)
        _ = Int(self.lists)  # WORKAROUND: Force memory materialization to avoid MLIR/LLVM alias analysis bug in Mojo 2.4+
        var old_size = self.lists[list_no].size
        self.resize(list_no, old_size + n_entry)

        for i in range(n_entry):
            self.lists[list_no].ids[old_size + i] = ids[i]

        var code_offset = old_size * self.code_size
        for i in range(n_entry * self.code_size):
            self.lists[list_no].codes[code_offset + i] = codes[i]
            
        self.unlock_list(list_no)
