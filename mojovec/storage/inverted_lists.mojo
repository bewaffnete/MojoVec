"""
Defines data structures and traits for managing inverted lists in vector storage.
"""

from std.memory.alloc import unsafe_alloc
from std.collections.span import Span
from std.math import max
from std.atomic import Atomic

trait InvertedListsTrait(Movable, Deinitable):
    """
    Abstract interface for an inverted lists storage container.
    """
    def list_size(self, list_no: Int) -> Int: ...
    def get_codes(self, list_no: Int) -> Span[UInt8, MutUntrackedOrigin]: ...
    def get_ids(self, list_no: Int) -> Span[Int, MutUntrackedOrigin]: ...
    def add_entries(
        mut self,
        list_no: Int,
        ids: Span[Int, _],
        codes: Span[UInt8, _],
    ): ...
    def resize(mut self, list_no: Int, new_size: Int): ...

@fieldwise_init
struct InvertedListBucket(Movable, Copyable, ImplicitlyCopyable):
    """
    Represents a single bucket within an inverted list.
    Designed as a plain data structure without ownership or automatic memory management 
    to avoid shallow-copy or double-free issues.
    """
    var size: Int
    var capacity: Int
    var ids: Pointer[Int, MutUntrackedOrigin]
    var codes: Pointer[UInt8, MutUntrackedOrigin]

struct ArrayInvertedLists(Movable, InvertedListsTrait):
    """
    An array-based implementation of inverted lists.
    Direct field access on the flat array is used to avoid unnecessary bucket copies.
    """
    var nlist: Int
    var code_size: Int
    var lists: Pointer[InvertedListBucket, MutUntrackedOrigin]
    var next_tickets: Pointer[UInt32, MutUntrackedOrigin]
    var now_serving: Pointer[UInt32, MutUntrackedOrigin]

    def __init__(out self, nlist: Int, code_size: Int):
        """
        Initializes the array inverted lists structure.

        Args:
            nlist: The total number of inverted lists (buckets).
            code_size: The byte size of each stored code.
        """
        self.nlist = nlist
        self.code_size = code_size
        self.lists = unsafe_alloc[InvertedListBucket](nlist)
        self.next_tickets = unsafe_alloc[UInt32](nlist)
        self.now_serving = unsafe_alloc[UInt32](nlist)
        for i in range(nlist):
            self.lists[unsafe_offset=i] = InvertedListBucket(
                size=0, capacity=0,
                ids=unsafe_alloc[Int](1),
                codes=unsafe_alloc[UInt8](1),
            )
            self.next_tickets[unsafe_offset=i] = 0
            self.now_serving[unsafe_offset=i] = 0

    def __init__(out self, *, deinit move: Self):
        self.nlist = move.nlist
        self.code_size = move.code_size
        self.lists = move.lists
        self.next_tickets = move.next_tickets
        self.now_serving = move.now_serving

    def __deinit__(deinit self):
        if self.nlist > 0 and Int(self.lists) != 0:
            for i in range(self.nlist):
                var ids_ptr = self.lists[unsafe_offset=i].ids
                var codes_ptr = self.lists[unsafe_offset=i].codes
                if Int(ids_ptr) != 0:
                    ids_ptr.unsafe_free()
                if Int(codes_ptr) != 0:
                    codes_ptr.unsafe_free()
            self.lists.unsafe_free()
            if Int(self.next_tickets) != 0:
                self.next_tickets.unsafe_free()
            if Int(self.now_serving) != 0:
                self.now_serving.unsafe_free()

    @always_inline
    def list_size(self, list_no: Int) -> Int:
        """
        Returns the number of elements currently stored in the specified list.
        """
        return self.lists[unsafe_offset=list_no].size

    @always_inline
    def get_codes(self, list_no: Int) -> Span[UInt8, MutUntrackedOrigin]:
        """
        Returns a pointer to the codes array for the specified list.
        """
        return Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=self.lists[unsafe_offset=list_no].codes,
            length=self.lists[unsafe_offset=list_no].size * self.code_size,
        )

    @always_inline
    def get_ids(self, list_no: Int) -> Span[Int, MutUntrackedOrigin]:
        """
        Returns a pointer to the IDs array for the specified list.
        """
        return Span[Int, MutUntrackedOrigin](
            unsafe_ptr=self.lists[unsafe_offset=list_no].ids,
            length=self.lists[unsafe_offset=list_no].size,
        )

    def resize(mut self, list_no: Int, new_size: Int):
        """
        Resizes the capacity of the specified list to accommodate new elements.
        """
        if new_size <= self.lists[unsafe_offset=list_no].capacity:
            self.lists[unsafe_offset=list_no].size = new_size
            return

        var old_cap = self.lists[unsafe_offset=list_no].capacity
        var new_cap = max(old_cap * 2, new_size)
        if old_cap == 0:
            new_cap = max(16, new_size)

        var new_ids = unsafe_alloc[Int](new_cap)
        var new_codes = unsafe_alloc[UInt8](new_cap * self.code_size)

        var old_size = self.lists[unsafe_offset=list_no].size
        if old_size > 0:
            for i in range(old_size):
                new_ids[unsafe_offset=i] = self.lists[unsafe_offset=list_no].ids[unsafe_offset=i]
            for i in range(old_size * self.code_size):
                new_codes[unsafe_offset=i] = self.lists[unsafe_offset=list_no].codes[unsafe_offset=i]

        if Int(self.lists[unsafe_offset=list_no].ids) != 0:
            self.lists[unsafe_offset=list_no].ids.unsafe_free()
        if Int(self.lists[unsafe_offset=list_no].codes) != 0:
            self.lists[unsafe_offset=list_no].codes.unsafe_free()

        self.lists[unsafe_offset=list_no].ids = new_ids
        self.lists[unsafe_offset=list_no].codes = new_codes
        self.lists[unsafe_offset=list_no].capacity = new_cap
        self.lists[unsafe_offset=list_no].size = new_size

    @always_inline
    def lock_list(self, list_no: Int):
        var ticket = Atomic.fetch_add(self.next_tickets.unsafe_offset(list_no), 1)
        while Atomic.load(self.now_serving.unsafe_offset(list_no)) != ticket:
            pass  # spin lock implementation

    @always_inline
    def unlock_list(self, list_no: Int):
        _ = Atomic.fetch_add(self.now_serving.unsafe_offset(list_no), 1)

    def add_entries(
        mut self,
        list_no: Int,
        ids: Span[Int, _],
        codes: Span[UInt8, _],
    ):
        """
        Adds multiple entries (codes and their corresponding IDs) to a specific list.
        """
        var n_entry = len(ids)
        var ids_ptr = ids.unsafe_ptr()
        var codes_ptr = codes.unsafe_ptr()
        self.lock_list(list_no)
        _ = Int(self.lists)  # WORKAROUND: Force memory materialization to avoid MLIR/LLVM alias analysis bug
        var old_size = self.lists[unsafe_offset=list_no].size
        self.resize(list_no, old_size + n_entry)

        for i in range(n_entry):
            self.lists[unsafe_offset=list_no].ids[unsafe_offset=old_size + i] = ids_ptr[unsafe_offset=i]

        var code_offset = old_size * self.code_size
        for i in range(n_entry * self.code_size):
            self.lists[unsafe_offset=list_no].codes[unsafe_offset=code_offset + i] = codes_ptr[unsafe_offset=i]
            
        self.unlock_list(list_no)
