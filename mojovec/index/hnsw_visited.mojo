from std.memory.alloc import unsafe_alloc
from std.atomic import Atomic
from std.sys.info import num_logical_cores
from std.sys.intrinsics import prefetch, PrefetchOptions


struct VisitedTable(Movable):
    """A table to track visited nodes during HNSW graph traversal."""
    var marks: Pointer[UInt8, MutUntrackedOrigin]
    var current_mark: UInt8
    var capacity: Int

    def __init__(out self, capacity: Int):
        """Initializes a visited table with the specified capacity."""
        self.capacity = capacity
        self.marks = unsafe_alloc[UInt8](capacity)
        for i in range(capacity):
            self.marks[unsafe_offset=i] = 0
        self.current_mark = 1
        
    def __deinit__(deinit self):
        """Frees the underlying memory of the visited table."""
        if Int(self.marks) != 0:
            self.marks.unsafe_free()

    def __init__(out self, *, deinit move: Self):
        """Moves the visited table."""
        self.capacity = move.capacity
        self.current_mark = move.current_mark
        self.marks = move.marks
        
    def advance(mut self):
        """Advances the current mark, clearing the table if the mark overflows."""
        # Reserve zero for freshly allocated/cleared entries and avoid relying
        # on integer overflow semantics in checked builds.
        if self.current_mark == 254:
            for i in range(self.capacity):
                self.marks[unsafe_offset=i] = 0
            self.current_mark = 1
        else:
            self.current_mark += 1

    def grow(mut self, new_capacity: Int):
        """Grows the visited table to the specified capacity."""
        if new_capacity <= self.capacity:
            return
        var new_marks = unsafe_alloc[UInt8](new_capacity)
        for i in range(self.capacity):
            new_marks[unsafe_offset=i] = self.marks[unsafe_offset=i]
        for i in range(self.capacity, new_capacity):
            new_marks[unsafe_offset=i] = 0
        self.marks.unsafe_free()
        self.marks = new_marks
        self.capacity = new_capacity
        
    @always_inline
    def is_visited(self, node: Int) -> Bool:
        """Checks if a node has been visited during the current traversal."""
        return self.marks[unsafe_offset=node] == self.current_mark

    @always_inline
    def set_visited(self, node: Int):
        """Marks a node as visited during the current traversal."""
        self.marks[unsafe_offset=node] = self.current_mark

    @always_inline
    def prefetch(self, node: Int):
        """Prefetches the visited mark for a node to hide memory latency."""
        comptime opts = PrefetchOptions().for_read().low_locality().to_data_cache()
        prefetch[opts](self.marks.unsafe_offset(node))

struct VisitedTablePool(Movable):
    """A thread-safe pool of visited tables for concurrent HNSW search operations."""
    var capacity: Int
    var num_tables: Int
    var tables: Pointer[VisitedTable, MutUntrackedOrigin]
    var locks: Pointer[UInt32, MutUntrackedOrigin]
    
    def __init__(out self, capacity: Int, num_tables: Int = 0):
        """Initializes one visited table per logical CPU worker by default."""
        self.capacity = capacity
        self.num_tables = num_tables
        if self.num_tables <= 0:
            self.num_tables = num_logical_cores()
        if self.num_tables <= 0:
            self.num_tables = 1
        self.tables = unsafe_alloc[VisitedTable](self.num_tables)
        self.locks = unsafe_alloc[UInt32](self.num_tables)
        for i in range(self.num_tables):
            var t = VisitedTable(capacity)
            self.tables.unsafe_offset(i).unsafe_write(t^)
            self.locks[unsafe_offset=i] = 0

    def __deinit__(deinit self):
        """Frees the underlying memory of the pool and its tables."""
        if Int(self.tables) != 0:
            for i in range(self.num_tables):
                _ = self.tables.unsafe_offset(i).unsafe_take_pointee()
            self.tables.unsafe_free()
        if Int(self.locks) != 0:
            self.locks.unsafe_free()
            
    def __init__(out self, *, deinit move: Self):
        """Moves the pool."""
        self.capacity = move.capacity
        self.num_tables = move.num_tables
        self.tables = move.tables
        self.locks = move.locks

    def acquire(self) -> Int:
        """Acquires an available visited table from the pool, blocking until one is available."""
        # Spin until we find a free table
        while True:
            for i in range(self.num_tables):
                var ptr = self.locks.unsafe_offset(i)
                var expected: UInt32 = 0
                if Atomic.load(ptr) == 0:
                    if Atomic.compare_exchange(ptr, expected, 1):
                        return i
                    
    def release(self, id: Int):
        """Releases a visited table back to the pool."""
        var ptr = self.locks.unsafe_offset(id)
        Atomic.store(ptr, 0)

    def grow(mut self, new_capacity: Int):
        """Grows all tables in the pool to the specified capacity."""
        if new_capacity <= self.capacity:
            return
        for i in range(self.num_tables):
            self.tables[unsafe_offset=i].grow(new_capacity)
        self.capacity = new_capacity

    def get(self, id: Int) -> Pointer[VisitedTable, MutUntrackedOrigin]:
        """Retrieves a pointer to the visited table by its ID."""
        return self.tables.unsafe_offset(id)
