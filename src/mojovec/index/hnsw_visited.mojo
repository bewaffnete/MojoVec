struct VisitedTable(Movable):
    var marks: UnsafePointer[UInt16, MutUntrackedOrigin]
    var current_mark: UInt16
    var capacity: Int
    
    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.marks = alloc[UInt16](capacity)
        for i in range(capacity):
            self.marks[i] = 0
        self.current_mark = 1
        
    def __del__(deinit self):
        if Int(self.marks) != 0:
            self.marks.free()
            
    def __init__(out self, *, deinit move: Self):
        self.capacity = move.capacity
        self.current_mark = move.current_mark
        self.marks = move.marks
        
    def advance(mut self):
        self.current_mark += 1
        if self.current_mark == 0:
            for i in range(self.capacity):
                self.marks[i] = 0
            self.current_mark = 1
            
    def grow(mut self, new_capacity: Int):
        if new_capacity <= self.capacity:
            return
        var new_marks = alloc[UInt16](new_capacity)
        for i in range(self.capacity):
            new_marks[i] = self.marks[i]
        for i in range(self.capacity, new_capacity):
            new_marks[i] = 0
        self.marks.free()
        self.marks = new_marks
        self.capacity = new_capacity
        
    @always_inline
    def is_visited(self, node: Int) -> Bool:
        return self.marks[node] == self.current_mark
        
    @always_inline
    def set_visited(self, node: Int):
        self.marks[node] = self.current_mark

from std.atomic import Atomic

struct VisitedTablePool(Movable):
    var capacity: Int
    var num_tables: Int
    var tables: UnsafePointer[VisitedTable, MutUntrackedOrigin]
    var locks: UnsafePointer[UInt32, MutUntrackedOrigin]
    
    def __init__(out self, capacity: Int, num_tables: Int = 128):
        self.capacity = capacity
        self.num_tables = num_tables
        self.tables = alloc[VisitedTable](num_tables)
        self.locks = alloc[UInt32](num_tables)
        for i in range(num_tables):
            # initialize table
            var t = VisitedTable(capacity)
            (self.tables + i).init_pointee_move(t^)
            # initialize lock
            self.locks[i] = 0
            
    def __del__(deinit self):
        if Int(self.tables) != 0:
            for i in range(self.num_tables):
                _ = (self.tables + i).take_pointee()
            self.tables.free()
        if Int(self.locks) != 0:
            self.locks.free()
            
    def __init__(out self, *, deinit move: Self):
        self.capacity = move.capacity
        self.num_tables = move.num_tables
        self.tables = move.tables
        self.locks = move.locks
        
    def acquire(self) -> Int:
        # Spin until we find a free table
        while True:
            for i in range(self.num_tables):
                var ptr = self.locks + i
                var expected: UInt32 = 0
                if Atomic.load(ptr) == 0:
                    if Atomic.compare_exchange(ptr, expected, 1):
                        self.tables[i].advance()
                        return i
                    
    def release(self, id: Int):
        var ptr = self.locks + id
        Atomic.store(ptr, 0)
        
    def get(self, id: Int) -> UnsafePointer[VisitedTable, MutUntrackedOrigin]:
        return self.tables + id
