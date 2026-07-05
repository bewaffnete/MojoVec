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
