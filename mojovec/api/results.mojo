from std.collections import List

@fieldwise_init
struct QueryResults(Movable, Copyable):
    var ids: List[List[Int]]
    var distances: List[List[Float32]]
