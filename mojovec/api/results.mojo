from std.collections import List

@fieldwise_init
struct QueryResults(Movable, Copyable):
    """
    Contains the IDs and distances resulting from a vector query.
    """
    var ids: List[List[Int]]
    var distances: List[List[Float32]]
