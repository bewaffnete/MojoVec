from std.memory.span import Span
from std.memory import alloc

def main():
    var dummy_ptr = alloc[UInt8](1)
    var empty_filter = Span[UInt8, _](ptr=dummy_ptr, length=0)
    print(empty_filter.length)
    dummy_ptr.free()
