from std.memory.span import Span

def main():
    var x: Span[UInt8, _]
    print(x.length)
