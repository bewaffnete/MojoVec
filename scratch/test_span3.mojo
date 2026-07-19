from std.memory.span import Span

def foo(x: Span[UInt8] = Span[UInt8]()):
    print(len(x))

def main():
    foo()
