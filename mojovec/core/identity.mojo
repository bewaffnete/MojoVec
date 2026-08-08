"""Operating-system-backed persistent collection identity generation."""

from std.ffi import external_call


def _random_collection_identity() raises -> UInt64:
    """Returns a non-zero 64-bit identity from the operating-system CSPRNG."""
    var storage = InlineArray[UInt64, 1](uninitialized=True)
    for _ in range(4):
        if external_call["getentropy", Int32](
            storage.unsafe_ptr(), 8
        ) == 0 and storage[0] != 0:
            return storage[0]
    raise Error("Unable to generate a random Collection identity.")
