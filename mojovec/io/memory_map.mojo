from std.ffi import external_call


comptime MMAP_PROT_READ = 1
comptime MMAP_PRIVATE = 2
comptime MMAP_ADVICE_RANDOM = 1
comptime MMAP_FAILED = -1


struct FileMemoryMap(Movable):
    """Owns one read-only private mapping of a complete index file."""

    var address: Int
    var size: Int

    def __init__(out self):
        self.address = 0
        self.size = 0

    def __init__(out self, *, deinit take: Self):
        self.address = take.address
        self.size = take.size

    def __del__(deinit self):
        if self.address != 0:
            _ = external_call["munmap", Int](self.address, self.size)

    @staticmethod
    def map_read_only(file_descriptor: Int, size: Int) raises -> Self:
        if size <= 0:
            raise Error("Cannot memory-map an empty index file.")
        var address = external_call["mmap", Int](
            Int(0),
            size,
            MMAP_PROT_READ,
            MMAP_PRIVATE,
            file_descriptor,
            Int(0),
        )
        if address == MMAP_FAILED:
            raise Error("mmap failed while loading the index.")
        # HNSW performs sparse, data-dependent reads. The hint is optional and
        # failure does not affect correctness.
        _ = external_call["madvise", Int](
            address,
            size,
            MMAP_ADVICE_RANDOM,
        )
        return Self(address=address, size=size)

    def __init__(out self, *, address: Int, size: Int):
        self.address = address
        self.size = size

    @always_inline
    def is_active(self) -> Bool:
        return self.address != 0

    def close(mut self):
        if self.address == 0:
            return
        _ = external_call["munmap", Int](self.address, self.size)
        self.address = 0
        self.size = 0
