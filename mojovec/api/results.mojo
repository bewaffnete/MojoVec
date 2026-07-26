from std.collections import List


@fieldwise_init
struct QueryResults(Movable, Copyable):
    """
    Contains the IDs and distances resulting from a vector query.
    """
    var ids: List[List[Int]]
    var distances: List[List[Float32]]


@fieldwise_init
struct CollectionStats(Copyable, Movable, Writable):
    """A snapshot of collection storage and HNSW configuration."""

    var active_count: Int
    var deleted_count: Int
    var total_count: Int
    var deleted_ratio: Float64
    var dimension: Int
    var quantized: Bool
    var M: Int
    var ef_construction: Int
    var ef_search: Int

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "CollectionStats(active_count=",
            self.active_count,
            ", deleted_count=",
            self.deleted_count,
            ", total_count=",
            self.total_count,
            ", deleted_ratio=",
            self.deleted_ratio,
            ", dimension=",
            self.dimension,
            ", quantized=",
            self.quantized,
            ", M=",
            self.M,
            ", ef_construction=",
            self.ef_construction,
            ", ef_search=",
            self.ef_search,
            ")",
        )


@fieldwise_init
struct CompactReport(Copyable, Movable, Writable):
    """Describes whether compaction ran and how much garbage it reclaimed."""

    var performed: Bool
    var before: CollectionStats
    var after: CollectionStats
    var reclaimed_records: Int
    var elapsed_seconds: Float64

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "CompactReport(performed=",
            self.performed,
            ", reclaimed_records=",
            self.reclaimed_records,
            ", elapsed_seconds=",
            self.elapsed_seconds,
            ", before=",
            self.before,
            ", after=",
            self.after,
            ")",
        )
