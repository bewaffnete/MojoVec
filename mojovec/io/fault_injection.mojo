comptime SnapshotFaultPoint = Int

comptime SNAPSHOT_FAULT_NONE: SnapshotFaultPoint = 0
comptime SNAPSHOT_FAULT_AFTER_HEADER: SnapshotFaultPoint = 1
comptime SNAPSHOT_FAULT_AFTER_PAYLOAD: SnapshotFaultPoint = 2
comptime SNAPSHOT_FAULT_BEFORE_PUBLISH: SnapshotFaultPoint = 3
comptime SNAPSHOT_FAULT_AFTER_PUBLISH: SnapshotFaultPoint = 4
comptime SNAPSHOT_FAULT_AFTER_CHECKPOINT_SNAPSHOT: SnapshotFaultPoint = 5

comptime BatchFaultPoint = Int

comptime BATCH_FAULT_NONE: BatchFaultPoint = 0
comptime BATCH_FAULT_AFTER_VECTOR_PREPARE: BatchFaultPoint = 1
comptime BATCH_FAULT_DURING_PAYLOAD_PREPARE: BatchFaultPoint = 2
comptime BATCH_FAULT_AFTER_PAYLOAD_PREPARE: BatchFaultPoint = 3
comptime BATCH_FAULT_AFTER_WAL_APPEND: BatchFaultPoint = 4


@always_inline
def inject_snapshot_fault(
    configured: SnapshotFaultPoint,
    current: SnapshotFaultPoint,
) raises:
    """Raises at one explicit durability boundary for deterministic tests."""
    if configured == current:
        raise Error("Injected snapshot persistence failure.")


@always_inline
def inject_batch_fault(
    configured: BatchFaultPoint,
    current: BatchFaultPoint,
) raises:
    """Raises at one explicit batch-write boundary for deterministic tests."""
    if configured == current:
        raise Error("Injected atomic batch failure.")
