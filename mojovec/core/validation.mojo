"""Shared validation for allocation-sensitive index parameters."""


comptime MAX_VECTOR_DIMENSION = 65_536
comptime MIN_HNSW_M = 2
comptime MAX_HNSW_M = 1_000
comptime MAX_HNSW_EF = 2_048


@always_inline
def _validate_vector_dimension(dimension: Int) raises:
    if dimension <= 0 or dimension > MAX_VECTOR_DIMENSION:
        raise Error("dimension must be between 1 and 65536.")


@always_inline
def _validate_hnsw_parameters(
    M: Int,
    ef_construction: Int,
    ef_search: Int,
) raises:
    if M < MIN_HNSW_M or M > MAX_HNSW_M:
        raise Error("M must be between 2 and 1000.")
    if ef_construction <= 0 or ef_construction > MAX_HNSW_EF:
        raise Error("ef_construction must be between 1 and 2048.")
    if ef_search <= 0 or ef_search > MAX_HNSW_EF:
        raise Error("ef_search must be between 1 and 2048.")
