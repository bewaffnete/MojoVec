from std.collections import Dict, List


comptime RRF_DEFAULT_K = 60
comptime RRF_DEFAULT_CANDIDATE_MULTIPLIER = 4


def _rrf_is_worse(
    score: Float64,
    internal_id: Int,
    other_score: Float64,
    other_internal_id: Int,
) -> Bool:
    return score < other_score or (
        score == other_score and internal_id > other_internal_id
    )


def _rrf_swap(
    mut ids: List[Int],
    mut scores: List[Float64],
    left: Int,
    right: Int,
):
    var swapped_id = ids[left]
    ids[left] = ids[right]
    ids[right] = swapped_id
    var swapped_score = scores[left]
    scores[left] = scores[right]
    scores[right] = swapped_score


def _rrf_sift_up(
    mut ids: List[Int],
    mut scores: List[Float64],
    start: Int,
):
    """Maintains a heap whose root is the worst retained candidate."""
    var child = start
    while child > 0:
        var parent = (child - 1) // 2
        if not _rrf_is_worse(
            scores[child],
            ids[child],
            scores[parent],
            ids[parent],
        ):
            break
        _rrf_swap(ids, scores, child, parent)
        child = parent


def _rrf_sift_down(
    mut ids: List[Int],
    mut scores: List[Float64],
    heap_size: Int,
):
    var parent = 0
    while True:
        var left = parent * 2 + 1
        if left >= heap_size:
            return
        var right = left + 1
        var worse_child = left
        if right < heap_size and _rrf_is_worse(
            scores[right],
            ids[right],
            scores[left],
            ids[left],
        ):
            worse_child = right
        if not _rrf_is_worse(
            scores[worse_child],
            ids[worse_child],
            scores[parent],
            ids[parent],
        ):
            return
        _rrf_swap(ids, scores, parent, worse_child)
        parent = worse_child


@fieldwise_init
struct RRFSearchResult(Movable):
    """One padded row of internal IDs and reciprocal-rank-fusion scores."""

    var internal_ids: List[Int]
    var scores: List[Float32]


def reciprocal_rank_fusion(
    vector_ids: List[Int],
    text_ids: List[Int],
    n_results: Int,
    rrf_k: Int = RRF_DEFAULT_K,
) raises -> RRFSearchResult:
    """
    Fuses two ranked internal-ID lists using equal-weight standard RRF.

    Each source contributes `1 / (rrf_k + rank)` with one-based ranks.
    Invalid padded IDs are ignored. Source rankings are expected to contain
    unique IDs, as MojoVec's HNSW and BM25 candidate paths do.
    """
    if n_results <= 0:
        raise Error("n_results must be positive.")
    if rrf_k <= 0:
        raise Error("rrf_k must be positive.")

    var accumulated = Dict[Int, Float64]()
    for index in range(len(vector_ids)):
        var internal_id = vector_ids[index]
        if internal_id < 0:
            continue
        accumulated[internal_id] = (
            1.0 / Float64(rrf_k + index + 1)
        )

    for index in range(len(text_ids)):
        var internal_id = text_ids[index]
        if internal_id < 0:
            continue
        var contribution = 1.0 / Float64(rrf_k + index + 1)
        if internal_id in accumulated:
            accumulated[internal_id] += contribution
        else:
            accumulated[internal_id] = contribution

    var top_ids = List[Int](capacity=n_results)
    var top_scores = List[Float64](capacity=n_results)
    for internal_id_ref in accumulated:
        var internal_id = Int(internal_id_ref)
        var score = accumulated[internal_id]
        if len(top_ids) < n_results:
            top_ids.append(internal_id)
            top_scores.append(score)
            _rrf_sift_up(top_ids, top_scores, len(top_ids) - 1)
            continue
        if _rrf_is_worse(
            top_scores[0],
            top_ids[0],
            score,
            internal_id,
        ):
            top_ids[0] = internal_id
            top_scores[0] = score
            _rrf_sift_down(top_ids, top_scores, len(top_ids))

    # In-place heapsort is O(k log(k)). Because the root is the worst retained
    # candidate, moving it to the shrinking tail produces descending scores
    # and ascending internal IDs for deterministic ties.
    var heap_size = len(top_ids)
    while heap_size > 1:
        _rrf_swap(top_ids, top_scores, 0, heap_size - 1)
        heap_size -= 1
        _rrf_sift_down(top_ids, top_scores, heap_size)

    var result_ids = List[Int](capacity=n_results)
    var result_scores = List[Float32](capacity=n_results)
    for index in range(len(top_ids)):
        result_ids.append(top_ids[index])
        result_scores.append(Float32(top_scores[index]))
    while len(result_ids) < n_results:
        result_ids.append(-1)
        result_scores.append(0.0)
    return RRFSearchResult(result_ids^, result_scores^)
