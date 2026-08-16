from std.collections import Dict, List
from std.math import log

from mojovec.api.text_analyzer import StandardBM25Analyzer


comptime BM25_DEFAULT_K1 = 1.2
comptime BM25_DEFAULT_B = 0.75


def _bm25_is_worse(
    score: Float64,
    internal_id: Int,
    other_score: Float64,
    other_internal_id: Int,
) -> Bool:
    return score < other_score or (
        score == other_score and internal_id > other_internal_id
    )


def _bm25_swap(
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


def _bm25_sift_up(
    mut ids: List[Int],
    mut scores: List[Float64],
    start: Int,
):
    """Maintains a heap whose root is the worst retained candidate."""
    var child = start
    while child > 0:
        var parent = (child - 1) // 2
        if not _bm25_is_worse(
            scores[child],
            ids[child],
            scores[parent],
            ids[parent],
        ):
            break
        _bm25_swap(ids, scores, child, parent)
        child = parent


def _bm25_sift_down(
    mut ids: List[Int],
    mut scores: List[Float64],
    heap_size: Int,
):
    """Restores the worst-candidate root after replacing it."""
    var parent = 0
    while True:
        var left = parent * 2 + 1
        if left >= heap_size:
            return
        var right = left + 1
        var worse_child = left
        if right < heap_size and _bm25_is_worse(
            scores[right],
            ids[right],
            scores[left],
            ids[left],
        ):
            worse_child = right
        if not _bm25_is_worse(
            scores[worse_child],
            ids[worse_child],
            scores[parent],
            ids[parent],
        ):
            return
        _bm25_swap(ids, scores, parent, worse_child)
        parent = worse_child


struct BM25Posting(Movable):
    """Internal IDs and term frequencies for one token."""

    var internal_ids: List[Int]
    var frequencies: List[Int]

    def __init__(out self):
        self.internal_ids = List[Int]()
        self.frequencies = List[Int]()

    def __init__(out self, *, deinit move: Self):
        self.internal_ids = move.internal_ids^
        self.frequencies = move.frequencies^

    def add(mut self, internal_id: Int, frequency: Int):
        self.internal_ids.append(internal_id)
        self.frequencies.append(frequency)


@fieldwise_init
struct BM25SearchResult(Movable):
    """One padded row of internal IDs and BM25 scores."""

    var internal_ids: List[Int]
    var scores: List[Float32]


struct BM25Index(Movable):
    """
    Incremental BM25 index over collection documents.

    Postings are append-only, matching the collection's versioned record
    storage. Updates and deletes only deactivate old internal IDs. Compaction
    naturally rebuilds postings from active documents.
    """

    var _term_lookup: Dict[String, Int]
    var _postings: List[BM25Posting]
    var _document_lengths: List[Int]
    var _active: List[UInt8]
    var _active_document_count: Int
    var _active_token_count: Int
    var _k1: Float64
    var _b: Float64
    var _analyzer: StandardBM25Analyzer

    def __init__(
        out self,
        k1: Float64 = BM25_DEFAULT_K1,
        b: Float64 = BM25_DEFAULT_B,
    ):
        self._term_lookup = Dict[String, Int]()
        self._postings = List[BM25Posting]()
        self._document_lengths = List[Int]()
        self._active = List[UInt8]()
        self._active_document_count = 0
        self._active_token_count = 0
        self._k1 = k1
        self._b = b
        self._analyzer = StandardBM25Analyzer()

    def __init__(out self, *, deinit move: Self):
        self._term_lookup = move._term_lookup^
        self._postings = move._postings^
        self._document_lengths = move._document_lengths^
        self._active = move._active^
        self._active_document_count = move._active_document_count
        self._active_token_count = move._active_token_count
        self._k1 = move._k1
        self._b = move._b
        self._analyzer = move._analyzer^

    def active_document_count(self) -> Int:
        return self._active_document_count

    def _ensure_slot(mut self, internal_id: Int):
        while len(self._document_lengths) <= internal_id:
            self._document_lengths.append(0)
            self._active.append(0)

    def _prepare_document(self, document: String) raises -> List[String]:
        """Performs all fallible document analysis before a batch commit."""
        return self._analyzer.analyze(document)

    def _add_prepared(mut self, internal_id: Int, tokens: List[String]):
        """Adds pre-tokenized content without introducing an error boundary."""
        self._ensure_slot(internal_id)
        if len(tokens) == 0:
            return

        var frequencies = Dict[String, Int]()
        for token in tokens:
            if token in frequencies:
                # The membership check and lookup share one local Dict with no
                # intervening mutation from another writer.
                try:
                    frequencies[token] += 1
                except:
                    continue
            else:
                frequencies[token] = 1

        for token_ref in frequencies:
            var token = token_ref.copy()
            var frequency: Int
            try:
                frequency = frequencies[token]
            except:
                continue
            var posting_index: Int
            if token in self._term_lookup:
                try:
                    posting_index = self._term_lookup[token]
                except:
                    continue
            else:
                posting_index = len(self._postings)
                self._term_lookup[token] = posting_index
                self._postings.append(BM25Posting())
            self._postings[posting_index].add(internal_id, frequency)

        self._document_lengths[internal_id] = len(tokens)
        self._active[internal_id] = 1
        self._active_document_count += 1
        self._active_token_count += len(tokens)

    def add(mut self, internal_id: Int, document: String) raises:
        var tokens = self._prepare_document(document)
        self._add_prepared(internal_id, tokens)

    def deactivate(mut self, internal_id: Int):
        if (
            internal_id < 0
            or internal_id >= len(self._active)
            or self._active[internal_id] == 0
        ):
            return
        self._active[internal_id] = 0
        self._active_document_count -= 1
        self._active_token_count -= self._document_lengths[internal_id]

    def _is_active(self, internal_id: Int) -> Bool:
        return (
            internal_id >= 0
            and internal_id < len(self._active)
            and self._active[internal_id] > 0
        )

    def _is_excluded(self, internal_id: Int, exclusion: List[UInt8]) -> Bool:
        return len(exclusion) > 0 and (
            internal_id >= len(exclusion) or exclusion[internal_id] > 0
        )

    def _empty_result(self, n_results: Int) -> BM25SearchResult:
        var ids = List[Int](capacity=n_results)
        var scores = List[Float32](capacity=n_results)
        for _ in range(n_results):
            ids.append(-1)
            scores.append(0.0)
        return BM25SearchResult(ids^, scores^)

    def search(
        self,
        query: String,
        n_results: Int,
        exclusion: List[UInt8],
    ) raises -> BM25SearchResult:
        """
        Returns the highest-scoring active internal IDs, padded to n_results.

        The optional exclusion byte mask filters candidates without changing
        corpus-wide IDF or average-document-length statistics.
        """
        if n_results <= 0:
            raise Error("n_results must be positive.")
        if len(exclusion) > 0 and len(exclusion) < len(self._active):
            raise Error("BM25 exclusion mask is smaller than the index.")
        if self._active_document_count == 0:
            return self._empty_result(n_results)

        var query_tokens = self._analyzer.analyze(query)
        if len(query_tokens) == 0:
            return self._empty_result(n_results)

        var unique_terms = Dict[String, Bool]()
        for token in query_tokens:
            unique_terms[token] = True

        var accumulated = Dict[Int, Float64]()
        var document_count = Float64(self._active_document_count)
        var average_length = Float64(self._active_token_count) / document_count

        for term in unique_terms:
            if term not in self._term_lookup:
                continue
            var posting_index = self._term_lookup[term]
            var document_frequency = 0
            for index in range(len(self._postings[posting_index].internal_ids)):
                if self._is_active(
                    self._postings[posting_index].internal_ids[index]
                ):
                    document_frequency += 1
            if document_frequency == 0:
                continue

            var idf = log(
                1.0
                + (document_count - Float64(document_frequency) + 0.5)
                / (Float64(document_frequency) + 0.5)
            )

            for index in range(len(self._postings[posting_index].internal_ids)):
                var internal_id = self._postings[posting_index].internal_ids[
                    index
                ]
                if not self._is_active(internal_id) or self._is_excluded(
                    internal_id, exclusion
                ):
                    continue
                var term_frequency = Float64(
                    self._postings[posting_index].frequencies[index]
                )
                var document_length = Float64(
                    self._document_lengths[internal_id]
                )
                var normalization = (
                    1.0 - self._b + self._b * document_length / average_length
                )
                var contribution = (
                    idf
                    * term_frequency
                    * (self._k1 + 1.0)
                    / (term_frequency + self._k1 * normalization)
                )
                if internal_id in accumulated:
                    accumulated[internal_id] += contribution
                else:
                    accumulated[internal_id] = contribution

        if len(accumulated) == 0:
            return self._empty_result(n_results)

        # Keep only top-k candidates in a managed min-heap. The root is the
        # worst retained candidate, making selection O(matches * log(k)).
        var top_ids = List[Int](capacity=n_results)
        var top_scores = List[Float64](capacity=n_results)
        for internal_id_ref in accumulated:
            var internal_id = Int(internal_id_ref)
            var score = accumulated[internal_id]
            if len(top_ids) < n_results:
                top_ids.append(internal_id)
                top_scores.append(score)
                _bm25_sift_up(
                    top_ids,
                    top_scores,
                    len(top_ids) - 1,
                )
                continue

            if _bm25_is_worse(
                top_scores[0],
                top_ids[0],
                score,
                internal_id,
            ):
                top_ids[0] = internal_id
                top_scores[0] = score
                _bm25_sift_down(
                    top_ids, top_scores, len(top_ids)
                )

        # The root is the worst candidate. In-place heapsort therefore emits
        # descending scores with ascending internal IDs in O(k log(k)).
        var heap_size = len(top_ids)
        while heap_size > 1:
            _bm25_swap(top_ids, top_scores, 0, heap_size - 1)
            heap_size -= 1
            _bm25_sift_down(top_ids, top_scores, heap_size)

        var result_ids = List[Int](capacity=n_results)
        var result_scores = List[Float32](capacity=n_results)
        for index in range(len(top_ids)):
            result_ids.append(top_ids[index])
            result_scores.append(Float32(top_scores[index]))
        while len(result_ids) < n_results:
            result_ids.append(-1)
            result_scores.append(0.0)
        return BM25SearchResult(result_ids^, result_scores^)
