from std.collections import List
from std.collections.span import Span
from std.memory.alloc import unsafe_alloc
from std.random.philox import Random
from std.math import sqrt
from std.testing import assert_true, assert_equal, assert_almost_equal, TestSuite
from max.algorithm import parallelize

from mojovec.core.types import (
    METRIC_L2,
    METRIC_INNER_PRODUCT,
    MODE_AUTO,
    MODE_SERIAL,
    MODE_RUNTIME,
    MODE_PTHREAD,
    set_hnsw_dispatch_mode,
    get_hnsw_dispatch_mode,
)
from mojovec.utils.distance_computer import StorageTrait
from mojovec.index.hnsw_visited import VisitedTablePool
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_flat_sq8 import IndexFlatSQ8
from mojovec.index.index_hnsw import IndexHNSW

def normalize_vectors(
    data: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    d: Int,
):
    """Normalizes n vectors of dimension d to unit L2 norm in-place."""
    for i in range(n):
        var norm_sq: Float32 = 0.0
        for j in range(d):
            var val = data[unsafe_offset=i * d + j]
            norm_sq += val * val
        var inv_norm: Float32 = 1.0 / sqrt(norm_sq) if norm_sq > 0.0 else 1.0
        for j in range(d):
            data[unsafe_offset=i * d + j] *= inv_norm

def test_hnsw_batch_size_matrix_boundary() raises:
    """Requirement 4.3: Batch size matrix testing, especially around transition point 256.

    Tests batch sizes: [1, 8, 32, 128, 255, 256, 257, 1024].
    Verifies that MODE_RUNTIME and MODE_PTHREAD (and MODE_SERIAL) produce
    identical labels and distances.
    """
    comptime dimension = 8
    comptime database_size = 128
    comptime max_batch = 1024
    comptime k = 4

    var storage = IndexFlat(dimension, METRIC_L2)
    var index = IndexHNSW[IndexFlat](storage^, dimension, METRIC_L2, M=8)

    var database = unsafe_alloc[Float32](database_size * dimension)
    for i in range(database_size):
        for j in range(dimension):
            database[unsafe_offset=i * dimension + j] = Float32(i * 3 + j) / 128.0
    index.add(
        Span[Float32](unsafe_ptr=database, length=database_size * dimension)
    )
    index.hnsw.efSearch = 16

    var all_queries = unsafe_alloc[Float32](max_batch * dimension)
    for i in range(max_batch):
        var src = (i * 11) % database_size
        for j in range(dimension):
            all_queries[unsafe_offset=i * dimension + j] = database[unsafe_offset=src * dimension + j]

    var empty_filter_storage = unsafe_alloc[UInt8](1)
    var empty_filter = Span[UInt8](unsafe_ptr=empty_filter_storage, length=0)

    # Batch sizes to test: 1, 8, 32, 128, 255, 256, 257, 1024
    var batch_sizes = [1, 8, 32, 128, 255, 256, 257, 1024]

    for b_idx in range(len(batch_sizes)):
        var batch = batch_sizes[b_idx]
        var q_span = Span[Float32](unsafe_ptr=all_queries, length=batch * dimension)

        var dist_s = unsafe_alloc[Float32](batch * k)
        var lbl_s = unsafe_alloc[Int](batch * k)
        var dist_r = unsafe_alloc[Float32](batch * k)
        var lbl_r = unsafe_alloc[Int](batch * k)
        var dist_p = unsafe_alloc[Float32](batch * k)
        var lbl_p = unsafe_alloc[Int](batch * k)
        var dist_a = unsafe_alloc[Float32](batch * k)
        var lbl_a = unsafe_alloc[Int](batch * k)

        var span_ds = Span[mut=True, Float32](unsafe_ptr=dist_s, length=batch * k)
        var span_ls = Span[mut=True, Int](unsafe_ptr=lbl_s, length=batch * k)
        var span_dr = Span[mut=True, Float32](unsafe_ptr=dist_r, length=batch * k)
        var span_lr = Span[mut=True, Int](unsafe_ptr=lbl_r, length=batch * k)
        var span_dp = Span[mut=True, Float32](unsafe_ptr=dist_p, length=batch * k)
        var span_lp = Span[mut=True, Int](unsafe_ptr=lbl_p, length=batch * k)
        var span_da = Span[mut=True, Float32](unsafe_ptr=dist_a, length=batch * k)
        var span_la = Span[mut=True, Int](unsafe_ptr=lbl_a, length=batch * k)

        # 1. Serial (baseline)
        set_hnsw_dispatch_mode(MODE_SERIAL)
        index.search(q_span, k, span_ds, span_ls, empty_filter)

        # 2. Runtime
        set_hnsw_dispatch_mode(MODE_RUNTIME)
        index.search(q_span, k, span_dr, span_lr, empty_filter)

        # 3. Pthread
        set_hnsw_dispatch_mode(MODE_PTHREAD)
        index.search(q_span, k, span_dp, span_lp, empty_filter)

        # 4. Auto
        set_hnsw_dispatch_mode(MODE_AUTO)
        index.search(q_span, k, span_da, span_la, empty_filter)

        # Compare all against Serial baseline
        for i in range(batch * k):
            assert_equal(lbl_r[unsafe_offset=i], lbl_s[unsafe_offset=i])
            assert_equal(lbl_p[unsafe_offset=i], lbl_s[unsafe_offset=i])
            assert_equal(lbl_a[unsafe_offset=i], lbl_s[unsafe_offset=i])
            assert_equal(dist_r[unsafe_offset=i], dist_s[unsafe_offset=i])
            assert_equal(dist_p[unsafe_offset=i], dist_s[unsafe_offset=i])
            assert_equal(dist_a[unsafe_offset=i], dist_s[unsafe_offset=i])

        dist_s.unsafe_free()
        lbl_s.unsafe_free()
        dist_r.unsafe_free()
        lbl_r.unsafe_free()
        dist_p.unsafe_free()
        lbl_p.unsafe_free()
        dist_a.unsafe_free()
        lbl_a.unsafe_free()

    database.unsafe_free()
    all_queries.unsafe_free()
    empty_filter_storage.unsafe_free()
    set_hnsw_dispatch_mode(MODE_AUTO)

def test_hnsw_concurrent_readers_single_collection() raises:
    """Requirement 4.3: Multiple concurrent threads calling search on the same collection.

    Verifies result integrity, lack of data races, and exact match with serial baseline.
    """
    comptime dimension = 8
    comptime database_size = 128
    comptime num_threads = 8
    comptime queries_per_thread = 32
    comptime total_queries = num_threads * queries_per_thread
    comptime k = 4

    var storage = IndexFlat(dimension, METRIC_L2)
    var index = IndexHNSW[IndexFlat](storage^, dimension, METRIC_L2, M=8)

    var database = unsafe_alloc[Float32](database_size * dimension)
    for i in range(database_size):
        for j in range(dimension):
            database[unsafe_offset=i * dimension + j] = Float32(i + j) / 128.0
    index.add(
        Span[Float32](unsafe_ptr=database, length=database_size * dimension)
    )
    index.hnsw.efSearch = 32

    var queries = unsafe_alloc[Float32](total_queries * dimension)
    for i in range(total_queries):
        var src = (i * 7) % database_size
        for j in range(dimension):
            queries[unsafe_offset=i * dimension + j] = database[unsafe_offset=src * dimension + j]

    var empty_filter_storage = unsafe_alloc[UInt8](1)
    var empty_filter = Span[UInt8](unsafe_ptr=empty_filter_storage, length=0)

    # Serial baseline
    var base_dist = unsafe_alloc[Float32](total_queries * k)
    var base_lbl = unsafe_alloc[Int](total_queries * k)
    var b_q_span = Span[Float32](unsafe_ptr=queries, length=total_queries * dimension)
    var b_d_span = Span[mut=True, Float32](unsafe_ptr=base_dist, length=total_queries * k)
    var b_l_span = Span[mut=True, Int](unsafe_ptr=base_lbl, length=total_queries * k)

    set_hnsw_dispatch_mode(MODE_SERIAL)
    index.search(b_q_span, k, b_d_span, b_l_span, empty_filter)

    # Concurrent search buffers
    var conc_dist = unsafe_alloc[Float32](total_queries * k)
    var conc_lbl = unsafe_alloc[Int](total_queries * k)
    var mismatch_count = unsafe_alloc[Int](num_threads)
    for i in range(num_threads):
        mismatch_count[unsafe_offset=i] = 0

    var index_ptr = Pointer(to=index)
    var queries_ptr = queries
    var conc_dist_ptr = conc_dist
    var conc_lbl_ptr = conc_lbl
    var ef_ptr = empty_filter_storage
    var base_lbl_ptr = base_lbl
    var base_dist_ptr = base_dist
    var mismatch_ptr = mismatch_count

    def reader_worker(tid: Int) {imm}:
        var q_sub = Span[Float32](
            unsafe_ptr=queries_ptr.unsafe_offset(tid * queries_per_thread * dimension),
            length=queries_per_thread * dimension,
        )
        var d_sub = Span[mut=True, Float32](
            unsafe_ptr=conc_dist_ptr.unsafe_offset(tid * queries_per_thread * k),
            length=queries_per_thread * k,
        )
        var l_sub = Span[mut=True, Int](
            unsafe_ptr=conc_lbl_ptr.unsafe_offset(tid * queries_per_thread * k),
            length=queries_per_thread * k,
        )
        var ef = Span[UInt8](unsafe_ptr=ef_ptr, length=0)
        index_ptr[].search(q_sub, k, d_sub, l_sub, ef)

        for qi in range(queries_per_thread * k):
            var offset = tid * queries_per_thread * k + qi
            if l_sub.unsafe_ptr()[unsafe_offset=qi] != base_lbl_ptr[unsafe_offset=offset]:
                mismatch_ptr[unsafe_offset=tid] += 1
            if d_sub.unsafe_ptr()[unsafe_offset=qi] != base_dist_ptr[unsafe_offset=offset]:
                mismatch_ptr[unsafe_offset=tid] += 1

    parallelize(reader_worker, num_threads, num_threads)

    for i in range(num_threads):
        assert_equal(mismatch_count[unsafe_offset=i], 0)

    database.unsafe_free()
    queries.unsafe_free()
    base_dist.unsafe_free()
    base_lbl.unsafe_free()
    conc_dist.unsafe_free()
    conc_lbl.unsafe_free()
    mismatch_count.unsafe_free()
    empty_filter_storage.unsafe_free()
    set_hnsw_dispatch_mode(MODE_AUTO)

def run_filter_and_metric_suite[StorageType: StorageTrait](
    mut index: IndexHNSW[StorageType],
    dimension: Int,
    database_size: Int,
    query_count: Int,
    k: Int,
    queries_ptr: Pointer[Float32, MutUntrackedOrigin],
) raises:
    var ep = index.hnsw.entry_point
    var query_span = Span[Float32](unsafe_ptr=queries_ptr, length=query_count * dimension)

    # 1. Filter: Exclude Entry Point
    var filter_ep = unsafe_alloc[UInt8](database_size)
    for i in range(database_size):
        filter_ep[unsafe_offset=i] = 1 if i == ep else 0
    var filter_ep_span = Span[UInt8](unsafe_ptr=filter_ep, length=database_size)

    # 2. Filter: Exclude Partial (odd IDs)
    var filter_odd = unsafe_alloc[UInt8](database_size)
    for i in range(database_size):
        filter_odd[unsafe_offset=i] = 1 if i % 2 != 0 else 0
    var filter_odd_span = Span[UInt8](unsafe_ptr=filter_odd, length=database_size)

    # 3. Filter: Exclude ALL
    var filter_all = unsafe_alloc[UInt8](database_size)
    for i in range(database_size):
        filter_all[unsafe_offset=i] = 1
    var filter_all_span = Span[UInt8](unsafe_ptr=filter_all, length=database_size)

    var dist_s = unsafe_alloc[Float32](query_count * k)
    var lbl_s = unsafe_alloc[Int](query_count * k)
    var dist_r = unsafe_alloc[Float32](query_count * k)
    var lbl_r = unsafe_alloc[Int](query_count * k)
    var dist_p = unsafe_alloc[Float32](query_count * k)
    var lbl_p = unsafe_alloc[Int](query_count * k)

    var s_ds = Span[mut=True, Float32](unsafe_ptr=dist_s, length=query_count * k)
    var s_ls = Span[mut=True, Int](unsafe_ptr=lbl_s, length=query_count * k)
    var s_dr = Span[mut=True, Float32](unsafe_ptr=dist_r, length=query_count * k)
    var s_lr = Span[mut=True, Int](unsafe_ptr=lbl_r, length=query_count * k)
    var s_dp = Span[mut=True, Float32](unsafe_ptr=dist_p, length=query_count * k)
    var s_lp = Span[mut=True, Int](unsafe_ptr=lbl_p, length=query_count * k)

    # Check 1: Exclude Entry Point
    set_hnsw_dispatch_mode(MODE_SERIAL)
    index.search(query_span, k, s_ds, s_ls, filter_ep_span)
    set_hnsw_dispatch_mode(MODE_RUNTIME)
    index.search(query_span, k, s_dr, s_lr, filter_ep_span)
    set_hnsw_dispatch_mode(MODE_PTHREAD)
    index.search(query_span, k, s_dp, s_lp, filter_ep_span)

    for i in range(query_count * k):
        assert_equal(lbl_r[unsafe_offset=i], lbl_s[unsafe_offset=i])
        assert_equal(lbl_p[unsafe_offset=i], lbl_s[unsafe_offset=i])
        assert_equal(dist_r[unsafe_offset=i], dist_s[unsafe_offset=i])
        assert_equal(dist_p[unsafe_offset=i], dist_s[unsafe_offset=i])
        assert_true(lbl_s[unsafe_offset=i] != ep)

    # Check 2: Exclude Odd candidates
    set_hnsw_dispatch_mode(MODE_SERIAL)
    index.search(query_span, k, s_ds, s_ls, filter_odd_span)
    set_hnsw_dispatch_mode(MODE_RUNTIME)
    index.search(query_span, k, s_dr, s_lr, filter_odd_span)
    set_hnsw_dispatch_mode(MODE_PTHREAD)
    index.search(query_span, k, s_dp, s_lp, filter_odd_span)

    for i in range(query_count * k):
        assert_equal(lbl_r[unsafe_offset=i], lbl_s[unsafe_offset=i])
        assert_equal(lbl_p[unsafe_offset=i], lbl_s[unsafe_offset=i])
        assert_equal(dist_r[unsafe_offset=i], dist_s[unsafe_offset=i])
        assert_equal(dist_p[unsafe_offset=i], dist_s[unsafe_offset=i])
        if lbl_s[unsafe_offset=i] != -1:
            assert_equal(lbl_s[unsafe_offset=i] % 2, 0)

    # Check 3: Exclude All candidates
    set_hnsw_dispatch_mode(MODE_SERIAL)
    index.search(query_span, k, s_ds, s_ls, filter_all_span)
    set_hnsw_dispatch_mode(MODE_RUNTIME)
    index.search(query_span, k, s_dr, s_lr, filter_all_span)
    set_hnsw_dispatch_mode(MODE_PTHREAD)
    index.search(query_span, k, s_dp, s_lp, filter_all_span)

    for i in range(query_count * k):
        assert_equal(lbl_s[unsafe_offset=i], -1)
        assert_equal(lbl_r[unsafe_offset=i], -1)
        assert_equal(lbl_p[unsafe_offset=i], -1)

    filter_ep.unsafe_free()
    filter_odd.unsafe_free()
    filter_all.unsafe_free()
    dist_s.unsafe_free()
    lbl_s.unsafe_free()
    dist_r.unsafe_free()
    lbl_r.unsafe_free()
    dist_p.unsafe_free()
    lbl_p.unsafe_free()

def test_hnsw_filters_and_metrics_flat_sq8() raises:
    """Requirement 4.3: Filters and metrics (L2, Cosine, IP) on Flat and SQ8."""
    comptime dimension = 8
    comptime database_size = 48
    comptime query_count = 16
    comptime k = 4

    var database = unsafe_alloc[Float32](database_size * dimension)
    for i in range(database_size):
        for j in range(dimension):
            database[unsafe_offset=i * dimension + j] = Float32(i + j + 1) / 48.0
    var db_span = Span[Float32](unsafe_ptr=database, length=database_size * dimension)

    var queries = unsafe_alloc[Float32](query_count * dimension)
    for i in range(query_count):
        for j in range(dimension):
            queries[unsafe_offset=i * dimension + j] = database[unsafe_offset=i * dimension + j]

    # 1. Flat L2
    var st_flat_l2 = IndexFlat(dimension, METRIC_L2)
    var idx_flat_l2 = IndexHNSW[IndexFlat](st_flat_l2^, dimension, METRIC_L2, M=8)
    idx_flat_l2.add(db_span)
    idx_flat_l2.hnsw.efSearch = 16
    run_filter_and_metric_suite(idx_flat_l2, dimension, database_size, query_count, k, queries)

    # 2. Flat IP
    var st_flat_ip = IndexFlat(dimension, METRIC_INNER_PRODUCT)
    var idx_flat_ip = IndexHNSW[IndexFlat](st_flat_ip^, dimension, METRIC_INNER_PRODUCT, M=8)
    idx_flat_ip.add(db_span)
    idx_flat_ip.hnsw.efSearch = 16
    run_filter_and_metric_suite(idx_flat_ip, dimension, database_size, query_count, k, queries)

    # 3. Flat Cosine (Normalized vectors + IP)
    var norm_db = unsafe_alloc[Float32](database_size * dimension)
    for i in range(database_size * dimension):
        norm_db[unsafe_offset=i] = database[unsafe_offset=i]
    normalize_vectors(norm_db, database_size, dimension)

    var norm_queries = unsafe_alloc[Float32](query_count * dimension)
    for i in range(query_count * dimension):
        norm_queries[unsafe_offset=i] = queries[unsafe_offset=i]
    normalize_vectors(norm_queries, query_count, dimension)

    var st_flat_cos = IndexFlat(dimension, METRIC_INNER_PRODUCT)
    var idx_flat_cos = IndexHNSW[IndexFlat](st_flat_cos^, dimension, METRIC_INNER_PRODUCT, M=8)
    idx_flat_cos.add(Span[Float32](unsafe_ptr=norm_db, length=database_size * dimension))
    idx_flat_cos.hnsw.efSearch = 16
    run_filter_and_metric_suite(idx_flat_cos, dimension, database_size, query_count, k, norm_queries)

    # 4. SQ8 L2
    var st_sq8_l2 = IndexFlatSQ8(dimension, METRIC_L2)
    var idx_sq8_l2 = IndexHNSW[IndexFlatSQ8](st_sq8_l2^, dimension, METRIC_L2, M=8)
    idx_sq8_l2.add(db_span)
    idx_sq8_l2.hnsw.efSearch = 16
    run_filter_and_metric_suite(idx_sq8_l2, dimension, database_size, query_count, k, queries)

    # 5. SQ8 IP
    var st_sq8_ip = IndexFlatSQ8(dimension, METRIC_INNER_PRODUCT)
    var idx_sq8_ip = IndexHNSW[IndexFlatSQ8](st_sq8_ip^, dimension, METRIC_INNER_PRODUCT, M=8)
    idx_sq8_ip.add(db_span)
    idx_sq8_ip.hnsw.efSearch = 16
    run_filter_and_metric_suite(idx_sq8_ip, dimension, database_size, query_count, k, queries)

    # 6. SQ8 Cosine
    var st_sq8_cos = IndexFlatSQ8(dimension, METRIC_INNER_PRODUCT)
    var idx_sq8_cos = IndexHNSW[IndexFlatSQ8](st_sq8_cos^, dimension, METRIC_INNER_PRODUCT, M=8)
    idx_sq8_cos.add(Span[Float32](unsafe_ptr=norm_db, length=database_size * dimension))
    idx_sq8_cos.hnsw.efSearch = 16
    run_filter_and_metric_suite(idx_sq8_cos, dimension, database_size, query_count, k, norm_queries)

    database.unsafe_free()
    queries.unsafe_free()
    norm_db.unsafe_free()
    norm_queries.unsafe_free()
    set_hnsw_dispatch_mode(MODE_AUTO)

def test_hnsw_visited_table_pool_starvation() raises:
    """Requirement 4.3: Behavior when concurrent workers exceed visited table pool capacity."""
    # Create VisitedTablePool with only 2 tables, run 16 concurrent workers
    var pool = VisitedTablePool(64, 2)
    var pool_ptr = Pointer(to=pool)

    comptime num_workers = 16
    comptime iterations_per_worker = 40

    var counter = unsafe_alloc[Int](num_workers)
    var errors = unsafe_alloc[Int](num_workers)
    for i in range(num_workers):
        counter[unsafe_offset=i] = 0
        errors[unsafe_offset=i] = 0
    var counter_ptr = counter
    var errors_ptr = errors

    def starvation_worker(wid: Int) {imm}:
        for _ in range(iterations_per_worker):
            var table_id = pool_ptr[].acquire()
            var table = pool_ptr[].get(table_id)
            table[].set_visited(wid)
            if not table[].is_visited(wid):
                errors_ptr[unsafe_offset=wid] += 1
            table[].advance()
            counter_ptr[unsafe_offset=wid] += 1
            pool_ptr[].release(table_id)

    parallelize(starvation_worker, num_workers, num_workers)

    for i in range(num_workers):
        assert_equal(errors[unsafe_offset=i], 0)
        assert_equal(counter[unsafe_offset=i], iterations_per_worker)

    counter.unsafe_free()
    errors.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
