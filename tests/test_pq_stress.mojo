from std.collections.span import Span
from std.memory.alloc import unsafe_alloc
from std.random.philox import Random
from std.time import perf_counter_ns
from std.ffi import external_call
from std.testing import assert_true, assert_equal, assert_almost_equal, TestSuite
from max.algorithm import parallelize

from mojovec.core.types import (
    MODE_AUTO,
    MODE_SERIAL,
    MODE_RUNTIME,
    MODE_PTHREAD,
    set_pq_dispatch_mode,
    get_pq_dispatch_mode,
)
from mojovec.quantization.pq import ProductQuantizer

def get_maxrss() -> Int64:
    """Returns max resident set size in bytes on macOS."""
    var usage = unsafe_alloc[Int64](18)
    for i in range(18):
        usage[unsafe_offset=i] = 0
    _ = external_call["getrusage", Int32](Int32(0), usage)
    var rss = usage[unsafe_offset=4]
    usage.unsafe_free()
    return rss

def compute_reconstruction_mse(
    orig: Span[Float32, _],
    recon: Span[Float32, _],
) -> Float32:
    var total_diff: Float32 = 0.0
    var n = len(orig)
    var orig_ptr = orig.unsafe_ptr()
    var recon_ptr = recon.unsafe_ptr()
    for i in range(n):
        var d = orig_ptr[unsafe_offset=i] - recon_ptr[unsafe_offset=i]
        total_diff += d * d
    return total_diff / Float32(n)

def test_pq_100_repetitions_memory_and_exact_match() raises:
    """Requirement 4.2: 100+ iterations (create -> train -> encode -> decode -> destroy).

    Verifies bit-exact centroids, codes, and reconstruction errors across
    MODE_SERIAL, MODE_RUNTIME, and MODE_PTHREAD, as well as absence of memory leaks.
    """
    comptime d = 16
    comptime n = 64
    comptime M = 4
    comptime ksub = 16
    comptime dsub = d // M

    var generator = Random(seed=UInt64(10042))
    var data = unsafe_alloc[Float32](n * d)
    var values = generator.step_uniform()
    for i in range(n * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        data[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0

    var data_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=data, length=n * d
    )

    var codes_s = unsafe_alloc[UInt8](n * M)
    var codes_r = unsafe_alloc[UInt8](n * M)
    var codes_p = unsafe_alloc[UInt8](n * M)
    var recon_s = unsafe_alloc[Float32](n * d)
    var recon_r = unsafe_alloc[Float32](n * d)
    var recon_p = unsafe_alloc[Float32](n * d)

    var span_cs = Span[mut=True, UInt8](unsafe_ptr=codes_s, length=n * M)
    var span_cr = Span[mut=True, UInt8](unsafe_ptr=codes_r, length=n * M)
    var span_cp = Span[mut=True, UInt8](unsafe_ptr=codes_p, length=n * M)
    var span_rs = Span[mut=True, Float32](unsafe_ptr=recon_s, length=n * d)
    var span_rr = Span[mut=True, Float32](unsafe_ptr=recon_r, length=n * d)
    var span_rp = Span[mut=True, Float32](unsafe_ptr=recon_p, length=n * d)

    # Warmup allocation before tracking memory
    set_pq_dispatch_mode(MODE_SERIAL)
    var pq_warmup = ProductQuantizer(d, M, ksub)
    pq_warmup.train(data_span)
    pq_warmup.compute_codes(data_span, span_cs)
    pq_warmup.decode(span_cs, span_rs)

    var initial_rss = get_maxrss()

    # 105 iterations loop
    for _ in range(105):
        # 1. SERIAL
        set_pq_dispatch_mode(MODE_SERIAL)
        var pqs = ProductQuantizer(d, M, ksub)
        pqs.train(data_span)
        pqs.compute_codes(data_span, span_cs)
        pqs.decode(span_cs, span_rs)
        var err_s = compute_reconstruction_mse(data_span, span_rs)

        # 2. RUNTIME
        set_pq_dispatch_mode(MODE_RUNTIME)
        var pqr = ProductQuantizer(d, M, ksub)
        pqr.train(data_span)
        pqr.compute_codes(data_span, span_cr)
        pqr.decode(span_cr, span_rr)
        var err_r = compute_reconstruction_mse(data_span, span_rr)

        # 3. PTHREAD
        set_pq_dispatch_mode(MODE_PTHREAD)
        var pqp = ProductQuantizer(d, M, ksub)
        pqp.train(data_span)
        pqp.compute_codes(data_span, span_cp)
        pqp.decode(span_cp, span_rp)
        var err_p = compute_reconstruction_mse(data_span, span_rp)

        # Verify bit-exact centroids
        var total_centroids = M * ksub * dsub
        for c in range(total_centroids):
            assert_equal(pqs.centroids[c], pqr.centroids[c])
            assert_equal(pqs.centroids[c], pqp.centroids[c])

        # Verify bit-exact codes
        for c in range(n * M):
            assert_equal(codes_s[unsafe_offset=c], codes_r[unsafe_offset=c])
            assert_equal(codes_s[unsafe_offset=c], codes_p[unsafe_offset=c])

        # Verify bit-exact decoded vectors
        for c in range(n * d):
            assert_equal(recon_s[unsafe_offset=c], recon_r[unsafe_offset=c])
            assert_equal(recon_s[unsafe_offset=c], recon_p[unsafe_offset=c])

        # Verify reconstruction error
        assert_almost_equal(err_s, err_r, atol=1e-6)
        assert_almost_equal(err_s, err_p, atol=1e-6)

    var final_rss = get_maxrss()
    # Memory growth across 105 iterations should be 0 or bounded by OS pages (< 4MB)
    var rss_diff = final_rss - initial_rss
    assert_true(rss_diff < 4 * 1024 * 1024)

    data.unsafe_free()
    codes_s.unsafe_free()
    codes_r.unsafe_free()
    codes_p.unsafe_free()
    recon_s.unsafe_free()
    recon_r.unsafe_free()
    recon_p.unsafe_free()
    set_pq_dispatch_mode(MODE_AUTO)

def run_parallel_pq_test(num_indices: Int, mode: Int) raises:
    set_pq_dispatch_mode(mode)
    comptime d = 16
    comptime n = 96
    comptime M = 4
    comptime ksub = 16

    var all_data = unsafe_alloc[Float32](num_indices * n * d)
    var generator = Random(seed=UInt64(54321 + num_indices))
    var values = generator.step_uniform()
    for i in range(num_indices * n * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        all_data[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0

    var pq_ptrs = unsafe_alloc[ProductQuantizer](num_indices)
    for idx in range(num_indices):
        var pq = ProductQuantizer(d, M, ksub)
        pq_ptrs.unsafe_offset(idx).unsafe_write(pq^)

    var pqs_addr = Int(pq_ptrs)
    var data_addr = Int(all_data)

    def worker(idx: Int) {imm}:
        var cur_data = Pointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=data_addr
        ).unsafe_offset(idx * n * d)
        var cur_span = Span[Float32, MutUntrackedOrigin](
            unsafe_ptr=cur_data, length=n * d
        )
        var cur_pq = Pointer[ProductQuantizer, MutUntrackedOrigin](
            unsafe_from_address=pqs_addr
        ).unsafe_offset(idx)
        cur_pq[].train(cur_span)

    parallelize(worker, num_indices, num_indices)

    # Verify all indices trained correctly
    for idx in range(num_indices):
        assert_true(pq_ptrs[unsafe_offset=idx].is_trained)
        # Check non-zero centroids
        var has_nonzero = False
        for c in range(M * ksub * (d // M)):
            if pq_ptrs[unsafe_offset=idx].centroids[c] != 0.0:
                has_nonzero = True
                break
        assert_true(has_nonzero)

    for idx in range(num_indices):
        _ = pq_ptrs.unsafe_offset(idx).unsafe_take_pointee()
    pq_ptrs.unsafe_free()
    all_data.unsafe_free()

def test_pq_parallel_training_independent_indices() raises:
    """Requirement 4.2: Concurrently train 4 and 8 independent PQ objects.

    Tests lack of data races, deadlocks, and crashes across all dispatch modes.
    """
    # 4 independent indices
    run_parallel_pq_test(4, MODE_SERIAL)
    run_parallel_pq_test(4, MODE_PTHREAD)
    run_parallel_pq_test(4, MODE_RUNTIME)

    # 8 independent indices
    run_parallel_pq_test(8, MODE_SERIAL)
    run_parallel_pq_test(8, MODE_PTHREAD)
    run_parallel_pq_test(8, MODE_RUNTIME)

    set_pq_dispatch_mode(MODE_AUTO)

def test_pq_training_performance_comparison() raises:
    """Requirement 4.2: Measure and compare PQ training time between runtime and pthread."""
    comptime d = 32
    comptime n = 256
    comptime M = 8
    comptime ksub = 32

    var generator = Random(seed=UInt64(777))
    var data = unsafe_alloc[Float32](n * d)
    var values = generator.step_uniform()
    for i in range(n * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        data[unsafe_offset=i] = values[i % 4]

    var data_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=data, length=n * d
    )

    # Serial measurement
    set_pq_dispatch_mode(MODE_SERIAL)
    var t_s0 = perf_counter_ns()
    var pq_s = ProductQuantizer(d, M, ksub)
    pq_s.train(data_span)
    var t_s1 = perf_counter_ns()
    var serial_time = Float64(t_s1 - t_s0) / 1e6

    # Runtime measurement
    set_pq_dispatch_mode(MODE_RUNTIME)
    var t_r0 = perf_counter_ns()
    var pq_r = ProductQuantizer(d, M, ksub)
    pq_r.train(data_span)
    var t_r1 = perf_counter_ns()
    var runtime_time = Float64(t_r1 - t_r0) / 1e6

    # Pthread measurement
    set_pq_dispatch_mode(MODE_PTHREAD)
    var t_p0 = perf_counter_ns()
    var pq_p = ProductQuantizer(d, M, ksub)
    pq_p.train(data_span)
    var t_p1 = perf_counter_ns()
    var pthread_time = Float64(t_p1 - t_p0) / 1e6

    assert_true(pq_s.is_trained)
    assert_true(pq_r.is_trained)
    assert_true(pq_p.is_trained)

    # Both parallel modes must execute without issue
    assert_true(runtime_time > 0.0)
    assert_true(pthread_time > 0.0)
    assert_true(serial_time > 0.0)

    data.unsafe_free()
    set_pq_dispatch_mode(MODE_AUTO)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
