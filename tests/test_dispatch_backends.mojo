from std.collections.span import Span
from std.memory.alloc import unsafe_alloc
from std.random.philox import Random
from std.testing import assert_true, assert_equal, assert_almost_equal, TestSuite

from mojovec.core.types import (
    METRIC_L2,
    MODE_AUTO,
    MODE_SERIAL,
    MODE_RUNTIME,
    MODE_PTHREAD,
    get_pq_dispatch_mode,
    set_pq_dispatch_mode,
    get_hnsw_dispatch_mode,
    set_hnsw_dispatch_mode,
)
from mojovec.quantization.pq import ProductQuantizer
from mojovec.index.index_flat import IndexFlat
from mojovec.index.index_flat_sq8 import IndexFlatSQ8
from mojovec.index.index_hnsw import IndexHNSW
from mojovec.utils.distances import l2_distance_simd


def test_pq_dispatch_mode_switch() raises:
    # Verify default is MODE_AUTO
    set_pq_dispatch_mode(MODE_AUTO)
    assert_equal(get_pq_dispatch_mode(), MODE_AUTO)

    # Test mode transitions
    set_pq_dispatch_mode(MODE_SERIAL)
    assert_equal(get_pq_dispatch_mode(), MODE_SERIAL)

    set_pq_dispatch_mode(MODE_RUNTIME)
    assert_equal(get_pq_dispatch_mode(), MODE_RUNTIME)

    set_pq_dispatch_mode(MODE_PTHREAD)
    assert_equal(get_pq_dispatch_mode(), MODE_PTHREAD)

    set_pq_dispatch_mode(MODE_AUTO)
    assert_equal(get_pq_dispatch_mode(), MODE_AUTO)


def test_pq_training_all_modes_identical() raises:
    comptime d = 16
    comptime n = 120
    comptime M = 4
    comptime ksub = 16
    comptime dsub = d // M

    var generator = Random(seed=UInt64(12345))
    var data = unsafe_alloc[Float32](n * d)
    var values = generator.step_uniform()
    for i in range(n * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        data[unsafe_offset=i] = values[i % 4] * 2.0 - 1.0

    var data_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=data, length=n * d
    )

    # Train Serial
    set_pq_dispatch_mode(MODE_SERIAL)
    var pq_serial = ProductQuantizer(d, M, ksub)
    pq_serial.train(data_span)
    assert_true(pq_serial.is_trained)

    # Train Runtime
    set_pq_dispatch_mode(MODE_RUNTIME)
    var pq_runtime = ProductQuantizer(d, M, ksub)
    pq_runtime.train(data_span)
    assert_true(pq_runtime.is_trained)

    # Train Pthread
    set_pq_dispatch_mode(MODE_PTHREAD)
    var pq_pthread = ProductQuantizer(d, M, ksub)
    pq_pthread.train(data_span)
    assert_true(pq_pthread.is_trained)

    # Train Auto
    set_pq_dispatch_mode(MODE_AUTO)
    var pq_auto = ProductQuantizer(d, M, ksub)
    pq_auto.train(data_span)
    assert_true(pq_auto.is_trained)

    # Centroids must be bit-exact across all modes
    var total_centroids = M * ksub * dsub
    for i in range(total_centroids):
        var expected = pq_serial.centroids[i]
        assert_equal(pq_runtime.centroids[i], expected)
        assert_equal(pq_pthread.centroids[i], expected)
        assert_equal(pq_auto.centroids[i], expected)

    # Codes computed with each quantizer must match bit-for-bit
    var codes_serial = unsafe_alloc[UInt8](n * M)
    var codes_runtime = unsafe_alloc[UInt8](n * M)
    var codes_pthread = unsafe_alloc[UInt8](n * M)
    var codes_auto = unsafe_alloc[UInt8](n * M)

    var span_cs = Span[mut=True, UInt8](unsafe_ptr=codes_serial, length=n * M)
    var span_cr = Span[mut=True, UInt8](unsafe_ptr=codes_runtime, length=n * M)
    var span_cp = Span[mut=True, UInt8](unsafe_ptr=codes_pthread, length=n * M)
    var span_ca = Span[mut=True, UInt8](unsafe_ptr=codes_auto, length=n * M)

    pq_serial.compute_codes(data_span, span_cs)
    pq_runtime.compute_codes(data_span, span_cr)
    pq_pthread.compute_codes(data_span, span_cp)
    pq_auto.compute_codes(data_span, span_ca)

    for i in range(n * M):
        var expected_code = codes_serial[unsafe_offset=i]
        assert_equal(codes_runtime[unsafe_offset=i], expected_code)
        assert_equal(codes_pthread[unsafe_offset=i], expected_code)
        assert_equal(codes_auto[unsafe_offset=i], expected_code)

    data.unsafe_free()
    codes_serial.unsafe_free()
    codes_runtime.unsafe_free()
    codes_pthread.unsafe_free()
    codes_auto.unsafe_free()
    set_pq_dispatch_mode(MODE_AUTO)


def test_pq_single_subspace() raises:
    comptime d = 8
    comptime n = 64
    comptime M = 1
    comptime ksub = 8

    var generator = Random(seed=UInt64(54321))
    var data = unsafe_alloc[Float32](n * d)
    var values = generator.step_uniform()
    for i in range(n * d):
        if i != 0 and i % 4 == 0:
            values = generator.step_uniform()
        data[unsafe_offset=i] = values[i % 4]

    var data_span = Span[Float32, MutUntrackedOrigin](
        unsafe_ptr=data, length=n * d
    )

    for m in range(4):
        var mode = MODE_AUTO
        if m == 0:
            mode = MODE_SERIAL
        elif m == 1:
            mode = MODE_RUNTIME
        elif m == 2:
            mode = MODE_PTHREAD
        set_pq_dispatch_mode(mode)
        var pq = ProductQuantizer(d, M, ksub)
        pq.train(data_span)
        assert_true(pq.is_trained)

    data.unsafe_free()
    set_pq_dispatch_mode(MODE_AUTO)


def test_hnsw_dispatch_mode_switch() raises:
    # Verify default is MODE_AUTO
    set_hnsw_dispatch_mode(MODE_AUTO)
    assert_equal(get_hnsw_dispatch_mode(), MODE_AUTO)

    # Test mode transitions
    set_hnsw_dispatch_mode(MODE_SERIAL)
    assert_equal(get_hnsw_dispatch_mode(), MODE_SERIAL)

    set_hnsw_dispatch_mode(MODE_RUNTIME)
    assert_equal(get_hnsw_dispatch_mode(), MODE_RUNTIME)

    set_hnsw_dispatch_mode(MODE_PTHREAD)
    assert_equal(get_hnsw_dispatch_mode(), MODE_PTHREAD)

    set_hnsw_dispatch_mode(MODE_AUTO)
    assert_equal(get_hnsw_dispatch_mode(), MODE_AUTO)


def test_hnsw_search_all_modes_identical_flat() raises:
    comptime dimension = 8
    comptime database_size = 48
    comptime query_count = 32
    comptime k = 4

    var storage = IndexFlat(dimension, METRIC_L2)
    var index = IndexHNSW[IndexFlat](storage^, dimension, METRIC_L2, M=8)

    var database = unsafe_alloc[Float32](database_size * dimension)
    for i in range(database_size):
        for j in range(dimension):
            database[unsafe_offset=i * dimension + j] = Float32(i * 3 + j) / 48.0
    index.add(
        Span[Float32](unsafe_ptr=database, length=database_size * dimension)
    )
    index.hnsw.efSearch = 16

    var queries = unsafe_alloc[Float32](query_count * dimension)
    for i in range(query_count):
        var src = i % database_size
        for j in range(dimension):
            queries[unsafe_offset=i * dimension + j] = database[unsafe_offset=src * dimension + j]

    var query_span = Span[Float32](unsafe_ptr=queries, length=query_count * dimension)

    var empty_filter_storage = unsafe_alloc[UInt8](1)
    var empty_filter = Span[UInt8](unsafe_ptr=empty_filter_storage, length=0)

    # Serial
    var dist_serial = unsafe_alloc[Float32](query_count * k)
    var lbl_serial = unsafe_alloc[Int](query_count * k)
    var span_ds = Span[mut=True, Float32](unsafe_ptr=dist_serial, length=query_count * k)
    var span_ls = Span[mut=True, Int](unsafe_ptr=lbl_serial, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_SERIAL)
    index.search(query_span, k, span_ds, span_ls, empty_filter)

    # Runtime
    var dist_runtime = unsafe_alloc[Float32](query_count * k)
    var lbl_runtime = unsafe_alloc[Int](query_count * k)
    var span_dr = Span[mut=True, Float32](unsafe_ptr=dist_runtime, length=query_count * k)
    var span_lr = Span[mut=True, Int](unsafe_ptr=lbl_runtime, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_RUNTIME)
    index.search(query_span, k, span_dr, span_lr, empty_filter)

    # Pthread
    var dist_pthread = unsafe_alloc[Float32](query_count * k)
    var lbl_pthread = unsafe_alloc[Int](query_count * k)
    var span_dp = Span[mut=True, Float32](unsafe_ptr=dist_pthread, length=query_count * k)
    var span_lp = Span[mut=True, Int](unsafe_ptr=lbl_pthread, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_PTHREAD)
    index.search(query_span, k, span_dp, span_lp, empty_filter)

    # Auto
    var dist_auto = unsafe_alloc[Float32](query_count * k)
    var lbl_auto = unsafe_alloc[Int](query_count * k)
    var span_da = Span[mut=True, Float32](unsafe_ptr=dist_auto, length=query_count * k)
    var span_la = Span[mut=True, Int](unsafe_ptr=lbl_auto, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_AUTO)
    index.search(query_span, k, span_da, span_la, empty_filter)

    for i in range(query_count * k):
        assert_equal(lbl_runtime[unsafe_offset=i], lbl_serial[unsafe_offset=i])
        assert_equal(dist_runtime[unsafe_offset=i], dist_serial[unsafe_offset=i])
        assert_equal(lbl_pthread[unsafe_offset=i], lbl_serial[unsafe_offset=i])
        assert_equal(dist_pthread[unsafe_offset=i], dist_serial[unsafe_offset=i])
        assert_equal(lbl_auto[unsafe_offset=i], lbl_serial[unsafe_offset=i])
        assert_equal(dist_auto[unsafe_offset=i], dist_serial[unsafe_offset=i])

    database.unsafe_free()
    queries.unsafe_free()
    dist_serial.unsafe_free()
    lbl_serial.unsafe_free()
    dist_runtime.unsafe_free()
    lbl_runtime.unsafe_free()
    dist_pthread.unsafe_free()
    lbl_pthread.unsafe_free()
    dist_auto.unsafe_free()
    lbl_auto.unsafe_free()
    empty_filter_storage.unsafe_free()


def test_hnsw_search_all_modes_identical_sq8() raises:
    comptime dimension = 8
    comptime database_size = 48
    comptime query_count = 32
    comptime k = 4

    var storage = IndexFlatSQ8(dimension, METRIC_L2)
    var index = IndexHNSW[IndexFlatSQ8](storage^, dimension, METRIC_L2, M=8)

    var database = unsafe_alloc[Float32](database_size * dimension)
    for i in range(database_size):
        for j in range(dimension):
            database[unsafe_offset=i * dimension + j] = Float32(i * 3 + j) / 48.0
    index.add(
        Span[Float32](unsafe_ptr=database, length=database_size * dimension)
    )
    index.hnsw.efSearch = 16

    var queries = unsafe_alloc[Float32](query_count * dimension)
    for i in range(query_count):
        var src = i % database_size
        for j in range(dimension):
            queries[unsafe_offset=i * dimension + j] = database[unsafe_offset=src * dimension + j]

    var query_span = Span[Float32](unsafe_ptr=queries, length=query_count * dimension)

    var empty_filter_storage = unsafe_alloc[UInt8](1)
    var empty_filter = Span[UInt8](unsafe_ptr=empty_filter_storage, length=0)

    # Serial
    var dist_serial = unsafe_alloc[Float32](query_count * k)
    var lbl_serial = unsafe_alloc[Int](query_count * k)
    var span_ds = Span[mut=True, Float32](unsafe_ptr=dist_serial, length=query_count * k)
    var span_ls = Span[mut=True, Int](unsafe_ptr=lbl_serial, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_SERIAL)
    index.search(query_span, k, span_ds, span_ls, empty_filter)

    # Runtime
    var dist_runtime = unsafe_alloc[Float32](query_count * k)
    var lbl_runtime = unsafe_alloc[Int](query_count * k)
    var span_dr = Span[mut=True, Float32](unsafe_ptr=dist_runtime, length=query_count * k)
    var span_lr = Span[mut=True, Int](unsafe_ptr=lbl_runtime, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_RUNTIME)
    index.search(query_span, k, span_dr, span_lr, empty_filter)

    # Pthread
    var dist_pthread = unsafe_alloc[Float32](query_count * k)
    var lbl_pthread = unsafe_alloc[Int](query_count * k)
    var span_dp = Span[mut=True, Float32](unsafe_ptr=dist_pthread, length=query_count * k)
    var span_lp = Span[mut=True, Int](unsafe_ptr=lbl_pthread, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_PTHREAD)
    index.search(query_span, k, span_dp, span_lp, empty_filter)

    # Auto
    var dist_auto = unsafe_alloc[Float32](query_count * k)
    var lbl_auto = unsafe_alloc[Int](query_count * k)
    var span_da = Span[mut=True, Float32](unsafe_ptr=dist_auto, length=query_count * k)
    var span_la = Span[mut=True, Int](unsafe_ptr=lbl_auto, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_AUTO)
    index.search(query_span, k, span_da, span_la, empty_filter)

    for i in range(query_count * k):
        assert_equal(lbl_runtime[unsafe_offset=i], lbl_serial[unsafe_offset=i])
        assert_equal(dist_runtime[unsafe_offset=i], dist_serial[unsafe_offset=i])
        assert_equal(lbl_pthread[unsafe_offset=i], lbl_serial[unsafe_offset=i])
        assert_equal(dist_pthread[unsafe_offset=i], dist_serial[unsafe_offset=i])
        assert_equal(lbl_auto[unsafe_offset=i], lbl_serial[unsafe_offset=i])
        assert_equal(dist_auto[unsafe_offset=i], dist_serial[unsafe_offset=i])

    database.unsafe_free()
    queries.unsafe_free()
    dist_serial.unsafe_free()
    lbl_serial.unsafe_free()
    dist_runtime.unsafe_free()
    lbl_runtime.unsafe_free()
    dist_pthread.unsafe_free()
    lbl_pthread.unsafe_free()
    dist_auto.unsafe_free()
    lbl_auto.unsafe_free()
    empty_filter_storage.unsafe_free()


def test_hnsw_search_with_filter_all_modes() raises:
    comptime dimension = 8
    comptime database_size = 32
    comptime query_count = 16
    comptime k = 3

    var storage = IndexFlat(dimension, METRIC_L2)
    var index = IndexHNSW[IndexFlat](storage^, dimension, METRIC_L2, M=8)

    var database = unsafe_alloc[Float32](database_size * dimension)
    for i in range(database_size):
        for j in range(dimension):
            database[unsafe_offset=i * dimension + j] = Float32(i + j) / 32.0
    index.add(
        Span[Float32](unsafe_ptr=database, length=database_size * dimension)
    )
    index.hnsw.efSearch = 16

    var queries = unsafe_alloc[Float32](query_count * dimension)
    for i in range(query_count):
        for j in range(dimension):
            queries[unsafe_offset=i * dimension + j] = database[unsafe_offset=i * dimension + j]

    var query_span = Span[Float32](unsafe_ptr=queries, length=query_count * dimension)

    # Filter (deletion mask): exclude odd numbered IDs (filter[id] == 1 means excluded)
    var filter_storage = unsafe_alloc[UInt8](database_size)
    for i in range(database_size):
        filter_storage[unsafe_offset=i] = UInt8(1 if i % 2 != 0 else 0)
    var filter_span = Span[UInt8](unsafe_ptr=filter_storage, length=database_size)

    # 1. Serial
    var dist_serial = unsafe_alloc[Float32](query_count * k)
    var lbl_serial = unsafe_alloc[Int](query_count * k)
    var span_ds = Span[mut=True, Float32](unsafe_ptr=dist_serial, length=query_count * k)
    var span_ls = Span[mut=True, Int](unsafe_ptr=lbl_serial, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_SERIAL)
    index.search(query_span, k, span_ds, span_ls, filter_span)

    # 2. Runtime
    var dist_runtime = unsafe_alloc[Float32](query_count * k)
    var lbl_runtime = unsafe_alloc[Int](query_count * k)
    var span_dr = Span[mut=True, Float32](unsafe_ptr=dist_runtime, length=query_count * k)
    var span_lr = Span[mut=True, Int](unsafe_ptr=lbl_runtime, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_RUNTIME)
    index.search(query_span, k, span_dr, span_lr, filter_span)

    # 3. Pthread
    var dist_pthread = unsafe_alloc[Float32](query_count * k)
    var lbl_pthread = unsafe_alloc[Int](query_count * k)
    var span_dp = Span[mut=True, Float32](unsafe_ptr=dist_pthread, length=query_count * k)
    var span_lp = Span[mut=True, Int](unsafe_ptr=lbl_pthread, length=query_count * k)
    set_hnsw_dispatch_mode(MODE_PTHREAD)
    index.search(query_span, k, span_dp, span_lp, filter_span)

    for i in range(query_count * k):
        assert_equal(lbl_runtime[unsafe_offset=i], lbl_serial[unsafe_offset=i])
        assert_equal(dist_runtime[unsafe_offset=i], dist_serial[unsafe_offset=i])
        assert_equal(lbl_pthread[unsafe_offset=i], lbl_serial[unsafe_offset=i])
        assert_equal(dist_pthread[unsafe_offset=i], dist_serial[unsafe_offset=i])
        if lbl_serial[unsafe_offset=i] != -1:
            # Filter must be respected (only even IDs allowed)
            assert_equal(lbl_serial[unsafe_offset=i] % 2, 0)

    database.unsafe_free()
    queries.unsafe_free()
    dist_serial.unsafe_free()
    lbl_serial.unsafe_free()
    dist_runtime.unsafe_free()
    lbl_runtime.unsafe_free()
    dist_pthread.unsafe_free()
    lbl_pthread.unsafe_free()
    filter_storage.unsafe_free()
    set_hnsw_dispatch_mode(MODE_AUTO)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
