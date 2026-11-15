"""Single-thread FP32 kernel sweep. Emits CSV; no dataset download required."""

from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns
from std.sys import simd_width_of
from mojovec.utils.distances import (
    l2_distance_simd,
    inner_product_simd,
    l2_distance_simd_batch4,
    inner_product_simd_batch4,
    l2_distance_short4,
    inner_product_short4,
    l2_distance_tuned,
    inner_product_tuned,
    l2_distance_tuned_batch4,
    inner_product_tuned_batch4,
)


@no_inline
def evaluate[
    width: Int, unroll: Int, mode: Int, guarded: Bool
](
    x: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    d: Int,
) -> Float32:
    comptime if mode == 0:
        return l2_distance_simd[width, unroll, guarded](x, y, d)
    elif mode == 1:
        return inner_product_simd[width, unroll, guarded](x, y, d)
    elif mode == 2:
        var values = l2_distance_simd_batch4[width](
            x,
            x.unsafe_offset(d),
            x.unsafe_offset(2 * d),
            x.unsafe_offset(3 * d),
            y,
            d,
        )
        return values[0] + values[1] + values[2] + values[3]
    elif mode == 3:
        var values = inner_product_simd_batch4[width](
            x,
            x.unsafe_offset(d),
            x.unsafe_offset(2 * d),
            x.unsafe_offset(3 * d),
            y,
            d,
        )
        return values[0] + values[1] + values[2] + values[3]
    elif mode == 4:
        return l2_distance_short4(x, y, d)
    elif mode == 5:
        return inner_product_short4(x, y, d)
    elif mode == 6:
        return l2_distance_tuned(x, y, d)
    elif mode == 7:
        return inner_product_tuned(x, y, d)
    elif mode == 8:
        var values = l2_distance_tuned_batch4(
            x,
            x.unsafe_offset(d),
            x.unsafe_offset(2 * d),
            x.unsafe_offset(3 * d),
            y,
            d,
        )
        return values[0] + values[1] + values[2] + values[3]
    else:
        var values = inner_product_tuned_batch4(
            x,
            x.unsafe_offset(d),
            x.unsafe_offset(2 * d),
            x.unsafe_offset(3 * d),
            y,
            d,
        )
        return values[0] + values[1] + values[2] + values[3]


@no_inline
def sweep[
    width: Int, unroll: Int, mode: Int, guarded: Bool
](
    x: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    d: Int,
    iterations: Int,
) -> Float64:
    var checksum: Float64 = 0.0
    for iteration in range(iterations):
        # Visit different rows; consume every result to prevent dead-code
        # elimination and avoid timing a loop-invariant distance.
        var row = (iteration * 73) & 255
        checksum += Float64(
            evaluate[width, unroll, mode, guarded](
                x.unsafe_offset(row * d), y, d
            )
        )
    return checksum


def measure[
    width: Int, unroll: Int, mode: Int, guarded: Bool
](
    x: Pointer[Float32, MutUntrackedOrigin],
    y: Pointer[Float32, MutUntrackedOrigin],
    d: Int,
    sample: Int,
):
    var iterations = 1024
    var elapsed: Int = 0
    var checksum: Float64 = 0.0
    # Calibrate each cell to >= 5 ms instead of timing tiny fixed loops.
    while elapsed < 5_000_000:
        var started = perf_counter_ns()
        checksum = sweep[width, unroll, mode, guarded](x, y, d, iterations)
        elapsed = Int(perf_counter_ns() - started)
        if elapsed < 5_000_000:
            iterations *= 2
    comptime distances_per_call = 4 if mode == 2 or mode == 3 or mode == 8 or mode == 9 else 1
    var ns = Float64(elapsed) / Float64(iterations * distances_per_call)
    print(
        String(
            sample,
            ",",
            mode,
            ",",
            d,
            ",",
            width,
            ",",
            unroll,
            ",",
            Int(guarded),
            ",",
            ns,
            ",",
            checksum,
        )
    )


def dimension_cases(d: Int, sample: Int):
    # Three spare rows make the batch4 kernel's last group valid.
    var x = unsafe_alloc[Float32](259 * d)
    var y = unsafe_alloc[Float32](d)
    for i in range(259 * d):
        x[unsafe_offset=i] = Float32((i * 17 + 3) % 127 - 63) / 64.0
    for i in range(d):
        y[unsafe_offset=i] = Float32((i * 29 + 5) % 127 - 63) / 64.0
    # Reverse specialization order on alternate rounds to reduce order bias.
    comptime for position in range(5):
        comptime width = 4 << position
        comptime reverse_width = 64 >> position
        comptime for power in range(4):
            comptime unroll = 1 << power
            comptime reverse_unroll = 8 >> power
            if sample % 2 == 0:
                measure[width, unroll, 0, True](x, y, d, sample)
                measure[width, unroll, 1, True](x, y, d, sample)
            else:
                measure[reverse_width, reverse_unroll, 0, True](x, y, d, sample)
                measure[reverse_width, reverse_unroll, 1, True](x, y, d, sample)
        measure[width, 1, 2, True](x, y, d, sample)
        measure[width, 1, 3, True](x, y, d, sample)
        if d < width:
            measure[width, 4, 0, False](x, y, d, sample)
            measure[width, 4, 1, False](x, y, d, sample)
    if d <= 4:
        measure[4, 1, 4, True](x, y, d, sample)
        measure[4, 1, 5, True](x, y, d, sample)
    comptime for mode in range(6, 10):
        measure[0, 0, mode, True](x, y, d, sample)
    x.unsafe_free()
    y.unsafe_free()


def main():
    print("# native_fp32_width=" + String(simd_width_of[Float32]()))
    print("sample,mode,dimension,width,unroll,guarded,ns_per_distance,checksum")
    for sample in range(5):
        for d in [1, 2, 3, 4, 8, 16, 32, 128, 384, 768, 1536]:
            dimension_cases(d, sample)
