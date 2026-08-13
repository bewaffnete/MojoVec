"""
Provides optimized SIMD implementations for computing various vector distances.
"""

from std.math import fma
@always_inline
def l2_distance_simd[simd_width: Int](
    x: UnsafePointer[Float32, _],
    y: UnsafePointer[Float32, _],
    d: Int,
    threshold: Float32 = Float32.MAX
) -> Float32:
    """
    Computes the squared L2 distance between two vectors of dimension `d` using SIMD instructions.
    Supports early termination if threshold is exceeded.
    """
    # These accumulators must be initialized explicitly. A default-constructed
    # SIMD value is not a portable zero and dimensions below ``simd_width``
    # reach the tail path without executing an FMA first (PQ commonly has
    # four-dimensional subvectors with an eight-lane kernel).
    var dist0 = SIMD[DType.float32, simd_width](0.0)
    var dist1 = SIMD[DType.float32, simd_width](0.0)
    var dist2 = SIMD[DType.float32, simd_width](0.0)
    var dist3 = SIMD[DType.float32, simd_width](0.0)

    var i = 0
    # Unroll 4x for Instruction Level Parallelism
    while i <= d - (simd_width * 4):
        var vx0 = x.load[width=simd_width](i)
        var vy0 = y.load[width=simd_width](i)
        var diff0 = vx0 - vy0
        dist0 = fma(diff0, diff0, dist0)

        var vx1 = x.load[width=simd_width](i + simd_width)
        var vy1 = y.load[width=simd_width](i + simd_width)
        var diff1 = vx1 - vy1
        dist1 = fma(diff1, diff1, dist1)

        var vx2 = x.load[width=simd_width](i + simd_width * 2)
        var vy2 = y.load[width=simd_width](i + simd_width * 2)
        var diff2 = vx2 - vy2
        dist2 = fma(diff2, diff2, dist2)

        var vx3 = x.load[width=simd_width](i + simd_width * 3)
        var vy3 = y.load[width=simd_width](i + simd_width * 3)
        var diff3 = vx3 - vy3
        dist3 = fma(diff3, diff3, dist3)

        i += simd_width * 4

    var dist = dist0 + dist1 + dist2 + dist3

    # 1x for remaining SIMD chunks
    while i <= d - simd_width:
        var vx = x.load[width=simd_width](i)
        var vy = y.load[width=simd_width](i)
        var diff = vx - vy
        dist = fma(diff, diff, dist)
        i += simd_width

    var res = dist.reduce_add()

    comptime if simd_width >= 32:
        while i <= d - 16:
            var vx = x.load[width=16](i)
            var vy = y.load[width=16](i)
            var diff = vx - vy
            var r = diff * diff
            res += r.reduce_add()
            i += 16

    comptime if simd_width >= 16:
        while i <= d - 8:
            var vx = x.load[width=8](i)
            var vy = y.load[width=8](i)
            var diff = vx - vy
            var r = diff * diff
            res += r.reduce_add()
            i += 8

    comptime if simd_width >= 8:
        while i <= d - 4:
            var vx = x.load[width=4](i)
            var vy = y.load[width=4](i)
            var diff = vx - vy
            var r = diff * diff
            res += r.reduce_add()
            i += 4

    # Handle remainder
    while i < d:
        var diff = x[i] - y[i]
        res += diff * diff
        i += 1

    return res

@always_inline
def inner_product_simd[simd_width: Int](x: UnsafePointer[Float32, _], y: UnsafePointer[Float32, _], d: Int) -> Float32:
    """
    Computes the inner product between two vectors of dimension `d` using SIMD instructions.

    Args:
        x: Pointer to the first vector.
        y: Pointer to the second vector.
        d: The dimensionality of the vectors.

    Returns:
        The computed inner product.
    """
    var prod0 = SIMD[DType.float32, simd_width](0.0)
    var prod1 = SIMD[DType.float32, simd_width](0.0)
    var prod2 = SIMD[DType.float32, simd_width](0.0)
    var prod3 = SIMD[DType.float32, simd_width](0.0)
    var i = 0

    while i <= d - (simd_width * 4):
        var vx0 = x.load[width=simd_width](i)
        var vy0 = y.load[width=simd_width](i)
        prod0 = fma(vx0, vy0, prod0)

        var vx1 = x.load[width=simd_width](i + simd_width)
        var vy1 = y.load[width=simd_width](i + simd_width)
        prod1 = fma(vx1, vy1, prod1)

        var vx2 = x.load[width=simd_width](i + simd_width * 2)
        var vy2 = y.load[width=simd_width](i + simd_width * 2)
        prod2 = fma(vx2, vy2, prod2)

        var vx3 = x.load[width=simd_width](i + simd_width * 3)
        var vy3 = y.load[width=simd_width](i + simd_width * 3)
        prod3 = fma(vx3, vy3, prod3)

        i += simd_width * 4

    var prod = prod0 + prod1 + prod2 + prod3

    while i <= d - simd_width:
        var vx = x.load[width=simd_width](i)
        var vy = y.load[width=simd_width](i)
        prod = fma(vx, vy, prod)
        i += simd_width

    var res = prod.reduce_add()

    comptime if simd_width >= 32:
        while i <= d - 16:
            var vx = x.load[width=16](i)
            var vy = y.load[width=16](i)
            var r = fma(vx, vy, SIMD[DType.float32, 16](0.0))
            res += r.reduce_add()
            i += 16

    comptime if simd_width >= 16:
        while i <= d - 8:
            var vx = x.load[width=8](i)
            var vy = y.load[width=8](i)
            var r = fma(vx, vy, SIMD[DType.float32, 8](0.0))
            res += r.reduce_add()
            i += 8

    comptime if simd_width >= 8:
        while i <= d - 4:
            var vx = x.load[width=4](i)
            var vy = y.load[width=4](i)
            var r = fma(vx, vy, SIMD[DType.float32, 4](0.0))
            res += r.reduce_add()
            i += 4

    # Handle remainder
    while i < d:
        res += x[i] * y[i]
        i += 1

    return res

from std.sys.intrinsics import llvm_intrinsic
import std.math as math

from std.sys.info import is_apple_gpu

@always_inline
def sq8_dot_product_simd(
    x: UnsafePointer[UInt8, _],
    y: UnsafePointer[UInt8, _],
    d: Int,
) -> UInt32:
    """
    Computes the dot product between two UInt8 vectors using explicit UDOT intrinsics
    when available, and falls back to generic SIMD multiplication otherwise.
    """
    comptime if is_apple_gpu():
        var acc0 = SIMD[DType.uint32, 4]()
        var acc1 = SIMD[DType.uint32, 4]()
        var acc2 = SIMD[DType.uint32, 4]()
        var acc3 = SIMD[DType.uint32, 4]()

        var i = 0
        while i <= d - 64:
            var vx0 = x.load[width=16](i)
            var vy0 = y.load[width=16](i)
            acc0 = llvm_intrinsic["llvm.aarch64.neon.udot.v4i32.v16i8", SIMD[DType.uint32, 4]](acc0, vx0, vy0)

            var vx1 = x.load[width=16](i + 16)
            var vy1 = y.load[width=16](i + 16)
            acc1 = llvm_intrinsic["llvm.aarch64.neon.udot.v4i32.v16i8", SIMD[DType.uint32, 4]](acc1, vx1, vy1)

            var vx2 = x.load[width=16](i + 32)
            var vy2 = y.load[width=16](i + 32)
            acc2 = llvm_intrinsic["llvm.aarch64.neon.udot.v4i32.v16i8", SIMD[DType.uint32, 4]](acc2, vx2, vy2)

            var vx3 = x.load[width=16](i + 48)
            var vy3 = y.load[width=16](i + 48)
            acc3 = llvm_intrinsic["llvm.aarch64.neon.udot.v4i32.v16i8", SIMD[DType.uint32, 4]](acc3, vx3, vy3)

            i += 64

        var acc = acc0 + acc1 + acc2 + acc3

        while i <= d - 16:
            var vx = x.load[width=16](i)
            var vy = y.load[width=16](i)
            acc = llvm_intrinsic["llvm.aarch64.neon.udot.v4i32.v16i8", SIMD[DType.uint32, 4]](acc, vx, vy)
            i += 16

        var res = acc.reduce_add()

        while i < d:
            res += UInt32(x[i]) * UInt32(y[i])
            i += 1

        return res
    else:
        var acc0 = SIMD[DType.uint32, 16]()
        var acc1 = SIMD[DType.uint32, 16]()
        var acc2 = SIMD[DType.uint32, 16]()
        var acc3 = SIMD[DType.uint32, 16]()

        var i = 0
        while i <= d - 64:
            var vx0 = x.load[width=16](i).cast[DType.uint16]()
            var vy0 = y.load[width=16](i).cast[DType.uint16]()
            acc0 += (vx0 * vy0).cast[DType.uint32]()

            var vx1 = x.load[width=16](i + 16).cast[DType.uint16]()
            var vy1 = y.load[width=16](i + 16).cast[DType.uint16]()
            acc1 += (vx1 * vy1).cast[DType.uint32]()

            var vx2 = x.load[width=16](i + 32).cast[DType.uint16]()
            var vy2 = y.load[width=16](i + 32).cast[DType.uint16]()
            acc2 += (vx2 * vy2).cast[DType.uint32]()

            var vx3 = x.load[width=16](i + 48).cast[DType.uint16]()
            var vy3 = y.load[width=16](i + 48).cast[DType.uint16]()
            acc3 += (vx3 * vy3).cast[DType.uint32]()

            i += 64

        var acc = acc0 + acc1 + acc2 + acc3

        while i <= d - 16:
            var vx = x.load[width=16](i).cast[DType.uint16]()
            var vy = y.load[width=16](i).cast[DType.uint16]()
            acc += (vx * vy).cast[DType.uint32]()
            i += 16

        var res = acc.reduce_add()

        while i <= d - 8:
            var vx = x.load[width=8](i).cast[DType.uint16]()
            var vy = y.load[width=8](i).cast[DType.uint16]()
            var r = (vx * vy).cast[DType.uint32]()
            res += r.reduce_add()
            i += 8

        while i <= d - 4:
            var vx = x.load[width=4](i).cast[DType.uint16]()
            var vy = y.load[width=4](i).cast[DType.uint16]()
            var r = (vx * vy).cast[DType.uint32]()
            res += r.reduce_add()
            i += 4

        while i < d:
            res += UInt32(x[i]) * UInt32(y[i])
            i += 1

        return res


@always_inline
def sq8_signed_dot_product_simd(
    x: UnsafePointer[Int8, _],
    y: UnsafePointer[Int8, _],
    d: Int,
) -> Int32:
    """Computes an Int8 dot product with SDOT on AArch64 and SIMD elsewhere."""
    comptime if is_apple_gpu():
        var acc0 = SIMD[DType.int32, 4]()
        var acc1 = SIMD[DType.int32, 4]()
        var acc2 = SIMD[DType.int32, 4]()
        var acc3 = SIMD[DType.int32, 4]()

        var i = 0
        while i <= d - 64:
            acc0 = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](acc0, x.load[width=16](i), y.load[width=16](i))
            acc1 = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](
                acc1,
                x.load[width=16](i + 16),
                y.load[width=16](i + 16),
            )
            acc2 = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](
                acc2,
                x.load[width=16](i + 32),
                y.load[width=16](i + 32),
            )
            acc3 = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](
                acc3,
                x.load[width=16](i + 48),
                y.load[width=16](i + 48),
            )
            i += 64

        var acc = acc0 + acc1 + acc2 + acc3
        while i <= d - 16:
            acc = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](acc, x.load[width=16](i), y.load[width=16](i))
            i += 16

        var result = acc.reduce_add()
        while i < d:
            result += Int32(x[i]) * Int32(y[i])
            i += 1
        return result
    else:
        var acc0 = SIMD[DType.int32, 16]()
        var acc1 = SIMD[DType.int32, 16]()
        var acc2 = SIMD[DType.int32, 16]()
        var acc3 = SIMD[DType.int32, 16]()

        var i = 0
        while i <= d - 64:
            var x0 = x.load[width=16](i).cast[DType.int16]()
            var y0 = y.load[width=16](i).cast[DType.int16]()
            acc0 += (x0 * y0).cast[DType.int32]()
            var x1 = x.load[width=16](i + 16).cast[DType.int16]()
            var y1 = y.load[width=16](i + 16).cast[DType.int16]()
            acc1 += (x1 * y1).cast[DType.int32]()
            var x2 = x.load[width=16](i + 32).cast[DType.int16]()
            var y2 = y.load[width=16](i + 32).cast[DType.int16]()
            acc2 += (x2 * y2).cast[DType.int32]()
            var x3 = x.load[width=16](i + 48).cast[DType.int16]()
            var y3 = y.load[width=16](i + 48).cast[DType.int16]()
            acc3 += (x3 * y3).cast[DType.int32]()
            i += 64

        var acc = acc0 + acc1 + acc2 + acc3
        while i <= d - 16:
            var vx = x.load[width=16](i).cast[DType.int16]()
            var vy = y.load[width=16](i).cast[DType.int16]()
            acc += (vx * vy).cast[DType.int32]()
            i += 16

        var result = acc.reduce_add()
        while i < d:
            result += Int32(x[i]) * Int32(y[i])
            i += 1
        return result

@always_inline
def sq8_l2_from_dot(norm_a: UInt32, norm_b: UInt32, dot: UInt32) -> UInt32:
    """
    Computes the squared L2 distance from precomputed norms and their dot product.
    Formula: `||a - b||^2 = ||a||^2 + ||b||^2 - 2(a dot b)`

    Args:
        norm_a: The squared norm of the first vector.
        norm_b: The squared norm of the second vector.
        dot: The dot product of the two vectors.

    Returns:
        The derived squared L2 distance.
    """
    var d = Int64(norm_a) + Int64(norm_b) - 2 * Int64(dot)
    return UInt32(math.max(d, 0))


from std.collections import InlineArray


@always_inline
def sq8_signed_dot_product_simd_batch4(
    x0: UnsafePointer[Int8, _],
    x1: UnsafePointer[Int8, _],
    x2: UnsafePointer[Int8, _],
    x3: UnsafePointer[Int8, _],
    query: UnsafePointer[Int8, _],
    d: Int,
) -> InlineArray[Int32, 4]:
    """Computes four signed SQ8 dots while loading each query block once."""
    var result = InlineArray[Int32, 4](uninitialized=True)
    comptime if is_apple_gpu():
        var acc0 = SIMD[DType.int32, 4]()
        var acc1 = SIMD[DType.int32, 4]()
        var acc2 = SIMD[DType.int32, 4]()
        var acc3 = SIMD[DType.int32, 4]()
        var i = 0
        while i <= d - 16:
            var q = query.load[width=16](i)
            acc0 = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](acc0, x0.load[width=16](i), q)
            acc1 = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](acc1, x1.load[width=16](i), q)
            acc2 = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](acc2, x2.load[width=16](i), q)
            acc3 = llvm_intrinsic[
                "llvm.aarch64.neon.sdot.v4i32.v16i8",
                SIMD[DType.int32, 4],
            ](acc3, x3.load[width=16](i), q)
            i += 16
        result[0] = acc0.reduce_add()
        result[1] = acc1.reduce_add()
        result[2] = acc2.reduce_add()
        result[3] = acc3.reduce_add()
        while i < d:
            var q = Int32(query[i])
            result[0] += Int32(x0[i]) * q
            result[1] += Int32(x1[i]) * q
            result[2] += Int32(x2[i]) * q
            result[3] += Int32(x3[i]) * q
            i += 1
        return result
    else:
        var acc0 = SIMD[DType.int32, 16]()
        var acc1 = SIMD[DType.int32, 16]()
        var acc2 = SIMD[DType.int32, 16]()
        var acc3 = SIMD[DType.int32, 16]()
        var i = 0
        while i <= d - 16:
            var q = query.load[width=16](i).cast[DType.int16]()
            acc0 += (
                x0.load[width=16](i).cast[DType.int16]() * q
            ).cast[DType.int32]()
            acc1 += (
                x1.load[width=16](i).cast[DType.int16]() * q
            ).cast[DType.int32]()
            acc2 += (
                x2.load[width=16](i).cast[DType.int16]() * q
            ).cast[DType.int32]()
            acc3 += (
                x3.load[width=16](i).cast[DType.int16]() * q
            ).cast[DType.int32]()
            i += 16
        result[0] = acc0.reduce_add()
        result[1] = acc1.reduce_add()
        result[2] = acc2.reduce_add()
        result[3] = acc3.reduce_add()
        while i < d:
            var q = Int32(query[i])
            result[0] += Int32(x0[i]) * q
            result[1] += Int32(x1[i]) * q
            result[2] += Int32(x2[i]) * q
            result[3] += Int32(x3[i]) * q
            i += 1
        return result


@always_inline
def l2_distance_simd_batch4[simd_width: Int](
    x0: UnsafePointer[Float32, MutUntrackedOrigin],
    x1: UnsafePointer[Float32, MutUntrackedOrigin],
    x2: UnsafePointer[Float32, MutUntrackedOrigin],
    x3: UnsafePointer[Float32, MutUntrackedOrigin],
    y: UnsafePointer[Float32, MutUntrackedOrigin],
    d: Int
) -> InlineArray[Float32, 4]:
    var dist0 = SIMD[DType.float32, simd_width](0.0)
    var dist1 = SIMD[DType.float32, simd_width](0.0)
    var dist2 = SIMD[DType.float32, simd_width](0.0)
    var dist3 = SIMD[DType.float32, simd_width](0.0)

    var i = 0
    while i <= d - simd_width:
        var vy = y.load[width=simd_width](i)

        var vx0 = x0.load[width=simd_width](i)
        var diff0 = vx0 - vy
        dist0 = fma(diff0, diff0, dist0)

        var vx1 = x1.load[width=simd_width](i)
        var diff1 = vx1 - vy
        dist1 = fma(diff1, diff1, dist1)

        var vx2 = x2.load[width=simd_width](i)
        var diff2 = vx2 - vy
        dist2 = fma(diff2, diff2, dist2)

        var vx3 = x3.load[width=simd_width](i)
        var diff3 = vx3 - vy
        dist3 = fma(diff3, diff3, dist3)

        i += simd_width

    var res0 = dist0.reduce_add()
    var res1 = dist1.reduce_add()
    var res2 = dist2.reduce_add()
    var res3 = dist3.reduce_add()

    # Handle remainder
    while i < d:
        var vy = y[i]

        var diff0 = x0[i] - vy
        res0 += diff0 * diff0

        var diff1 = x1[i] - vy
        res1 += diff1 * diff1

        var diff2 = x2[i] - vy
        res2 += diff2 * diff2

        var diff3 = x3[i] - vy
        res3 += diff3 * diff3

        i += 1

    var ret = InlineArray[Float32, 4](uninitialized=True)
    var ret_ptr = ret.unsafe_ptr()
    ret_ptr[0] = res0
    ret_ptr[1] = res1
    ret_ptr[2] = res2
    ret_ptr[3] = res3
    return ret

@always_inline
def inner_product_simd_batch4[simd_width: Int](
    x0: UnsafePointer[Float32, MutUntrackedOrigin],
    x1: UnsafePointer[Float32, MutUntrackedOrigin],
    x2: UnsafePointer[Float32, MutUntrackedOrigin],
    x3: UnsafePointer[Float32, MutUntrackedOrigin],
    y: UnsafePointer[Float32, MutUntrackedOrigin],
    d: Int
) -> InlineArray[Float32, 4]:
    var ip0 = SIMD[DType.float32, simd_width](0.0)
    var ip1 = SIMD[DType.float32, simd_width](0.0)
    var ip2 = SIMD[DType.float32, simd_width](0.0)
    var ip3 = SIMD[DType.float32, simd_width](0.0)

    var i = 0
    while i <= d - simd_width:
        var vy = y.load[width=simd_width](i)

        var vx0 = x0.load[width=simd_width](i)
        ip0 = fma(vx0, vy, ip0)

        var vx1 = x1.load[width=simd_width](i)
        ip1 = fma(vx1, vy, ip1)

        var vx2 = x2.load[width=simd_width](i)
        ip2 = fma(vx2, vy, ip2)

        var vx3 = x3.load[width=simd_width](i)
        ip3 = fma(vx3, vy, ip3)

        i += simd_width

    var res0 = ip0.reduce_add()
    var res1 = ip1.reduce_add()
    var res2 = ip2.reduce_add()
    var res3 = ip3.reduce_add()

    while i < d:
        var vy = y[i]
        res0 += x0[i] * vy
        res1 += x1[i] * vy
        res2 += x2[i] * vy
        res3 += x3[i] * vy
        i += 1

    var ret = InlineArray[Float32, 4](uninitialized=True)
    var ret_ptr = ret.unsafe_ptr()
    ret_ptr[0] = res0
    ret_ptr[1] = res1
    ret_ptr[2] = res2
    ret_ptr[3] = res3
    return ret
