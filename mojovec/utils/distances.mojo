"""
Provides optimized SIMD implementations for computing various vector distances.
"""

from std.math import fma
@always_inline
def l2_distance_simd[simd_width: Int](x: UnsafePointer[Float32, MutUntrackedOrigin], y: UnsafePointer[Float32, MutUntrackedOrigin], d: Int) -> Float32:
    """
    Computes the squared L2 distance between two vectors of dimension `d` using SIMD instructions.
    
    Args:
        x: Pointer to the first vector.
        y: Pointer to the second vector.
        d: The dimensionality of the vectors.
        
    Returns:
        The squared L2 distance.
    """
    var dist0 = SIMD[DType.float32, simd_width]()
    var dist1 = SIMD[DType.float32, simd_width]()
    var dist2 = SIMD[DType.float32, simd_width]()
    var dist3 = SIMD[DType.float32, simd_width]()
    
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
    
    # Handle remainder
    while i < d:
        var diff = x[i] - y[i]
        res += diff * diff
        i += 1
        
    return res

@always_inline
def inner_product_simd[simd_width: Int](x: UnsafePointer[Float32, MutUntrackedOrigin], y: UnsafePointer[Float32, MutUntrackedOrigin], d: Int) -> Float32:
    """
    Computes the inner product between two vectors of dimension `d` using SIMD instructions.
    
    Args:
        x: Pointer to the first vector.
        y: Pointer to the second vector.
        d: The dimensionality of the vectors.
        
    Returns:
        The computed inner product.
    """
    var prod0 = SIMD[DType.float32, simd_width]()
    var prod1 = SIMD[DType.float32, simd_width]()
    var prod2 = SIMD[DType.float32, simd_width]()
    var prod3 = SIMD[DType.float32, simd_width]()
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
    
    # Handle remainder
    while i < d:
        res += x[i] * y[i]
        i += 1
        
    return res

from std.sys.intrinsics import llvm_intrinsic
import std.math as math

@always_inline
def sq8_dot_product_simd(
    x: UnsafePointer[UInt8, MutUntrackedOrigin],
    y: UnsafePointer[UInt8, MutUntrackedOrigin],
    d: Int,
) -> UInt32:
    """
    Computes the dot product between two UInt8 vectors using SIMD.
    Works universally across architectures (ARM, x86) while maintaining
    a fast intermediate 16-bit multiplication to assist LLVM idiom recognition.
    
    Args:
        x: Pointer to the first vector.
        y: Pointer to the second vector.
        d: The dimensionality of the vectors.
        
    Returns:
        The computed dot product.
    """
    var acc0 = SIMD[DType.uint32, 16]()
    var acc1 = SIMD[DType.uint32, 16]()
    var acc2 = SIMD[DType.uint32, 16]()
    var acc3 = SIMD[DType.uint32, 16]()
    
    var i = 0
    # Unroll 4x (4 * 16 bytes = 64 bytes per iteration)
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
    
    # 1x for remaining 16-byte chunks
    while i <= d - 16:
        var vx = x.load[width=16](i).cast[DType.uint16]()
        var vy = y.load[width=16](i).cast[DType.uint16]()
        acc += (vx * vy).cast[DType.uint32]()
        i += 16
        
    var res = acc.reduce_add()
    
    # Handle remainder (scalar)
    while i < d:
        res += UInt32(x[i]) * UInt32(y[i])
        i += 1
        
    return res

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
