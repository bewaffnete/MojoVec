# distances.mojo

from std.math import fma
@always_inline
def l2_distance_simd[simd_width: Int](x: UnsafePointer[Float32, MutUntrackedOrigin], y: UnsafePointer[Float32, MutUntrackedOrigin], d: Int) -> Float32:
    """
    Computes the squared L2 distance between two vectors of dimension d using SIMD.
    """
    var dist = SIMD[DType.float32, simd_width]()
    var i = 0
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
    Computes the inner product between two vectors of dimension d using SIMD.
    """
    var prod = SIMD[DType.float32, simd_width]()
    var i = 0
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
