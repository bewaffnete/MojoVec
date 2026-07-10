# distances.mojo

from std.math import fma
@always_inline
def l2_distance_simd[simd_width: Int](x: UnsafePointer[Float32, MutUntrackedOrigin], y: UnsafePointer[Float32, MutUntrackedOrigin], d: Int) -> Float32:
    """
    Computes the squared L2 distance between two vectors of dimension d using SIMD.
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
    Computes the inner product between two vectors of dimension d using SIMD.
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
