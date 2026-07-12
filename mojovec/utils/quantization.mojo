# quantization.mojo
from std.math import min, max

@always_inline
def encode_8bit_simd[simd_width: Int](
    x: SIMD[DType.float32, simd_width], 
    vmin: SIMD[DType.float32, simd_width], 
    vdiff: SIMD[DType.float32, simd_width]
) -> SIMD[DType.uint8, simd_width]:
    
    # Avoid division by zero and normalize
    var xi = SIMD[DType.float32, simd_width]()
    for i in range(simd_width):
        if vdiff[i] == 0.0:
            xi[i] = 0.0
        else:
            xi[i] = (x[i] - vmin[i]) / vdiff[i]
    
    # clamp
    xi = xi.clamp(0.0, 1.0)
    
    # scale and round
    var scaled = (xi * 255.0 + 0.5).cast[DType.uint8]()
    return scaled

@always_inline
def decode_8bit_simd[simd_width: Int](
    code: SIMD[DType.uint8, simd_width],
    vmin: SIMD[DType.float32, simd_width],
    vdiff: SIMD[DType.float32, simd_width]
) -> SIMD[DType.float32, simd_width]:
    
    var xi = code.cast[DType.float32]() / 255.0
    var x = xi * vdiff + vmin
    return x
