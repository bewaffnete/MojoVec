# quantizer.mojo

from ..core.types import QuantizerType, QT_8bit, QT_fp16
from ..utils.quantization import encode_8bit_simd, decode_8bit_simd

@fieldwise_init
struct ScalarQuantizer(Movable, Copyable):
    var d: Int
    var qtype: QuantizerType
    var is_trained: Bool
    
    var vmin: UnsafePointer[Float32, MutUntrackedOrigin]
    var vdiff: UnsafePointer[Float32, MutUntrackedOrigin]
    
    def __init__(out self, d: Int, qtype: QuantizerType):
        self.d = d
        self.qtype = qtype
        self.is_trained = (qtype == QT_fp16)
        
        self.vmin = alloc[Float32](self.d)
        self.vdiff = alloc[Float32](self.d)
            
    def __del__(deinit self):
        self.vmin.free()
        self.vdiff.free()
            
    def code_size(self) -> Int:
        if self.qtype == QT_8bit:
            return self.d
        elif self.qtype == QT_fp16:
            return self.d * 2
        return self.d

    def train(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        if self.qtype == QT_fp16:
            return
        if n == 0:
            return
            
        var vmax = alloc[Float32](self.d)
        for j in range(self.d):
            self.vmin[j] = x[j]
            vmax[j] = x[j]
            
        for i in range(1, n):
            var x_ptr = x + (i * self.d)
            for j in range(self.d):
                if x_ptr[j] < self.vmin[j]:
                    self.vmin[j] = x_ptr[j]
                if x_ptr[j] > vmax[j]:
                    vmax[j] = x_ptr[j]
                    
        for i in range(self.d):
            self.vdiff[i] = vmax[i] - self.vmin[i]
            
        vmax.free()
        self.is_trained = True

    def encode(self, x: UnsafePointer[Float32, MutUntrackedOrigin], codes: UnsafePointer[UInt8, MutUntrackedOrigin]):
        if self.qtype == QT_fp16:
            var codes_fp16 = codes.bitcast[Float16]()
            for i in range(self.d):
                codes_fp16[i] = x[i].cast[DType.float16]()
        elif self.qtype == QT_8bit:
            for i in range(self.d):
                var vdiff_safe = self.vdiff[i]
                if vdiff_safe == 0.0:
                    codes[i] = 0
                else:
                    var xi = (x[i] - self.vmin[i]) / vdiff_safe
                    if xi < 0.0:
                        codes[i] = 0
                    elif xi > 1.0:
                        codes[i] = 255
                    else:
                        codes[i] = (xi * 255.0 + 0.5).cast[DType.uint8]()

    def decode(self, codes: UnsafePointer[UInt8, MutUntrackedOrigin], x: UnsafePointer[Float32, MutUntrackedOrigin]):
        if self.qtype == QT_fp16:
            var codes_fp16 = codes.bitcast[Float16]()
            for i in range(self.d):
                x[i] = codes_fp16[i].cast[DType.float32]()
        elif self.qtype == QT_8bit:
            for i in range(self.d):
                var xi = codes[i].cast[DType.float32]() / 255.0
                x[i] = xi * self.vdiff[i] + self.vmin[i]
