"""
Provides implementations for vector quantization, including scalar quantization for 8-bit and fp16.
"""

from ..core.types import QuantizerType, QT_8bit, QT_fp16
from ..utils.quantization import encode_8bit_simd, decode_8bit_simd
from std.collections import List
from std.memory import ArcPointer
from std.memory.span import Span

struct _ScalarQuantizerCalibration(Movable):
    """Shared, managed storage for immutable post-training calibration data."""

    var vmin: List[Float32]
    var vdiff: List[Float32]

    def __init__(out self, d: Int):
        self.vmin = List[Float32](unsafe_uninit_length=d)
        self.vdiff = List[Float32](unsafe_uninit_length=d)


struct ScalarQuantizer(Copyable):
    """
    A scalar quantizer that encodes and decodes vectors using 8-bit or 16-bit precision.
    """
    var d: Int
    var qtype: QuantizerType
    var is_trained: Bool
    
    # Search creates lightweight copies of a quantizer for distance computers.
    # ArcPointer makes those copies O(1) and gives the calibration arrays one
    # unambiguous lifetime. The arrays are only mutated by train(), before the
    # index is published for querying.
    var _calibration: ArcPointer[_ScalarQuantizerCalibration]
    
    def __init__(out self, d: Int, qtype: QuantizerType):
        """
        Initializes the scalar quantizer.
        
        Args:
            d: The dimensionality of the vectors.
            qtype: The target quantization type (e.g., QT_8bit or QT_fp16).
        """
        self.d = d
        self.qtype = qtype
        self.is_trained = (qtype == QT_fp16)
        
        self._calibration = ArcPointer(_ScalarQuantizerCalibration(d))

    def __init__(out self, *, copy: Self):
        self.d = copy.d
        self.qtype = copy.qtype
        self.is_trained = copy.is_trained
        self._calibration = copy._calibration

    @always_inline
    def vmin_at(self, i: Int) -> Float32:
        """Returns the trained minimum for one vector component."""
        return self._calibration[].vmin[i]

    @always_inline
    def vdiff_at(self, i: Int) -> Float32:
        """Returns the trained range for one vector component."""
        return self._calibration[].vdiff[i]
            
    def code_size(self) -> Int:
        """
        Returns the byte size required to store a single encoded vector.
        
        Returns:
            The size in bytes based on the quantization type.
        """
        if self.qtype == QT_8bit:
            return self.d
        elif self.qtype == QT_fp16:
            return self.d * 2
        return self.d

    def train(mut self, x: Span[Float32, _]):
        """
        Trains the quantizer on a provided set of vectors to determine value ranges.
        
        Args:
            x: The flattened training vectors. Its length must be divisible by
                the quantizer dimension.
        """
        if self.qtype == QT_fp16:
            return
        var n = len(x) // self.d
        if n == 0:
            return

        var x_data = x.unsafe_ptr()
        var vmax = List[Float32](unsafe_uninit_length=self.d)
        var vmin_ptr = self._calibration[].vmin.unsafe_ptr()
        var vdiff_ptr = self._calibration[].vdiff.unsafe_ptr()
        var vmax_ptr = vmax.unsafe_ptr()
        for j in range(self.d):
            vmin_ptr[j] = x_data[j]
            vmax_ptr[j] = x_data[j]
            
        for i in range(1, n):
            var x_ptr = x_data + (i * self.d)
            for j in range(self.d):
                if x_ptr[j] < vmin_ptr[j]:
                    vmin_ptr[j] = x_ptr[j]
                if x_ptr[j] > vmax_ptr[j]:
                    vmax_ptr[j] = x_ptr[j]
                    
        for i in range(self.d):
            vdiff_ptr[i] = vmax_ptr[i] - vmin_ptr[i]

        self.is_trained = True

    def encode(self, x: UnsafePointer[Float32, _], codes: UnsafePointer[UInt8, MutUntrackedOrigin]):
        """
        Encodes a single vector into its quantized representation.
        
        Args:
            x: A pointer to the input vector.
            codes: A pointer to the output buffer for the quantized code.
        """
        if self.qtype == QT_fp16:
            var codes_fp16 = codes.bitcast[Float16]()
            for i in range(self.d):
                codes_fp16[i] = x[i].cast[DType.float16]()
        elif self.qtype == QT_8bit:
            var vmin_ptr = self._calibration[].vmin.unsafe_ptr()
            var vdiff_ptr = self._calibration[].vdiff.unsafe_ptr()
            for i in range(self.d):
                var vdiff_safe = vdiff_ptr[i]
                if vdiff_safe == 0.0:
                    codes[i] = 0
                else:
                    var xi = (x[i] - vmin_ptr[i]) / vdiff_safe
                    if xi < 0.0:
                        codes[i] = 0
                    elif xi > 1.0:
                        codes[i] = 255
                    else:
                        codes[i] = (xi * 255.0 + 0.5).cast[DType.uint8]()

    def decode(self, codes: UnsafePointer[UInt8, MutUntrackedOrigin], x: UnsafePointer[Float32, MutUntrackedOrigin]):
        """
        Decodes a quantized representation back into a float32 vector.
        
        Args:
            codes: A pointer to the quantized code.
            x: A pointer to the output buffer for the reconstructed vector.
        """
        if self.qtype == QT_fp16:
            var codes_fp16 = codes.bitcast[Float16]()
            for i in range(self.d):
                x[i] = codes_fp16[i].cast[DType.float32]()
        elif self.qtype == QT_8bit:
            var vmin_ptr = self._calibration[].vmin.unsafe_ptr()
            var vdiff_ptr = self._calibration[].vdiff.unsafe_ptr()
            for i in range(self.d):
                var xi = codes[i].cast[DType.float32]() / 255.0
                x[i] = xi * vdiff_ptr[i] + vmin_ptr[i]
