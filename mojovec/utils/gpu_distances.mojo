from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer

@always_inline
def l2_kernel_single_thread(
    x: UnsafePointer[Float32, MutUntrackedOrigin],
    y: UnsafePointer[Float32, MutUntrackedOrigin],
    out: UnsafePointer[Float32, MutUntrackedOrigin],
    d: Int
):
    var tid = global_idx.x
    if tid == 0:
        var sum: Float32 = 0.0
        for i in range(d):
            var diff = x[i] - y[i]
            sum += diff * diff
        out[0] = sum

struct GPUDistanceComputer:
    var ctx: DeviceContext
    var out_buf: DeviceBuffer[DType.float32]
    var out_host: UnsafePointer[Float32, MutUntrackedOrigin]
    
    def __init__(out self):
        self.ctx = DeviceContext()
        self.out_buf = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.out_host = UnsafePointer[Float32, MutUntrackedOrigin].alloc(1)
        
    def __del__(deinit self):
        self.out_host.free()
        
    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx
        self.out_buf = move.out_buf
        self.out_host = move.out_host

    @always_inline
    def compute(self, x: UnsafePointer[Float32, MutUntrackedOrigin], y: UnsafePointer[Float32, MutUntrackedOrigin], d: Int) -> Float32:
        self.ctx.enqueue_function[l2_kernel_single_thread](x, y, self.out_buf.unsafe_ptr(), d, grid_dim=1, block_dim=1)
        self.ctx.synchronize()
        # copy back
        self.ctx.enqueue_copy(self.out_host, self.out_buf)
        self.ctx.synchronize()
        return self.out_host[0]

var gpu_comp = GPUDistanceComputer()

@always_inline
def gpu_l2_distance(x: UnsafePointer[Float32, MutUntrackedOrigin], y: UnsafePointer[Float32, MutUntrackedOrigin], d: Int) -> Float32:
    return gpu_comp.compute(x, y, d)
