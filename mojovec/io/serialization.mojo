from std.io.file import FileHandle
from std.memory.span import Span
from std.memory import alloc
from ..core.index import Index
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from ..index.index_flat import IndexFlat
from ..index.index_ivf_flat import IndexIVFFlat
from ..index.index_ivf_pq import IndexIVFPQ
from ..storage.inverted_lists import ArrayInvertedLists
from ..quantization.pq import ProductQuantizer
from ..index.index_hnsw import IndexHNSW
from ..index.hnsw_graph import HNSWGraph
comptime MAGIC_FLAT: Int = 0x4d4a4f46
comptime MAGIC_HNSW: Int = 0x4d4a4f48
comptime MAGIC_IVF_FLAT: Int = 0x4d4a4f49
comptime MAGIC_IVF_PQ: Int = 0x4d4a4f50
comptime MAGIC_INVLISTS: Int = 0x4d4a4f4c
comptime MAGIC_PQ: Int = 0x4d4a4f51

# --- Primitive I/O ---

def write_int(mut f: FileHandle, val: Int) raises:
    var ptr = alloc[Int](1)
    ptr[0] = val
    var span = Span[UInt8, MutUntrackedOrigin](ptr=ptr.bitcast[UInt8](), length=8)
    f.write_bytes(span)
    ptr.free()

def read_int(mut f: FileHandle) raises -> Int:
    var read_data = f.read_bytes(8)
    var ptr = read_data.unsafe_ptr().bitcast[Int]()
    var val = ptr[0]
    _ = len(read_data)
    return val

def write_bool(mut f: FileHandle, val: Bool) raises:
    var ptr = alloc[Bool](1)
    ptr[0] = val
    var span = Span[UInt8, MutUntrackedOrigin](ptr=ptr.bitcast[UInt8](), length=1)
    f.write_bytes(span)
    ptr.free()

def read_bool(mut f: FileHandle) raises -> Bool:
    var read_data = f.read_bytes(1)
    var ptr = read_data.unsafe_ptr().bitcast[Bool]()
    var val = ptr[0]
    _ = len(read_data)
    return val

def write_unsafe_pointer_float32(mut f: FileHandle, ptr: UnsafePointer[Float32, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var span = Span[UInt8, MutUntrackedOrigin](ptr=ptr.bitcast[UInt8](), length=count * 4)
    f.write_bytes(span)

def read_unsafe_pointer_float32(mut f: FileHandle, ptr: UnsafePointer[Float32, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var read_data = f.read_bytes(count * 4)
    var src = read_data.unsafe_ptr().bitcast[Float32]()
    for i in range(count):
        ptr[i] = src[i]
    _ = len(read_data)

def write_unsafe_pointer_uint8(mut f: FileHandle, ptr: UnsafePointer[UInt8, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var span = Span[UInt8, MutUntrackedOrigin](ptr=ptr, length=count)
    f.write_bytes(span)

def read_unsafe_pointer_uint8(mut f: FileHandle, ptr: UnsafePointer[UInt8, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var read_data = f.read_bytes(count)
    var src = read_data.unsafe_ptr()
    for i in range(count):
        ptr[i] = src[i]
    _ = len(read_data)

def write_unsafe_pointer_int(mut f: FileHandle, ptr: UnsafePointer[Int, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var span = Span[UInt8, MutUntrackedOrigin](ptr=ptr.bitcast[UInt8](), length=count * 8)
    f.write_bytes(span)

def read_unsafe_pointer_int(mut f: FileHandle, ptr: UnsafePointer[Int, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var read_data = f.read_bytes(count * 8)
    var src = read_data.unsafe_ptr().bitcast[Int]()
    for i in range(count):
        ptr[i] = src[i]
    _ = len(read_data)

# --- IndexFlat ---

def write_index_flat(mut f: FileHandle, index: IndexFlat) raises:
    write_int(f, MAGIC_FLAT)
    write_int(f, index.d)
    write_int(f, index.ntotal)
    write_int(f, index.capacity)
    var metric = 0
    if index.metric_type == METRIC_INNER_PRODUCT: metric = 1
    write_int(f, metric)
    
    write_unsafe_pointer_float32(f, index.codes, index.capacity * index.d)

def read_index_flat(mut f: FileHandle) raises -> IndexFlat:
    var magic = read_int(f)
    if magic != MAGIC_FLAT: raise Error("Invalid magic for IndexFlat")
    
    var d = read_int(f)
    var ntotal = read_int(f)
    var capacity = read_int(f)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var index = IndexFlat(d, metric)
    index.ntotal = ntotal
    index.capacity = capacity
    
    # We allocated d * 1024 inside init, so we must reallocate if capacity is larger
    if Int(index.codes) != 0: index.codes.free()
    index.codes = alloc[Float32](capacity * d)
    
    read_unsafe_pointer_float32(f, index.codes, capacity * d)
    return index^

# --- HNSWGraph and IndexHNSW ---

def write_hnsw_graph(mut f: FileHandle, graph: HNSWGraph) raises:
    write_int(f, graph.M)
    write_int(f, graph.efConstruction)
    write_int(f, graph.efSearch)
    write_int(f, graph.max_level)
    write_int(f, graph.entry_point)
    write_int(f, graph.ntotal)
    write_int(f, graph.capacity)
    write_int(f, graph.neighbors_capacity)
    
    write_unsafe_pointer_int(f, graph.levels, graph.capacity)
    write_unsafe_pointer_int(f, graph.offsets, graph.capacity + 1)
    
    if graph.neighbors_capacity > 0:
        var span_neighbors = Span[UInt8, MutUntrackedOrigin](ptr=graph.neighbors.bitcast[UInt8](), length=graph.neighbors_capacity * 4)
        f.write_bytes(span_neighbors)
    
    write_unsafe_pointer_int(f, graph.cum_nneighbor_per_level, 33)

def read_hnsw_graph(mut f: FileHandle, mut graph: HNSWGraph) raises:
    graph.M = read_int(f)
    graph.efConstruction = read_int(f)
    graph.efSearch = read_int(f)
    graph.max_level = read_int(f)
    graph.entry_point = read_int(f)
    graph.ntotal = read_int(f)
    
    var capacity = read_int(f)
    var neighbors_capacity = read_int(f)
    
    if capacity > graph.capacity:
        graph.capacity = capacity
        if Int(graph.levels) != 0: graph.levels.free()
        if Int(graph.offsets) != 0: graph.offsets.free()
        graph.levels = alloc[Int](capacity)
        graph.offsets = alloc[Int](capacity + 1)
        
    if neighbors_capacity > graph.neighbors_capacity:
        graph.neighbors_capacity = neighbors_capacity
        if Int(graph.neighbors) != 0: graph.neighbors.free()
        graph.neighbors = alloc[Int32](neighbors_capacity)
        
    read_unsafe_pointer_int(f, graph.levels, capacity)
    read_unsafe_pointer_int(f, graph.offsets, capacity + 1)
    
    if neighbors_capacity > 0:
        var read_data = f.read_bytes(neighbors_capacity * 4)
        var src = read_data.unsafe_ptr().bitcast[Int32]()
        for i in range(neighbors_capacity):
            graph.neighbors[i] = src[i]
        _ = len(read_data)
        
    read_unsafe_pointer_int(f, graph.cum_nneighbor_per_level, 33)

def write_index_hnsw(mut f: FileHandle, index: IndexHNSW[IndexFlat]) raises:
    write_int(f, MAGIC_HNSW)
    write_int(f, index.d)
    write_int(f, index.ntotal)
    write_bool(f, index.is_trained)
    var metric = 0
    if index.metric_type == METRIC_INNER_PRODUCT: metric = 1
    write_int(f, metric)
    
    write_index_flat(f, index.storage)
    write_hnsw_graph(f, index.hnsw)

def read_index_hnsw(mut f: FileHandle) raises -> IndexHNSW[IndexFlat]:
    var magic = read_int(f)
    if magic != MAGIC_HNSW: raise Error("Invalid magic for IndexHNSW")
    
    var d = read_int(f)
    var ntotal = read_int(f)
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var storage = read_index_flat(f)
    var index = IndexHNSW[IndexFlat](storage^, d, metric, M=32)
    index.ntotal = ntotal
    index.is_trained = is_trained
    read_hnsw_graph(f, index.hnsw)
    return index^

# --- ArrayInvertedLists ---

def write_invlists(mut f: FileHandle, invlists: ArrayInvertedLists) raises:
    write_int(f, MAGIC_INVLISTS)
    write_int(f, invlists.nlist)
    write_int(f, invlists.code_size)
    
    _ = Int(invlists.lists) # Alias analysis workaround
    for i in range(invlists.nlist):
        write_int(f, invlists.lists[i].size)
        write_int(f, invlists.lists[i].capacity)

        write_unsafe_pointer_int(f, invlists.lists[i].ids, invlists.lists[i].size)
        write_unsafe_pointer_uint8(f, invlists.lists[i].codes, invlists.lists[i].size * invlists.code_size)

def read_invlists(mut f: FileHandle, mut invlists: ArrayInvertedLists) raises:
    var magic = read_int(f)
    if magic != MAGIC_INVLISTS: raise Error("Invalid magic for ArrayInvertedLists")
    
    var nlist = read_int(f)
    var code_size = read_int(f)
    
    # We assume invlists is already initialized. Just update it.
    # Need to free old lists if any? Wait, we can just let it resize.
    invlists.nlist = nlist
    invlists.code_size = code_size
    
    for i in range(nlist):
        var size = read_int(f)
        var capacity = read_int(f)
        
        invlists.resize(i, capacity)
        _ = Int(invlists.lists)
        invlists.lists[i].size = size
        
        var list_codes = invlists.get_codes(i)
        var list_ids = invlists.get_ids(i)
        
        read_unsafe_pointer_int(f, list_ids, size)
        read_unsafe_pointer_uint8(f, list_codes, size * code_size)

# --- ProductQuantizer ---

def write_pq(mut f: FileHandle, pq: ProductQuantizer) raises:
    write_int(f, MAGIC_PQ)
    write_int(f, pq.d)
    write_int(f, pq.M)
    write_int(f, pq.ksub)
    write_bool(f, pq.is_trained)
    write_unsafe_pointer_float32(f, pq.centroids, pq.M * pq.ksub * pq.dsub)

def read_pq(mut f: FileHandle, mut pq: ProductQuantizer) raises:
    var magic = read_int(f)
    if magic != MAGIC_PQ: raise Error("Invalid magic for ProductQuantizer")
    
    pq.d = read_int(f)
    pq.M = read_int(f)
    pq.ksub = read_int(f)
    pq.dsub = pq.d // pq.M
    pq.is_trained = read_bool(f)
    
    if Int(pq.centroids) != 0: pq.centroids.free()
    pq.centroids = alloc[Float32](pq.M * pq.ksub * pq.dsub)
    read_unsafe_pointer_float32(f, pq.centroids, pq.M * pq.ksub * pq.dsub)

# --- IndexIVFFlat ---

def write_index_ivf_flat(mut f: FileHandle, index: IndexIVFFlat[IndexFlat]) raises:
    write_int(f, MAGIC_IVF_FLAT)
    write_int(f, index.d)
    write_int(f, index.nlist)
    write_int(f, index.nprobe)
    write_int(f, index.ntotal)
    write_bool(f, index.is_trained)
    
    var metric = 0
    if index.metric_type == METRIC_INNER_PRODUCT: metric = 1
    write_int(f, metric)
    
    write_index_flat(f, index.quantizer[0])
    write_invlists(f, index.invlists)

def read_index_ivf_flat(mut f: FileHandle) raises -> IndexIVFFlat[IndexFlat]:
    var magic = read_int(f)
    if magic != MAGIC_IVF_FLAT: raise Error("Invalid magic for IndexIVFFlat")
    
    var d = read_int(f)
    var nlist = read_int(f)
    var nprobe = read_int(f)
    var ntotal = read_int(f)
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(read_index_flat(f))
    
    var index = IndexIVFFlat[IndexFlat](quantizer, d, nlist, metric)
    index.nprobe = nprobe
    index.ntotal = ntotal
    index.is_trained = is_trained
    
    read_invlists(f, index.invlists)
    return index^

# --- IndexIVFPQ ---

def write_index_ivf_pq(mut f: FileHandle, index: IndexIVFPQ[IndexFlat]) raises:
    write_int(f, MAGIC_IVF_PQ)
    write_int(f, index.d)
    write_int(f, index.nlist)
    write_int(f, index.M)
    write_int(f, index.nprobe)
    write_int(f, index.ntotal)
    write_bool(f, index.is_trained)
    
    var metric = 0
    if index.metric_type == METRIC_INNER_PRODUCT: metric = 1
    write_int(f, metric)
    
    write_index_flat(f, index.quantizer[0])
    write_invlists(f, index.invlists)
    write_pq(f, index.pq)

def read_index_ivf_pq(mut f: FileHandle) raises -> IndexIVFPQ[IndexFlat]:
    var magic = read_int(f)
    if magic != MAGIC_IVF_PQ: raise Error("Invalid magic for IndexIVFPQ")
    
    var d = read_int(f)
    var nlist = read_int(f)
    var M = read_int(f)
    var nprobe = read_int(f)
    var ntotal = read_int(f)
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var quantizer = alloc[IndexFlat](1)
    quantizer.init_pointee_move(read_index_flat(f))
    
    var index = IndexIVFPQ[IndexFlat](quantizer, d, nlist, M, metric)
    index.nprobe = nprobe
    index.ntotal = ntotal
    index.is_trained = is_trained
    
    read_invlists(f, index.invlists)
    read_pq(f, index.pq)
    return index^
