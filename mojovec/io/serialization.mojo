from std.io.file import FileHandle
from std.collections import List
from std.memory.span import Span
from std.memory import alloc
from std.os import SEEK_CUR, SEEK_SET
from ..core.index import Index
from ..core.types import MetricType, METRIC_L2, METRIC_INNER_PRODUCT
from ..index.index_flat import IndexFlat, _alloc_aligned, _free_aligned
from ..index.index_ivf_flat import IndexIVFFlat
from ..index.index_ivf_pq import IndexIVFPQ
from ..index.index_flat_sq8 import IndexFlatSQ8
from ..storage.inverted_lists import ArrayInvertedLists
from ..quantization.pq import ProductQuantizer
from ..index.index_hnsw import IndexHNSW
from ..index.hnsw_graph import HNSWGraph
from mojovec.io.memory_map import FileMemoryMap
from mojovec.core.validation import (
    _validate_hnsw_parameters,
    _validate_vector_dimension,
)
comptime MAGIC_FLAT: Int = 0x4d4a4f46
comptime MAGIC_HNSW: Int = 0x4d4a4f48
comptime MAGIC_IVF_FLAT: Int = 0x4d4a4f49
comptime MAGIC_IVF_PQ: Int = 0x4d4a4f50
comptime MAGIC_INVLISTS: Int = 0x4d4a4f4c
comptime MAGIC_PQ: Int = 0x4d4a4f51
comptime MAGIC_FLAT_SQ8: Int = 0x4d4a4f59
comptime MAGIC_HNSW_SQ8: Int = 0x4d4a4f5A
comptime MAGIC_FLAT_MMAP: Int = 0x4d4a4f54
comptime MAGIC_FLAT_SQ8_MMAP: Int = 0x4d4a4f5B
comptime MAGIC_HNSW_GRAPH_MMAP: Int = 0x4d4a4f56
comptime MAGIC_HNSW_MMAP: Int = 0x4d4a4f57
comptime MAGIC_HNSW_SQ8_MMAP: Int = 0x4d4a4f5C
comptime MMAP_DATA_ALIGNMENT = 64
comptime MAX_SERIALIZED_BYTE_COUNT = 9_223_372_036_854_775_807

# --- Primitive I/O ---

@always_inline
def check_size_limit(size: Int, max_allowed: Int) raises:
    if size < 0 or size > max_allowed:
        raise Error("Security Error: Deserialized size is out of valid range or exceeds safety limits")


@always_inline
def checked_byte_count(count: Int, item_size: Int) raises -> Int:
    """Multiplies serialized element counts without signed overflow."""
    if (
        count < 0
        or item_size <= 0
        or count > MAX_SERIALIZED_BYTE_COUNT // item_size
    ):
        raise Error("Serialized byte count overflows the supported range.")
    return count * item_size


def _read_exact_bytes(mut f: FileHandle, byte_count: Int) raises -> List[UInt8]:
    if byte_count < 0:
        raise Error("Serialized byte count cannot be negative.")
    var data = f.read_bytes(byte_count)
    if len(data) != byte_count:
        raise Error("Unexpected end of serialized data.")
    return data^


def write_int(mut f: FileHandle, val: Int) raises:
    var ptr = alloc[Int](1)
    ptr[0] = val
    var span = Span[UInt8](ptr=ptr.bitcast[UInt8](), length=8)
    f.write_bytes(span)
    ptr.free()

def read_int(mut f: FileHandle) raises -> Int:
    var read_data = _read_exact_bytes(f, 8)
    var ptr = read_data.unsafe_ptr().bitcast[Int]()
    var val = ptr[0]
    _ = len(read_data)
    return val


def write_uint64(mut f: FileHandle, value: UInt64) raises:
    var storage = InlineArray[UInt64, 1](uninitialized=True)
    storage[0] = value
    f.write_all(
        Span[UInt8](
            ptr=storage.unsafe_ptr().bitcast[UInt8](),
            length=8,
        )
    )


def read_uint64(mut f: FileHandle) raises -> UInt64:
    var data = _read_exact_bytes(f, 8)
    var value = data.unsafe_ptr().bitcast[UInt64]()[0]
    _ = len(data)
    return value


def _align_mmap_offset(offset: Int) -> Int:
    return (
        (offset + MMAP_DATA_ALIGNMENT - 1)
        // MMAP_DATA_ALIGNMENT
        * MMAP_DATA_ALIGNMENT
    )


def _write_padding_to(mut f: FileHandle, target: Int) raises:
    var current = Int(f.seek(0, SEEK_CUR))
    if target < current:
        raise Error("Invalid memory-mapped serialization offset.")
    var count = target - current
    if count == 0:
        return
    var padding = List[UInt8](unsafe_uninit_length=count)
    for index in range(count):
        padding[index] = 0
    f.write_bytes(padding)


def _validate_mmap_region(
    byte_offset: Int,
    byte_count: Int,
    file_size: Int,
) raises:
    if (
        file_size < 0
        or byte_offset < 0
        or byte_count < 0
        or byte_offset % MMAP_DATA_ALIGNMENT != 0
        or byte_offset > file_size
        or byte_count > file_size - byte_offset
    ):
        raise Error("Invalid memory-mapped index region.")

def write_bool(mut f: FileHandle, val: Bool) raises:
    var ptr = alloc[Bool](1)
    ptr[0] = val
    var span = Span[UInt8](ptr=ptr.bitcast[UInt8](), length=1)
    f.write_bytes(span)
    ptr.free()

def read_bool(mut f: FileHandle) raises -> Bool:
    var read_data = _read_exact_bytes(f, 1)
    var value = read_data[0]
    if value > 1:
        raise Error("Invalid serialized Bool value.")
    return value == 1

def write_unsafe_pointer_float32(mut f: FileHandle, ptr: UnsafePointer[Float32, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var span = Span[UInt8](ptr=ptr.bitcast[UInt8](), length=count * 4)
    f.write_bytes(span)

def read_unsafe_pointer_float32(mut f: FileHandle, mut ptr: UnsafePointer[Float32, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var read_data = _read_exact_bytes(f, checked_byte_count(count, 4))
    var src = read_data.unsafe_ptr().bitcast[Float32]()
    for i in range(count):
        ptr[i] = src[i]
    _ = len(read_data)

def write_unsafe_pointer_uint8(mut f: FileHandle, ptr: UnsafePointer[UInt8, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var span = Span[UInt8](ptr=ptr, length=count)
    f.write_bytes(span)

def read_unsafe_pointer_uint8(mut f: FileHandle, mut ptr: UnsafePointer[UInt8, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var read_data = _read_exact_bytes(f, count)
    var src = read_data.unsafe_ptr()
    for i in range(count):
        ptr[i] = src[i]
    _ = len(read_data)

def write_unsafe_pointer_uint32(mut f: FileHandle, ptr: UnsafePointer[UInt32, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var span = Span[UInt8](ptr=ptr.bitcast[UInt8](), length=count * 4)
    f.write_bytes(span)

def read_unsafe_pointer_uint32(mut f: FileHandle, mut ptr: UnsafePointer[UInt32, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var read_data = _read_exact_bytes(f, checked_byte_count(count, 4))
    var src = read_data.unsafe_ptr().bitcast[UInt32]()
    for i in range(count):
        ptr[i] = src[i]
    _ = len(read_data)

def write_unsafe_pointer_int(mut f: FileHandle, ptr: UnsafePointer[Int, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var span = Span[UInt8](ptr=ptr.bitcast[UInt8](), length=count * 8)
    f.write_bytes(span)

def read_unsafe_pointer_int(mut f: FileHandle, mut ptr: UnsafePointer[Int, MutUntrackedOrigin], count: Int) raises:
    if count == 0: return
    var read_data = _read_exact_bytes(f, checked_byte_count(count, 8))
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
    _validate_vector_dimension(d)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var capacity = read_int(f)
    check_size_limit(capacity, 1_000_000_000)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var index = IndexFlat(d, metric, initial_capacity=0)
    index.ntotal = ntotal
    _free_aligned(index.codes)
    index.capacity = capacity
    index.codes = _alloc_aligned(capacity * d)
    read_unsafe_pointer_float32(f, index.codes, capacity * d)
        
    return index^

def write_index_flat_sq8(mut f: FileHandle, index: IndexFlatSQ8) raises:
    write_int(f, MAGIC_FLAT_SQ8)
    write_int(f, index.d)
    write_int(f, index.ntotal)
    write_int(f, index.capacity)
    var metric = 0
    if index.metric_type == METRIC_INNER_PRODUCT: metric = 1
    write_int(f, metric)
    
    # Write SQ8 params
    var float32_params = alloc[Float32](3)
    float32_params[0] = index.global_min
    float32_params[1] = index.global_max
    float32_params[2] = index.scale
    write_unsafe_pointer_float32(f, float32_params, 3)
    float32_params.free()
    
    # Write data
    if index.capacity > 0:
        write_unsafe_pointer_float32(f, index.codes_f32, index.capacity * index.d)
        write_unsafe_pointer_uint8(f, index.codes_u8, index.capacity * index.d)
        write_unsafe_pointer_uint32(f, index.norms_u32, index.capacity)

def read_index_flat_sq8(mut f: FileHandle) raises -> IndexFlatSQ8:
    var magic = read_int(f)
    if magic != MAGIC_FLAT_SQ8: raise Error("Invalid magic for IndexFlatSQ8")
    
    var d = read_int(f)
    _validate_vector_dimension(d)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var capacity = read_int(f)
    check_size_limit(capacity, 1_000_000_000)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var index = IndexFlatSQ8(d, metric, initial_capacity=0)
    index.ntotal = ntotal
    index.capacity = capacity
    
    var float32_params = alloc[Float32](3)
    read_unsafe_pointer_float32(f, float32_params, 3)
    index.global_min = float32_params[0]
    index.global_max = float32_params[1]
    index.scale = float32_params[2]
    float32_params.free()
    
    if Int(index.codes_f32) != 0: index.codes_f32.free()
    if Int(index.codes_u8) != 0: index.codes_u8.free()
    if Int(index.norms_u32) != 0: index.norms_u32.free()
    
    index.codes_f32 = alloc[Float32](capacity * d)
    index.codes_u8 = alloc[UInt8](capacity * d)
    index.norms_u32 = alloc[UInt32](capacity)
    
    if capacity > 0:
        read_unsafe_pointer_float32(f, index.codes_f32, capacity * d)
        read_unsafe_pointer_uint8(f, index.codes_u8, capacity * d)
        read_unsafe_pointer_uint32(f, index.norms_u32, capacity)
        
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
        var span_neighbors = Span[UInt8](ptr=graph.neighbors.bitcast[UInt8](), length=graph.neighbors_capacity * 4)
        f.write_bytes(span_neighbors)
    
    write_unsafe_pointer_int(f, graph.cum_nneighbor_per_level, 33)

def read_hnsw_graph(mut f: FileHandle, mut graph: HNSWGraph) raises:
    graph.M = read_int(f)
    graph.efConstruction = read_int(f)
    graph.efSearch = read_int(f)
    _validate_hnsw_parameters(
        graph.M,
        graph.efConstruction,
        graph.efSearch,
    )
    graph.max_level = read_int(f)
    if graph.max_level < -1 or graph.max_level > 32:
        raise Error("Invalid HNSW maximum level.")
    graph.entry_point = read_int(f)
    if graph.entry_point < -1 or graph.entry_point > 1_000_000_000:
        raise Error("Invalid HNSW entry point.")
    graph.ntotal = read_int(f)
    check_size_limit(graph.ntotal, 1_000_000_000)
    if (
        (
            graph.ntotal == 0
            and (graph.max_level != -1 or graph.entry_point != -1)
        )
        or (
            graph.ntotal > 0
            and (graph.max_level < 0 or graph.entry_point >= graph.ntotal)
        )
    ):
        raise Error("HNSW graph entry point does not match its size.")
    
    var capacity = read_int(f)
    check_size_limit(capacity, 1_000_000_000)
    var neighbors_capacity = read_int(f)
    check_size_limit(neighbors_capacity, 2_000_000_000)
    
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
    _validate_vector_dimension(d)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var storage = read_index_flat(f)
    var index = IndexHNSW[IndexFlat](storage^, d, metric, M=32)
    index.ntotal = ntotal
    index.is_trained = is_trained
    read_hnsw_graph(f, index.hnsw)
    index.vt_pool.grow(index.hnsw.capacity)
    return index^

def write_index_hnsw_sq8(mut f: FileHandle, index: IndexHNSW[IndexFlatSQ8]) raises:
    write_int(f, MAGIC_HNSW_SQ8)
    write_int(f, index.d)
    write_int(f, index.ntotal)
    write_bool(f, index.is_trained)
    var metric = 0
    if index.metric_type == METRIC_INNER_PRODUCT: metric = 1
    write_int(f, metric)
    
    write_index_flat_sq8(f, index.storage)
    write_hnsw_graph(f, index.hnsw)

def read_index_hnsw_sq8(mut f: FileHandle) raises -> IndexHNSW[IndexFlatSQ8]:
    var magic = read_int(f)
    if magic != MAGIC_HNSW_SQ8: raise Error("Invalid magic for IndexHNSW SQ8")
    
    var d = read_int(f)
    _validate_vector_dimension(d)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var storage = read_index_flat_sq8(f)
    var index = IndexHNSW[IndexFlatSQ8](storage^, d, metric, M=32)
    index.ntotal = ntotal
    index.is_trained = is_trained
    read_hnsw_graph(f, index.hnsw)
    index.vt_pool.grow(index.hnsw.capacity)
    return index^


# --- Memory-mappable Flat/SQ8 HNSW (Collection format V5+) ---

def write_index_flat_mmap(mut f: FileHandle, index: IndexFlat) raises:
    var start = Int(f.seek(0, SEEK_CUR))
    # Persist only live values. Heap capacity is an implementation detail and
    # may contain an uninitialized tail that must not leak into the file.
    var serialized_capacity = index.ntotal
    var codes_count = serialized_capacity * index.d
    var codes_offset = _align_mmap_offset(start + 7 * 8)
    write_int(f, MAGIC_FLAT_MMAP)
    write_int(f, index.d)
    write_int(f, index.ntotal)
    write_int(f, serialized_capacity)
    write_int(f, 1 if index.metric_type == METRIC_INNER_PRODUCT else 0)
    write_int(f, codes_offset)
    write_int(f, codes_count)
    _write_padding_to(f, codes_offset)
    write_unsafe_pointer_float32(f, index.codes, codes_count)


def read_index_flat_mmap(
    mut f: FileHandle,
    file_size: Int,
) raises -> IndexFlat:
    if read_int(f) != MAGIC_FLAT_MMAP:
        raise Error("Invalid magic for memory-mapped IndexFlat.")
    var d = read_int(f)
    _validate_vector_dimension(d)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var capacity = read_int(f)
    check_size_limit(capacity, 1_000_000_000)
    if ntotal != capacity:
        raise Error("Memory-mapped IndexFlat must omit unused capacity.")
    var metric_int = read_int(f)
    if metric_int != 0 and metric_int != 1:
        raise Error("Invalid IndexFlat metric.")
    var codes_offset = read_int(f)
    var codes_count = read_int(f)
    var expected_codes_count = checked_byte_count(capacity, d)
    if codes_count != expected_codes_count:
        raise Error("Invalid memory-mapped IndexFlat code count.")
    var codes_bytes = checked_byte_count(codes_count, 4)
    _validate_mmap_region(codes_offset, codes_bytes, file_size)
    _ = f.seek(UInt64(codes_offset + codes_bytes), SEEK_SET)

    var metric = (
        METRIC_INNER_PRODUCT if metric_int == 1 else METRIC_L2
    )
    var mapping = FileMemoryMap.map_read_only(f.handle, file_size)
    var codes_address = mapping.address + codes_offset
    var index = IndexFlat(d, metric, initial_capacity=0)
    _free_aligned(index.codes)
    index.ntotal = ntotal
    index.capacity = capacity
    index.codes = UnsafePointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=codes_address
    )
    index._mapping = mapping^
    return index^


def write_index_flat_sq8_mmap(
    mut f: FileHandle,
    index: IndexFlatSQ8,
) raises:
    var start = Int(f.seek(0, SEEK_CUR))
    var serialized_capacity = index.ntotal
    var f32_count = serialized_capacity * index.d
    var u8_count = serialized_capacity * index.d
    var norms_count = serialized_capacity
    var f32_offset = _align_mmap_offset(start + 100)
    var u8_offset = _align_mmap_offset(f32_offset + f32_count * 4)
    var norms_offset = _align_mmap_offset(u8_offset + u8_count)

    write_int(f, MAGIC_FLAT_SQ8_MMAP)
    write_int(f, index.d)
    write_int(f, index.ntotal)
    write_int(f, serialized_capacity)
    write_int(f, 1 if index.metric_type == METRIC_INNER_PRODUCT else 0)
    var parameters = alloc[Float32](3)
    parameters[0] = index.global_min
    parameters[1] = index.global_max
    parameters[2] = index.scale
    write_unsafe_pointer_float32(f, parameters, 3)
    parameters.free()
    write_int(f, f32_offset)
    write_int(f, f32_count)
    write_int(f, u8_offset)
    write_int(f, u8_count)
    write_int(f, norms_offset)
    write_int(f, norms_count)

    _write_padding_to(f, f32_offset)
    write_unsafe_pointer_float32(f, index.codes_f32, f32_count)
    _write_padding_to(f, u8_offset)
    write_unsafe_pointer_uint8(f, index.codes_u8, u8_count)
    _write_padding_to(f, norms_offset)
    write_unsafe_pointer_uint32(f, index.norms_u32, norms_count)


def read_index_flat_sq8_mmap(
    mut f: FileHandle,
    file_size: Int,
) raises -> IndexFlatSQ8:
    if read_int(f) != MAGIC_FLAT_SQ8_MMAP:
        raise Error("Invalid magic for memory-mapped IndexFlatSQ8.")
    var d = read_int(f)
    _validate_vector_dimension(d)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var capacity = read_int(f)
    check_size_limit(capacity, 1_000_000_000)
    if ntotal != capacity:
        raise Error("Memory-mapped IndexFlatSQ8 must omit unused capacity.")
    var metric_int = read_int(f)
    if metric_int != 0 and metric_int != 1:
        raise Error("Invalid IndexFlatSQ8 metric.")
    var parameters = alloc[Float32](3)
    read_unsafe_pointer_float32(f, parameters, 3)
    var global_min = parameters[0]
    var global_max = parameters[1]
    var scale = parameters[2]
    parameters.free()
    var f32_offset = read_int(f)
    var f32_count = read_int(f)
    var u8_offset = read_int(f)
    var u8_count = read_int(f)
    var norms_offset = read_int(f)
    var norms_count = read_int(f)
    var expected_vector_count = checked_byte_count(capacity, d)
    if (
        f32_count != expected_vector_count
        or u8_count != expected_vector_count
        or norms_count != capacity
    ):
        raise Error("Invalid memory-mapped IndexFlatSQ8 array sizes.")
    var f32_bytes = checked_byte_count(f32_count, 4)
    var norms_bytes = checked_byte_count(norms_count, 4)
    _validate_mmap_region(f32_offset, f32_bytes, file_size)
    _validate_mmap_region(u8_offset, u8_count, file_size)
    _validate_mmap_region(norms_offset, norms_bytes, file_size)
    _ = f.seek(UInt64(norms_offset + norms_bytes), SEEK_SET)

    var metric = (
        METRIC_INNER_PRODUCT if metric_int == 1 else METRIC_L2
    )
    var mapping = FileMemoryMap.map_read_only(f.handle, file_size)
    var base = mapping.address
    var index = IndexFlatSQ8(d, metric, initial_capacity=0)
    index.codes_f32.free()
    index.codes_u8.free()
    index.norms_u32.free()
    index.ntotal = ntotal
    index.capacity = capacity
    index.global_min = global_min
    index.global_max = global_max
    index.scale = scale
    index.codes_f32 = UnsafePointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=base + f32_offset
    )
    index.codes_u8 = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=base + u8_offset
    )
    index.norms_u32 = UnsafePointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=base + norms_offset
    )
    index._mapping = mapping^
    return index^


def write_hnsw_graph_mmap(
    mut f: FileHandle,
    graph: HNSWGraph,
) raises:
    var start = Int(f.seek(0, SEEK_CUR))
    var serialized_capacity = graph.ntotal
    var levels_count = serialized_capacity
    var offsets_count = serialized_capacity + 1
    var neighbors_count = 0
    if graph.ntotal > 0:
        var last_node = graph.ntotal - 1
        neighbors_count = (
            graph.offsets[last_node]
            + graph.cum_nneighbor_per_level[graph.levels[last_node] + 1]
        )
    var cumulative_count = 33
    var levels_offset = _align_mmap_offset(start + 17 * 8)
    var offsets_offset = _align_mmap_offset(
        levels_offset + levels_count * 8
    )
    var neighbors_offset = _align_mmap_offset(
        offsets_offset + offsets_count * 8
    )
    var cumulative_offset = _align_mmap_offset(
        neighbors_offset + neighbors_count * 4
    )

    write_int(f, MAGIC_HNSW_GRAPH_MMAP)
    write_int(f, graph.M)
    write_int(f, graph.efConstruction)
    write_int(f, graph.efSearch)
    write_int(f, graph.max_level)
    write_int(f, graph.entry_point)
    write_int(f, graph.ntotal)
    write_int(f, serialized_capacity)
    write_int(f, neighbors_count)
    write_int(f, levels_offset)
    write_int(f, levels_count)
    write_int(f, offsets_offset)
    write_int(f, offsets_count)
    write_int(f, neighbors_offset)
    write_int(f, neighbors_count)
    write_int(f, cumulative_offset)
    write_int(f, cumulative_count)

    _write_padding_to(f, levels_offset)
    write_unsafe_pointer_int(f, graph.levels, levels_count)
    _write_padding_to(f, offsets_offset)
    write_unsafe_pointer_int(f, graph.offsets, serialized_capacity)
    # The runtime graph does not use offsets[ntotal], but serializing this
    # sentinel makes the mapped offsets region complete and safe to detach.
    write_int(f, neighbors_count)
    _write_padding_to(f, neighbors_offset)
    if neighbors_count > 0:
        f.write_bytes(
            Span[UInt8](
                ptr=graph.neighbors.bitcast[UInt8](),
                length=neighbors_count * 4,
            )
        )
    _write_padding_to(f, cumulative_offset)
    write_unsafe_pointer_int(
        f,
        graph.cum_nneighbor_per_level,
        cumulative_count,
    )


def read_hnsw_graph_mmap(
    mut f: FileHandle,
    mut graph: HNSWGraph,
    file_size: Int,
) raises:
    if read_int(f) != MAGIC_HNSW_GRAPH_MMAP:
        raise Error("Invalid magic for memory-mapped HNSW graph.")
    var M = read_int(f)
    var ef_construction = read_int(f)
    var ef_search = read_int(f)
    _validate_hnsw_parameters(M, ef_construction, ef_search)
    var max_level = read_int(f)
    if max_level < -1 or max_level > 32:
        raise Error("Invalid HNSW maximum level.")
    var entry_point = read_int(f)
    if entry_point < -1 or entry_point > 1_000_000_000:
        raise Error("Invalid HNSW entry point.")
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var capacity = read_int(f)
    check_size_limit(capacity, 1_000_000_000)
    var neighbors_capacity = read_int(f)
    check_size_limit(neighbors_capacity, 2_000_000_000)
    if ntotal != capacity:
        raise Error("Memory-mapped HNSW graph must omit unused capacity.")
    if (
        (ntotal == 0 and (max_level != -1 or entry_point != -1))
        or (ntotal > 0 and (max_level < 0 or entry_point >= ntotal))
    ):
        raise Error("HNSW graph entry point does not match its size.")

    var levels_offset = read_int(f)
    var levels_count = read_int(f)
    var offsets_offset = read_int(f)
    var offsets_count = read_int(f)
    var neighbors_offset = read_int(f)
    var neighbors_count = read_int(f)
    var cumulative_offset = read_int(f)
    var cumulative_count = read_int(f)
    if (
        levels_count != capacity
        or offsets_count != capacity + 1
        or neighbors_count != neighbors_capacity
        or cumulative_count != 33
    ):
        raise Error("Invalid memory-mapped HNSW graph array sizes.")
    var levels_bytes = checked_byte_count(levels_count, 8)
    var offsets_bytes = checked_byte_count(offsets_count, 8)
    var neighbors_bytes = checked_byte_count(neighbors_count, 4)
    var cumulative_bytes = checked_byte_count(cumulative_count, 8)
    _validate_mmap_region(levels_offset, levels_bytes, file_size)
    _validate_mmap_region(offsets_offset, offsets_bytes, file_size)
    _validate_mmap_region(
        neighbors_offset, neighbors_bytes, file_size
    )
    _validate_mmap_region(
        cumulative_offset, cumulative_bytes, file_size
    )
    _ = f.seek(UInt64(cumulative_offset + cumulative_bytes), SEEK_SET)

    var mapping = FileMemoryMap.map_read_only(f.handle, file_size)
    var base = mapping.address
    graph.levels.free()
    graph.offsets.free()
    graph.neighbors.free()
    graph.cum_nneighbor_per_level.free()
    graph.M = M
    graph.efConstruction = ef_construction
    graph.efSearch = ef_search
    graph.max_level = max_level
    graph.entry_point = entry_point
    graph.ntotal = ntotal
    graph.capacity = capacity
    graph.neighbors_capacity = neighbors_capacity
    graph.levels = UnsafePointer[Int, MutUntrackedOrigin](
        unsafe_from_address=base + levels_offset
    )
    graph.offsets = UnsafePointer[Int, MutUntrackedOrigin](
        unsafe_from_address=base + offsets_offset
    )
    graph.neighbors = UnsafePointer[Int32, MutUntrackedOrigin](
        unsafe_from_address=base + neighbors_offset
    )
    graph.cum_nneighbor_per_level = UnsafePointer[
        Int, MutUntrackedOrigin
    ](unsafe_from_address=base + cumulative_offset)
    graph._mapping = mapping^


def write_index_hnsw_mmap(
    mut f: FileHandle,
    index: IndexHNSW[IndexFlat],
) raises:
    write_int(f, MAGIC_HNSW_MMAP)
    write_int(f, index.d)
    write_int(f, index.ntotal)
    write_bool(f, index.is_trained)
    write_int(f, 1 if index.metric_type == METRIC_INNER_PRODUCT else 0)
    write_index_flat_mmap(f, index.storage)
    write_hnsw_graph_mmap(f, index.hnsw)


def read_index_hnsw_mmap(
    mut f: FileHandle,
    file_size: Int,
    expected_count: Int = -1,
) raises -> IndexHNSW[IndexFlat]:
    if read_int(f) != MAGIC_HNSW_MMAP:
        raise Error("Invalid magic for memory-mapped IndexHNSW.")
    var d = read_int(f)
    _validate_vector_dimension(d)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    if expected_count >= 0 and ntotal != expected_count:
        raise Error("Collection metadata and Flat index size differ.")
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    if metric_int != 0 and metric_int != 1:
        raise Error("Invalid IndexHNSW metric.")
    var metric = (
        METRIC_INNER_PRODUCT if metric_int == 1 else METRIC_L2
    )
    var storage = read_index_flat_mmap(f, file_size)
    if (
        storage.d != d
        or storage.ntotal != ntotal
        or storage.metric_type != metric
    ):
        raise Error("HNSW and Flat storage headers differ.")
    var index = IndexHNSW[IndexFlat](storage^, d, metric, M=32)
    index.ntotal = ntotal
    index.is_trained = is_trained
    read_hnsw_graph_mmap(f, index.hnsw, file_size)
    if index.hnsw.ntotal != ntotal:
        raise Error("HNSW graph and index sizes differ.")
    index.vt_pool.grow(index.hnsw.capacity)
    return index^


def write_index_hnsw_sq8_mmap(
    mut f: FileHandle,
    index: IndexHNSW[IndexFlatSQ8],
) raises:
    write_int(f, MAGIC_HNSW_SQ8_MMAP)
    write_int(f, index.d)
    write_int(f, index.ntotal)
    write_bool(f, index.is_trained)
    write_int(f, 1 if index.metric_type == METRIC_INNER_PRODUCT else 0)
    write_index_flat_sq8_mmap(f, index.storage)
    write_hnsw_graph_mmap(f, index.hnsw)


def read_index_hnsw_sq8_mmap(
    mut f: FileHandle,
    file_size: Int,
    expected_count: Int = -1,
) raises -> IndexHNSW[IndexFlatSQ8]:
    if read_int(f) != MAGIC_HNSW_SQ8_MMAP:
        raise Error("Invalid magic for memory-mapped IndexHNSW SQ8.")
    var d = read_int(f)
    _validate_vector_dimension(d)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    if expected_count >= 0 and ntotal != expected_count:
        raise Error("Collection metadata and SQ8 index size differ.")
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    if metric_int != 0 and metric_int != 1:
        raise Error("Invalid IndexHNSW SQ8 metric.")
    var metric = (
        METRIC_INNER_PRODUCT if metric_int == 1 else METRIC_L2
    )
    var storage = read_index_flat_sq8_mmap(f, file_size)
    if (
        storage.d != d
        or storage.ntotal != ntotal
        or storage.metric_type != metric
    ):
        raise Error("HNSW and SQ8 storage headers differ.")
    var index = IndexHNSW[IndexFlatSQ8](storage^, d, metric, M=32)
    index.ntotal = ntotal
    index.is_trained = is_trained
    read_hnsw_graph_mmap(f, index.hnsw, file_size)
    if index.hnsw.ntotal != ntotal:
        raise Error("HNSW graph and index sizes differ.")
    index.vt_pool.grow(index.hnsw.capacity)
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
    check_size_limit(nlist, 1_000_000)
    var code_size = read_int(f)
    check_size_limit(code_size, 65536)
    if nlist != invlists.nlist or code_size != invlists.code_size:
        raise Error("Inverted-list shape does not match its index header.")
    
    for i in range(nlist):
        var size = read_int(f)
        var capacity = read_int(f)
        check_size_limit(capacity, 1_000_000_000)
        check_size_limit(size, capacity)

        
        invlists.resize(i, capacity)
        _ = Int(invlists.lists)
        invlists.lists[i].size = size
        
        var list_codes = invlists.get_codes(i).unsafe_ptr()
        var list_ids = invlists.get_ids(i).unsafe_ptr()
        
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
    
    var d = read_int(f)
    check_size_limit(d, 65536)
    var M = read_int(f)
    check_size_limit(M, 65536)
    var ksub = read_int(f)
    check_size_limit(ksub, 256)
    if d <= 0 or M <= 0 or M > d or d % M != 0:
        raise Error("Invalid ProductQuantizer shape.")
    if ksub <= 0 or ksub > 256:
        raise Error("Invalid ProductQuantizer centroid count.")
    if d != pq.d or M != pq.M or ksub != pq.ksub:
        raise Error("ProductQuantizer shape does not match its index header.")
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
    
    write_index_flat(f, index.quantizer)
    write_invlists(f, index.invlists)

def read_index_ivf_flat(mut f: FileHandle) raises -> IndexIVFFlat[IndexFlat]:
    var magic = read_int(f)
    if magic != MAGIC_IVF_FLAT: raise Error("Invalid magic for IndexIVFFlat")
    
    var d = read_int(f)
    _validate_vector_dimension(d)
    var nlist = read_int(f)
    check_size_limit(nlist, 1_000_000)
    var nprobe = read_int(f)
    check_size_limit(nprobe, 1_000_000)
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var quantizer = read_index_flat(f)
    var index = IndexIVFFlat[IndexFlat](quantizer^, d, nlist, metric)
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
    
    write_index_flat(f, index.quantizer)
    write_invlists(f, index.invlists)
    write_pq(f, index.pq)

def read_index_ivf_pq(mut f: FileHandle) raises -> IndexIVFPQ[IndexFlat]:
    var magic = read_int(f)
    if magic != MAGIC_IVF_PQ: raise Error("Invalid magic for IndexIVFPQ")
    
    var d = read_int(f)
    _validate_vector_dimension(d)
    var nlist = read_int(f)
    check_size_limit(nlist, 1_000_000)
    var M = read_int(f)
    check_size_limit(M, d)
    if M <= 0 or d % M != 0:
        raise Error("Invalid IndexIVFPQ subvector count.")
    var nprobe = read_int(f)
    check_size_limit(nprobe, nlist)
    if nprobe <= 0:
        raise Error("Invalid IndexIVFPQ nprobe.")
    var ntotal = read_int(f)
    check_size_limit(ntotal, 1_000_000_000)
    var is_trained = read_bool(f)
    var metric_int = read_int(f)
    if metric_int != 0 and metric_int != 1:
        raise Error("Invalid IndexIVFPQ metric.")
    
    var metric = METRIC_L2
    if metric_int == 1: metric = METRIC_INNER_PRODUCT
        
    var quantizer = read_index_flat(f)
    if (
        quantizer.d != d
        or (
            is_trained and quantizer.ntotal != nlist
        )
        or (
            not is_trained and quantizer.ntotal != 0
        )
        or quantizer.metric_type != metric
    ):
        raise Error("IndexIVFPQ coarse quantizer header mismatch.")
    var index = IndexIVFPQ[IndexFlat](quantizer^, d, nlist, M, metric)
    index.nprobe = nprobe
    index.ntotal = ntotal
    index.is_trained = is_trained
    
    read_invlists(f, index.invlists)
    read_pq(f, index.pq)
    if index.invlists.code_size != M:
        raise Error("IndexIVFPQ code size does not match M.")
    if index.pq.is_trained != is_trained:
        raise Error("IndexIVFPQ training state mismatch.")
    return index^
