# heap.mojo

struct HeapResult:
    var dist: Float32
    var label: Int
    def __init__(out self, dist: Float32, label: Int):
        self.dist = dist
        self.label = label


@always_inline
def max_heap_replace_top[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int, origin2], k: Int, dist: Float32, label: Int):
    """
    Replaces the top element (max) of the max-heap with a new element and sifts down.
    Assumes the heap has size k and the top is at index 0.
    """
    heap_distances[0] = dist
    heap_labels[0] = label
    
    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var largest = i
        
        if left < k and heap_distances[left] > heap_distances[largest]:
            largest = left
        if right < k and heap_distances[right] > heap_distances[largest]:
            largest = right
            
        if largest != i:
            # Swap
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[largest]
            heap_distances[largest] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[largest]
            heap_labels[largest] = tmp_label
            
            i = largest
        else:
            break

@always_inline
def max_heap_push[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int, origin2], current_size: Int, dist: Float32, label: Int):
    """
    Pushes a new element into the max-heap and sifts up.
    """
    var i = current_size
    heap_distances[i] = dist
    heap_labels[i] = label
    
    while i > 0:
        var parent = (i - 1) // 2
        if heap_distances[parent] < heap_distances[i]:
            # Swap
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[parent]
            heap_distances[parent] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[parent]
            heap_labels[parent] = tmp_label
            
            i = parent
        else:
            break

@always_inline
def min_heap_push[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int, origin2], current_size: Int, dist: Float32, label: Int):
    var i = current_size
    heap_distances[i] = dist
    heap_labels[i] = label
    
    while i > 0:
        var parent = (i - 1) // 2
        if heap_distances[parent] > heap_distances[i]:
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[parent]
            heap_distances[parent] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[parent]
            heap_labels[parent] = tmp_label
            
            i = parent
        else:
            break

@always_inline
def min_heap_pop[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int, origin2], current_size: Int) -> HeapResult:
    var popped_dist = heap_distances[0]
    var popped_label = heap_labels[0]
    
    var last_idx = current_size - 1
    if last_idx == 0:
        return HeapResult(popped_dist, popped_label)
        
    heap_distances[0] = heap_distances[last_idx]
    heap_labels[0] = heap_labels[last_idx]
    
    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var smallest = i
        
        if left < last_idx and heap_distances[left] < heap_distances[smallest]:
            smallest = left
        if right < last_idx and heap_distances[right] < heap_distances[smallest]:
            smallest = right
            
        if smallest != i:
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[smallest]
            heap_distances[smallest] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[smallest]
            heap_labels[smallest] = tmp_label
            
            i = smallest
        else:
            break
            
    return HeapResult(popped_dist, popped_label)

@always_inline
def max_heap_pop[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int, origin2], current_size: Int) -> HeapResult:
    var popped_dist = heap_distances[0]
    var popped_label = heap_labels[0]
    
    var last_idx = current_size - 1
    if last_idx == 0:
        return HeapResult(popped_dist, popped_label)
        
    heap_distances[0] = heap_distances[last_idx]
    heap_labels[0] = heap_labels[last_idx]
    
    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var largest = i
        
        if left < last_idx and heap_distances[left] > heap_distances[largest]:
            largest = left
        if right < last_idx and heap_distances[right] > heap_distances[largest]:
            largest = right
            
        if largest != i:
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[largest]
            heap_distances[largest] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[largest]
            heap_labels[largest] = tmp_label
            
            i = largest
        else:
            break
            
    return HeapResult(popped_dist, popped_label)

# heap.mojo

struct HeapResult32:
    var dist: Float32
    var label: Int32
    def __init__(out self, dist: Float32, label: Int32):
        self.dist = dist
        self.label = label


@always_inline
def max_heap_replace_top[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int32, origin2], k: Int, dist: Float32, label: Int32):
    """
    Replaces the top element (max) of the max-heap with a new element and sifts down.
    Assumes the heap has size k and the top is at index 0.
    """
    heap_distances[0] = dist
    heap_labels[0] = label
    
    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var largest = i
        
        if left < k and heap_distances[left] > heap_distances[largest]:
            largest = left
        if right < k and heap_distances[right] > heap_distances[largest]:
            largest = right
            
        if largest != i:
            # Swap
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[largest]
            heap_distances[largest] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[largest]
            heap_labels[largest] = tmp_label
            
            i = largest
        else:
            break

@always_inline
def max_heap_push[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int32, origin2], current_size: Int, dist: Float32, label: Int32):
    """
    Pushes a new element into the max-heap and sifts up.
    """
    var i = current_size
    heap_distances[i] = dist
    heap_labels[i] = label
    
    while i > 0:
        var parent = (i - 1) // 2
        if heap_distances[parent] < heap_distances[i]:
            # Swap
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[parent]
            heap_distances[parent] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[parent]
            heap_labels[parent] = tmp_label
            
            i = parent
        else:
            break

@always_inline
def min_heap_push[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int32, origin2], current_size: Int, dist: Float32, label: Int32):
    var i = current_size
    heap_distances[i] = dist
    heap_labels[i] = label
    
    while i > 0:
        var parent = (i - 1) // 2
        if heap_distances[parent] > heap_distances[i]:
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[parent]
            heap_distances[parent] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[parent]
            heap_labels[parent] = tmp_label
            
            i = parent
        else:
            break

@always_inline
def min_heap_pop[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int32, origin2], current_size: Int) -> HeapResult32:
    var popped_dist = heap_distances[0]
    var popped_label = heap_labels[0]
    
    var last_idx = current_size - 1
    if last_idx == 0:
        return HeapResult32(popped_dist, popped_label)
        
    heap_distances[0] = heap_distances[last_idx]
    heap_labels[0] = heap_labels[last_idx]
    
    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var smallest = i
        
        if left < last_idx and heap_distances[left] < heap_distances[smallest]:
            smallest = left
        if right < last_idx and heap_distances[right] < heap_distances[smallest]:
            smallest = right
            
        if smallest != i:
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[smallest]
            heap_distances[smallest] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[smallest]
            heap_labels[smallest] = tmp_label
            
            i = smallest
        else:
            break
            
    return HeapResult32(popped_dist, popped_label)

@always_inline
def max_heap_pop[origin1: MutOrigin, origin2: MutOrigin](heap_distances: UnsafePointer[Float32, origin1], heap_labels: UnsafePointer[Int32, origin2], current_size: Int) -> HeapResult32:
    var popped_dist = heap_distances[0]
    var popped_label = heap_labels[0]
    
    var last_idx = current_size - 1
    if last_idx == 0:
        return HeapResult32(popped_dist, popped_label)
        
    heap_distances[0] = heap_distances[last_idx]
    heap_labels[0] = heap_labels[last_idx]
    
    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var largest = i
        
        if left < last_idx and heap_distances[left] > heap_distances[largest]:
            largest = left
        if right < last_idx and heap_distances[right] > heap_distances[largest]:
            largest = right
            
        if largest != i:
            var tmp_dist = heap_distances[i]
            heap_distances[i] = heap_distances[largest]
            heap_distances[largest] = tmp_dist
            
            var tmp_label = heap_labels[i]
            heap_labels[i] = heap_labels[largest]
            heap_labels[largest] = tmp_label
            
            i = largest
        else:
            break
            
    return HeapResult32(popped_dist, popped_label)