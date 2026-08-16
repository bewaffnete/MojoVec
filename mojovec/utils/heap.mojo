"""
Provides min-heap and max-heap data structures for maintaining nearest neighbor search results.
"""

struct HeapResult(TrivialRegisterPassable):
    """
    Represents a single element extracted from the heap, containing a distance and an integer label.
    """
    var dist: Float32
    var label: Int
    def __init__(out self, dist: Float32, label: Int):
        self.dist = dist
        self.label = label


@always_inline
def max_heap_replace_top[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int, origin2], k: Int, dist: Float32, label: Int):
    """
    Replaces the top element of a max-heap with a new element and sifts down to maintain the heap property.

    Args:
        heap_distances: Pointer to the array of distances representing the heap.
        heap_labels: Pointer to the array of corresponding labels.
        k: The maximum size of the heap.
        dist: The new distance to insert.
        label: The new label to insert.
    """
    heap_distances[unsafe_offset=0] = dist
    heap_labels[unsafe_offset=0] = label

    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var largest = i

        if left < k and heap_distances[unsafe_offset=left] > heap_distances[unsafe_offset=largest]:
            largest = left
        if right < k and heap_distances[unsafe_offset=right] > heap_distances[unsafe_offset=largest]:
            largest = right

        if largest != i:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=largest]
            heap_distances[unsafe_offset=largest] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=largest]
            heap_labels[unsafe_offset=largest] = tmp_label

            i = largest
        else:
            break

@always_inline
def max_heap_push[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int, origin2], current_size: Int, dist: Float32, label: Int):
    """
    Pushes a new element into the max-heap and sifts up to maintain the heap property.

    Args:
        heap_distances: Pointer to the heap distances array.
        heap_labels: Pointer to the heap labels array.
        current_size: The current number of elements in the heap before pushing.
        dist: The new distance to insert.
        label: The new label to insert.
    """
    var i = current_size
    heap_distances[unsafe_offset=i] = dist
    heap_labels[unsafe_offset=i] = label

    while i > 0:
        var parent = (i - 1) // 2
        if heap_distances[unsafe_offset=parent] < heap_distances[unsafe_offset=i]:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=parent]
            heap_distances[unsafe_offset=parent] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=parent]
            heap_labels[unsafe_offset=parent] = tmp_label

            i = parent
        else:
            break

@always_inline
def min_heap_push[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int, origin2], current_size: Int, dist: Float32, label: Int):
    """
    Pushes a new element into the min-heap and sifts up to maintain the heap property.

    Args:
        heap_distances: Pointer to the heap distances array.
        heap_labels: Pointer to the heap labels array.
        current_size: The current number of elements in the heap before pushing.
        dist: The new distance to insert.
        label: The new label to insert.
    """
    var i = current_size
    heap_distances[unsafe_offset=i] = dist
    heap_labels[unsafe_offset=i] = label

    while i > 0:
        var parent = (i - 1) // 2
        if heap_distances[unsafe_offset=parent] > heap_distances[unsafe_offset=i]:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=parent]
            heap_distances[unsafe_offset=parent] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=parent]
            heap_labels[unsafe_offset=parent] = tmp_label

            i = parent
        else:
            break

@always_inline
def min_heap_pop[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int, origin2], current_size: Int) -> HeapResult:
    """
    Pops and returns the minimum element from the min-heap.

    Args:
        heap_distances: Pointer to the heap distances array.
        heap_labels: Pointer to the heap labels array.
        current_size: The current number of elements in the heap.

    Returns:
        The extracted `HeapResult`.
    """
    var popped_dist = heap_distances[unsafe_offset=0]
    var popped_label = heap_labels[unsafe_offset=0]

    var last_idx = current_size - 1
    if last_idx == 0:
        return HeapResult(popped_dist, popped_label)

    heap_distances[unsafe_offset=0] = heap_distances[unsafe_offset=last_idx]
    heap_labels[unsafe_offset=0] = heap_labels[unsafe_offset=last_idx]

    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var smallest = i

        if left < last_idx and heap_distances[unsafe_offset=left] < heap_distances[unsafe_offset=smallest]:
            smallest = left
        if right < last_idx and heap_distances[unsafe_offset=right] < heap_distances[unsafe_offset=smallest]:
            smallest = right

        if smallest != i:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=smallest]
            heap_distances[unsafe_offset=smallest] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=smallest]
            heap_labels[unsafe_offset=smallest] = tmp_label

            i = smallest
        else:
            break

    return HeapResult(popped_dist, popped_label)

@always_inline
def max_heap_pop[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int, origin2], current_size: Int) -> HeapResult:
    """
    Pops and returns the maximum element from the max-heap.

    Args:
        heap_distances: Pointer to the heap distances array.
        heap_labels: Pointer to the heap labels array.
        current_size: The current number of elements in the heap.

    Returns:
        The extracted `HeapResult`.
    """
    var popped_dist = heap_distances[unsafe_offset=0]
    var popped_label = heap_labels[unsafe_offset=0]

    var last_idx = current_size - 1
    if last_idx == 0:
        return HeapResult(popped_dist, popped_label)

    heap_distances[unsafe_offset=0] = heap_distances[unsafe_offset=last_idx]
    heap_labels[unsafe_offset=0] = heap_labels[unsafe_offset=last_idx]

    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var largest = i

        if left < last_idx and heap_distances[unsafe_offset=left] > heap_distances[unsafe_offset=largest]:
            largest = left
        if right < last_idx and heap_distances[unsafe_offset=right] > heap_distances[unsafe_offset=largest]:
            largest = right

        if largest != i:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=largest]
            heap_distances[unsafe_offset=largest] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=largest]
            heap_labels[unsafe_offset=largest] = tmp_label

            i = largest
        else:
            break

    return HeapResult(popped_dist, popped_label)

struct HeapResult32(TrivialRegisterPassable):
    """
    Represents a single element extracted from the heap, containing a distance and an Int32 label.
    """
    var dist: Float32
    var label: Int32
    def __init__(out self, dist: Float32, label: Int32):
        self.dist = dist
        self.label = label


@always_inline
def max_heap_replace_top[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int32, origin2], k: Int, dist: Float32, label: Int32):
    """
    Replaces the top element (max) of the max-heap with a new element and sifts down.

    Args:
        heap_distances: Pointer to the array of distances representing the heap.
        heap_labels: Pointer to the array of corresponding labels.
        k: The maximum size of the heap.
        dist: The new distance to insert.
        label: The new label to insert.
    """
    heap_distances[unsafe_offset=0] = dist
    heap_labels[unsafe_offset=0] = label

    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var largest = i

        if left < k and heap_distances[unsafe_offset=left] > heap_distances[unsafe_offset=largest]:
            largest = left
        if right < k and heap_distances[unsafe_offset=right] > heap_distances[unsafe_offset=largest]:
            largest = right

        if largest != i:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=largest]
            heap_distances[unsafe_offset=largest] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=largest]
            heap_labels[unsafe_offset=largest] = tmp_label

            i = largest
        else:
            break

@always_inline
def max_heap_push[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int32, origin2], current_size: Int, dist: Float32, label: Int32):
    """
    Pushes a new element into the max-heap and sifts up.

    Args:
        heap_distances: Pointer to the heap distances array.
        heap_labels: Pointer to the heap labels array.
        current_size: The current number of elements in the heap before pushing.
        dist: The new distance to insert.
        label: The new label to insert.
    """
    var i = current_size
    heap_distances[unsafe_offset=i] = dist
    heap_labels[unsafe_offset=i] = label

    while i > 0:
        var parent = (i - 1) // 2
        if heap_distances[unsafe_offset=parent] < heap_distances[unsafe_offset=i]:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=parent]
            heap_distances[unsafe_offset=parent] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=parent]
            heap_labels[unsafe_offset=parent] = tmp_label

            i = parent
        else:
            break

@always_inline
def min_heap_push[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int32, origin2], current_size: Int, dist: Float32, label: Int32):
    """
    Pushes a new element into the min-heap and sifts up.

    Args:
        heap_distances: Pointer to the heap distances array.
        heap_labels: Pointer to the heap labels array.
        current_size: The current number of elements in the heap before pushing.
        dist: The new distance to insert.
        label: The new label to insert.
    """
    var i = current_size
    heap_distances[unsafe_offset=i] = dist
    heap_labels[unsafe_offset=i] = label

    while i > 0:
        var parent = (i - 1) // 2
        if heap_distances[unsafe_offset=parent] > heap_distances[unsafe_offset=i]:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=parent]
            heap_distances[unsafe_offset=parent] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=parent]
            heap_labels[unsafe_offset=parent] = tmp_label

            i = parent
        else:
            break

@always_inline
def min_heap_pop[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int32, origin2], current_size: Int) -> HeapResult32:
    """
    Pops and returns the minimum element from the min-heap.

    Args:
        heap_distances: Pointer to the heap distances array.
        heap_labels: Pointer to the heap labels array.
        current_size: The current number of elements in the heap.

    Returns:
        The extracted `HeapResult32`.
    """
    var popped_dist = heap_distances[unsafe_offset=0]
    var popped_label = heap_labels[unsafe_offset=0]

    var last_idx = current_size - 1
    if last_idx == 0:
        return HeapResult32(popped_dist, popped_label)

    heap_distances[unsafe_offset=0] = heap_distances[unsafe_offset=last_idx]
    heap_labels[unsafe_offset=0] = heap_labels[unsafe_offset=last_idx]

    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var smallest = i

        if left < last_idx and heap_distances[unsafe_offset=left] < heap_distances[unsafe_offset=smallest]:
            smallest = left
        if right < last_idx and heap_distances[unsafe_offset=right] < heap_distances[unsafe_offset=smallest]:
            smallest = right

        if smallest != i:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=smallest]
            heap_distances[unsafe_offset=smallest] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=smallest]
            heap_labels[unsafe_offset=smallest] = tmp_label

            i = smallest
        else:
            break

    return HeapResult32(popped_dist, popped_label)

@always_inline
def max_heap_pop[origin1: MutOrigin, origin2: MutOrigin](heap_distances: Pointer[Float32, origin1], heap_labels: Pointer[Int32, origin2], current_size: Int) -> HeapResult32:
    """
    Pops and returns the maximum element from the max-heap.

    Args:
        heap_distances: Pointer to the heap distances array.
        heap_labels: Pointer to the heap labels array.
        current_size: The current number of elements in the heap.

    Returns:
        The extracted `HeapResult32`.
    """
    var popped_dist = heap_distances[unsafe_offset=0]
    var popped_label = heap_labels[unsafe_offset=0]

    var last_idx = current_size - 1
    if last_idx == 0:
        return HeapResult32(popped_dist, popped_label)

    heap_distances[unsafe_offset=0] = heap_distances[unsafe_offset=last_idx]
    heap_labels[unsafe_offset=0] = heap_labels[unsafe_offset=last_idx]

    var i = 0
    while True:
        var left = 2 * i + 1
        var right = 2 * i + 2
        var largest = i

        if left < last_idx and heap_distances[unsafe_offset=left] > heap_distances[unsafe_offset=largest]:
            largest = left
        if right < last_idx and heap_distances[unsafe_offset=right] > heap_distances[unsafe_offset=largest]:
            largest = right

        if largest != i:
            var tmp_dist = heap_distances[unsafe_offset=i]
            heap_distances[unsafe_offset=i] = heap_distances[unsafe_offset=largest]
            heap_distances[unsafe_offset=largest] = tmp_dist

            var tmp_label = heap_labels[unsafe_offset=i]
            heap_labels[unsafe_offset=i] = heap_labels[unsafe_offset=largest]
            heap_labels[unsafe_offset=largest] = tmp_label

            i = largest
        else:
            break

    return HeapResult32(popped_dist, popped_label)
