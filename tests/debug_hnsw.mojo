from mojovec.index.index_hnsw import IndexHNSW
from mojovec.index.index_flat import IndexFlat
from mojovec.core.types import METRIC_L2
from std.random import rand
from std.memory import alloc

def main() raises:
    var d = 2
    var nb = 10
    var xb = alloc[Float32](nb * d)
    for i in range(nb * d):
        xb[i] = Float32(i) # Deterministic points
        
    var storage = IndexFlat(d, METRIC_L2)
    var hnsw = IndexHNSW[IndexFlat](storage^, d, METRIC_L2, M=2)
    
    hnsw.add(nb, xb)
    
    # Print the graph
    for i in range(nb):
        var level = hnsw.hnsw.levels[i]
        pass  # print("Node", i, "level", level)
        for l in range(level + 1):
            var neighbors_info = hnsw.hnsw.get_neighbors(i, l)
            var neighbors = neighbors_info.ptr
            pass  # print("  level", l, "neighbors: ", end="")
            for j in range(neighbors_info.max_links):
                var neigh = neighbors[j]
                if neigh == -1:
                    break
                pass  # print(neigh, end=" ")
            pass  # print("")
    xb.free()
