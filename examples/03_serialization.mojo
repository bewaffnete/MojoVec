"""
MojoVec Example 3: Saving and Loading Indexes (Serialization)
-------------------------------------------------------------
This example demonstrates how to persist your built index to disk and reload it later.
MojoVec includes custom binary serialization specifically optimized for its own index structures.
"""

from mojovec import IndexFlat, METRIC_L2
from mojovec import write_index_flat, read_index_flat
from std.memory import alloc
from std.random import rand

def main() raises:
    var d = 128
    var num_vectors = 1000

    print("1. Creating and populating a Flat index...")
    var index = IndexFlat(d, METRIC_L2)
    
    var xb = alloc[Float32](num_vectors * d)
    rand(xb, num_vectors * d)
    index.add(num_vectors, xb)

    print("2. Saving the index to disk ('my_index.bin')...")
    # write_index_flat will serialize the Flat index vectors
    var f_w = open("my_index.bin", "w")
    write_index_flat(f_w, index)
    f_w.close()
    print("Index saved successfully!")

    print("3. Loading the index from disk...")
    # read_index_flat restores the entire state
    var f_r = open("my_index.bin", "r")
    var loaded_index = read_index_flat(f_r)
    f_r.close()
    
    print("Index loaded! Ready for querying.")

    # Cleanup
    xb.free()
