from src.mojovec.clustering.kmeans import KMeans
from std.memory import alloc
from std.random import random_float64

def main() raises:
    var n = 1000
    var d = 8
    var k = 5
    var x = alloc[Float32](n * d)
    
    # Create 5 distinct clusters with noise
    for c in range(5):
        for i in range(200):
            var idx = c * 200 + i
            for j in range(d):
                var noise = Float32(random_float64(-1.0, 1.0))
                x[idx * d + j] = Float32(c * 10 + j) + noise
                
    var kmeans = KMeans(d, k, 20)
    kmeans.train(n, x)
    
    def assert_true(cond: Bool, msg: String = "Assertion failed") raises:
        if not cond:
            raise Error(msg)

    print("KMeans completed.")
    for c in range(k):
        assert_true(kmeans.counts[c] == 200, "Count mismatch")
        print("Centroid", c, "count:", kmeans.counts[c])
        print("C[0]:", kmeans.centroids[c * d])
        
    print("All KMeans tests passed!")
    x.free()
