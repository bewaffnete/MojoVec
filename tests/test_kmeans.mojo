from src.mojovec.clustering.kmeans import KMeans
from std.memory import alloc
from std.random import rand

def main() raises:
    var n = 1000
    var d = 16
    var k = 10
    var x = alloc[Float32](n * d)
    rand(x, n * d)
    var kmeans = KMeans(d, k, 5)
    kmeans.train(n, x)
    x.free()
    print("KMeans test passed")
