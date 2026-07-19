from std.memory.span import Span
from mojovec.clustering.kmeans import KMeans
from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises, TestSuite
from std.memory import alloc
from std.random import rand

def test_kmeans() raises:
    var n = 1000
    var d = 16
    var k = 10
    var x = alloc[Float32](n * d)
    rand(x, n * d)
    var kmeans = KMeans(d, k, 5)
    kmeans.train(n, x)
    x.free()

def test_kmeans_k_greater_than_n() raises:
    var n = 5
    var d = 16
    var k = 10
    var x = alloc[Float32](n * d)
    for i in range(n * d): x[i] = Float32(i)
    var kmeans = KMeans(d, k, 5)
    kmeans.train(n, x) # Should not crash
    x.free()

def test_kmeans_identical_points() raises:
    var n = 100
    var d = 16
    var k = 5
    var x = alloc[Float32](n * d)
    for i in range(n * d): x[i] = 1.0 # all identical
    var kmeans = KMeans(d, k, 5)
    kmeans.train(n, x) # Should not crash or produce NaN centroids
    
    # Assert no NaNs
    for i in range(k * d):
        assert_true(kmeans.centroids[i] == kmeans.centroids[i], "NaN found in centroids!")
    
    x.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
