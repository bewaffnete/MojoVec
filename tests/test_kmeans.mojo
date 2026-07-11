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

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
