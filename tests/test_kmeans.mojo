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
    kmeans.train(Span[Float32, MutUntrackedOrigin](ptr=x, length=n * d))
    x.free()

def test_kmeans_k_greater_than_n() raises:
    var n = 5
    var d = 16
    var k = 10
    var x = alloc[Float32](n * d)
    for i in range(n * d): x[i] = Float32(i)
    var kmeans = KMeans(d, k, 5)
    kmeans.train(Span[Float32, MutUntrackedOrigin](ptr=x, length=n * d))
    x.free()

def test_kmeans_identical_points() raises:
    var n = 100
    var d = 16
    var k = 5
    var x = alloc[Float32](n * d)
    for i in range(n * d): x[i] = 1.0 # all identical
    var kmeans = KMeans(d, k, 5)
    kmeans.train(Span[Float32, MutUntrackedOrigin](ptr=x, length=n * d))
    
    # Assert no NaNs
    for i in range(k * d):
        assert_true(kmeans.centroids[i] == kmeans.centroids[i], "NaN found in centroids!")
    
    x.free()


def test_kmeans_training_is_reproducible() raises:
    var n = 257
    var d = 8
    var k = 17
    var x = alloc[Float32](n * d)
    for i in range(n * d):
        x[i] = Float32((i * 37 + i // d * 11) % 101) / 50.0 - 1.0

    var data = Span[Float32, MutUntrackedOrigin](
        ptr=x,
        length=n * d,
    )
    var first = KMeans(d, k, 5)
    var second = KMeans(d, k, 5)
    first.train(data)
    second.train(data)

    for i in range(k * d):
        assert_almost_equal(
            first.centroids[i],
            second.centroids[i],
            atol=1e-6,
        )

    x.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
