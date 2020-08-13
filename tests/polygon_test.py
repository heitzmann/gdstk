import numpy
import gdstk


def test_orientation():
    poly = gdstk.Polygon([(0, 0), (1, 0), 1j])
    assert poly.area() == 0.5
    numpy.testing.assert_array_equal(poly.points, [[0, 0], [1, 0], [0, 1]])

    poly = gdstk.Polygon([(0, 0), 1j, (1, 0)])
    assert poly.area() == 0.5
    numpy.testing.assert_array_equal(poly.points, [[1, 0], [0, 1], [0, 0]])

    poly = gdstk.Polygon([(0, 0), (1, 0), 1 + 1j, 1j])
    assert poly.area() == 1.0
    numpy.testing.assert_array_equal(poly.points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    poly = gdstk.Polygon([(0, 0), 1j, 1 + 1j, (1, 0)])
    assert poly.area() == 1.0
    numpy.testing.assert_array_equal(poly.points, [[1, 0], [1, 1], [0, 1], [0, 0]])


if __name__ == "__main__":
    test_orientation()
