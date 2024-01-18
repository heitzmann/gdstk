#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

from copy import deepcopy
import pytest
import numpy
import gdstk

from conftest import assert_same_shape, assert_close


def test_area_size():
    poly = gdstk.Polygon([(0, 0), (1, 0), 1j])
    assert poly.layer == 0
    assert poly.datatype == 0
    assert poly.size == 3
    assert poly.area() == 0.5
    numpy.testing.assert_array_equal(poly.points, [[0, 0], [1, 0], [0, 1]])

    poly = gdstk.Polygon([(0, 0), 1j, (1, 0)], 1, 2)
    assert poly.layer == 1
    assert poly.datatype == 2
    assert poly.size == 3
    assert poly.area() == 0.5
    numpy.testing.assert_array_equal(poly.points, [[0, 0], [0, 1], [1, 0]])

    poly = gdstk.Polygon([(0, 0), (1, 0), 1 + 1j, 1j])
    assert poly.size == 4
    assert poly.area() == 1.0
    numpy.testing.assert_array_equal(poly.points, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_perimeter_size():
    poly1 = gdstk.Polygon([(0, 0), (2, 0), 2 + 2j, 2j])
    assert poly1.layer == 0
    assert poly1.datatype == 0
    assert poly1.size == 4
    assert poly1.perimeter() == 8.0
    numpy.testing.assert_array_equal(poly1.points, [[0, 0], [2, 0], [2, 2], [0, 2]])

    poly2 = gdstk.Polygon([(0, 0), (1, 0), 1 + 1j, 1j], 1, 2)
    assert poly2.layer == 1
    assert poly2.datatype == 2
    assert poly2.size == 4
    assert poly2.perimeter() == 4.0
    numpy.testing.assert_array_equal(poly2.points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    poly3 = gdstk.Polygon([(0, 0), (1, 0), 1 + 1j, 1j])
    assert poly3.size == 4
    assert poly3.perimeter() == 4.0
    numpy.testing.assert_array_equal(poly3.points, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_bounding_box():
    poly = gdstk.Polygon([-1 + 0j, -2j, 3 + 0j, 4j])
    assert poly.bounding_box() == ((-1, -2), (3, 4))


def test_contain():
    r = gdstk.rectangle((0, 0), (20, 10))
    pts = [
        [(1, 1), (-1, -1)],
        [(2, 2), (-2, 2), (2, -2)],
        [(5, 5), (10, 5)],
        [(-1, -1), (-2, -2)],
    ]
    assert r.contain(pts[0][0]) == True
    assert r.contain(pts[0][1]) == False
    assert r.contain(*pts[0]) == (True, False)
    assert r.contain(*pts[1]) == (True, False, False)
    assert r.contain_any(*pts[1]) == True
    assert r.contain_all(*pts[1]) == False
    assert r.contain_any(*pts[2]) == True
    assert r.contain_all(*pts[2]) == True
    assert r.contain_any(*pts[3]) == False
    assert r.contain_all(*pts[3]) == False


def test_copy():
    points = [[-1, 0], [0, -2], [3, 0], [0, 4]]
    p1 = gdstk.Polygon(points, 5, 6)
    p2 = p1.copy()
    assert p1 is not p2
    assert p1.layer == 5
    assert p2.layer == 5
    assert p1.datatype == 6
    assert p2.datatype == 6
    numpy.testing.assert_array_equal(p1.points, points)
    numpy.testing.assert_array_equal(p2.points, points)


def test_deepcopy():
    points = [[-1, 0], [0, -2], [3, 0], [0, 4]]
    p1 = gdstk.Polygon(points, 5, 6)
    p2 = deepcopy(p1)
    assert p1 is not p2
    assert p1.layer == 5
    assert p2.layer == 5
    assert p1.datatype == 6
    assert p2.datatype == 6
    numpy.testing.assert_array_equal(p1.points, points)
    numpy.testing.assert_array_equal(p2.points, points)

    p1.layer = 12
    assert p1.layer == 12
    assert p2.layer == 5


def test_fillet(proof_cells):
    p1 = gdstk.Polygon([(0, 0), (1.2, 0), (1.2, 0.3), (1, 0.3), (1.5, 1), (0, 1.5)])
    p2 = p1.copy().translate(2, 0)
    p1.fillet(0.3, tolerance=1e-3)
    p2.fillet([0.3, 0, 0.1, 0, 0.5, 10], tolerance=1e-3)
    assert_same_shape(proof_cells["Polygon.fillet"].polygons, [p1, p2])


def test_fracture():
    poly = gdstk.racetrack((0, 0), 10, 20, 1, vertical=False)
    frac = poly.fracture(12, 1e-3)
    assert_same_shape(poly, frac)
    poly = gdstk.racetrack((0, 50), 10, 20, 1, vertical=True)
    frac = poly.fracture(12, 1e-3)
    assert_same_shape(poly, frac)


def test_mirror():
    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.mirror(1j)
    numpy.testing.assert_array_equal(poly.points, [[0, 0], [-1, 0], [0, 1]])
    poly.mirror(-1j, 1 + 0j)
    numpy.testing.assert_array_equal(poly.points, [[1, -1], [1, -2], [2, -1]])


def test_rotate():
    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.rotate(numpy.pi / 2, 0.5 + 0.5j)
    assert_close(poly.points, [[1, 0], [1, 1], [0, 0]])


def test_scale():
    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.scale(0.5)
    assert_close(poly.points, [[0, 0], [0.5, 0], [0, 0.5]])

    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.scale(-2, 3, 1 + 0j)
    assert_close(poly.points, [[3, 0], [1, 0], [3, 3]])


def test_translate():
    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.translate(-1, 2)
    assert_close(poly.points, [[-1, 2], [0, 2], [-1, 3]])


def test_transform():
    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.transform()
    assert_close(poly.points, [[0, 0], [1, 0], [0, 1]])

    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.transform(2, True, numpy.pi / 2, -1j)
    assert_close(poly.points, [[0, -1], [0, 1], [2, -1]])

    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.transform(matrix=[[1, 2], [3, 4]])
    assert_close(poly.points, [[0, 0], [1, 3], [2, 4]])

    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.transform(matrix=[[1, 2], [3, 4], [1, -0.5]])
    assert_close(poly.points, [[0, 0], [0.5, 1.5], [4, 8]])

    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.transform(matrix=[[1, 2, 3], [4, 5, 6]])
    assert_close(poly.points, [[3, 6], [4, 10], [5, 11]])

    poly = gdstk.Polygon([0j, 1 + 0j, 1j])
    poly.transform(matrix=[[1, 2, 3], [4, 5, 6], [3, 2, -1]])
    assert_close(poly.points, [[-3, -6], [2, 5], [5, 11]])
