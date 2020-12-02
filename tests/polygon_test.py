#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

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


def test_bounding_box():
    poly = gdstk.Polygon([-1 + 0j, -2j, 3 + 0j, 4j])
    assert poly.bounding_box() == ((-1, -2), (3, 4))


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
