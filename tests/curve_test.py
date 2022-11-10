#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pytest
import numpy
import gdstk


def test_init():
    curve = gdstk.Curve(1j)
    assert curve.tolerance == 1e-2
    curve.tolerance = 1e-1
    assert curve.tolerance == 1e-1
    numpy.testing.assert_array_equal(curve.points(), [[0, 1]])

    curve = gdstk.Curve((4, 5), 1e-3)
    assert curve.tolerance == 1e-3
    numpy.testing.assert_array_equal(curve.points(), [[4, 5]])


def test_points():
    points = [(0, 1), (1, 0), (-1, -1)]
    curve = gdstk.Curve(points[0])
    curve.segment(points[1])
    numpy.testing.assert_array_equal(curve.points(), points[:2])
    curve.segment(points[2])
    numpy.testing.assert_array_equal(curve.points(), points)

    points = [(0, 1), (1, 0), (1, 0), (-1, -1), (0.02, 1.02)]
    curve = gdstk.Curve(points[0], 1e-1)
    curve.segment(points[1:])
    numpy.testing.assert_array_equal(curve.points(), points[:-1])
