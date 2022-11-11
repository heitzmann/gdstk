#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import numpy
import gdstk
from tutorial_images import draw


def init_image():
    curve = gdstk.Curve((3, 4), tolerance=1e-3)
    curve.segment((1, 1), True)
    curve.turn(1, -numpy.pi / 2)
    curve.segment((1, -1), True)
    polygon = gdstk.Polygon(curve.points())
    return gdstk.Cell("init").add(polygon)


def segment_image():
    curve = gdstk.Curve((1, 0))
    curve.segment((0, 1))
    curve.segment([0j, -1 + 0j])
    curve.segment([(0, -1), (2, -1)], True)
    polygon = gdstk.Polygon(curve.points())
    return gdstk.Cell("segment").add(polygon)


def cubic_image():
    curve = gdstk.Curve((0, 0), tolerance=1e-3)
    curve.cubic([(1, -2), (2, -2), (3, 0)])
    curve.cubic([(2.7, 1), (1.8, 1), (1.5, 0), (1.3, -0.2), (0.3, -0.2), (0, 0)])
    polygon = gdstk.Polygon(curve.points())
    return gdstk.Cell("cubic").add(polygon)


def cubic_smooth_image():
    curve = gdstk.Curve((0, 0), tolerance=1e-3)
    curve.cubic([1 + 0j, 1.5 + 0.5j, 1 + 1j])
    curve.cubic_smooth([1j, 0j])
    polygon = gdstk.Polygon(curve.points())
    return gdstk.Cell("cubic_smooth").add(polygon)


def bezier_image():
    points = [(4, 1), (4, 3), (0, 5), (-4, 3), (-4, -2), (0, -4), (0, 0)]
    curve = gdstk.Curve((0, 0))
    curve.segment(points)
    control_poly = gdstk.Polygon(curve.points(), datatype=1)
    curve = gdstk.Curve((0, 0), tolerance=1e-3)
    curve.bezier(points)
    polygon = gdstk.Polygon(curve.points())
    return gdstk.Cell("bezier").add(polygon, control_poly)


def interpolation_image():
    points = [(4, 1), (4, 3), (0, 5), (-4, 3), (-4, -2), (0, -4)]
    curve = gdstk.Curve((0, 0))
    curve.segment(points)
    control_poly_1 = gdstk.Polygon(curve.points(), datatype=1)
    curve = gdstk.Curve((0, 0), tolerance=1e-3)
    curve.interpolation(points, cycle=True)
    polygon_1 = gdstk.Polygon(curve.points())

    half_pi = numpy.pi / 2
    angles = [half_pi, None, None, None, -half_pi, -half_pi, None]
    curve = gdstk.Curve((4, -9))
    curve.segment(points, relative=True)
    control_poly_2 = gdstk.Polygon(curve.points(), datatype=1)
    curve = gdstk.Curve((4, -9), tolerance=1e-3)
    curve.interpolation(points, angles, cycle=True, relative=True)
    polygon_2 = gdstk.Polygon(curve.points())
    return gdstk.Cell("interpolation").add(
        polygon_1, control_poly_1, polygon_2, control_poly_2
    )


def arc_image():
    curve = gdstk.Curve((-0.6, 0), tolerance=1e-3)
    curve.segment((1, 0), True)
    curve.arc(1, 0, numpy.pi / 2)
    polygon_1 = gdstk.Polygon(curve.points())

    curve = gdstk.Curve((0.6, 0), tolerance=1e-3)
    curve.segment((1, 0), True)
    curve.arc((2**-0.5, 0.4), -numpy.pi / 4, 3 * numpy.pi / 4, -numpy.pi / 4)
    polygon_2 = gdstk.Polygon(curve.points())
    return gdstk.Cell("arc").add(polygon_1, polygon_2)


def parametric_image():
    def top(u):
        x = 4 * u
        y = 1 - numpy.cos(4 * numpy.pi * u)
        return (x, y)

    curve = gdstk.Curve((-2, 0), tolerance=1e-3)
    curve.parametric(top)
    curve.parametric(lambda u: (4 - 2 * u**0.5) * numpy.exp(-1.5j * numpy.pi * u) - 4)
    polygon = gdstk.Polygon(curve.points())
    return gdstk.Cell("parametric").add(polygon)


def commands_image():
    curve = gdstk.Curve((0, 0), tolerance=1e-3)
    curve.commands("l", 1, 1, "a", 1, -numpy.pi / 2, "l", 1, -1, "S", 1, -2, 0, -2)
    polygon = gdstk.Polygon(curve.points())
    return gdstk.Cell("commands").add(polygon)


def tolerance_image():
    curve = gdstk.Curve((-2.5, 0), tolerance=1e-1)
    curve.arc((2, 3), 0, numpy.pi)
    polygon_1 = gdstk.Polygon(curve.points())
    assert polygon_1.size == 7

    curve = gdstk.Curve((2.5, 0), tolerance=1e-3)
    curve.arc((2, 3), 0, numpy.pi)
    polygon_2 = gdstk.Polygon(curve.points())
    assert polygon_2.size == 62
    return gdstk.Cell("tolerance").add(polygon_1, polygon_2)


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "curve"
    path.mkdir(parents=True, exist_ok=True)

    draw(init_image(), path)
    draw(segment_image(), path)
    draw(cubic_image(), path)
    draw(cubic_smooth_image(), path)
    draw(bezier_image(), path)
    draw(interpolation_image(), path)
    draw(arc_image(), path)
    draw(parametric_image(), path)
    draw(commands_image(), path)
    draw(tolerance_image(), path)
