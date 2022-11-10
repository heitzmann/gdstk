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
    path = gdstk.FlexPath(
        [(0, 5), (0, 0), (5, 0), (15, 10), (15, -5)],
        [0.8, 0.8, 0.8, 0.8],
        1.0,
        joins=["natural", "bevel", "miter", "round"],
        ends=["flush", "extended", (0.4, 0.8), "round"],
        layer=[0, 1, 2, 3],
    )
    return gdstk.Cell("init").add(path)


def init0_image():
    points = [(0, 8), (0, 0), (8, 0), (18, 13), (18, -8)]
    path_1 = gdstk.FlexPath(points, 1, datatype=1)
    path_2 = gdstk.FlexPath(points, 1, bend_radius=3)
    return gdstk.Cell("init0").add(path_2, path_1)


def init1_image():
    def custom_broken_join(p0, v0, p1, v1, center, width):
        p0 = numpy.array(p0)
        v0 = numpy.array(v0)
        p1 = numpy.array(p1)
        v1 = numpy.array(v1)
        center = numpy.array(center)
        # Calculate intersection point p between lines defined by
        # p0 + u0 * v0 (for all u0) and p1 + u1 * v1 (for all u1)
        den = v1[1] * v0[0] - v1[0] * v0[1]
        lim = 1e-12 * (v0[0] ** 2 + v0[1] ** 2) * (v1[0] ** 2 + v1[1] ** 2)
        if den**2 < lim:
            # Lines are parallel: use mid-point
            u0 = u1 = 0
            p = 0.5 * (p0 + p1)
        else:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            u0 = (v1[1] * dx - v1[0] * dy) / den
            u1 = (v0[1] * dx - v0[0] * dy) / den
            p = 0.5 * (p0 + v0 * u0 + p1 + v1 * u1)
        if u0 <= 0 and u1 >= 0:
            # Inner corner
            return [p]
        # Outer corner
        return [p0, center, p1]

    def custom_pointy_end(p0, v0, p1, v1):
        p0 = numpy.array(p0)
        v0 = numpy.array(v0)
        p1 = numpy.array(p1)
        v1 = numpy.array(v1)
        r = 0.5 * numpy.sqrt(numpy.sum((p0 - p1) ** 2))
        v0 /= numpy.sqrt(numpy.sum(v0**2))
        v1 /= numpy.sqrt(numpy.sum(v1**2))
        return [p0, 0.5 * (p0 + p1) + 0.5 * (v0 - v1) * r, p1]

    path = gdstk.FlexPath(
        [(0, 5), (0, 0), (5, 0), (15, 10), (15, -5)],
        3,
        joins=custom_broken_join,
        ends=custom_pointy_end,
    )
    return gdstk.Cell("init1").add(path)


def horizontal_image():
    path = gdstk.FlexPath((0, 0), 0.2)
    path.horizontal(2, width=0.4, relative=True)
    path.horizontal(2, offset=[0.4], relative=True)
    path.horizontal(2, relative=True)
    assert (
        numpy.max(
            numpy.abs(
                path.spine()
                - numpy.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
            )
        )
        == 0
    )
    return gdstk.Cell("horizontal").add(path)


def segment_image():
    points = [(1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    path_1 = gdstk.FlexPath((0, 0), 0.2)
    path_1.segment(points, 0.6)
    path_2 = gdstk.FlexPath((3, 0), [0.1, 0.1], 0.2)
    path_2.segment(points, offset=0.6, relative=True)
    return gdstk.Cell("segment").add(path_1, path_2)


def cubic_image():
    path = gdstk.FlexPath((0, 0), 0.2, tolerance=1e-3)
    path.cubic([(0, 1), (1, 1), (1, 0)])
    path.cubic([(1, -1), (2, -1), (2.5, -0.5), (3, 0), (3, 1), (2, 1)], width=0.5)
    return gdstk.Cell("cubic").add(path)


def cubic_smooth_image():
    path = gdstk.FlexPath((0, 0), 0.2, tolerance=1e-3)
    path.cubic([(0, 1), (1, 1), (1, 0)])
    path.cubic_smooth([(2, -1), (2.5, -0.5), (3, 1), (2, 1)], width=0.5)
    return gdstk.Cell("cubic_smooth").add(path)


def bezier_image():
    path = gdstk.FlexPath((0, 0), 0.2, tolerance=1e-3)
    path.bezier([(4, 1), (4, 3), (0, 5), (-4, 3), (-4, -2), (0, -4)])
    return gdstk.Cell("bezier").add(path)


def interpolation_image():
    half_pi = numpy.pi / 2
    points = [(4, 1), (4, 3), (0, 5), (-4, 3), (-4, -2), (0, -4)]
    angles = [half_pi, None, None, None, -half_pi, -half_pi, None]
    path_1 = gdstk.FlexPath((0, 0), 0.2, tolerance=1e-3)
    path_1.interpolation(points, cycle=True)
    path_2 = gdstk.FlexPath((6, -8), 0.2, tolerance=1e-3)
    path_2.interpolation(points, angles, cycle=True, relative=True)
    return gdstk.Cell("interpolation").add(path_1, path_2)


def arc_image():
    path = gdstk.FlexPath((0, 0), [0.2, 0.3], 0.4, tolerance=1e-3)
    path.vertical(5)
    path.arc(2.5, numpy.pi, 0)
    path.arc(5, -numpy.pi, -numpy.pi / 2)
    return gdstk.Cell("arc").add(path)


def parametric_image():
    def spiral(u):
        rad = 2 * u**0.5
        ang = 3 * numpy.pi * u
        return (rad * numpy.cos(ang), rad * numpy.sin(ang))

    path = gdstk.FlexPath((0, 0), 0.2, tolerance=1e-3)
    path.parametric(spiral)
    return gdstk.Cell("parametric").add(path)


def commands_image():
    path = gdstk.FlexPath((0, 0), [0.2, 0.4, 0.2], 0.5, tolerance=1e-3)
    path.commands(
        "l",
        3,
        4,
        "A",
        2,
        numpy.arctan2(3, -4),
        numpy.pi / 2,
        "h",
        0.5,
        "a",
        3,
        -numpy.pi,
    )
    return gdstk.Cell("commands").add(path)


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "flexpath"
    path.mkdir(parents=True, exist_ok=True)

    draw(init_image(), path)
    draw(init0_image(), path)
    draw(init1_image(), path)
    draw(horizontal_image(), path)
    draw(segment_image(), path)
    draw(cubic_image(), path)
    draw(cubic_smooth_image(), path)
    draw(bezier_image(), path)
    draw(interpolation_image(), path)
    draw(arc_image(), path)
    draw(parametric_image(), path)
    draw(commands_image(), path)
