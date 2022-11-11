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
    fpath = gdstk.FlexPath((0, 0), 0.5, tolerance=1e-3)
    fpath.arc(1, 0, numpy.pi / 2)
    fpath.arc(1, 0, -numpy.pi / 2)
    rpath = gdstk.RobustPath((3, 0), 0.5, tolerance=1e-3)
    rpath.arc(1, 0, numpy.pi / 2)
    rpath.arc(1, 0, -numpy.pi / 2)
    return gdstk.Cell("init").add(fpath, rpath)


def init1_image():
    path = gdstk.RobustPath(
        (0, 0),
        [0.8, 0.8, 0.8, 0.8],
        1.0,
        ends=["flush", "extended", (0.4, 0.8), "round"],
        tolerance=1e-3,
        layer=[0, 1, 2, 3],
    )
    path.horizontal(5)
    return gdstk.Cell("init1").add(path)


def segment_image():
    path = gdstk.RobustPath((0, 0), 0.2, tolerance=1e-3)
    path.segment((5, 0), 1.0)
    path.segment((5, 3), (0.2, "constant"))
    path.segment((0, 3), (1.2, "smooth"))
    path.segment((0, 7), lambda u: 0.2 + 0.8 * numpy.cos(2 * numpy.pi * u) ** 2)
    return gdstk.Cell("segment").add(path)


def segment1_image():
    path = gdstk.RobustPath((0, 0), [0.2, 0.2], 0.3, tolerance=1e-3)
    path.segment((5, 0), offset=0.8)
    path.segment((5, 3), offset=(0.3, "constant"))
    path.segment((0, 3), offset=(1.0, "smooth"))
    path.segment(
        (0, 7),
        offset=[
            lambda u: -0.2 - 0.8 * numpy.cos(2 * numpy.pi * u) ** 2,
            lambda u: 0.2 + 0.8 * numpy.cos(2 * numpy.pi * u) ** 2,
        ],
    )
    return gdstk.Cell("segment1").add(path)


def cubic_image():
    path = gdstk.RobustPath((0, 0), 0.2, tolerance=1e-3)
    path.cubic([(0.5, 0.8), (0.5, 2.2), (0, 3)])
    path.cubic([(0.8, -0.5), (2.2, -0.5), (3, 0)], relative=True)
    path.cubic([(-0.5, -0.8), (-0.5, -2.2), (0, -3)], relative=True)
    return gdstk.Cell("cubic").add(path)


def cubic_smooth_image():
    path = gdstk.RobustPath((0, 0), 0.2, tolerance=1e-3)
    path.cubic([(0.5, 0.8), (0.5, 2.2), (0, 3)])
    path.cubic_smooth([(3.5, 0.8), (3, 0)], relative=True)
    path.cubic_smooth([(-0.5, -2.2), (0, -3)], relative=True)
    return gdstk.Cell("cubic_smooth").add(path)


def bezier_image():
    path = gdstk.RobustPath((0, 0), 0.2, tolerance=1e-3)
    path.bezier(
        [(4, 1), (4, 3), (0, 5), (-4, 3), (-4, -2), (0, -4)],
        width=lambda u: 0.2 + 0.8 * u**2,
    )
    return gdstk.Cell("bezier").add(path)


def interpolation_image():
    half_pi = numpy.pi / 2
    points = [(4, 1), (4, 3), (0, 5), (-4, 3), (-4, -2), (0, -4)]
    angles = [half_pi, None, None, None, -half_pi, -half_pi, None]
    path_1 = gdstk.RobustPath((0, 0), 0.2, tolerance=1e-3)
    path_1.interpolation(points, cycle=True)
    path_2 = gdstk.RobustPath((6, -8), 0.2, tolerance=1e-3)
    path_2.interpolation(points, angles, cycle=True, relative=True)
    return gdstk.Cell("interpolation").add(path_1, path_2)


def arc_image():
    path = gdstk.RobustPath((0, 0), [0.2, 0.3], 0.4, tolerance=1e-3)
    path.vertical(5)
    path.arc(2.5, numpy.pi, 0)
    path.arc((3, 5), -numpy.pi, -numpy.pi / 2)
    return gdstk.Cell("arc").add(path)


def parametric_image():
    def spiral(u):
        rad = 2 * u**0.5
        ang = 3 * numpy.pi * u
        return (rad * numpy.cos(ang), rad * numpy.sin(ang))

    path = gdstk.RobustPath((0, 0), 0.2, tolerance=1e-3)
    path.parametric(spiral, width=lambda u: 0.2 + 0.6 * u**2)
    return gdstk.Cell("parametric").add(path)


def commands_image():
    path = gdstk.RobustPath((0, 0), [0.2, 0.4, 0.2], 0.5, tolerance=1e-3)
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
    path = pathlib.Path(__file__).parent.absolute() / "robustpath"
    path.mkdir(parents=True, exist_ok=True)

    draw(init_image(), path)
    draw(init1_image(), path)
    draw(segment_image(), path)
    draw(segment1_image(), path)
    draw(cubic_image(), path)
    draw(cubic_smooth_image(), path)
    draw(bezier_image(), path)
    draw(interpolation_image(), path)
    draw(arc_image(), path)
    draw(parametric_image(), path)
    draw(commands_image(), path)
