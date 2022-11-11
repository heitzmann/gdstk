#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import numpy
import gdspy
import gdstk


def bench_gdspy(output=None):
    sp1 = gdspy.FlexPath(
        [(0, 0), (3, 0), (3, 2), (5, 3), (3, 4), (0, 4)],
        1,
        gdsii_path=True,
        datatype=1,
    )
    sp1.smooth([(0, 2), (2, 2), (4, 3), (5, 1)], relative=True)
    sp2 = gdspy.FlexPath(
        [(12, 0), (8, 0), (8, 3), (10, 2)],
        [0.3, 0.2, 0.4],
        0.5,
        ends=["extended", "flush", "round"],
        corners=["bevel", "miter", "round"],
        datatype=2,
    )
    sp2.arc(2, -0.5 * numpy.pi, 0.5 * numpy.pi)
    sp2.arc(1, 0.5 * numpy.pi, 1.5 * numpy.pi)

    points = [(0, 0), (0, 10), (20, 0), (18, 15), (8, 15)]
    sp3 = gdspy.FlexPath(
        points,
        0.5,
        corners="circular bend",
        bend_radius=5,
        gdsii_path=True,
        datatype=4,
    )
    sp4 = gdspy.FlexPath(points, 0.5, layer=1, gdsii_path=True, datatype=5)
    if output:
        cell = gdspy.Cell("MAIN", exclude_from_current=True)
        cell.add([sp1, sp2, sp3, sp4])
        cell.write_svg(output, 30)


def bench_gdstk(output=None):
    sp1 = gdstk.FlexPath(
        [(0, 0), (3, 0), (3, 2), (5, 3), (3, 4), (0, 4)],
        1,
        simple_path=True,
        datatype=1,
    )
    sp1.interpolation([(0, 2), (2, 2), (4, 3), (5, 1)], relative=True)

    sp2 = gdstk.FlexPath(
        [(12, 0), (8, 0), (8, 3), (10, 2)],
        [0.3, 0.2, 0.4],
        0.5,
        ends=["extended", "flush", "round"],
        joins=["bevel", "miter", "round"],
        datatype=2,
    )
    sp2.arc(2, -0.5 * numpy.pi, 0.5 * numpy.pi)
    sp2.arc(1, 0.5 * numpy.pi, 1.5 * numpy.pi)

    points = [(0, 0), (0, 10), (20, 0), (18, 15), (8, 15)]
    sp3 = gdstk.FlexPath(points, 0.5, bend_radius=5, simple_path=True, datatype=4)
    sp4 = gdstk.FlexPath(points, 0.5, simple_path=True, layer=1, datatype=5)
    if output:
        cell = gdstk.Cell("MAIN")
        cell.add(sp1, sp2, sp3, sp4)
        cell.write_svg(output, 30)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
