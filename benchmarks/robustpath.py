#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import numpy
import gdspy
import gdstk


def bench_gdspy(output=None):
    rp = gdspy.RobustPath(
        (50, 0),
        [2, 0.5, 1, 1],
        [0, 0, -1, 1],
        ends=["extended", "round", "flush", "flush"],
        layer=[0, 2, 1, 1],
        datatype=[0, 1, 2, 3],
        max_points=8190,
    )
    rp.segment((45, 0))
    rp.segment(
        (5, 0),
        width=[lambda u: 2 + 16 * u * (1 - u), 0.5, 1, 1],
        offset=[
            0,
            lambda u: 8 * u * (1 - u) * numpy.cos(12 * numpy.pi * u),
            lambda u: -1 - 8 * u * (1 - u),
            lambda u: 1 + 8 * u * (1 - u),
        ],
    )
    rp.segment((0, 0))
    rp.smooth(
        [(5, 10)],
        angles=[0.5 * numpy.pi, 0],
        width=0.5,
        offset=[-0.25, 0.25, -0.75, 0.75],
    )
    rp.bezier([(0, 10), (10, 10), (10, -10), (20, -10), (20, 0), (30, 0)])
    if output:
        cell = gdspy.Cell("MAIN", exclude_from_current=True)
        cell.add(rp)
        cell.write_svg(output, 10)


def bench_gdstk(output=None):
    # Offset signs must be inverted to be compatible with gdspy!
    rp = gdstk.RobustPath(
        (50, 0),
        [2, 0.5, 1, 1],
        [0, 0, 1, -1],
        ends=["extended", "round", "flush", "flush"],
        layer=[0, 2, 1, 1],
        datatype=[0, 1, 2, 3],
    )
    rp.segment((45, 0))
    rp.segment(
        (5, 0),
        width=[lambda u: 2 + 16 * u * (1 - u), 0.5, 1, 1],
        offset=[
            0,
            lambda u: -8 * u * (1 - u) * numpy.cos(12 * numpy.pi * u),
            lambda u: 1 + 8 * u * (1 - u),
            lambda u: -1 - 8 * u * (1 - u),
        ],
    )
    rp.segment((0, 0))
    rp.interpolation(
        [(5, 10)],
        angles=[0.5 * numpy.pi, 0],
        width=0.5,
        offset=[0.25, -0.25, 0.75, -0.75],
    )
    rp.bezier(
        [(0, 10), (10, 10), (10, -10), (20, -10), (20, 0), (30, 0)], relative=True
    )
    if output:
        cell = gdstk.Cell("MAIN")
        cell.add(rp)
        cell.write_svg(output, 10)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
