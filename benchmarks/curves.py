#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import numpy
import gdspy
import gdstk


def bench_gdspy(output=None):
    r = 0.15
    c1 = gdspy.Curve(1 / 2 - r, -(3**0.5) / 6, 1e-3)
    c1.arc(r, numpy.pi, numpy.pi * 2 / 3)
    c1.l((1 - 2 * r) * numpy.exp(1j * numpy.pi * 2 / 3))
    c1.arc(r, -numpy.pi / 3, -numpy.pi * 2 / 3)
    c1.l((1 - 2 * r) * numpy.exp(-1j * numpy.pi * 2 / 3))
    c1.arc(r, numpy.pi / 3, 0)
    p1 = gdspy.Polygon(c1.get_points())
    z0 = 2 * r * numpy.exp(-1j * numpy.pi / 6)
    z1 = r * numpy.exp(1j * numpy.pi / 6)
    z2 = r * numpy.exp(1j * numpy.pi * 5 / 6)
    z3 = 2 * r * numpy.exp(-1j * numpy.pi * 5 / 6)
    c2 = gdspy.Curve(0, -r, 1e-3)
    c2.I(
        [
            (z0.real, z0.imag),
            (z1.real, z1.imag),
            (0, 2 * r),
            (z2.real, z2.imag),
            (z3.real, z3.imag),
        ],
        cycle=True,
    )
    p2 = gdspy.Polygon(c2.get_points(), layer=1)
    if output:
        cell = gdspy.Cell("MAIN", exclude_from_current=True)
        cell.add([p1, p2])
        cell.write_svg(output, 300)


def bench_gdstk(output=None):
    r = 0.15
    c1 = gdstk.Curve((1 / 2 - r, -(3**0.5) / 6), 1e-3)
    c1.arc(r, numpy.pi, numpy.pi * 2 / 3)
    c1.segment((1 - 2 * r) * numpy.exp(1j * numpy.pi * 2 / 3), relative=True)
    c1.arc(r, -numpy.pi / 3, -numpy.pi * 2 / 3)
    c1.segment((1 - 2 * r) * numpy.exp(-1j * numpy.pi * 2 / 3), relative=True)
    c1.arc(r, numpy.pi / 3, 0)
    p1 = gdstk.Polygon(c1.points())
    c2 = gdstk.Curve((0, -r), 1e-3)
    c2.interpolation(
        [
            2 * r * numpy.exp(-1j * numpy.pi / 6),
            r * numpy.exp(1j * numpy.pi / 6),
            (0, 2 * r),
            r * numpy.exp(1j * numpy.pi * 5 / 6),
            2 * r * numpy.exp(-1j * numpy.pi * 5 / 6),
        ],
        cycle=True,
    )
    p2 = gdstk.Polygon(c2.points(), layer=1)
    if output:
        cell = gdstk.Cell("MAIN")
        cell.add(p1, p2)
        cell.write_svg(output, 300)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
