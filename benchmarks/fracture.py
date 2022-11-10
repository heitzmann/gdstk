#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import numpy
import gdspy
import gdstk


def pts():
    t = numpy.linspace(0, numpy.pi * 2, 201)[:-1]
    r = 1 + 4 * numpy.cos(2 * t + 0.5) ** 2
    return numpy.vstack((r * numpy.cos(t), r * numpy.sin(t))).T


def bench_gdspy(output=None):
    p = gdspy.Polygon(pts()).fracture(10)
    if output:
        c = gdspy.Cell("MAIN", exclude_from_current=True)
        c.add(p)
        c.write_svg(output, 50)


def bench_gdstk(output=None):
    p = gdstk.Polygon(pts()).fracture(10)
    if output:
        c = gdstk.Cell("MAIN")
        c.add(*p)
        c.write_svg(output, 50)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
