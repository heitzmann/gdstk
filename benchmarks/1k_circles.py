#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import tempfile
import pathlib
import numpy
import gdspy
import gdstk


def bench_gdspy():
    n = 8000
    r = 1
    cell = gdspy.Cell("MAIN", exclude_from_current=True)
    for i in range(1000):
        cell.add(gdspy.Round((i, 0), r, number_of_points=n, max_points=0))
    name = pathlib.Path(tempfile.gettempdir()) / "gsdpy.gds"
    lib = gdspy.GdsLibrary()
    lib.add(cell)
    lib.write_gds(name)


def bench_gdstk():
    n = 8000
    r = 1
    length = 2 * r * numpy.sin(numpy.pi / n)
    cell = gdstk.Cell("MAIN")
    for i in range(1000):
        cell.add(gdstk.regular_polygon((i, 0), length, n))
    name = pathlib.Path(tempfile.gettempdir()) / "gsdtk.gds"
    lib = gdstk.Library()
    lib.add(cell)
    lib.write_gds(name, 0)


if __name__ == "__main__":
    bench_gdspy()
    bench_gdstk()
