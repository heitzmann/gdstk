#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import gdspy
import gdstk


def bench_gdspy(output=None):
    poly = gdspy.Round((0, 0), 1.5, number_of_points=6, layer=1)
    orig = gdspy.Cell("OFF", exclude_from_current=True)
    orig.add(poly)
    ref = gdspy.CellArray(orig, 4, 1, (2, 0), origin=(-3, 5))
    off = gdspy.offset([poly, ref], 0.2, "round", layer=0)
    boo = gdspy.boolean(off, [poly, ref], "not", layer=2)
    if output:
        cell = gdspy.Cell("MAIN", exclude_from_current=True)
        cell.add([ref, poly, off, boo])
        cell.write_svg(output, 50)


def bench_gdstk(output=None):
    poly = gdstk.regular_polygon((0, 0), 1.5, 6, layer=1)
    orig = gdstk.Cell("OFF")
    orig.add(poly)
    ref = gdstk.Reference(orig, (-3, 5), columns=4, spacing=(2, 0))
    off = gdstk.offset([poly, ref], 0.2, "bevel", layer=0)
    boo = gdstk.boolean(off, [poly, ref], "not", layer=2)
    if output:
        cell = gdstk.Cell("MAIN")
        cell.add(ref, poly, *off, *boo)
        cell.write_svg(output, 50)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
