#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import numpy
import gdspy
import gdstk


def bench_gdspy(output=None):
    p = gdspy.Polygon([(0, 0), (1, 0), (0, 1)])
    fp = gdspy.FlexPath([(0, 0), (1, 0), (0.5, -0.5)], 0.1, ends="round")
    c1 = gdspy.Cell("REF", exclude_from_current=True)
    c1.add([p, fp])
    r = gdspy.CellArray(c1, columns=3, rows=2, spacing=(2, 2), rotation=30)
    c2 = gdspy.Cell("MAIN", exclude_from_current=True)
    c2.add(r)
    bb = c2.get_bounding_box()
    if output:
        c2.add(gdspy.Rectangle(*bb, layer=1))
        c2.write_svg(output, 100)


def bench_gdstk(output=None):
    p = gdstk.Polygon([(0, 0), (1, 0), (0, 1)])
    fp = gdstk.FlexPath([(0, 0), (1, 0), (0.5, -0.5)], 0.1, ends="round")
    c1 = gdstk.Cell("REF")
    c1.add(p, fp)
    r = gdstk.Reference(c1, columns=3, rows=2, spacing=(2, 2), rotation=30* numpy.pi/180)
    c2 = gdstk.Cell("MAIN")
    c2.add(r)
    bb = c2.bounding_box()
    if output:
        c2.add(gdstk.rectangle(*bb, layer=1))
        c2.write_svg(output, 100)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
