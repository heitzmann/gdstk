#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import numpy
import gdspy
import gdstk


def bench_gdspy(output=None):
    p = gdspy.Polygon([(0, 0), (1, 0), (0, 1)])
    fp = gdspy.FlexPath([(-1, 0.5), (1, 0), (0.5, -0.5)], [0.1, 0.1], 0.3, ends="round")
    rp = gdspy.RobustPath((0, 0), [0.1, 0.1], 0.3).smooth([(2, 0.5), (2, 1), (-1, 4)])
    c1 = gdspy.Cell("REF", exclude_from_current=True)
    c1.add([p, fp, rp])
    r1 = gdspy.CellArray(c1, columns=3, rows=2, spacing=(2, 2), rotation=30)
    r2 = gdspy.CellArray(c1, origin=(8, 0), columns=5, rows=4, spacing=(2, 2))
    r3 = gdspy.CellArray(c1, origin=(9, 1), columns=4, rows=3, spacing=(2, 2))
    c2 = gdspy.Cell("MAIN", exclude_from_current=True)
    c2.add([r1, r2, r3])
    bb = c2.get_bounding_box()
    if output:
        c2.add(gdspy.Rectangle(*bb, layer=1))
        c2.write_svg(output, 100)


def bench_gdstk(output=None):
    p = gdstk.Polygon([(0, 0), (1, 0), (0, 1)])
    fp = gdstk.FlexPath([(-1, 0.5), (1, 0), (0.5, -0.5)], [0.1, 0.1], 0.3, ends="round")
    rp = gdstk.RobustPath((0, 0), [0.1, 0.1], 0.3).interpolation(
        [(2, 0.5), (2, 1), (-1, 4)]
    )
    c1 = gdstk.Cell("REF")
    c1.add(p, fp, rp)
    r1 = gdstk.Reference(
        c1, columns=3, rows=2, spacing=(2, 2), rotation=30 * numpy.pi / 180
    )
    r2 = gdstk.Reference(c1, origin=(8, 0), columns=5, rows=4, spacing=(2, 2))
    r3 = gdstk.Reference(c1, origin=(9, 1), columns=4, rows=3, spacing=(2, 2))
    c2 = gdstk.Cell("MAIN")
    c2.add(r1, r2, r3)
    bb = c2.bounding_box()
    if output:
        c2.add(gdstk.rectangle(*bb, layer=1))
        c2.write_svg(output, 100)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
