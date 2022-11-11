#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import gdspy
import gdstk


def bench_gdspy(output=None):
    c1 = gdspy.Cell("REF", exclude_from_current=True)
    c1.add(gdspy.Rectangle((0, 0), (10, 10)))
    c1.add(
        gdspy.FlexPath([(0, 0), (10, 0), (10, 10), (0, 10)], [0.1, 0.1], 0.3, layer=1)
    )
    c1.add(gdspy.Label("Label", (5, 5), anchor="o"))
    c2 = gdspy.Cell("MAIN", exclude_from_current=True)
    c2.add(gdspy.CellArray(c1, columns=3, rows=2, spacing=(20, 20)))
    c2.flatten()
    c1.remove_polygons(lambda *_: True)
    c1.remove_paths(lambda *_: True)
    c1.remove_labels(lambda *_: True)
    if output:
        c2.write_svg(output, 10)


def bench_gdstk(output=None):
    c1 = gdstk.Cell("REF")
    c1.add(gdstk.rectangle((0, 0), (10, 10)))
    c1.add(
        gdstk.FlexPath([(0, 0), (10, 0), (10, 10), (0, 10)], [0.1, 0.1], 0.3, layer=1)
    )
    c1.add(gdstk.Label("Label", (5, 5), anchor="o"))
    c2 = gdstk.Cell("MAIN")
    c2.add(gdstk.Reference(c1, columns=3, rows=2, spacing=(20, 20)))
    c2.flatten()
    c1.remove(*c1.polygons, *c1.paths, *c1.labels)
    if output:
        c2.write_svg(output, 10)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
