#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import tempfile
import pathlib
import gdspy
import gdstk


def bench_gdspy():
    cell = gdspy.Cell("MAIN", exclude_from_current=True)
    for i in range(10000):
        cell.add(gdspy.Rectangle((i, 0), (i + 1, 1)))
    name = pathlib.Path(tempfile.gettempdir()) / "gsdpy.gds"
    lib = gdspy.GdsLibrary()
    lib.add(cell)
    lib.write_gds(name)


def bench_gdstk():
    cell = gdstk.Cell("MAIN")
    for i in range(10000):
        cell.add(gdstk.rectangle((i, 0), (i + 1, 1)))
    name = pathlib.Path(tempfile.gettempdir()) / "gsdtk.gds"
    lib = gdstk.Library()
    lib.add(cell)
    lib.write_gds(name)


if __name__ == "__main__":
    bench_gdspy()
    bench_gdstk()
