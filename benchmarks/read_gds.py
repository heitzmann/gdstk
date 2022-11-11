#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import gdspy
import gdstk


def bench_gdspy():
    lib = gdspy.GdsLibrary(infile="tests/proof_lib.gds")


def bench_gdstk():
    lib = gdstk.read_gds("tests/proof_lib.gds")


if __name__ == "__main__":
    bench_gdspy()
    bench_gdstk()
