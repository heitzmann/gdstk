#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import gdspy
import gdstk


def bench_gdspy():
    d = gdspy.get_binary_cells("tests/proof_lib.gds")


def bench_gdstk():
    d = gdstk.read_rawcells("tests/proof_lib.gds")


if __name__ == "__main__":
    bench_gdspy()
    bench_gdstk()
