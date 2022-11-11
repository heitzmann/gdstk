#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import numpy
import gdstk
from tutorial_images import draw


def init_image():
    frame = gdstk.rectangle((-2, -1), (2, 1), datatype=1)
    label_o = gdstk.Label("Center", (0, 0), rotation=numpy.pi / 6)
    label_n = gdstk.Label("North", (0, 1), "n")
    label_s = gdstk.Label("South", (0, -1), "s")
    label_e = gdstk.Label("East", (2, 0), "e")
    label_w = gdstk.Label("West", (-2, 0), "w")
    label_ne = gdstk.Label("Northeast", (2, 1), "ne")
    label_se = gdstk.Label("Southeast", (2, -1), "se")
    label_nw = gdstk.Label("Northwest", (-2, 1), "nw")
    label_sw = gdstk.Label("Southwest", (-2, -1), "sw")
    return gdstk.Cell("init").add(
        frame,
        label_o,
        label_n,
        label_s,
        label_e,
        label_w,
        label_ne,
        label_se,
        label_nw,
        label_sw,
    )


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "label"
    path.mkdir(parents=True, exist_ok=True)

    draw(init_image(), path)
