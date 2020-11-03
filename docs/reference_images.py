#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020-2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import numpy
import gdstk
from tutorial_images import draw


def init_image():
    polygons = gdstk.text("F", 10, (0, 0))
    f_cell = gdstk.Cell("F_CELL")
    f_cell.add(*polygons)
    ref = gdstk.Reference(f_cell, rotation=numpy.pi / 2)
    array_ref = gdstk.Reference(f_cell, columns=3, rows=2, spacing=(8, 10))
    return gdstk.Cell("init").add(ref, array_ref)


def bounding_box_image():
    polygons = gdstk.text("F", 10, (0, 0))
    f_cell = gdstk.Cell("F_CELL")
    f_cell.add(*polygons)
    array_ref = gdstk.Reference(
        f_cell, rotation=numpy.pi / 4, columns=3, rows=2, spacing=(8, 10)
    )
    bbox = array_ref.bounding_box()
    # print(bbox)
    polygon_bb = gdstk.rectangle(*bbox, datatype=1)
    return gdstk.Cell("bounding_box").add(array_ref, polygon_bb)


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "reference"
    path.mkdir(parents=True, exist_ok=True)

    draw(init_image(), path)
    draw(bounding_box_image(), path)
