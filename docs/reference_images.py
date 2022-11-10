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
    assert bbox == (
        (-12.816310409006173, 1.7677669529663689),
        (11.313708498984761, 27.66555281392367),
    )
    polygon_bb = gdstk.rectangle(*bbox, datatype=1)
    return gdstk.Cell("bounding_box").add(array_ref, polygon_bb)


def convex_hull_image():
    polygons = gdstk.text("F", 10, (0, 0))
    f_cell = gdstk.Cell("F_CELL")
    f_cell.add(*polygons)
    array_ref = gdstk.Reference(
        f_cell, rotation=numpy.pi / 4, columns=3, rows=2, spacing=(8, 10)
    )
    hull = array_ref.convex_hull()
    error = hull - numpy.array(
        [
            [1.14904852, 27.66555281],
            [-12.81631041, 13.70019389],
            [-0.88388348, 1.76776695],
            [11.3137085, 13.96535893],
            [9.98788328, 17.94283457],
            [8.66205807, 20.15254326],
        ]
    )
    assert numpy.abs(error).max() < 1e-8
    polygon_hull = gdstk.Polygon(hull, datatype=1)
    return gdstk.Cell("convex_hull").add(array_ref, polygon_hull)


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "reference"
    path.mkdir(parents=True, exist_ok=True)

    draw(init_image(), path)
    draw(bounding_box_image(), path)
    draw(convex_hull_image(), path)
