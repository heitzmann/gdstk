#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import numpy
import gdstk
from tutorial_images import draw


def bounding_box_image():
    polygons = gdstk.text("F", 10, (0, 0))
    f_cell = gdstk.Cell("F_CELL")
    f_cell.add(*polygons)
    array_ref = gdstk.Reference(
        f_cell, rotation=numpy.pi / 4, columns=3, rows=2, spacing=(8, 10)
    )
    path = gdstk.FlexPath([(-5, 0), (0, -5), (5, 0)], 1, gdsii_path=True)
    main_cell = gdstk.Cell("MAIN")
    main_cell.add(array_ref, path)
    bbox = main_cell.bounding_box()
    assert bbox == (
        (-12.816310409006173, -5.707106781186548),
        (11.313708498984761, 27.66555281392367),
    )
    polygon_bb = gdstk.rectangle(*bbox, datatype=1)
    main_cell.name = "bounding_box"
    return main_cell.add(polygon_bb)


def flatten_image():
    poly1 = gdstk.Polygon([(0, 0), (1, 0), (0.5, 1)])
    cell1 = gdstk.Cell("CELL_1")
    cell1.add(poly1)
    poly2 = gdstk.Polygon([(1, 0), (1.5, 1), (0.5, 1)], layer=1)
    ref = gdstk.Reference(cell1, columns=2, rows=2, spacing=(1, 1))
    cell2 = gdstk.Cell("CELL_2")
    cell2.add(poly2, ref)
    assert len(cell2.polygons) == 1
    assert len(cell2.references) == 1
    assert len(cell2.dependencies(True)) == 1
    cell2.flatten()
    assert len(cell2.polygons) == 5
    assert len(cell2.references) == 0
    assert len(cell2.dependencies(True)) == 0
    cell2.name = "flatten"
    return cell2


def write_svg_image():
    poly1 = gdstk.ellipse((0, 0), (13, 10), datatype=1)  # (layer, datatype) = (0, 1)
    poly2 = gdstk.ellipse((0, 0), (10, 7), layer=1)  # (layer, datatype) = (1, 0)
    cell = gdstk.Cell("SVG")
    cell.add(poly1, poly2)
    # cell.write_svg(
    #     "example.svg",
    #     background="none",
    #     style={(0, 1): {"fill": "none", "stroke": "black", "stroke-dasharray": "8,8"}},
    #     pad="5%",
    # )
    cell.name = "write_svg"
    return cell


def remove_image():
    polygons = gdstk.text("Filter dots\nin i and j!", 8, (0, 0))
    cell = gdstk.Cell("FILTERED")
    cell.add(*polygons)
    dots = [poly for poly in cell.polygons if poly.area() < 2]
    cell.remove(*dots)
    cell.name = "remove"
    return cell


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "cell"
    path.mkdir(parents=True, exist_ok=True)

    draw(bounding_box_image(), path)
    draw(flatten_image(), path)
    draw(write_svg_image(), path)
    draw(remove_image(), path)
