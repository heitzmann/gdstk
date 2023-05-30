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
    polygon_1 = gdstk.Polygon([(0, 0), (1, 0), 1 + 1.5j, 1j])
    polygon_2 = gdstk.Polygon([0j, (-1, 1), (-1, 0)], 2, 2)
    return gdstk.Cell("init").add(polygon_1, polygon_2)


def bounding_box_image():
    polygon = gdstk.Polygon([(0, 1), (1, 2), (3, -1)])
    bbox = polygon.bounding_box()
    polygon_bb = gdstk.rectangle(*bbox, datatype=1)
    return gdstk.Cell("bounding_box").add(polygon, polygon_bb)


def fillet_image():
    points = [(0, 0), (1.2, 0), (1.2, 0.3), (1, 0.3), (1.5, 1), (0, 1.5)]
    polygon_1 = gdstk.Polygon(points, datatype=1)
    polygon_2 = gdstk.Polygon(points).fillet(0.3, tolerance=1e-3)
    return gdstk.Cell("fillet").add(polygon_2, polygon_1)


def fracture_image():
    polygon = gdstk.racetrack((0, 0), 30, 60, 40, tolerance=1e-3)
    poly_list = polygon.fracture()
    return gdstk.Cell("fracture").add(*poly_list)


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "polygon"
    path.mkdir(parents=True, exist_ok=True)

    draw(init_image(), path)
    draw(bounding_box_image(), path)
    draw(fillet_image(), path)
    draw(fracture_image(), path)
