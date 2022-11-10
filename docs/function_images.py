#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import numpy
import gdstk
from tutorial_images import draw


def cross_image():
    cross1 = gdstk.cross((0, 0), 10, 1)
    cross2 = gdstk.cross((0.5, 0.5), 5, 0.75, layer=1)
    return gdstk.Cell("cross").add(cross1, cross2)


def regular_polygon_image():
    poly3 = gdstk.regular_polygon((0, 0), 9, 3)
    poly4 = gdstk.regular_polygon((10, 0), 7, 4, layer=1)
    poly5 = gdstk.regular_polygon((0, 10), 5, 5, layer=2)
    poly6 = gdstk.regular_polygon((10, 10), 4, 6, layer=3)
    return gdstk.Cell("regular_polygon").add(poly3, poly4, poly5, poly6)


def ellipse_image():
    circle = gdstk.ellipse((0, 0), 40)
    ellipse = gdstk.ellipse((100, 0), (40, 30), layer=1)
    ring = gdstk.ellipse((0, 100), 40, inner_radius=(30, 20), layer=2)
    circle_slice = gdstk.ellipse(
        (100, 100),
        40,
        initial_angle=-numpy.pi / 4,
        final_angle=5 * numpy.pi / 4,
        layer=3,
    )
    ring_slice = gdstk.ellipse(
        (50, 200), (70, 30), (60, 20), -3 * numpy.pi / 4, numpy.pi / 2, layer=4
    )
    return gdstk.Cell("ellipse").add(circle, ellipse, ring, circle_slice, ring_slice)


def racetrack_image():
    racetrack1 = gdstk.racetrack((0, 0), 8, 5)
    racetrack2 = gdstk.racetrack((18, 0), 8, 5, 3, True)
    return gdstk.Cell("racetrack").add(racetrack1, racetrack2)


def text_image():
    text = gdstk.text(f"Created with\nGDSTK {gdstk.__version__}", 1, (0, 0))
    rect = gdstk.rectangle((0, -5 / 4), (5 * 9 / 16, -1 / 4), datatype=1)
    return gdstk.Cell("text").add(*text, rect)


def contour_image():
    y, x = numpy.mgrid[0.5:1.5:128j, -1:1:255j]
    data = numpy.sin(4 * numpy.pi * x * y) - 0.8 * numpy.sin(2 * numpy.pi * y)
    contours = gdstk.contour(data, 0, 1 / 127, 0.01 / 127)
    rect = gdstk.rectangle((0, 0), (2, 1), datatype=1)
    return gdstk.Cell("contour").add(*contours, rect)


def offset_image():
    text = gdstk.text("#A", 10, (0, 0), datatype=1)
    circle = gdstk.ellipse(
        (5, 11), 5, initial_angle=0, final_angle=numpy.pi, datatype=1
    )
    path = gdstk.FlexPath([(0, -1), (5, -10), (10, -1)], 1, datatype=1)
    dilated = gdstk.offset(text + [circle, path], 0.4)
    eroded = gdstk.offset(text + [circle, path], -0.4, use_union=True, layer=1)
    return gdstk.Cell("offset").add(*text, circle, path, *dilated, *eroded)


def boolean_image():
    circle = gdstk.ellipse((0, 0), 50)
    path = gdstk.FlexPath((-50, 30), [5, 5], 10)
    path.interpolation(
        [(20, 15), (0, 0), (-20, -15), (50, -30)], angles=[0.3, None, None, None, 0.3]
    )
    text = gdstk.text("GDSTK", 40, (-2.5 * 40 * 9 / 16, -40 / 2))
    result = gdstk.boolean(circle, text + [path], "not")
    return gdstk.Cell("boolean").add(*result)


def slice_image():
    triangle = gdstk.regular_polygon((-10, 0), 8, 3)
    ring = gdstk.ellipse((10, 0), 8, 5, layer=1)
    result = gdstk.slice([triangle, ring], (-10, -5, 0, 6, 14), "x")
    assert len(result) == 6
    assert all(len(polys) == s for polys, s in zip(result, [1, 1, 0, 1, 2, 1]))
    return gdstk.Cell("slice").add(*[p for polys in result for p in polys])


def read_rawcells_example():
    cell1 = gdstk.Cell("CELL_1")
    cell1.add(gdstk.rectangle((0, 0), (2, 1)))
    cell2 = gdstk.Cell("CELL_2")
    cell2.add(gdstk.Reference(cell1, (-1, 0)))
    library = gdstk.Library()
    library.add(cell1, cell2)
    library.write_gds("test.gds")
    raw_cells = gdstk.read_rawcells("test.gds")
    assert tuple(sorted(raw_cells.keys())) == ("CELL_1", "CELL_2")
    assert len(raw_cells["CELL_1"].dependencies(True)) == 0
    assert len(raw_cells["CELL_2"].dependencies(True)) == 1
    deps = raw_cells["CELL_2"].dependencies(True)
    assert deps[0] is raw_cells["CELL_1"]


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "function"
    path.mkdir(parents=True, exist_ok=True)

    draw(cross_image(), path)
    draw(regular_polygon_image(), path)
    draw(ellipse_image(), path)
    draw(racetrack_image(), path)
    draw(text_image(), path)
    draw(contour_image(), path)
    draw(offset_image(), path)
    draw(boolean_image(), path)
    draw(slice_image(), path)
    read_rawcells_example()
