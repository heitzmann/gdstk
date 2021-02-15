#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pytest
import numpy
import gdstk

from conftest import assert_same_shape, assert_close


@pytest.fixture
def tree():
    p1 = gdstk.Polygon(((0, 0), (0, 1), (1, 0)), 0, 0)
    p2 = gdstk.Polygon(((2, 0), (2, 1), (1, 0)), 1, 1)
    l1 = gdstk.Label("label1", (0, 0), layer=11)
    l2 = gdstk.Label("label2", (2, 1), layer=12)
    c1 = gdstk.Cell("tree1")
    c1.add(p1)
    c1.add(l1)
    c2 = gdstk.Cell("tree2")
    c2.add(l2)
    c2.add(p2)
    c2.add(gdstk.Reference(c1))
    c3 = gdstk.Cell("tree3")
    c3.add(gdstk.Reference(c2, (0, 0), columns=3, rows=2, spacing=(3, 3)))
    return c3, c2, c1


def test_add_element():
    p = gdstk.Polygon(((0, 0), (1, 0), (0, 1)))
    l = gdstk.Label("label", (0, 0))
    c = gdstk.Cell("c_add_element")
    assert c.add(p) is c
    assert c.add(p, l) is c
    polygons = c.polygons
    assert len(polygons) == 2
    assert polygons[0] is p
    assert polygons[1] is p
    assert len(c.labels) == 1
    assert c.labels[0] is l


def test_copy():
    p = gdstk.Polygon(((0, 0), (1, 0), (0, 1)))
    lbl = gdstk.Label("label", (0, 0))
    cref = gdstk.Cell("ref").add(gdstk.rectangle((-1, -1), (-2, -2)))
    ref = gdstk.Reference(cref)
    cell = gdstk.Cell("original")
    cell.add(p, lbl, ref)
    shallow_copy = cell.copy("copy_0", deep_copy=False)
    assert len(shallow_copy.polygons) == len(cell.polygons)
    assert p in shallow_copy.polygons
    assert len(shallow_copy.labels) == len(cell.labels)
    assert lbl in shallow_copy.labels
    assert len(shallow_copy.references) == len(cell.references)
    assert ref in shallow_copy.references
    deep_copy = cell.copy("copy_1")
    assert len(deep_copy.polygons) == len(cell.polygons)
    assert p not in deep_copy.polygons
    assert len(deep_copy.labels) == len(cell.labels)
    assert lbl not in deep_copy.labels
    assert len(deep_copy.references) == len(cell.references)
    assert ref not in deep_copy.references
    assert deep_copy.references[0].cell is ref.cell


def test_copy_transform(proof_cells):
    ref_cell1 = gdstk.Cell("Reference 1")
    ref_cell1.add(*gdstk.text("F.", 10, (0, 0)))
    ref_cell1.add(gdstk.Label("LaBeL", (2.4, 8.7), "s"))
    ref_cell1.add(
        gdstk.FlexPath(8 + 4j, 1, simple_path=True, layer=3).arc(2, 0, numpy.pi / 2)
    )
    ref_cell1.add(
        gdstk.RobustPath(7.5 + 7j, 1, simple_path=True, layer=4).bezier(
            [-2 + 1j, -2 + 3j, 4j, 6j, -3 + 6j], relative=True
        )
    )

    ref_cell2 = gdstk.Cell("Reference 2")
    ref_cell2.add(*gdstk.text("^", 10, (0, 5), layer=1))
    ref_cell2.add(gdstk.Reference(ref_cell1))

    cell = gdstk.Cell("Original cell")
    cell.add(gdstk.rectangle((-1, -0.5), (1, 0.5), layer=2))
    cell.add(gdstk.Reference(ref_cell2))
    cell.add(gdstk.Reference(ref_cell1, (10, 7), numpy.pi / 4, 0.5, True))
    cell.add(
        gdstk.Reference(ref_cell1, (-7, 15), -numpy.pi / 3, 0.5, True, 3, 2, (5, 4))
    )
    cell.add(
        gdstk.Reference(ref_cell2, (-7, 23), numpy.pi / 3, 0.5, True, 3, 2, (5, 8))
    )
    cell_copy = cell.copy("Cell.copy", (-10, -10), numpy.pi / 2, 2, True).flatten()
    for path in cell_copy.paths:
        cell_copy.add(*path.to_polygons())
    assert_same_shape(proof_cells["Cell.copy"].polygons, cell_copy.polygons)


def test_remove(tree):
    c3, c2, c1 = tree
    p1 = c1.polygons[0]
    l1 = c1.labels[0]
    c1.remove(p1)
    assert len(c1.polygons) == 0
    assert len(c1.labels) == 1
    c1.remove(p1, l1)
    assert len(c1.polygons) == 0
    assert len(c1.labels) == 0


def test_filter():
    polys = [
        gdstk.rectangle((0, 0), (1, 1), layer=l, datatype=t)
        for t in range(3)
        for l in range(3)
    ]
    labels = [
        gdstk.Label("FILTER", (1, 1), layer=l, texttype=t)
        for t in range(3)
        for l in range(3)
    ]
    paths = [
        gdstk.FlexPath([0j, 1j], [0.1, 0.1, 0.1], 0.5, layer=[0, 1, 2], datatype=t)
        for t in range(3)
    ] + [
        gdstk.RobustPath(0j, [0.1, 0.1], 0.5, layer=[1, 2], datatype=t)
        for t in range(3)
    ]
    layers = [1, 2]
    types = [0]
    for op, test in [
        ("and", lambda a, b: a and b),
        ("or", lambda a, b: a or b),
        ("xor", lambda a, b: (a and not b) or (b and not a)),
        ("nand", lambda a, b: not (a and b)),
        ("nor", lambda a, b: not (a or b)),
        ("nxor", lambda a, b: not ((a and not b) or (b and not a))),
    ]:
        path_results = [
            [test(a in layers, b in types) for a, b in zip(path.layers, path.datatypes)]
            for path in paths
        ]
        cell = gdstk.Cell(op)
        cell.add(*polys, *labels, *paths)
        cell.filter(layers, types, op)

        cell_polys = cell.polygons
        for poly in polys:
            if test(poly.layer in layers, poly.datatype in types):
                assert poly not in cell_polys
            else:
                assert poly in cell_polys
        cell_labels = cell.labels
        for label in labels:
            if test(label.layer in layers, label.texttype in types):
                assert label not in cell_labels
            else:
                assert label in cell_labels
        cell_paths = cell.paths
        for path, results in zip(paths, path_results):
            if all(results):
                assert path not in cell_paths
            else:
                assert path in cell_paths
                assert len(path.layers) == len(results) - sum(results)
                assert all(
                    not test(a in layers, b in types)
                    for a, b in zip(path.layers, path.datatypes)
                )


def test_area():
    c = gdstk.Cell("c_area")
    c.add(gdstk.rectangle((0, 0), (1, 1), layer=0))
    c.add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    c.add(gdstk.rectangle((1, 1), (2, 2), layer=1))
    c.add(gdstk.rectangle((1, 1), (2, 2), datatype=2))
    assert c.area() == 4.0
    assert c.area(True) == {(0, 0): 1.0, (1, 0): 2.0, (0, 2): 1}


def test_flatten(tree):
    c3, c2, c1 = tree
    c3.flatten()
    polygons = c3.polygons
    assert len(polygons) == 12
    for i in range(12):
        assert polygons[i].layer == 0 or polygons[i].layer == 1
        assert polygons[i].layer == polygons[i].datatype
    assert len(c3.labels) == 12


def test_bb(tree):
    c3, c2, c1 = tree
    assert_close(c3.bounding_box(), ((0, 0), (8, 4)))
    p2 = gdstk.Polygon(((-1, 2), (-1, 1), (0, 2)), 2, 2)
    c2.add(p2)
    assert_close(c3.bounding_box(), ((-1, 0), (8, 5)))
    p1 = gdstk.Polygon(((0, 3), (0, 2), (1, 3)), 3, 3)
    c1.add(p1)
    assert_close(c3.bounding_box(), ((-1, 0), (8, 6)))
