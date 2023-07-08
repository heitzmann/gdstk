#!/usr/bin/env python

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
    ref_cell1.add(gdstk.FlexPath(8 + 4j, 1, simple_path=True, layer=3).arc(2, 0, numpy.pi / 2))
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
    cell.add(gdstk.Reference(ref_cell1, (-7, 15), -numpy.pi / 3, 0.5, True, 3, 2, (5, 4)))
    cell.add(gdstk.Reference(ref_cell2, (-7, 23), numpy.pi / 3, 0.5, True, 3, 2, (5, 8)))
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
        gdstk.rectangle((0, 0), (1, 1), layer=l, datatype=t) for t in range(3) for l in range(3)
    ]
    labels = [
        gdstk.Label("FILTER", (1, 1), layer=l, texttype=t) for t in range(3) for l in range(3)
    ]
    paths = [
        gdstk.FlexPath([0j, 1j], [0.1, 0.1, 0.1], 0.5, layer=[0, 1, 2], datatype=t)
        for t in range(3)
    ] + [gdstk.RobustPath(0j, [0.1, 0.1], 0.5, layer=[1, 2], datatype=t) for t in range(3)]
    spec = [(1, 0), (2, 0)]
    for remove in (True, False):
        cell = gdstk.Cell(str(remove))
        cell.add(*polys, *labels, *paths)
        cell.filter(spec, remove)

        cell_polys = cell.polygons
        for poly in polys:
            if ((poly.layer, poly.datatype) in spec) == remove:
                assert poly not in cell_polys
            else:
                assert poly in cell_polys

        cell_labels = cell.labels
        for label in labels:
            if ((label.layer, label.texttype) in spec) == remove:
                assert label not in cell_labels
            else:
                assert label in cell_labels

        path_results = [
            [(tag in spec) == remove for tag in zip(path.layers, path.datatypes)] for path in paths
        ]
        cell_paths = cell.paths
        for path, results in zip(paths, path_results):
            if all(results):
                assert path not in cell_paths
            else:
                assert path in cell_paths
                assert len(path.layers) == len(results) - sum(results)
                assert all((tag in spec) != remove for tag in zip(path.layers, path.datatypes))


def test_remap(tree):
    c3, c2, c1 = tree

    c2.remap({(0, 0): (10, 10), (12, 0): (0, 12)})
    assert c1.labels[0].layer == 11
    assert c1.labels[0].texttype == 0
    assert c1.polygons[0].layer == 0
    assert c1.polygons[0].datatype == 0
    assert c2.labels[0].layer == 0
    assert c2.labels[0].texttype == 12
    assert c2.polygons[0].layer == 1
    assert c2.polygons[0].datatype == 1

    c1.remap({(0, 0): (10, 10), (12, 0): (0, 12)})
    assert c1.labels[0].layer == 11
    assert c1.labels[0].texttype == 0
    assert c1.polygons[0].layer == 10
    assert c1.polygons[0].datatype == 10


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


def test_bb_label_repetition():
    lbl = gdstk.Label("label", (1, 2))
    lbl.repetition = gdstk.Repetition(x_offsets=(1, 3, -2))
    c_lbl = gdstk.Cell("A")
    c_lbl.add(lbl)
    assert_close(c_lbl.bounding_box(), ((-1, 2), (4, 2)))
    ref = gdstk.Reference(c_lbl)
    ref.repetition = gdstk.Repetition(y_offsets=(-1, 2, -4))
    c_ref = gdstk.Cell("B")
    c_ref.add(ref)
    assert_close(c_ref.bounding_box(), ((-1, -2), (4, 4)))
    ref.rotation = numpy.pi / 4
    a = (-1 + 2j) * numpy.exp(0.25j * numpy.pi)
    b = (4 + 2j) * numpy.exp(0.25j * numpy.pi)
    assert_close(c_ref.bounding_box(), ((a.real, a.imag - 4), (b.real, b.imag + 2)))


def test_bb_polygon_repetition():
    pol = gdstk.rectangle((0, 0), (1, 1))
    pol.repetition = gdstk.Repetition(x_offsets=(1, 3, -2))
    c_pol = gdstk.Cell("C")
    c_pol.add(pol)
    assert_close(c_pol.bounding_box(), ((-2, 0), (4, 1)))
    ref = gdstk.Reference(c_pol)
    ref.repetition = gdstk.Repetition(y_offsets=(-1, 2, -4))
    c_ref = gdstk.Cell("D")
    c_ref.add(ref)
    assert_close(c_ref.bounding_box(), ((-2, -4), (4, 3)))
    ref.rotation = numpy.pi / 4
    a = (-2 + 1j) * numpy.exp(0.25j * numpy.pi)
    b = (-2 + 0j) * numpy.exp(0.25j * numpy.pi)
    c = (4 + 0j) * numpy.exp(0.25j * numpy.pi)
    d = (4 + 1j) * numpy.exp(0.25j * numpy.pi)
    assert_close(c_ref.bounding_box(), ((a.real, b.imag - 4), (c.real, d.imag + 2)))


def test_bb_flexpath_repetition():
    pth = gdstk.FlexPath([0.5 + 0j, 0.5 + 1j], 1)
    pth.repetition = gdstk.Repetition(x_offsets=(1, 3, -2))
    c_pth = gdstk.Cell("E")
    c_pth.add(pth)
    assert_close(c_pth.bounding_box(), ((-2, 0), (4, 1)))
    ref = gdstk.Reference(c_pth)
    ref.repetition = gdstk.Repetition(y_offsets=(-1, 2, -4))
    c_ref = gdstk.Cell("F")
    c_ref.add(ref)
    assert_close(c_ref.bounding_box(), ((-2, -4), (4, 3)))
    ref.rotation = numpy.pi / 4
    a = (-2 + 1j) * numpy.exp(0.25j * numpy.pi)
    b = (-2 + 0j) * numpy.exp(0.25j * numpy.pi)
    c = (4 + 0j) * numpy.exp(0.25j * numpy.pi)
    d = (4 + 1j) * numpy.exp(0.25j * numpy.pi)
    assert_close(c_ref.bounding_box(), ((a.real, b.imag - 4), (c.real, d.imag + 2)))


def test_bb_robustpath_repetition():
    pth = gdstk.RobustPath(0.5j, 1).segment((1, 0.5))
    pth.repetition = gdstk.Repetition(x_offsets=(1, 3, -2))
    c_pth = gdstk.Cell("G")
    c_pth.add(pth)
    assert_close(c_pth.bounding_box(), ((-2, 0), (4, 1)))
    ref = gdstk.Reference(c_pth)
    ref.repetition = gdstk.Repetition(y_offsets=(-1, 2, -4))
    c_ref = gdstk.Cell("H")
    c_ref.add(ref)
    assert_close(c_ref.bounding_box(), ((-2, -4), (4, 3)))
    ref.rotation = numpy.pi / 4
    a = (-2 + 1j) * numpy.exp(0.25j * numpy.pi)
    b = (-2 + 0j) * numpy.exp(0.25j * numpy.pi)
    c = (4 + 0j) * numpy.exp(0.25j * numpy.pi)
    d = (4 + 1j) * numpy.exp(0.25j * numpy.pi)
    assert_close(c_ref.bounding_box(), ((a.real, b.imag - 4), (c.real, d.imag + 2)))


def test_get_polygons_depth(tree):
    c3, c2, c1 = tree
    polys = c3.get_polygons()
    assert len(polys) == 12
    polys = c3.get_polygons(depth=0)
    assert len(polys) == 0
    polys = c3.get_polygons(depth=1)
    assert len(polys) == 6


def test_get_polygons_filter(tree):
    c3, c2, c1 = tree
    with pytest.raises(ValueError):
        _ = c3.get_polygons(layer=3)
    with pytest.raises(ValueError):
        _ = c3.get_polygons(datatype=3)
    polys = c3.get_polygons(layer=0, datatype=0)
    assert len(polys) == 6
    polys = c3.get_polygons(layer=1, datatype=1)
    assert len(polys) == 6
    polys = c3.get_polygons(layer=0, datatype=1)
    assert len(polys) == 0
    polys = c3.get_polygons(layer=1, datatype=0)
    assert len(polys) == 0


def test_get_paths(tree):
    c3, c2, c1 = tree
    c1.add(gdstk.FlexPath([(0, 0), (1, 1)], [0.1, 0.1], layer=[0, 1], datatype=[2, 3]))
    c2.add(gdstk.RobustPath((0, 0), [0.1, 0.1], layer=[0, 1], datatype=[2, 3]))
    paths = c3.get_paths()
    assert len(paths) == 12
    assert paths[0].num_paths == 2
    assert all(p is not None for p in paths)
    paths = c3.get_paths(depth=1)
    assert len(paths) == 6
    assert paths[0].num_paths == 2
    assert all(p is not None for p in paths)
    paths = c3.get_paths(depth=1, layer=1, datatype=3)
    assert len(paths) == 6
    assert paths[0].num_paths == 1
    assert all(p is not None for p in paths)


def test_get_labels(tree):
    c3, c2, c1 = tree
    labels = c3.get_labels()
    assert len(labels) == 12
    labels = c3.get_labels(depth=1)
    assert len(labels) == 6
    labels = c3.get_labels(depth=1, layer=11, texttype=0)
    assert len(labels) == 0
    labels = c3.get_labels(depth=2, layer=11, texttype=0)
    assert len(labels) == 6
