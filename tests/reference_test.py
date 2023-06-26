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


def test_noreference():
    name = "ca_noreference"
    ref = gdstk.Reference(name, (1, -1), numpy.pi / 2, 2.1, True)
    assert ref.cell == name
    assert ref.cell_name == name
    assert ref.bounding_box() is None
    assert ref.origin == (1, -1)
    assert ref.rotation == numpy.pi / 2
    assert ref.magnification == 2.1
    assert ref.x_reflection


def test_x_reflection():
    c = gdstk.Cell("test")
    ref = gdstk.Reference(c, x_reflection=False)
    assert not ref.x_reflection

    ref.x_reflection = True
    assert ref.x_reflection

    ref.x_reflection = False
    assert not ref.x_reflection


def test_empty():
    name = "ca_empty"
    c = gdstk.Cell(name)
    ref = gdstk.Reference(c, (1, -1), numpy.pi / 2, 2, True, 2, 3, (3, 2))
    assert ref.cell is c
    assert ref.cell_name == name
    assert ref.bounding_box() is None


def test_notempty():
    name = "ca_notempty"
    c = gdstk.Cell(name)
    ref = gdstk.Reference(c, (1, -1), numpy.pi / 2, 2, True, 2, 3, (3, 2))
    ref.origin = (0, 0)
    c.add(gdstk.rectangle((0, 0), (1, 2), 2, 3))
    assert_close(ref.bounding_box(), ((0, 0), (8, 5)))
    assert_same_shape(
        [gdstk.rectangle((0, 0), (8, 2)), gdstk.rectangle((0, 3), (8, 5))],
        gdstk.Cell("TMP").add(ref).flatten().polygons,
    )
    assert ref.cell_name == name


def test_label_bounding_box():
    c = gdstk.Cell("CELL")
    l = gdstk.Label("Label", (2, 3))
    c.add(l)
    bb = c.bounding_box()
    assert bb[0][0] == 2 and bb[0][1] == 3
    assert bb[1][0] == 2 and bb[1][1] == 3
    ref = gdstk.Reference(c, (-1, 1))
    bb = ref.bounding_box()
    assert bb[0][0] == 1 and bb[0][1] == 4
    assert bb[1][0] == 1 and bb[1][1] == 4
    ang = numpy.pi / 4
    x = ref.origin[0] + l.origin[0] * numpy.cos(ang) - l.origin[1] * numpy.sin(ang)
    y = ref.origin[1] + l.origin[0] * numpy.sin(ang) + l.origin[1] * numpy.cos(ang)
    ref.rotation = ang
    bb = ref.bounding_box()
    assert_close(bb, ((x, y), (x, y)))


def _create_cell_reference(raw=False):
    if raw:
        c = gdstk.RawCell("CELL")
    else:
        c = gdstk.Cell("CELL")
    return gdstk.Reference(c)


def _copy_and_set_properties(ref):
    ref_copy = ref.copy()
    ref_copy.set_gds_property(101, "test")
    return ref_copy


def test_copy_with_cell():
    ref_copy = _copy_and_set_properties(_create_cell_reference(raw=False))
    assert ref_copy.get_gds_property(101) == "test"
    assert ref_copy.cell.name == "CELL"
    assert ref_copy.cell_name == "CELL"


def test_copy_with_rawcell():
    ref_copy = _copy_and_set_properties(_create_cell_reference(raw=True))
    assert ref_copy.get_gds_property(101) == "test"
    assert ref_copy.cell.name == "CELL"
    assert ref_copy.cell_name == "CELL"


def test_gds_array(tmpdir):
    lib = gdstk.Library("Library")
    ref_cell = lib.new_cell("Base")
    ref_cell.add(*gdstk.text("F", 1, (0, 0)))

    for rect in (True, False):
        for cols in (1, 3):
            for rows in (1, 3):
                if cols * rows == 1:
                    continue
                for dx in (-3, 0, 3):
                    for dy in (-4, 0, 4):
                        if rect:
                            rep = gdstk.Repetition(
                                columns=cols, rows=rows, spacing=(dx, dy)
                            )
                        else:
                            rep = gdstk.Repetition(
                                columns=cols, rows=rows, v1=(dx, 0), v2=(0, dy)
                            )
                        for rot in range(-3, 4):
                            for refl in (True, False):
                                cell = lib.new_cell(
                                    f"{'RECT' if rect else 'REGL'}"
                                    f"_{cols}_{rows}_{dx}_{dy}"
                                    f"_{rot}{'_X' if refl else ''}"
                                )
                                ref = gdstk.Reference(
                                    ref_cell,
                                    (-0.5, -1),
                                    0.5 * numpy.pi * rot,
                                    x_reflection=refl,
                                )
                                ref.repetition = rep
                                cell.add(ref)

    fname = str(tmpdir.join("aref_test.gds"))
    lib.write_gds(fname)
    lib2 = gdstk.read_gds(fname)

    cell_dict = {cell.name: cell for cell in lib.cells}
    for cell in lib2.cells:
        if len(cell.references) == 0:
            continue
        assert len(cell.references) == 1
        assert cell.references[0].repetition.size > 1
        assert cell.name in cell_dict
        polygons1 = cell_dict[cell.name].get_polygons()
        polygons2 = cell.get_polygons()
        assert len(polygons1) == len(polygons2)
        assert_same_shape(polygons1, polygons2)


def test_get_polygons_depth(tree):
    c3, c2, c1 = tree
    r3 = c3.references[0]
    polys = r3.get_polygons()
    assert len(polys) == 12
    polys = r3.get_polygons(depth=0)
    assert len(polys) == 6
    polys = r3.get_polygons(depth=1)
    assert len(polys) == 12


def test_get_polygons_filter(tree):
    c3, c2, c1 = tree
    r3 = c3.references[0]
    with pytest.raises(ValueError):
        _ = r3.get_polygons(layer=3)
    with pytest.raises(ValueError):
        _ = r3.get_polygons(datatype=3)
    polys = r3.get_polygons(layer=0, datatype=0)
    assert len(polys) == 6
    polys = r3.get_polygons(layer=1, datatype=1)
    assert len(polys) == 6
    polys = r3.get_polygons(layer=0, datatype=1)
    assert len(polys) == 0
    polys = r3.get_polygons(layer=1, datatype=0)
    assert len(polys) == 0


def test_get_paths(tree):
    c3, c2, c1 = tree
    r3 = c3.references[0]
    c1.add(gdstk.FlexPath([(0, 0), (1, 1)], [0.1, 0.1], layer=[0, 1], datatype=[2, 3]))
    c2.add(gdstk.RobustPath((0, 0), [0.1, 0.1], layer=[0, 1], datatype=[2, 3]))
    paths = r3.get_paths()
    assert len(paths) == 12
    assert paths[0].num_paths == 2
    assert all(p is not None for p in paths)
    paths = r3.get_paths(depth=0)
    assert len(paths) == 6
    assert paths[0].num_paths == 2
    assert all(p is not None for p in paths)
    paths = r3.get_paths(depth=0, layer=1, datatype=3)
    assert len(paths) == 6
    assert paths[0].num_paths == 1
    assert all(p is not None for p in paths)


def test_get_labels(tree):
    c3, c2, c1 = tree
    r3 = c3.references[0]
    labels = r3.get_labels()
    assert len(labels) == 12
    labels = r3.get_labels(depth=0)
    assert len(labels) == 6
    labels = r3.get_labels(depth=0, layer=11, texttype=0)
    assert len(labels) == 0
    labels = r3.get_labels(depth=1, layer=11, texttype=0)
    assert len(labels) == 6
