#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

from datetime import datetime
import hashlib
import pytest
import numpy
import gdstk


@pytest.fixture
def tree():
    c = [gdstk.Cell("tree_" + str(i)) for i in range(8)]
    lib = gdstk.Library()
    lib.add(*c)
    c[0].add(gdstk.Reference(c[1]))
    c[0].add(gdstk.Reference(c[3]))
    c[1].add(gdstk.Reference(c[2]))
    c[1].add(gdstk.Reference(c[2], columns=2, rows=1, spacing=(0, 0)))
    c[1].add(gdstk.Reference(c[3], columns=2, rows=1, spacing=(0, 0)))
    c[4].add(gdstk.Reference(c[3]))
    c[6].add(gdstk.Reference(c[5], columns=2, rows=1, spacing=(0, 0)))
    return lib, c


def test_top_level_1(tree):
    lib, c = tree
    tl = lib.top_level()
    assert len(tl) == 4 and c[0] in tl and c[4] in tl and c[6] in tl and c[7] in tl


def test_top_level_2(tree):
    lib, c = tree
    c[7].add(gdstk.Reference(c[0]))
    c[7].add(gdstk.Reference(c[4]))
    c[7].add(gdstk.Reference(c[6]))
    assert lib.top_level() == [c[7]]


def test_top_level_3(tree):
    lib, c = tree
    c[7].add(gdstk.Reference(c[0]))
    c[3].add(gdstk.Reference(c[4]))
    c[2].add(gdstk.Reference(c[6]))
    c[1].add(gdstk.Reference(c[7]))
    assert lib.top_level() == []


def test_rw_gds(tmpdir):
    lib = gdstk.Library("lib", unit=2e-3, precision=1e-5)
    c1 = gdstk.Cell("gl_rw_gds_1")
    c1.add(gdstk.rectangle((0, -1), (1, 2), 2, 4))
    c1.add(gdstk.Label("label", (1, -1), "w", 10, 1.5, True, 5, 6))
    c2 = gdstk.Cell("gl_rw_gds_2")
    c2.add(gdstk.ellipse((0, 0), 1))
    c3 = gdstk.Cell("gl_rw_gds_3")
    c3.add(gdstk.Reference(c1, (0, 1), -90, 2, True))
    c4 = gdstk.Cell("gl_rw_gds_4")
    c4.add(
        gdstk.Reference(
            c2,
            (-1, -2),
            columns=2,
            rows=3,
            spacing=(1, 4),
            rotation=numpy.pi,
            magnification=0.5,
            x_reflection=True,
        )
    )
    lib.add(c1, c2, c3, c4)

    fname1 = str(tmpdir.join("test1.gds"))
    lib.write_gds(fname1, max_points=20)
    lib1 = gdstk.read_gds(fname1, unit=1e-3)
    assert lib1.name == "lib"
    assert len(lib1.cells) == 4
    cells = {c.name: c for c in lib1.cells}
    assert set(cells.keys()) == {
        "gl_rw_gds_1",
        "gl_rw_gds_2",
        "gl_rw_gds_3",
        "gl_rw_gds_4",
    }
    c = cells["gl_rw_gds_1"]
    assert len(c.polygons) == len(c.labels) == 1
    assert c.polygons[0].area() == 12.0
    assert c.polygons[0].layer == 2
    assert c.polygons[0].datatype == 4
    assert c.labels[0].text == "label"
    assert c.labels[0].origin[0] == 2 and c.labels[0].origin[1] == -2
    assert c.labels[0].anchor == "w"
    assert c.labels[0].rotation == 10
    assert c.labels[0].magnification == 1.5
    assert c.labels[0].x_reflection == True
    assert c.labels[0].layer == 5
    assert c.labels[0].texttype == 6

    c = cells["gl_rw_gds_2"]
    assert len(c.polygons) == 2
    assert isinstance(c.polygons[0], gdstk.Polygon) and isinstance(
        c.polygons[1], gdstk.Polygon
    )

    c = cells["gl_rw_gds_3"]
    assert len(c.references) == 1
    assert isinstance(c.references[0], gdstk.Reference)
    assert c.references[0].cell == cells["gl_rw_gds_1"]
    assert c.references[0].origin[0] == 0 and c.references[0].origin[1] == 2
    assert c.references[0].rotation == -90
    assert c.references[0].magnification == 2
    assert c.references[0].x_reflection == True

    c = cells["gl_rw_gds_4"]
    assert len(c.references) == 1
    assert isinstance(c.references[0], gdstk.Reference)
    assert c.references[0].cell == cells["gl_rw_gds_2"]
    assert c.references[0].origin[0] == -2 and c.references[0].origin[1] == -4
    assert c.references[0].rotation == numpy.pi
    assert c.references[0].magnification == 0.5
    assert c.references[0].x_reflection == True
    print("spacing", c.references[0].repetition.spacing)
    print("v1", c.references[0].repetition.v1)
    print("v2", c.references[0].repetition.v2)
    print("offsets", c.references[0].repetition.offsets)
    assert c.references[0].repetition.columns == 2
    assert c.references[0].repetition.rows == 3
    assert c.references[0].repetition.v1 == (-2.0, 0.0)
    assert c.references[0].repetition.v2 == (0.0, 8.0)


def test_replace(tree, tmpdir):
    lib, c = tree
    fname = tmpdir.join("tree.gds")
    lib.write_gds(fname)
    rc = gdstk.read_rawcells(fname)
    c3 = gdstk.Cell(c[3].name)
    c2 = rc[c[2].name]
    lib.replace(c2, c3)
    assert c[2] not in lib.cells
    assert c[3] not in lib.cells
    assert c2 in lib.cells
    assert c3 in lib.cells
    assert c[0].references[1].cell is c3
    assert c[1].references[0].cell is c2
    assert c[1].references[1].cell is c2
    assert c[1].references[2].cell is c3
    assert c[4].references[0].cell is c3


def hash_file(fname):
    with open(fname, "rb") as fin:
        md5 = hashlib.md5(fin.read()).digest()
    return md5


def test_time_changes_gds_hash(tmpdir):
    fn1 = str(tmpdir.join("nofreeze1.gds"))
    fn2 = str(tmpdir.join("nofreeze2.gds"))
    date1 = datetime(1988, 8, 28)
    date2 = datetime(2037, 12, 25)
    lib = gdstk.Library(name="speedy")
    lib.new_cell("empty")
    lib.write_gds(fn1, timestamp=date1)
    hash1 = hash_file(fn1)
    lib.write_gds(fn2, timestamp=date2)
    hash2 = hash_file(fn2)
    assert hash1 != hash2


def test_frozen_gds_has_constant_hash(tmpdir):
    fn1 = str(tmpdir.join("freeze1.gds"))
    fn2 = str(tmpdir.join("freeze2.gds"))
    frozen_date = datetime(1988, 8, 28)
    lib = gdstk.Library(name="Elsa")
    lib.new_cell("empty")
    lib.write_gds(fn1, timestamp=frozen_date)
    hash1 = hash_file(fn1)
    lib.write_gds(fn2, timestamp=frozen_date)
    hash2 = hash_file(fn2)
    assert hash1 == hash2


def test_frozen_gds_with_cell_has_constant_hash(tmpdir):
    fn1 = str(tmpdir.join("freezec1.gds"))
    fn2 = str(tmpdir.join("freezec2.gds"))
    frozen_date = datetime(1988, 8, 28)
    lib = gdstk.Library(name="Elsa")
    cell = gdstk.Cell(name="Anna")
    cell.add(gdstk.rectangle((0, 0), (100, 1000)))
    lib.add(cell)
    lib.write_gds(fn1, timestamp=frozen_date)
    hash1 = hash_file(fn1)
    lib.write_gds(fn2, timestamp=frozen_date)
    hash2 = hash_file(fn2)
    assert hash1 == hash2


def test_frozen_gds_with_cell_array_has_constant_hash(tmpdir):
    fn1 = str(tmpdir.join("freezea1.gds"))
    fn2 = str(tmpdir.join("freezea2.gds"))
    frozen_date = datetime(1988, 8, 28)
    lib = gdstk.Library(name="Elsa")
    cell = gdstk.Cell(name="Anna")
    cell.add(gdstk.rectangle((0, 0), (100, 1000)))
    cell2 = gdstk.Cell(name="Olaf")
    cell2.add(gdstk.rectangle((0, 0), (50, 100)))
    cell_array = gdstk.Reference(
        cell2, columns=5, rows=2, spacing=(60, 120), origin=(1000, 0)
    )
    cell.add(cell_array)
    lib.add(cell)
    lib.write_gds(fn1, timestamp=frozen_date)
    hash1 = hash_file(fn1)
    lib.write_gds(fn2, timestamp=frozen_date)
    hash2 = hash_file(fn2)
    assert hash1 == hash2
