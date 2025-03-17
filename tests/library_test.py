#!/usr/bin/env python

# Copyright 2021 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import hashlib
import pathlib
from datetime import datetime
from typing import Callable, Union

import numpy
import pytest

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


def test_mapping(tree):
    empty = gdstk.Library("Empty")
    assert len(empty) == 0
    with pytest.raises(TypeError):
        _ = empty[0]
    with pytest.raises(KeyError):
        _ = empty["X"]

    lib, c = tree
    assert len(lib) == 8
    assert c[0] is lib["tree_0"]
    with pytest.raises(KeyError):
        _ = lib["X"]


@pytest.fixture
def sample_library():
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
    return lib


def test_remap(sample_library):
    sample_library.remap({(0, 0): (10, 11), (2, 4): (4, 8)})
    polygon = sample_library["gl_rw_gds_1"].polygons[0]
    assert polygon.layer == 4 and polygon.datatype == 8
    label = sample_library["gl_rw_gds_1"].labels[0]
    assert label.layer == 5 and label.texttype == 6
    sample_library.remap({(0, 0): (10, 11), (2, 4): (4, 8)})
    polygon = sample_library["gl_rw_gds_2"].polygons[0]
    assert polygon.layer == 10 and polygon.datatype == 11


def test_gds_info(tmpdir, sample_library):
    fname = str(tmpdir.join("test.gds"))
    sample_library.write_gds(fname, max_points=20)
    info = gdstk.gds_info(fname)
    assert set(info["cell_names"]) == {
        "gl_rw_gds_1",
        "gl_rw_gds_2",
        "gl_rw_gds_3",
        "gl_rw_gds_4",
    }
    assert info["layers_and_datatypes"] == {(0, 0), (2, 4)}
    assert info["layers_and_texttypes"] == {(5, 6)}
    assert info["num_polygons"] == 3
    assert info["num_paths"] == 0
    assert info["num_references"] == 2
    assert info["num_labels"] == 1
    assert info["unit"] == 2e-3
    assert info["precision"] == 1e-5


def test_rw_gds(tmpdir, sample_library):
    fname = str(tmpdir.join("test.gds"))
    sample_library.write_gds(fname, max_points=20)
    library = gdstk.read_gds(fname, unit=1e-3)

    assert library.name == "lib"
    assert len(library.cells) == 4
    cells = {c.name: c for c in library.cells}
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
    assert c.labels[0].x_reflection
    assert c.labels[0].layer == 5
    assert c.labels[0].texttype == 6

    c = cells["gl_rw_gds_2"]
    assert len(c.polygons) == 2
    assert isinstance(c.polygons[0], gdstk.Polygon) and isinstance(c.polygons[1], gdstk.Polygon)

    c = cells["gl_rw_gds_3"]
    assert len(c.references) == 1
    assert isinstance(c.references[0], gdstk.Reference)
    assert c.references[0].cell == cells["gl_rw_gds_1"]
    assert c.references[0].origin[0] == 0 and c.references[0].origin[1] == 2
    assert c.references[0].rotation == -90
    assert c.references[0].magnification == 2
    assert c.references[0].x_reflection

    c = cells["gl_rw_gds_4"]
    assert len(c.references) == 1
    assert isinstance(c.references[0], gdstk.Reference)
    assert c.references[0].cell == cells["gl_rw_gds_2"]
    assert c.references[0].origin[0] == -2 and c.references[0].origin[1] == -4
    assert c.references[0].rotation == numpy.pi
    assert c.references[0].magnification == 0.5
    assert c.references[0].x_reflection
    assert c.references[0].repetition.columns == 2
    assert c.references[0].repetition.rows == 3
    assert c.references[0].repetition.v1 == pytest.approx((-2.0, 0.0))
    assert c.references[0].repetition.v2 == pytest.approx((0.0, 8.0))


def test_rw_gds_filter(tmpdir, sample_library):
    fname = str(tmpdir.join("test.gds"))
    sample_library.write_gds(fname, max_points=20)
    library = gdstk.read_gds(fname, unit=1e-3, filter={(0, 0)})

    assert library.name == "lib"
    assert len(library.cells) == 4
    cells = {c.name: c for c in library.cells}
    assert set(cells.keys()) == {
        "gl_rw_gds_1",
        "gl_rw_gds_2",
        "gl_rw_gds_3",
        "gl_rw_gds_4",
    }
    c = cells["gl_rw_gds_1"]
    assert len(c.polygons) == 0
    assert len(c.labels) == 1
    assert c.labels[0].text == "label"
    assert c.labels[0].origin[0] == 2 and c.labels[0].origin[1] == -2
    assert c.labels[0].anchor == "w"
    assert c.labels[0].rotation == 10
    assert c.labels[0].magnification == 1.5
    assert c.labels[0].x_reflection
    assert c.labels[0].layer == 5
    assert c.labels[0].texttype == 6

    c = cells["gl_rw_gds_2"]
    assert len(c.polygons) == 2
    assert isinstance(c.polygons[0], gdstk.Polygon) and isinstance(c.polygons[1], gdstk.Polygon)

    c = cells["gl_rw_gds_3"]
    assert len(c.references) == 1
    assert isinstance(c.references[0], gdstk.Reference)
    assert c.references[0].cell == cells["gl_rw_gds_1"]
    assert c.references[0].origin[0] == 0 and c.references[0].origin[1] == 2
    assert c.references[0].rotation == -90
    assert c.references[0].magnification == 2
    assert c.references[0].x_reflection

    c = cells["gl_rw_gds_4"]
    assert len(c.references) == 1
    assert isinstance(c.references[0], gdstk.Reference)
    assert c.references[0].cell == cells["gl_rw_gds_2"]
    assert c.references[0].origin[0] == -2 and c.references[0].origin[1] == -4
    assert c.references[0].rotation == numpy.pi
    assert c.references[0].magnification == 0.5
    assert c.references[0].x_reflection
    assert c.references[0].repetition.columns == 2
    assert c.references[0].repetition.rows == 3
    assert c.references[0].repetition.v1 == pytest.approx((-2.0, 0.0))
    assert c.references[0].repetition.v2 == pytest.approx((0.0, 8.0))


def test_read_gds_missing_refs(tmpdir):
    c1 = gdstk.Cell("c1")
    c1.add(gdstk.rectangle((0, -1), (1, 2), 2, 4))

    r1 = gdstk.Reference(c1, magnification=2)
    c2 = gdstk.Cell("c2")
    c2.add(r1)

    lib = gdstk.Library()
    lib.add(c2)

    fname = str(tmpdir.join("test_missing_refs.gds"))
    lib.write_gds(fname)

    with pytest.warns(RuntimeWarning):
        lib2 = gdstk.read_gds(fname)

    assert len(lib2.cells) == 1
    assert lib2.cells[0].name == "c2"


def test_large_layer_number(tmpdir):
    c1 = gdstk.Cell("CELL")
    c1.add(gdstk.rectangle((0, 0), (1, 1), 255, 256))
    c1.add(gdstk.rectangle((0, 0), (1, 1), 50000, 50001))
    c1.add(gdstk.rectangle((0, 0), (1, 1), 60100, 60101))
    c1.add(gdstk.rectangle((0, 0), (1, 1), 2**16 - 1, 2**16))
    c1.add(gdstk.rectangle((0, 0), (1, 1), 100000, 100001))

    lib = gdstk.Library()
    lib.add(c1)

    expected = {
        (255, 256),
        (50000, 50001),
        (60100, 60101),
        (2**16 - 1, 2**16),
        (100000, 100001),
    }
    assert lib.layers_and_datatypes() == expected

    fname = str(tmpdir.join("test_large_layer_num.gds"))
    lib.write_gds(fname)

    lib2 = gdstk.read_gds(fname)
    assert lib2.layers_and_datatypes() == expected

    info = gdstk.gds_info(fname)
    assert info["layers_and_datatypes"] == expected


# def test_rw_oas_filter(tmpdir, sample_library):
#     fname = str(tmpdir.join("test.oas"))
#     sample_library.write_oas(fname)
#     library = gdstk.read_oas(fname, unit=1e-3, filter={(0, 0)})

#     assert library.name == "LIB"
#     assert len(library.cells) == 4
#     cells = {c.name: c for c in library.cells}
#     assert set(cells.keys()) == {
#         "gl_rw_gds_1",
#         "gl_rw_gds_2",
#         "gl_rw_gds_3",
#         "gl_rw_gds_4",
#     }
#     c = cells["gl_rw_gds_1"]
#     assert len(c.polygons) == 0
#     assert len(c.labels) == 1
#     assert c.labels[0].text == "label"
#     assert c.labels[0].origin[0] == 2 and c.labels[0].origin[1] == -2
#     assert c.labels[0].anchor == "sw"
#     assert c.labels[0].rotation == 0
#     assert c.labels[0].magnification == 1
#     assert c.labels[0].x_reflection == False
#     assert c.labels[0].layer == 5
#     assert c.labels[0].texttype == 6

#     c = cells["gl_rw_gds_2"]
#     assert len(c.polygons) == 1
#     assert isinstance(c.polygons[0], gdstk.Polygon)

#     c = cells["gl_rw_gds_3"]
#     assert len(c.references) == 1
#     assert isinstance(c.references[0], gdstk.Reference)
#     assert c.references[0].cell == cells["gl_rw_gds_1"]
#     assert c.references[0].origin[0] == 0 and c.references[0].origin[1] == 2
#     assert c.references[0].rotation == -90
#     assert c.references[0].magnification == 2
#     assert c.references[0].x_reflection == True

#     c = cells["gl_rw_gds_4"]
#     assert len(c.references) == 1
#     assert isinstance(c.references[0], gdstk.Reference)
#     assert c.references[0].cell == cells["gl_rw_gds_2"]
#     assert c.references[0].origin[0] == -2 and c.references[0].origin[1] == -4
#     assert c.references[0].rotation == numpy.pi
#     assert c.references[0].magnification == 0.5
#     assert c.references[0].x_reflection == True
#     assert c.references[0].repetition.columns == 2
#     assert c.references[0].repetition.rows == 3
#     assert c.references[0].repetition.v1 == (-2.0, 0.0)
#     assert c.references[0].repetition.v2 == (0.0, 8.0)


def test_rw_oas(tmpdir, sample_library):
    fname = str(tmpdir.join("test.oas"))
    sample_library.write_oas(fname)
    library = gdstk.read_oas(fname, unit=1e-3)

    assert library.name == "LIB"
    assert len(library.cells) == 4
    cells = {c.name: c for c in library.cells}
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
    assert c.labels[0].anchor == "sw"
    assert c.labels[0].rotation == 0
    assert c.labels[0].magnification == 1
    assert not c.labels[0].x_reflection
    assert c.labels[0].layer == 5
    assert c.labels[0].texttype == 6

    c = cells["gl_rw_gds_2"]
    assert len(c.polygons) == 1
    assert isinstance(c.polygons[0], gdstk.Polygon)

    c = cells["gl_rw_gds_3"]
    assert len(c.references) == 1
    assert isinstance(c.references[0], gdstk.Reference)
    assert c.references[0].cell == cells["gl_rw_gds_1"]
    assert c.references[0].origin[0] == 0 and c.references[0].origin[1] == 2
    assert c.references[0].rotation == -90
    assert c.references[0].magnification == 2
    assert c.references[0].x_reflection

    c = cells["gl_rw_gds_4"]
    assert len(c.references) == 1
    assert isinstance(c.references[0], gdstk.Reference)
    assert c.references[0].cell == cells["gl_rw_gds_2"]
    assert c.references[0].origin[0] == -2 and c.references[0].origin[1] == -4
    assert c.references[0].rotation == numpy.pi
    assert c.references[0].magnification == 0.5
    assert c.references[0].x_reflection
    assert c.references[0].repetition.columns == 2
    assert c.references[0].repetition.rows == 3
    assert c.references[0].repetition.v1 == (-2.0, 0.0)
    assert c.references[0].repetition.v2 == (0.0, 8.0)


def test_replace(tree, tmpdir):
    lib, c = tree
    fname = str(tmpdir.join("tree.gds"))
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
    cell_array = gdstk.Reference(cell2, columns=5, rows=2, spacing=(60, 120), origin=(1000, 0))
    cell.add(cell_array)
    lib.add(cell)
    lib.write_gds(fn1, timestamp=frozen_date)
    hash1 = hash_file(fn1)
    lib.write_gds(fn2, timestamp=frozen_date)
    hash2 = hash_file(fn2)
    assert hash1 == hash2


def test_layers_and_types(sample_library):
    ld = sample_library.layers_and_datatypes()
    assert ld == {(2, 4), (0, 0)}
    lt = sample_library.layers_and_texttypes()
    assert lt == {(5, 6)}


def test_rename_cell():
    c1 = gdstk.Cell("C1")
    c2 = gdstk.Cell("C2")
    c3 = gdstk.Cell("C3")
    c4 = gdstk.Cell("C4")
    c4.add(gdstk.Reference("C1"))
    c4.add(gdstk.Reference(c3))
    c3.add(gdstk.Reference("C2"))
    c2.add(gdstk.Reference("C1"))
    lib = gdstk.Library("TEST")
    lib.add(c1, c2, c3, c4)
    lib.rename_cell("C1", "c1")
    assert c1.name == "c1"
    assert c2.name == "C2"
    assert c3.name == "C3"
    assert c2.references[0].cell == "c1"
    assert c4.references[0].cell == "c1"
    assert c4.references[1].cell is c3
    lib.rename_cell(c1, "C1")
    assert c1.name == "C1"
    assert c2.name == "C2"
    assert c3.name == "C3"
    assert c2.references[0].cell == "C1"
    assert c4.references[0].cell == "C1"
    assert c4.references[1].cell is c3


# def test_replace_cell():
#     c0 = gdstk.Cell("C0")
#     c1 = gdstk.Cell("C1")
#     c2 = gdstk.Cell("C2")
#     c3 = gdstk.Cell("C3")
#     c4 = gdstk.Cell("C4")
#     c4.add(gdstk.Reference("C1"))
#     c4.add(gdstk.Reference(c3))
#     c3.add(gdstk.Reference("C2"))
#     c2.add(gdstk.Reference(c1))
#     lib = gdstk.Library("TEST")
#     lib.add(c1, c2, c3, c4)
#     lib.replace_cell(c1, c0)
#     assert c1 not in lib.cells
#     assert c0 in lib.cells
#     assert c4.references[0].cell == "c0"
#     assert c2.references[0].cell == c0


def test_roundtrip_path_ends(tmpdir: pathlib.Path):
    for path_type in gdstk.FlexPath, gdstk.RobustPath:
        path = path_type(
            (0, 0),
            0.2,
            ends=(0.1, 0.0),
            simple_path=True,
        )

        path.vertical(2)

        cell = gdstk.Cell("path_test")
        cell.add(path)

        lib = gdstk.Library("path_test")
        lib.add(cell)

        lib.write_gds(tmpdir / "path_test.gds")
        lib.write_oas(tmpdir / "path_test.oas")

        gds_path = gdstk.read_gds(tmpdir / "path_test.gds").top_level()[0].get_paths()[0]
        oas_path = gdstk.read_oas(tmpdir / "path_test.oas").top_level()[0].get_paths()[0]

        assert path.ends == gds_path.ends and path.ends == oas_path.ends, (
            f"expected: {path.ends}, gds: {gds_path.ends}, oas: {oas_path.ends}"
        )


@pytest.mark.parametrize(
    "write_f, read_f",
    (
        (gdstk.Library.write_oas, gdstk.read_oas),
        (gdstk.Library.write_gds, gdstk.read_gds),
    ),
)
def test_write_min_length_path(
    write_f: Callable[[gdstk.Library, Union[str, pathlib.Path]], None],
    read_f: Callable[Union[str, pathlib.Path], gdstk.Library],
    tmp_path: pathlib.Path,
):
    source = gdstk.read_oas(pathlib.Path(__file__).parent / "min_length_path.oas")
    assert source.cells[0].paths

    rw_path = tmp_path / "out"
    write_f(source, rw_path)
    rw = read_f(rw_path)
    assert rw.cells[0].paths

    assert numpy.array_equal(
        source.cells[0].paths[0].spine(),
        rw.cells[0].paths[0].spine(),
    )


@pytest.mark.parametrize(
    "simple_path",
    (True, False),
)
@pytest.mark.parametrize(
    "write_f",
    (gdstk.Library.write_oas, gdstk.Library.write_gds),
)
def test_empty_path_warning(
    simple_path: bool,
    write_f: Callable[[gdstk.Library, Union[str, pathlib.Path]], None],
    tmp_path: pathlib.Path,
):
    lib = gdstk.Library("test")
    tol = lib.precision / lib.unit

    # Simple paths are written to GDS & OASIS with PATH records; non-simple
    # paths are serialized as polygons.
    path = gdstk.FlexPath(
        ((0, 0), (tol / 2, 0)), width=0.01, tolerance=tol, simple_path=simple_path
    )

    cell = gdstk.Cell("top")
    cell.add(path)
    lib.add(cell)

    with pytest.warns(RuntimeWarning, match="Empty path"):
        write_f(lib, tmp_path / "out")
