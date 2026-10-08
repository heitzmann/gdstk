#!/usr/bin/env python

import os
import pathlib
import sys
from datetime import datetime
from xml.etree import ElementTree

import gdstk
import pytest


@pytest.fixture(
    params=[
        ("ascii", "layout"),
        ("ascii", "t\u00e4st"),
        ("\u6d4b\u8bd5", "layout"),
        ("nested_\u00e4", "mask_\U0001f9ea"),
        ("ascii", "te\u0308st"),
    ],
    ids=["ascii", "umlaut", "chinese-parent", "emoji", "decomposed"],
)
def file_path(tmp_path, request):
    parent, name = request.param
    directory = tmp_path / parent
    directory.mkdir()
    return directory / name


@pytest.fixture(params=[str, pathlib.Path, os.fsencode], ids=["str", "path", "bytes"])
def path_argument(request):
    return request.param


@pytest.fixture
def layout():
    library = gdstk.Library("layout", unit=2e-6, precision=1e-9)
    library.new_cell("TOP").add(gdstk.rectangle((0, 0), (2, 3), layer=5))
    return library


@pytest.mark.parametrize("writer", ["gds", "stream", "oas", "svg"])
def test_write_exact_path(file_path, path_argument, layout, writer):
    extension = {"stream": "gds"}.get(writer, writer)
    target = file_path.with_suffix("." + extension)
    argument = path_argument(target)
    if writer == "gds":
        layout.write_gds(argument)
    elif writer == "oas":
        layout.write_oas(argument, validation="crc32")
    elif writer == "svg":
        layout["TOP"].write_svg(argument)
    else:
        stream = gdstk.GdsWriter(argument, unit=layout.unit, precision=layout.precision)
        stream.write(layout["TOP"])
        stream.close()
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]
    assert target.is_file()
    if writer == "svg":
        assert ElementTree.parse(target).getroot().tag == "{http://www.w3.org/2000/svg}svg"
    else:
        read = (gdstk.read_oas if writer == "oas" else gdstk.read_gds)(argument, unit=layout.unit)
        assert read.unit == layout.unit
        assert read.precision == layout.precision
        assert read["TOP"].polygons[0].area() == 6


@pytest.mark.parametrize(
    "reader", ["read_gds", "read_rawcells", "gds_units", "gds_info", "gds_timestamp"]
)
def test_read_existing_gds_path(tmp_path, file_path, path_argument, layout, reader):
    # Move a known-good ASCII export with Python, so reading does not depend on
    # GDSTK first writing the same (possibly incorrect) path.
    seed = tmp_path / "seed.gds"
    original = datetime(2024, 1, 2, 3, 4, 5)
    layout.write_gds(seed, timestamp=original)
    target = file_path.with_suffix(".gds")
    seed.rename(target)
    argument = path_argument(target)
    if reader == "read_gds":
        assert gdstk.read_gds(argument)["TOP"].polygons[0].area() == 6
    elif reader == "read_rawcells":
        raw = gdstk.read_rawcells(argument)
        assert set(raw) == {"TOP"}
        output = tmp_path / "raw-copy.gds"
        stream = gdstk.GdsWriter(output, unit=layout.unit, precision=layout.precision)
        stream.write(raw["TOP"])
        stream.close()
        assert gdstk.read_gds(output)["TOP"].polygons[0].area() == 6
    elif reader == "gds_units":
        assert gdstk.gds_units(argument) == (layout.unit, layout.precision)
    elif reader == "gds_info":
        info = gdstk.gds_info(argument)
        assert info["cell_names"] == ["TOP"]
        assert info["num_polygons"] == 1
    else:
        assert gdstk.gds_timestamp(argument) == original


@pytest.mark.parametrize("reader", ["read_oas", "oas_precision", "oas_validate"])
def test_read_existing_oas_path(tmp_path, file_path, path_argument, layout, reader):
    seed = tmp_path / "seed.oas"
    layout.write_oas(seed, validation="crc32")
    target = file_path.with_suffix(".oas")
    seed.rename(target)
    argument = path_argument(target)
    if reader == "read_oas":
        assert gdstk.read_oas(argument, unit=layout.unit)["TOP"].polygons[0].area() == 6
    elif reader == "oas_precision":
        assert gdstk.oas_precision(argument) == layout.precision
    else:
        valid, signature = gdstk.oas_validate(argument)
        assert valid is True
        assert signature != 0


@pytest.mark.parametrize(
    "reader",
    [
        gdstk.read_gds,
        gdstk.read_rawcells,
        gdstk.gds_units,
        gdstk.gds_info,
        gdstk.gds_timestamp,
        gdstk.read_oas,
        gdstk.oas_precision,
        gdstk.oas_validate,
    ],
)
def test_missing_input_file(tmp_path, reader):
    with pytest.raises(OSError):
        reader(tmp_path / "missing")
    with pytest.raises(ValueError):
        reader(str(tmp_path / "embedded\0nul"))


@pytest.mark.parametrize("writer", ["gds", "stream", "oas", "svg"])
def test_missing_output_parent(tmp_path, layout, writer):
    argument = tmp_path / "missing" / "output"
    with pytest.raises(TypeError if writer == "stream" else OSError):
        if writer == "gds":
            layout.write_gds(argument)
        elif writer == "oas":
            layout.write_oas(argument)
        elif writer == "svg":
            layout["TOP"].write_svg(argument)
        else:
            gdstk.GdsWriter(argument)
    assert not argument.parent.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths use UTF-8")
def test_invalid_utf8_does_not_create_substituted_file(tmp_path, layout):
    # A permissive UTF-8 conversion would write to replacement-character paths.
    for suffix in (b"\xff.gds", b"\xed\xa0\x80.gds", b"\xc0\xaf.gds"):
        argument = os.fsencode(tmp_path) + b"/" + suffix
        with pytest.raises(OSError):
            layout.write_gds(argument)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ANSI path regression")
@pytest.mark.parametrize("writer", ["gds", "oas", "stream", "svg"])
def test_unicode_output_does_not_overwrite_ansi_path(tmp_path, layout, writer):
    extension = "gds" if writer == "stream" else writer
    name = "t\u00e4st." + extension
    ansi_name = name.encode("utf-8").decode("mbcs", errors="replace")
    if ansi_name == name:
        pytest.skip("ANSI codepage is already UTF-8")
    unrelated = tmp_path / ansi_name
    unrelated.write_bytes(b"unrelated existing content")
    target = tmp_path / name
    if writer == "gds":
        layout.write_gds(target)
    elif writer == "oas":
        layout.write_oas(target)
    elif writer == "svg":
        layout["TOP"].write_svg(target)
    else:
        stream = gdstk.GdsWriter(target)
        stream.write(layout["TOP"])
        stream.close()
    assert target.is_file()
    assert unrelated.read_bytes() == b"unrelated existing content"


@pytest.mark.parametrize("writer", ["gds", "oas", "stream", "svg"])
def test_invalid_output_path(tmp_path, layout, writer):
    for argument, error in [
        (tmp_path, TypeError if writer == "stream" else OSError),
        (str(tmp_path / "nul\0name"), ValueError),
    ]:
        with pytest.raises(error):
            if writer == "gds":
                layout.write_gds(argument)
            elif writer == "oas":
                layout.write_oas(argument)
            elif writer == "svg":
                layout["TOP"].write_svg(argument)
            else:
                gdstk.GdsWriter(argument)
    assert list(tmp_path.iterdir()) == []
