#!/usr/bin/env python

# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import pytest
import gdstk


def test_layernames_roundtrip(tmp_path):
    """Test OASIS LAYERNAME record read/write round-trip.
    
    Validates that layer_names dict property correctly serializes to and
    deserializes from OASIS LAYERNAME records (types 11 and 12).
    """
    # Create a library with layer names
    lib = gdstk.Library('test_layernames')
    cell = lib.new_cell('TOP')
    cell.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    cell.add(gdstk.rectangle((5, 5), (6, 6), layer=2, datatype=0))
    
    # Set layer names
    expected_names = {(1, 0): ('METAL1', 'METAL1_TEXT'), (2, 0): ('VIA1', 'drawing')}
    lib.layer_names = expected_names
    assert lib.layer_names == expected_names, "Failed to set layer_names"
    
    # Write to OASIS and read back
    oas_file = tmp_path / "test_layernames.oas"
    lib.write_oas(str(oas_file))
    assert oas_file.exists(), "OASIS file not written"
    
    lib2 = gdstk.read_oas(str(oas_file))
    assert lib2.layer_names == expected_names, f"Round-trip failed: {lib2.layer_names} != {expected_names}"


def test_layernames_empty():
    """Test that empty layer_names dict is correctly handled."""
    lib = gdstk.Library('empty_test')
    cell = lib.new_cell('TOP')
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    
    # Default should be empty dict
    assert lib.layer_names == {}, "Default layer_names should be empty dict"
    
    # Setting empty dict should work
    lib.layer_names = {}
    assert lib.layer_names == {}, "Failed to set empty layer_names"


def test_layernames_single_name():
    """Test layer_names with only data name (no text name)."""
    lib = gdstk.Library('single_test')
    cell = lib.new_cell('TOP')
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    
    # Set only data name, empty text name
    lib.layer_names = {(1, 0): ('DATA_LAYER', '')}
    assert lib.layer_names == {(1, 0): ('DATA_LAYER', '')}, "Failed to set single name"


def test_layernames_multiple_layers():
    """Test layer_names with multiple layer/datatype combinations."""
    lib = gdstk.Library('multi_test')
    lib.new_cell('TOP')
    
    names = {
        (1, 0): ('METAL1', 'M1_TEXT'),
        (2, 0): ('VIA1', ''),
        (3, 1): ('METAL2', 'M2_TEXT'),
        (10, 5): ('CUSTOM', 'CUSTOM_TXT'),
    }
    lib.layer_names = names
    assert lib.layer_names == names, "Failed to set multiple layer names"
