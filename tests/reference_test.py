#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020-2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pytest
import numpy
import gdstk

from conftest import assert_same_shape, assert_close


def test_noreference():
    name = "ca_noreference"
    ref = gdstk.Reference(name, (1, -1), numpy.pi / 2, 2.1, True)
    assert ref.cell == name
    assert ref.bounding_box() is None
    assert ref.origin == (1, -1)
    assert ref.rotation == numpy.pi / 2
    assert ref.magnification == 2.1
    assert ref.x_reflection == True


def test_empty():
    name = "ca_empty"
    c = gdstk.Cell(name)
    ref = gdstk.Reference(c, (1, -1), numpy.pi / 2, 2, True, 2, 3, (3, 2))
    assert ref.cell is c
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
        1e-12,
    )
