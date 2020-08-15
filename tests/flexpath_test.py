#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020-2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pytest
import numpy
import gdstk

from conftest import assert_same_shape


def test_init():
    path = gdstk.FlexPath(0j, 2)
    assert path.layers == (0,)
    assert path.datatypes == (0,)
    assert path.num_paths == 1
    assert path.size == 1

    path = gdstk.FlexPath(0j, [2, 2], layer=3, datatype=[4, 5])
    assert path.layers == (3, 3)
    assert path.datatypes == (4, 5)
    assert path.num_paths == 2
    assert path.size == 1

    path = gdstk.FlexPath((0j, (1, 1)), 2, [-1, 1])
    assert path.layers == (0, 0)
    assert path.datatypes == (0, 0)
    assert path.num_paths == 2

    path = gdstk.FlexPath((1j, -1j), [2, 2], [-1, 1], layer=3, datatype=[4, 5])
    assert path.layers == (3, 3)
    assert path.datatypes == (4, 5)
    assert path.num_paths == 2


def test_transforms(proof_cells):
    for scale_width in [True, False]:
        path0 = gdstk.FlexPath(
            (0j, 1j, 0.5 + 1j,),
            [0.1, 0.2],
            0.3,
            tolerance=1e-4,
            scale_width=scale_width,
        )
        path0.turn(0.4, -numpy.pi, [0.2, 0.1]).segment((-0.2, 0), relative=True)
        paths = [
            path0,
            path0.copy().mirror((1.5, 0)),
            path0.copy().mirror((1.5, 0), (1.5, 1)),
            path0.copy().scale(2, (3, 0)),
            path0.copy().scale(-2, (-1, 0)),
            path0.copy().rotate(numpy.pi / 2, (2, 1)).translate(0.2, -0.3),
        ]
        assert_same_shape(
            proof_cells[f"FlexPath: scale_width {scale_width}"].polygons, paths, 3e-3
        )
