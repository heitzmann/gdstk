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


def test_init():
    path = gdstk.RobustPath(0j, 2)
    assert path.layers == (0,)
    assert path.datatypes == (0,)
    assert path.num_paths == 1
    assert path.size == 0
    path.set_layers(1)
    path.set_datatypes(2)
    assert path.layers == (1,)
    assert path.datatypes == (2,)

    path = gdstk.RobustPath(0j, [2, 2], layer=3, datatype=[4, 5])
    assert path.layers == (3, 3)
    assert path.datatypes == (4, 5)
    assert path.num_paths == 2
    assert path.size == 0
    path.set_layers(5, 4)
    path.set_datatypes(3, 2)
    assert path.layers == (5, 4)
    assert path.datatypes == (3, 2)

    path = gdstk.RobustPath((1, 1), 2, [-1, 1]).horizontal(10)
    assert path.layers == (0, 0)
    assert path.datatypes == (0, 0)
    assert path.num_paths == 2
    assert path.size == 1

    path = gdstk.RobustPath(-1j, [2, 2], [-1, 1], layer=3, datatype=[4, 5])
    assert path.layers == (3, 3)
    assert path.datatypes == (4, 5)
    assert path.num_paths == 2


def test_transforms(proof_cells):
    for scale_width in [True, False]:
        path0 = gdstk.RobustPath(
            0j,
            [0.1, 0.2],
            0.3,
            tolerance=1e-4,
            scale_width=scale_width,
        )
        path0.vertical(1).horizontal(0.5).turn(0.4, -numpy.pi, [0.2, 0.1]).segment(
            (-0.2, 0), relative=True
        )
        paths = [
            path0,
            path0.copy().mirror((1.5, 0)),
            path0.copy().mirror((1.5, 0), (1.5, 1)),
            path0.copy().scale(2, (3, 0)),
            path0.copy().scale(-2, (-1, 0)),
            path0.copy().rotate(numpy.pi / 2, (2, 1)).translate(0.2, -0.3),
        ]
        assert_same_shape(
            proof_cells[f"RobustPath: scale_width {scale_width}"].polygons, paths
        )


@pytest.fixture(scope="session")
def robust_path():
    path = gdstk.RobustPath((-1, 0), [0.1, 0], 0, tolerance=1e-4)
    path.segment((1, 1), width=[0.2, 0.1], offset=0.2)
    path.arc(
        2, 0.5 * numpy.pi, 0, width=(0.15, "constant"), offset=[-0.1, (0, "smooth")]
    )
    return path


def test_position(robust_path):
    assert_close(robust_path.position(-0.1), (-1, 0))
    assert_close(robust_path.position(0), (-1, 0))
    assert_close(robust_path.position(0.5), (0, 0.5))
    assert_close(robust_path.position(1), (1, 1))
    assert_close(robust_path.position(1.5), (1 + 2 ** 0.5, -1 + 2 ** 0.5))
    assert_close(robust_path.position(2), (3, -1))
    assert_close(robust_path.position(2.1), (3, -1))


def test_gradient(robust_path):
    assert_close(robust_path.gradient(-0.1), (2, 1))
    assert_close(robust_path.gradient(0), (2, 1))
    assert_close(robust_path.gradient(0.5), (2, 1))
    assert_close(robust_path.gradient(1), (2, 1))
    assert_close(robust_path.gradient(1, False), (numpy.pi, 0))
    assert_close(robust_path.gradient(1.5), (numpy.pi / 2 ** 0.5, -numpy.pi / 2 ** 0.5))
    assert_close(robust_path.gradient(2), (0, -numpy.pi))
    assert_close(robust_path.gradient(2, False), (0, -numpy.pi))
    assert_close(robust_path.gradient(2.1), (0, -numpy.pi))


def test_widths(robust_path):
    assert_close(robust_path.widths(-0.1), (0.1, 0))
    assert_close(robust_path.widths(0), (0.1, 0))
    assert_close(robust_path.widths(0.5), (0.15, 0.05))
    assert_close(robust_path.widths(1), (0.2, 0.1))
    assert_close(robust_path.widths(1, False), (0.15, 0.15))
    assert_close(robust_path.widths(1.5), (0.15, 0.15))
    assert_close(robust_path.widths(2), (0.15, 0.15))
    assert_close(robust_path.widths(2, False), (0.15, 0.15))
    assert_close(robust_path.widths(2.1), (0.15, 0.15))


def test_offsets(robust_path):
    assert_close(robust_path.offsets(-0.1), (0, 0))
    assert_close(robust_path.offsets(0), (0, 0))
    assert_close(robust_path.offsets(0.5), (-0.05, 0.05))
    assert_close(robust_path.offsets(1), (-0.1, 0.1))
    assert_close(robust_path.offsets(1, False), (-0.1, 0.1))
    assert_close(robust_path.offsets(1.5), (-0.1, 0.05))
    assert_close(robust_path.offsets(2), (-0.1, 0))
    assert_close(robust_path.offsets(2, False), (-0.1, 0))
    assert_close(robust_path.offsets(2.1), (-0.1, 0))
