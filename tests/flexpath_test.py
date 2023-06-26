#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

from copy import deepcopy
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
    assert path.tolerance == 1e-2
    assert path.simple_path == False
    assert path.scale_width == True
    assert path.joins == ("natural",)
    assert path.ends == ("flush",)
    assert path.bend_radius == (0,)
    assert path.bend_function == (None,)
    path.set_layers(1)
    path.set_datatypes(2)
    path.tolerance = 1e-3
    path.simple_path = True
    path.set_joins("round")
    path.set_ends((1, 2))
    path.set_bend_radius(0.1)
    path.set_bend_function(None)
    assert path.layers == (1,)
    assert path.datatypes == (2,)
    assert path.tolerance == 1e-3
    assert path.simple_path == True
    assert path.joins == ("round",)
    assert path.ends == ((1, 2),)
    assert path.bend_radius == (0.1,)
    assert path.bend_function == (None,)

    path = gdstk.FlexPath(0j, [2, 2], layer=3, datatype=[4, 5])
    assert path.layers == (3, 3)
    assert path.datatypes == (4, 5)
    assert path.num_paths == 2
    assert path.size == 1
    path.set_layers(5, 4)
    path.set_datatypes(3, 2)
    assert path.layers == (5, 4)
    assert path.datatypes == (3, 2)

    path = gdstk.FlexPath((0j, (1, 1)), 2, [-1, 1])
    assert path.layers == (0, 0)
    assert path.datatypes == (0, 0)
    assert path.num_paths == 2

    path = gdstk.FlexPath((1j, -1j), [2, 2], [-1, 1], layer=3, datatype=[4, 5])
    assert path.layers == (3, 3)
    assert path.datatypes == (4, 5)
    assert path.num_paths == 2
    assert path.bend_function == (None, None)
    dummy = lambda *args: (0j,)
    path.set_bend_function(dummy, None)
    assert path.bend_function == (dummy, None)


def test_transforms(proof_cells):
    for scale_width in [True, False]:
        path0 = gdstk.FlexPath(
            (
                0j,
                1j,
                0.5 + 1j,
            ),
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
            proof_cells[f"FlexPath: scale_width {scale_width}"].polygons, paths
        )


def test_points():
    path = gdstk.FlexPath((0j, 10j), 2).horizontal(10, 2, 1).vertical(0, 1, [1])
    numpy.testing.assert_array_equal(path.spine(), [[0, 0], [0, 10], [10, 10], [10, 0]])
    numpy.testing.assert_array_equal(path.widths(), [[2], [2], [2], [1]])
    numpy.testing.assert_array_equal(path.offsets(), [[0], [0], [0], [1]])
    path_spines = path.path_spines()
    assert len(path_spines) == 1
    numpy.testing.assert_array_equal(
        path_spines[0], [[0, 0], [0, 10], [10, 10], [11, 0]]
    )

    path = (
        gdstk.FlexPath((0j, 10j), [2, 2], 2)
        .horizontal(10, 2, 1)
        .vertical(0, 1, [-2, 1])
    )
    numpy.testing.assert_array_equal(path.spine(), [[0, 0], [0, 10], [10, 10], [10, 0]])
    numpy.testing.assert_array_equal(path.widths(), [[2, 2], [2, 2], [2, 2], [1, 1]])
    numpy.testing.assert_array_equal(
        path.offsets(), [[-1, 1], [-1, 1], [-0.5, 0.5], [-2, 1]]
    )
    path_spines = path.path_spines()
    assert len(path_spines) == 2

def test_deepcopy():
    path = gdstk.FlexPath(0j, 2)
    path2 = deepcopy(path)

    assert path is not path2
    
    assert path.layers == path2.layers
    assert path.datatypes == path2.datatypes
    assert path.num_paths == path2.num_paths
    assert path.size == path2.size
    assert path.tolerance == path2.tolerance
    assert path.simple_path == path2.simple_path
    assert path.scale_width == path2.scale_width
    assert path.joins == path2.joins
    assert path.ends == path2.ends
    assert path.bend_radius == path2.bend_radius
    assert path.bend_function == path2.bend_function

    path2.set_layers(1)
    assert path.layers == (0,)
    assert path2.layers == (1,)
