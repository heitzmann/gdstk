#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pytest
import gdstk


def test_inside():
    ring = gdstk.ellipse((0, 0), 1, inner_radius=0.5, tolerance=1e-3)
    circle = gdstk.ellipse((0, 0), 0.5, tolerance=1e-3)

    points = [(0, 0), (0.2, 0), (-0.1, -0.8), (0.9, 0.7), (-0.4, 0.4)]
    truth_ring = tuple(0.25 <= p[0] ** 2 + p[1] ** 2 <= 1 for p in points)
    truth_circle = tuple(p[0] ** 2 + p[1] ** 2 <= 0.25 for p in points)

    assert ring.contain(*points) == truth_ring
    assert ring.contain_all(*points) == all(truth_ring)
    assert ring.contain_any(*points) == any(truth_ring)

    assert circle.contain(*points) == truth_circle
    assert circle.contain_all(*points) == all(truth_circle)
    assert circle.contain_any(*points) == any(truth_circle)

    assert gdstk.inside(points, [ring, circle]) == tuple(
        r or c for r, c in zip(truth_ring, truth_circle)
    )
    assert gdstk.all_inside(points, [ring, circle]) == all(
        [r or c for r, c in zip(truth_ring, truth_circle)]
    )
    assert gdstk.any_inside(points, [ring, circle]) == any(
        [r or c for r, c in zip(truth_ring, truth_circle)]
    )

    polys = [gdstk.rectangle((0, 0), (10, 10)), gdstk.rectangle((10, 0), (20, 10))]
    for pts, _any, _all in (
        ([(1, 1), (-1, -1)], True, False),
        ([(2, 2), (-2, 2), (2, -2)], True, False),
        ([(5, 5), (10, 5)], True, True),
        ([(-1, -1), (-2, -2)], False, False),
        ([(2, 3)], True, True),
    ):
        assert gdstk.any_inside(pts, polys) == _any
        assert gdstk.all_inside(pts, polys) == _all
