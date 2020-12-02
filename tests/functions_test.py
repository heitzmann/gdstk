#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pytest
import gdstk


def test_inside():
    r = gdstk.rectangle((0, 0), (20, 10))
    pts = [
        [(1, 1), (-1, -1)],
        [(2, 2), (-2, 2), (2, -2)],
        [(5, 5), (10, 5)],
        [(-1, -1), (-2, -2)],
        [(2, 3)],
    ]
    assert gdstk.inside(pts[0], r) == (True, False)
    assert gdstk.inside(pts[1], r) == (True, False, False)
    assert gdstk.inside(pts[1], r, "any") == (True,)
    assert gdstk.inside(pts[1], r, "all") == (False,)
    assert gdstk.inside(pts[4], r) == (True,)
    assert gdstk.inside(pts, r, "any") == (True, True, True, False, True)
    assert gdstk.inside(pts, r, "all") == (False, False, True, False, True)
