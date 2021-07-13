#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pytest
import gdstk


def test_inside():
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
