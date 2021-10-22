#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import math
import gdstk

from conftest import assert_close


def test_label_transform():
    label = gdstk.Label("hello world", (5, 5))
    label.transform(
        magnification=2, x_reflection=True, rotation=math.pi, translation=(2, 2)
    )

    assert_close(label.magnification, 2)
    assert label.x_reflection is True
    assert_close(label.rotation, math.pi)
    assert_close(label.origin, (-8, 12))

    label.transform(0.5, True, 0, (-2, -2))
    assert_close(label.magnification, 1)
    assert label.x_reflection is False
    assert_close(label.rotation, math.pi * -1)
    assert_close(label.origin, (-6, -8))
