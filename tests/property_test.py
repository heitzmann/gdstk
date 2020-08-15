#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020-2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import gdstk


def test_properties():
    for obj in [
        gdstk.Polygon([-1 + 0j, -2j, 3 + 0j, 4j]),
        gdstk.FlexPath((0j, 1j), 0.1),
        gdstk.RobustPath(0j, 0.1),
        gdstk.Label("Label", 0j),
        gdstk.Reference("EMPTY"),
    ]:
        assert obj.get_property(12) is None
        assert obj.delete_property(12) is obj
        obj.set_property(13, "Property text")
        assert obj.get_property(12) is None
        assert obj.get_property(13) == "Property text"
        obj.delete_property(13)
        assert obj.get_property(13) is None
        obj.set_property(13, "Second text")
        obj.set_property(13, "Third text")
        assert obj.get_property(13) == "Third text"
