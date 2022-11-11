#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import gdspy
import gdstk

gdspy_ring = gdspy.Round((0, 0), 1, inner_radius=0.5, tolerance=1e-3, max_points=0)
gdspy_circle = gdspy.Round((0, 0), 0.5, tolerance=1e-3, max_points=0)
gdstk_ring = gdstk.ellipse((0, 0), 1, inner_radius=0.5, tolerance=1e-3)
gdstk_circle = gdstk.ellipse((0, 0), 0.5, tolerance=1e-3)
points = [(0, 0), (0.2, 0), (-0.1, -0.8), (0.9, 0.7), (-0.4, 0.4)] * 10


def bench_gdspy():
    gdspy.inside(points, gdspy_ring)
    gdspy.inside([points], gdspy_ring, "any")
    gdspy.inside([points], gdspy_ring, "all")
    gdspy.inside(points, [gdspy_ring, gdspy_circle])
    gdspy.inside([points], [gdspy_ring, gdspy_circle], "any")
    gdspy.inside([points], [gdspy_ring, gdspy_circle], "all")


def bench_gdstk():
    gdstk_ring.contain(*points)
    gdstk_ring.contain_any(*points)
    gdstk_ring.contain_all(*points)
    gdstk.inside(points, [gdstk_ring, gdstk_circle])
    gdstk.any_inside(points, [gdstk_ring, gdstk_circle])
    gdstk.all_inside(points, [gdstk_ring, gdstk_circle])


if __name__ == "__main__":
    bench_gdspy()
    bench_gdstk()
