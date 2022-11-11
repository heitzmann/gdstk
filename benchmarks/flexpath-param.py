#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import numpy
import gdspy
import gdstk


def bench_gdspy(output=None):
    def broken(p0, v0, p1, v1, p2, w):
        den = v1[1] * v0[0] - v1[0] * v0[1]
        lim = 1e-12 * (v0[0] ** 2 + v0[1] ** 2) * (v1[0] ** 2 + v1[1] ** 2)
        if den**2 < lim:
            u0 = u1 = 0
            p = 0.5 * (p0 + p1)
        else:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            u0 = (v1[1] * dx - v1[0] * dy) / den
            u1 = (v0[1] * dx - v0[0] * dy) / den
            p = 0.5 * (p0 + v0 * u0 + p1 + v1 * u1)
        if u0 <= 0 and u1 >= 0:
            return [p]
        return [p0, p2, p1]

    def pointy(p0, v0, p1, v1):
        r = 0.5 * numpy.sqrt(numpy.sum((p0 - p1) ** 2))
        v0 /= numpy.sqrt(numpy.sum(v0**2))
        v1 /= numpy.sqrt(numpy.sum(v1**2))
        return [p0, 0.5 * (p0 + p1) + 0.5 * (v0 - v1) * r, p1]

    sp0 = gdspy.FlexPath(
        [(0, 0), (0, 1)],
        [0.1, 0.3, 0.5],
        offset=[-0.2, 0, 0.4],
        layer=[0, 1, 2],
        corners=broken,
        ends=pointy,
        datatype=3,
    )
    sp0.segment((3, 3), offset=[-0.5, -0.1, 0.5])
    sp0.segment((4, 1), width=[0.2, 0.2, 0.2], offset=[-0.2, 0, 0.2])
    sp0.segment((0, -1), relative=True)

    def spiral(u):
        r = 2 - u
        theta = 5 * u * numpy.pi
        return (r * numpy.cos(theta) - 2, r * numpy.sin(theta))

    sp1 = gdspy.FlexPath([(2, 6)], 0.2, layer=2, max_points=8190)
    sp1.parametric(spiral)

    if output:
        cell = gdspy.Cell("MAIN", exclude_from_current=True)
        cell.add([sp0, sp1])
        cell.write_svg(output, 50)


def bench_gdstk(output=None):
    def broken(p0, v0, p1, v1, p2, w):
        p0 = numpy.array(p0)
        v0 = numpy.array(v0)
        p1 = numpy.array(p1)
        v1 = numpy.array(v1)
        p2 = numpy.array(p2)
        den = v1[1] * v0[0] - v1[0] * v0[1]
        lim = 1e-12 * (v0[0] ** 2 + v0[1] ** 2) * (v1[0] ** 2 + v1[1] ** 2)
        if den**2 < lim:
            u0 = u1 = 0
            p = 0.5 * (p0 + p1)
        else:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            u0 = (v1[1] * dx - v1[0] * dy) / den
            u1 = (v0[1] * dx - v0[0] * dy) / den
            p = 0.5 * (p0 + v0 * u0 + p1 + v1 * u1)
        if u0 <= 0 and u1 >= 0:
            return [p]
        return [p0, p2, p1]

    def pointy(p0, v0, p1, v1):
        p0 = numpy.array(p0)
        v0 = numpy.array(v0)
        p1 = numpy.array(p1)
        v1 = numpy.array(v1)
        r = 0.5 * numpy.sqrt(numpy.sum((p0 - p1) ** 2))
        return [p0, 0.5 * (p0 + p1) + 0.5 * (v0 - v1) * r, p1]

    sp0 = gdstk.FlexPath(
        [(0, 0), (0, 1)],
        [0.1, 0.3, 0.5],
        offset=[-0.2, 0, 0.4],
        layer=[0, 1, 2],
        joins=broken,
        ends=pointy,
        datatype=3,
    )
    sp0.segment((3, 3), offset=[-0.5, -0.1, 0.5])
    sp0.segment((4, 1), width=[0.2, 0.2, 0.2], offset=[-0.2, 0, 0.2])
    sp0.segment((0, -1), relative=True)

    def spiral(u):
        r = 2 - u
        theta = 5 * u * numpy.pi
        return (r * numpy.cos(theta) - 2, r * numpy.sin(theta))

    sp1 = gdstk.FlexPath((2, 6), 0.2, layer=2)
    sp1.parametric(spiral)

    if output:
        cell = gdstk.Cell("MAIN")
        cell.add(sp0, sp1)
        cell.write_svg(output, 50)


if __name__ == "__main__":
    bench_gdspy("/tmp/gdspy.svg")
    bench_gdstk("/tmp/gdstk.svg")
