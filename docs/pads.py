# Copyright 2022 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
from tutorial_images import draw
import numpy
import gdstk


if __name__ == "__main__":

    def filleted_pad(pad_radius, fillet_radius=0, tolerance=0.01):
        def _f(p0, v0, p1, v1):
            p0 = numpy.array(p0)
            v0 = numpy.array(v0)
            p1 = numpy.array(p1)
            v1 = numpy.array(v1)

            half_trace_width = 0.5 * numpy.sqrt(numpy.sum((p0 - p1) ** 2))
            a = half_trace_width + fillet_radius
            c = pad_radius + fillet_radius
            b = (c**2 - a**2) ** 0.5
            alpha = numpy.arccos(a / c)
            gamma = numpy.arctan2(v0[1], v0[0]) + 0.5 * numpy.pi

            curve = gdstk.Curve(p0 - v0 * b, tolerance=tolerance)
            if fillet_radius > 0:
                curve.arc(fillet_radius, gamma, gamma - alpha)
            curve.arc(pad_radius, gamma - numpy.pi - alpha, gamma + alpha)
            if fillet_radius > 0:
                curve.arc(fillet_radius, gamma - numpy.pi + alpha, gamma - numpy.pi)

            return curve.points()

        return _f

    main = gdstk.Cell("Main")

    # Create a bus with 4 traces
    bus = gdstk.FlexPath(
        [(0, 0), (10, 5)], [3] * 4, offset=15, joins="round", ends=filleted_pad(5, 3)
    )
    bus.segment((20, 10), offset=6)
    bus.segment([(40, 20), (40, 50), (80, 50)])
    bus.segment((100, 50), offset=12)
    main.add(bus)
    main.name = "pads"
    path = pathlib.Path(__file__).parent.absolute()
    draw(main, path / "how-tos")
