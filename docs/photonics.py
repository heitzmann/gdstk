# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import numpy
import gdstk


def alignment_mark(lib):
    cross = gdstk.cross((0, 0), 50, 3, layer=1)
    lib.new_cell("Alignment Mark").add(cross)


def directional_coupler(lib):
    path = gdstk.RobustPath((0, 0), [0.5, 0.5], 2, simple_path=True, layer=1)
    path.segment((0.1, 0), relative=True)
    path.segment((2.2, 0), offset=(0.6, "smooth"), relative=True)
    path.segment((0.4, 0), relative=True)
    path.segment((2.2, 0), offset=(2, "smooth"), relative=True)
    path.segment((0.1, 0), relative=True)
    lib.new_cell("Directinal Coupler").add(path)


def mach_zehnder_interferometer(lib):
    cell = lib.new_cell("MZI")
    cell.add(gdstk.Reference("Directinal Coupler", (0, 0)))
    cell.add(gdstk.Reference("Directinal Coupler", (75, 0)))

    points = numpy.array([(5, 1), (25, 1), (25, 40), (55, 40), (55, 1), (75, 1)])
    arm1 = gdstk.FlexPath(points, 0.5, bend_radius=15, simple_path=True, layer=1)
    points[:, 1] *= -1
    arm2 = gdstk.FlexPath(points, 0.5, bend_radius=15, simple_path=True, layer=1)
    points = numpy.array([(25, 20), (25, 40), (55, 40), (55, 20)])
    heater1 = gdstk.FlexPath(points, 2, bend_radius=15, simple_path=True, layer=10)
    points[:, 1] *= -1
    heater2 = gdstk.FlexPath(points, 2, bend_radius=15, simple_path=True, layer=10)
    cell.add(arm1, arm2, heater1, heater2)


if __name__ == "__main__":
    lib = gdstk.Library("Photonics")

    alignment_mark(lib)
    directional_coupler(lib)
    mach_zehnder_interferometer(lib)

    path = pathlib.Path(__file__).parent.absolute()
    lib.write_gds(path / "photonics.gds")
