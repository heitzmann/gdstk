# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import gdstk


def make_first_lib(filename):
    lib = gdstk.Library("First")
    main = lib.new_cell("Main")
    main.add(*gdstk.text("First library", 10, (0, 0)))
    ref1 = lib.new_cell("Square")
    ref1.add(gdstk.rectangle((-15, 0), (-5, 10)))
    main.add(gdstk.Reference(ref1))
    ref2 = lib.new_cell("Circle")
    ref2.add(gdstk.ellipse((0, 0), 4))
    ref1.add(gdstk.Reference(ref2, (-10, 5)))
    lib.write_gds(filename)


def make_second_lib(filename):
    lib = gdstk.Library("Second")
    main = lib.new_cell("Main")
    main.add(*gdstk.text("Second library", 10, (0, 0)))
    ref = lib.new_cell("Circle")
    ref.add(gdstk.ellipse((-10, 5), 5))
    main.add(gdstk.Reference(ref))
    lib.write_gds(filename)


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute()

    # First we create the two libraries we'll be merging
    make_first_lib(path / "lib1.gds")
    make_second_lib(path / "lib2.gds")

    # Now we load the existing libraries
    lib1 = gdstk.read_gds(path / "lib1.gds")
    lib2 = gdstk.read_gds(path / "lib2.gds")

    # We add all cells from the second library to the first
    lib1_cell_names = {c.name for c in lib1.cells}
    for cell in lib2.cells:
        # We must check that all names are unique within the merged library
        if cell.name in lib1_cell_names:
            cell.name += "-lib2"
            assert cell.name not in lib1_cell_names
        # Now we add the cell and update the set of names
        lib1.add(cell)
        lib1_cell_names.add(cell.name)

    lib1.write_gds(path / "merging.gds")
