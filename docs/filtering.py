# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

from tutorial_images import draw
import pathlib
import gdstk


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute()

    # Load existing library
    lib = gdstk.read_gds(path / "layout.gds")

    for cell in lib.cells:
        # Remove any polygons in layer 2
        cell.filter([(2, 0)], paths=False, labels=False)
        # Remove any paths in layer 10
        cell.filter([(10, 0)], polygons=False, labels=False)

    lib.write_gds(path / "filtered-layout.gds")

    main = lib.top_level()[0]
    main.name = "filtering"
    draw(main, path / "how-tos")
