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
        for pol in cell.polygons:
            # Remove any polygons in layer 2
            if pol.layer == 2:
                cell.remove(pol)
        for pth in cell.paths:
            # Because this is a loaded library, paths will have only 1 element.
            assert pth.num_paths == 1
            # Remove any paths in layer 10
            if pth.layers[0] == 10:
                cell.remove(pth)

    lib.write_gds(path / "filtered-layout.gds")

    main = lib.top_level()[0]
    main.name = "filtering"
    draw(main, path / "how-tos")
