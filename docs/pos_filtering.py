# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

from tutorial_images import draw
import pathlib
import gdstk


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute()

    unit = gdstk.Cell("Unit")
    unit.add(gdstk.cross((0, 0), 1, 0.2))

    main = gdstk.Cell("Main")

    # Create repeating pattern using references
    d = 2
    ref1 = gdstk.Reference(unit, columns=11, rows=6, spacing=(d, d * 3**0.5))
    ref2 = gdstk.Reference(
        unit, (d / 2, d * 3**0.5 / 2), columns=10, rows=5, spacing=(d, d * 3**0.5)
    )
    main.add(ref1, ref2)
    main.flatten()

    hole = gdstk.text("PY", 8 * d, (0.5 * d, 0), layer=1)
    for pol in main.polygons:
        if gdstk.any_inside(pol.points, hole):
            main.remove(pol)

    main.add(*hole)

    gdstk.Library().add(main).write_gds(path / "pos_filtering.gds")

    main.name = "pos_filtering"
    draw(main, path / "how-tos")
