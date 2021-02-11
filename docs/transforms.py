# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
from tutorial_images import draw
import numpy
import gdstk


if __name__ == "__main__":
    n = 3  # Number of unit cells around defect
    d = 0.2  # Unit cell size
    r = 0.05  # Circle radius
    s = 1.5  # Scaling factor

    # Create a simple unit cell
    unit_cell = gdstk.Cell("Unit Cell")
    unit_cell.add(gdstk.ellipse((0, 0), r, tolerance=1e-3))

    # Build a resonator from a unit cell grid with a defect inside
    ressonator = gdstk.Cell("Resonator")
    patches = [
        gdstk.Reference(
            unit_cell, (-n * d, -n * d), columns=2 * n + 1, rows=n, spacing=(d, d)
        ),
        gdstk.Reference(
            unit_cell, (-n * d, d), columns=2 * n + 1, rows=n, spacing=(d, d)
        ),
        gdstk.Reference(unit_cell, (-n * d, 0), columns=n, rows=1, spacing=(d, d)),
        gdstk.Reference(unit_cell, (d, 0), columns=n, rows=1, spacing=(d, d)),
    ]
    # Defect
    rect = gdstk.rectangle((-r / 2, -r / 2), (r / 2, r / 2))
    # Path for illustration
    path = gdstk.FlexPath(
        [(-n * d, 0), (n * d, 0)],
        r,
        ends=(r, r),
        simple_path=True,
        scale_width=False,
        layer=1,
    )
    ressonator.add(rect, path, *patches)

    # Main output cell with the original resonator,…
    main = gdstk.Cell("Main")
    main.add(gdstk.Reference(ressonator))
    main.add(*gdstk.text("Original", d, ((n + 1) * d, -d / 2)))

    # … a copy created by scaling a reference to the original resonator,…
    main.add(gdstk.Reference(ressonator, (0, (1 + s) * (n + 1) * d), magnification=s))
    main.add(
        *gdstk.text("Reference\nscaling", d, (s * (n + 1) * d, (1 + s) * (n + 1) * d))
    )

    # … and another copy created by copying and scaling the Cell itself.
    ressonator_copy = ressonator.copy("Resonator Copy", magnification=s)
    main.add(gdstk.Reference(ressonator_copy, (0, (1 + 3 * s) * (n + 1) * d)))
    main.add(
        *gdstk.text(
            "Cell copy\nscaling", d, (s * (n + 1) * d, (1 + 3 * s) * (n + 1) * d)
        )
    )
    main.name = "transforms"
    path = pathlib.Path(__file__).parent.absolute() / "how-tos"
    path.mkdir(parents=True, exist_ok=True)
    draw(main, path)
