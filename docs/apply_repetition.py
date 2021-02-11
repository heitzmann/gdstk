# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
from tutorial_images import draw
import numpy
import gdstk


if __name__ == "__main__":
    # X-explicit repetition
    vline = gdstk.FlexPath([(3, 2), (3, 3.5)], 0.1, simple_path=True)
    vline.repetition = gdstk.Repetition(x_offsets=[0.2, 0.6, 1.4, 3.0])

    # Y-explicit repetition
    hline = gdstk.RobustPath((3, 2), 0.05, simple_path=True)
    hline.segment((6, 2))
    hline.repetition = gdstk.Repetition(y_offsets=[0.1, 0.3, 0.7, 1.5])

    # Create all copies
    vlines = vline.apply_repetition()
    hlines = hline.apply_repetition()

    # Include original elements for boolean operation
    vlines.append(vline)
    hlines.append(hline)

    result = gdstk.boolean(vlines, hlines, "or")

    main = gdstk.Cell("Main")
    main.add(*result)
    main.name = "apply_repetition"
    path = pathlib.Path(__file__).parent.absolute() / "how-tos"
    path.mkdir(parents=True, exist_ok=True)
    draw(main, path)
