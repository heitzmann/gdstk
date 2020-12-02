# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

from tutorial_images import draw
import pathlib
import numpy
import gdstk
import pcell


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute()

    # Check library units. In this case it is using the default units.
    units = gdstk.gds_units(path / "photonics.gds")
    print(f"Using unit = {units[0]}, precision = {units[1]}")

    # Load the library as a dictionary of RawCell
    pdk = gdstk.read_rawcells(path / "photonics.gds")

    # Cell holding a single device (MZI)
    dev_cell = gdstk.Cell("Device")
    dev_cell.add(gdstk.Reference(pdk["MZI"], (-40, 0)))

    # Create a grating coupler using the function imported from the
    # pcell module (pcell.py) created earlier.
    grating = pcell.grating(0.62, layer=2)
    # Add 4 grating couplers to the device cell: one for each port.
    dev_cell.add(
        gdstk.Reference(
            grating, (-200, -150), rotation=numpy.pi / 2, columns=2, spacing=(300, 0)
        ),
        gdstk.Reference(
            grating, (200, 150), rotation=-numpy.pi / 2, columns=2, spacing=(300, 0)
        ),
    )

    # Create a waveguide connecting a grating to a MZI port.
    waveguide = gdstk.FlexPath((-220, -150), 20, bend_radius=15, layer=1)
    # Grating background
    waveguide.segment((20, 0), relative=True)
    # Linear taper
    waveguide.segment((-100, -150), 0.5)
    # Connection to MZI
    waveguide.segment([(-70, -150), (-70, -1), (-40, -1)])
    # Since the device is symmetrical, we can create a cell with the
    # waveguide geometry and reuse it for all 4 ports.
    wg_cell = gdstk.Cell("Waveguide")
    wg_cell.add(waveguide)

    dev_cell.add(gdstk.Reference(wg_cell))
    dev_cell.add(gdstk.Reference(wg_cell, x_reflection=True))
    dev_cell.add(gdstk.Reference(wg_cell, rotation=numpy.pi))
    dev_cell.add(gdstk.Reference(wg_cell, rotation=numpy.pi, x_reflection=True))

    # Main cell with 2 devices and lithography alignment marks
    main = gdstk.Cell("Main")
    main.add(
        gdstk.Reference(dev_cell, (250, 250)),
        gdstk.Reference(dev_cell, (250, 750)),
        gdstk.Reference(pdk["Alignment Mark"], columns=2, rows=3, spacing=(500, 500)),
    )

    lib = gdstk.Library()
    lib.add(main, *main.dependencies(True))
    lib.write_gds(path / "layout.gds")

    pdk_lib = gdstk.read_gds(path / "photonics.gds")
    pdk = {c.name: c for c in pdk_lib.cells}
    for cell in [dev_cell, main]:
        for x in cell.references:
            if isinstance(x.cell, gdstk.RawCell):
                ref = gdstk.Reference(pdk[x.cell.name], x.origin)
                ref.repetition = x.repetition
                cell.remove(x)
                cell.add(ref)
    main.name = "layout"
    draw(main, path / "how-tos")
