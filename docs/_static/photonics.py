######################################################################
#                                                                    #
#  Copyright 2020-2020 Lucas Heitzmann Gabrielli.                    #
#  This file is part of gdstk, distributed under the terms of the    #
#  Boost Software License - Version 1.0.  See the accompanying       #
#  LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>            #
#                                                                    #
######################################################################

import pathlib
import numpy
import gdstk


def grating(
    period,
    number_of_teeth,
    fill_frac,
    width,
    position,
    direction,
    lda=1,
    sin_theta=0,
    focus_distance=-1,
    focus_width=-1,
    tolerance=0.001,
    layer=0,
    datatype=0,
):
    """
    Straight or focusing grating.

    Args:
        period: Grating period.
        number_of_teeth: Number of teeth in the grating.
        fill_frac: Filling fraction of the teeth (wrt period).
        width: Width of the grating.
        position: Grating position (feed point).
        direction: One of {"+x", "-x", "+y", "-y"}.
        lda: Free-space wavelength.
        sin_theta: Sine of incidence angle.
        focus_distance: Focus distance (negative for straight grating).
        focus_width: If non-negative, the focusing area is included in
          the result (usually for negative resists) and this is the
          width of the waveguide connecting to the grating.
        tolerance: Polygonal approximation tolerance.
        layer: GDSII layer number
        datatype: GDSII data type number

    Return:
        List of Polygon
    """
    if focus_distance < 0:
        path = gdstk.FlexPath(
            (
                position[0] - 0.5 * width,
                position[1] + 0.5 * (number_of_teeth - 1 + fill_frac) * period,
            ),
            [period * fill_frac] * number_of_teeth,
            period,
            layer=layer,
            datatype=datatype,
        )
        p = path.horizontal(0.5 * width).to_polygons()
    else:
        p = []
        neff = lda / float(period) + sin_theta
        qmin = int(focus_distance / float(period) + 0.5)
        c3 = neff ** 2 - sin_theta ** 2
        w = 0.5 * width
        for q in range(qmin, qmin + number_of_teeth):
            c1 = q * lda * sin_theta
            c2 = (q * lda) ** 2
            curve = lambda t: (
                width * t - w,
                (c1 + neff * numpy.sqrt(c2 - c3 * (width * t - w) ** 2)) / c3,
            )
            path = gdstk.FlexPath(
                curve(0),
                period * fill_frac,
                tolerance=tolerance,
                layer=layer,
                datatype=datatype,
            )
            path.parametric(curve, relative=False)
            p.extend(path.to_polygons())
        p0 = p[0].points
        sz = p0.shape[0] // 2
        if focus_width == 0:
            p0 = numpy.vstack((p0[sz + 1 :], p0[:1], [position]))
            p[0] = gdstk.Polygon(p0, layer=layer, datatype=datatype)
        elif focus_width > 0:
            p0 = numpy.vstack(
                (
                    p0[sz + 1 :],
                    p0[:1],
                    [
                        (position[0] + 0.5 * focus_width, position[1]),
                        (position[0] - 0.5 * focus_width, position[1]),
                    ],
                )
            )
            p[0] = gdstk.Polygon(p0, layer=layer, datatype=datatype)
    if direction == "-x":
        return [x.rotate(0.5 * numpy.pi, position) for x in p]
    elif direction == "+x":
        return [x.rotate(-0.5 * numpy.pi, position) for x in p]
    elif direction == "-y":
        return [x.rotate(numpy.pi, position) for x in p]
    else:
        return p


if __name__ == "__main__":
    # Examples
    lib = gdstk.Library()

    # Negative resist example
    width = 0.45
    bend_radius = 50.0
    ring_radius = 20.0
    taper_len = 50.0
    input_gap = 150.0
    io_gap = 500.0
    wg_gap = 20.0
    ring_gaps = [0.06 + 0.02 * i for i in range(8)]

    ring = lib.new_cell("NRing")
    ring.add(
        gdstk.ellipse(
            (ring_radius, 0), ring_radius, ring_radius - width, tolerance=0.001
        )
    )

    grat = lib.new_cell("NGrat")
    grat.add(
        *grating(
            0.626,
            28,
            0.5,
            19,
            (0, 0),
            "+y",
            1.55,
            numpy.sin(numpy.pi * 8 / 180),
            21.5,
            width,
            tolerance=0.001,
        )
    )

    taper = lib.new_cell("NTaper")
    taper.add(gdstk.FlexPath((0, 0), 0.12).vertical(taper_len, width=width))

    c = lib.new_cell("Negative")
    for i, gap in enumerate(ring_gaps):
        path = gdstk.FlexPath(
            (input_gap * i, taper_len),
            width=width,
            bend_radius=bend_radius,
            gdsii_path=True,
        )
        path.segment((0, 600 - wg_gap * i), relative=True)
        path.segment((io_gap, 0), relative=True)
        path.segment((0, 300 + wg_gap * i), relative=True)
        c.add(path)
        c.add(gdstk.Reference(ring, (input_gap * i + width / 2 + gap, 300)))
    c.add(
        gdstk.Reference(taper, (0, 0), columns=len(ring_gaps), spacing=(input_gap, 0))
    )
    c.add(
        gdstk.Reference(
            grat,
            (io_gap, 900 + taper_len),
            columns=len(ring_gaps),
            spacing=(input_gap, 0),
        )
    )

    # Positive resist example
    width = 0.45
    ring_radius = 20.0
    big_margin = 10.0
    small_margin = 5.0
    taper_len = 50.0
    bus_len = 400.0
    input_gap = 150.0
    io_gap = 500.0
    wg_gap = 20.0
    ring_gaps = [0.06 + 0.02 * i for i in range(8)]

    ring_bus = gdstk.FlexPath(
        [(0, taper_len), (0, taper_len + bus_len)],
        [small_margin] * 2,
        small_margin + width,
    )

    p = gdstk.FlexPath((0, 0), [small_margin] * 2, small_margin + width)
    p.segment((0, 21.5), offset=small_margin + 19)
    grat = lib.new_cell("PGrat")
    grat.add(p)
    grat.add(
        *grating(
            0.626,
            28,
            0.5,
            19,
            (0, 0),
            "+y",
            1.55,
            numpy.sin(numpy.pi * 8 / 180),
            21.5,
            tolerance=0.001,
        )
    )

    taper = lib.new_cell("PTaper")
    poly = gdstk.Polygon([(0.12 / 2, 0), (0.12 / 2 + big_margin, 0),
        (width / 2 + small_margin, taper_len), (width/2, taper_len)])
    taper.add(poly)
    poly = poly.copy().mirror((0, 1))
    taper.add(poly)

    c = lib.new_cell("Positive")
    for i, gap in enumerate(ring_gaps):
        path = gdstk.FlexPath(
            [(input_gap * i, taper_len + bus_len)],
            width=[small_margin] * 2,
            offset=small_margin + width,
            bend_radius=bend_radius,
            gdsii_path=True,
        )
        path.segment((0, 600 - bus_len - wg_gap * i), relative=True)
        path.segment((io_gap, 0), relative=True)
        path.segment((0, 300 + wg_gap * i), relative=True)
        c.add(path)
        dx = width / 2 + gap
        ring_margin = gdstk.rectangle(
            (dx, 300 - ring_radius - big_margin),
            (dx + 2 * ring_radius + big_margin, 300 + ring_radius + big_margin),
        )
        ring_hole = gdstk.ellipse(
            (dx + ring_radius, 300), ring_radius, ring_radius - width, tolerance=0.001
        )
        polys = gdstk.boolean(
            gdstk.boolean(ring_bus, ring_margin, "or", precision=1e-4),
            ring_hole,
            "not",
            precision=1e-4,
        )
        c.add(*[p.translate(input_gap * i, 0) for p in polys])
    c.add(
        gdstk.Reference(taper, (0, 0), columns=len(ring_gaps), spacing=(input_gap, 0))
    )
    c.add(
        gdstk.Reference(
            grat,
            (io_gap, 900 + taper_len),
            columns=len(ring_gaps),
            spacing=(input_gap, 0),
        )
    )

    # Save to a gds file
    lib.write_gds(pathlib.Path(__file__).parent / "photonics.gds")
