#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
import numpy
import gdstk


def draw(cell, path):
    bb = cell.bounding_box()
    scaling = 300 / (1.1 * (bb[1][0] - bb[0][0]))
    name = path / (cell.name + ".svg")
    cell.write_svg(
        name,
        scaling=scaling,
        background="none",
        shape_style={
            (0, 1): {"fill": "none", "stroke": "black", "stroke-dasharray": "8,8"}
        },
        label_style={(3, 2): {"stroke": "red", "fill": "none", "font-size": "32px"}},
        pad="5%",
    )
    print(f"Saving {name} (scaling {scaling})")


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute() / "tutorial"
    path.mkdir(parents=True, exist_ok=True)

    # Polygons
    # Create a polygon from a list of vertices
    points = [(0, 0), (2, 2), (2, 6), (-6, 6), (-6, -6), (-4, -4), (-4, 4), (0, 4)]
    poly = gdstk.Polygon(points)
    draw(gdstk.Cell("polygons").add(poly), path)

    # Holes
    # Manually connect the hole to the outer boundary
    cutout = gdstk.Polygon(
        [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0), (2, 2), (2, 3), (3, 3), (3, 2), (2, 2)]
    )
    draw(gdstk.Cell("holes").add(cutout), path)

    # Circles
    # Circle centered at (0, 0), with radius 2 and tolerance 0.1
    circle = gdstk.ellipse((0, 0), 2, tolerance=0.01)

    # To create an ellipse, simply pass a list with 2 radii.
    # Because the tolerance is small (resulting a large number of
    # vertices), the ellipse is fractured in 2 polygons.
    ellipse = gdstk.ellipse((4, 0), [1, 2], tolerance=1e-4)

    # Circular arc example
    arc = gdstk.ellipse(
        (2, 4),
        2,
        inner_radius=1,
        initial_angle=-0.2 * numpy.pi,
        final_angle=1.2 * numpy.pi,
        tolerance=0.01,
    )
    draw(gdstk.Cell("circles").add(circle, ellipse, arc), path)

    # Curves
    # Construct a curve made of a sequence of line segments
    c1 = gdstk.Curve((0, 0)).segment([(1, 0), (2, 1), (2, 2), (0, 2)])
    p1 = gdstk.Polygon(c1.points())

    # Construct another curve using relative coordinates
    c2 = gdstk.Curve((3, 1)).segment([(1, 0), (2, 1), (2, 2), (0, 2)], relative=True)
    p2 = gdstk.Polygon(c2.points())
    draw(gdstk.Cell("curves").add(p1, p2), path)

    # Curves 1
    # Use complex numbers to facilitate writing polar coordinates
    c3 = gdstk.Curve(2j).segment(4 * numpy.exp(1j * numpy.pi / 6), relative=True)
    # Elliptical arcs have syntax similar to gdstk.ellipse
    c3.arc((4, 2), 0.5 * numpy.pi, -0.5 * numpy.pi)
    p3 = gdstk.Polygon(c3.points())
    draw(gdstk.Cell("curves_1").add(p3), path)

    # Curves 2
    # Cubic Bezier curves can be easily created
    c4 = gdstk.Curve((0, 0), tolerance=1e-3)
    c4.cubic([(0, 1), (1, 1), (1, 0)])
    # Smooth continuation:
    c4.cubic_smooth([(1, -1), (1, 0)], relative=True)

    # Similarly for quadratic Bezier curves
    c4.quadratic([(0.5, 1), (1, 0)], relative=True)
    c4.quadratic_smooth((1, 0), relative=True)

    # Smooth interpolating curve
    c4.interpolation([(4, -1), (3, -2), (2, -1.5), (1, -2), (0, -1), (0, 0)])
    p4 = gdstk.Polygon(c4.points())
    # draw
    ref_poly = gdstk.Polygon(
        [
            0j,
            1j,
            1 + 1j,
            1 + 0j,
            1 - 1j,
            2 - 1j,
            2 + 0j,
            2.5 + 1j,
            3 + 0j,
            3.5 - 1j,
            4 + 0j,
            4 - 1j,
            3 - 2j,
            2 - 1.5j,
            1 - 2j,
            -1j,
        ],
        datatype=1,
    )
    draw(gdstk.Cell("curves_2").add(p4, ref_poly), path)

    # Transformations
    poly = gdstk.rectangle((-2, -2), (2, 2))
    poly.rotate(numpy.pi / 4)
    poly.scale(1, 0.5)
    draw(gdstk.Cell("transformations").add(poly), path)

    # Layer and Datatype
    # Layer/datatype definitions for each step in the fabrication
    ld = {
        "full etch": {"layer": 1, "datatype": 3},
        "partial etch": {"layer": 2, "datatype": 3},
        "lift-off": {"layer": 0, "datatype": 7},
    }

    p1 = gdstk.rectangle((-3, -3), (3, 3), **ld["full etch"])
    p2 = gdstk.rectangle((-5, -3), (-3, 3), **ld["partial etch"])
    p3 = gdstk.rectangle((5, -3), (3, 3), **ld["partial etch"])
    p4 = gdstk.regular_polygon((0, 0), 2, 6, **ld["lift-off"])
    draw(gdstk.Cell("layer_and_datatype").add(p1, p2, p3, p4), path)

    # References
    # Create a cell with a component that is used repeatedly
    contact = gdstk.Cell("CONTACT")
    contact.add(p1, p2, p3, p4)

    # Create a cell with the complete device
    device = gdstk.Cell("DEVICE")
    device.add(cutout)
    # Add 2 references to the component changing size and orientation
    ref1 = gdstk.Reference(contact, (3.5, 1), magnification=0.25)
    ref2 = gdstk.Reference(contact, (1, 3.5), magnification=0.25, rotation=numpy.pi / 2)
    device.add(ref1, ref2)

    # The final layout has several repetitions of the complete device
    main = gdstk.Cell("MAIN")
    main.add(gdstk.Reference(device, (0, 0), columns=3, rows=2, spacing=(6, 7)))
    # draw
    main.name = "references"
    draw(main, path)
    main.name = "MAIN"

    # Flexible Paths
    # Path defined by a sequence of points and stored as a GDSII path
    fp1 = gdstk.FlexPath(
        [(0, 0), (3, 0), (3, 2), (5, 3), (3, 4), (0, 4)], 1, simple_path=True
    )

    # Other construction methods can still be used
    fp1.interpolation([(0, 2), (2, 2), (4, 3), (5, 1)], relative=True)

    # Multiple parallel paths separated by 0.5 with different widths,
    # end caps, and joins.  Because of the join specification, they
    # cannot be stared as GDSII paths, only as polygons.
    fp2 = gdstk.FlexPath(
        [(12, 0), (8, 0), (8, 3), (10, 2)],
        [0.3, 0.2, 0.4],
        0.5,
        ends=["extended", "flush", "round"],
        joins=["bevel", "miter", "round"],
    )
    fp2.arc(2, -0.5 * numpy.pi, 0.5 * numpy.pi)
    fp2.arc(1, 0.5 * numpy.pi, 1.5 * numpy.pi)
    draw(gdstk.Cell("flexible_paths").add(fp1, fp2), path)

    # Flexible Paths 1
    # Path created with automatic bends of radius 5
    points = [(0, 0), (0, 10), (20, 0), (18, 15), (8, 15)]
    fp3 = gdstk.FlexPath(points, 0.5, bend_radius=5, simple_path=True)

    # Same path, generated with natural joins, for comparison
    fp4 = gdstk.FlexPath(points, 0.5, layer=1, simple_path=True)
    draw(gdstk.Cell("flexible_paths_2").add(fp3, fp4), path)

    # Flexible Paths 2
    # Straight segment showing the possibility of width and offset changes
    fp5 = gdstk.FlexPath((0, 0), [0.5, 0.5], 1)
    fp5.horizontal(2)
    fp5.horizontal(4, width=0.8, offset=1.8)
    fp5.horizontal(6)
    draw(gdstk.Cell("flexible_paths_3").add(fp5), path)

    # Robust Paths
    # Create 4 parallel paths in different layers
    rp = gdstk.RobustPath(
        (0, 50),
        [2, 0.5, 1, 1],
        [0, 0, -1, 1],
        ends=["extended", "round", "flush", "flush"],
        layer=[1, 0, 2, 2],
    )
    rp.segment((0, 45))
    rp.segment(
        (0, 5),
        width=[lambda u: 2 + 16 * u * (1 - u), 0.5, 1, 1],
        offset=[
            0,
            lambda u: 8 * u * (1 - u) * numpy.cos(12 * numpy.pi * u),
            lambda u: -1 - 8 * u * (1 - u),
            lambda u: 1 + 8 * u * (1 - u),
        ],
    )
    rp.segment((0, 0))
    rp.interpolation(
        [(15, 5)],
        angles=[0, 0.5 * numpy.pi],
        width=0.5,
        offset=[-0.25, 0.25, -0.75, 0.75],
    )
    rp.parametric(
        lambda u: numpy.array((4 * numpy.sin(6 * numpy.pi * u), 45 * u)),
        offset=[
            lambda u: -0.25 * numpy.cos(24 * numpy.pi * u),
            lambda u: 0.25 * numpy.cos(24 * numpy.pi * u),
            -0.75,
            0.75,
        ],
    )
    draw(gdstk.Cell("robust_paths").add(rp), path)

    # Text
    # Label centered at (1, 3)
    label = gdstk.Label("Sample label", (5, 3), texttype=2)

    # Horizontal text with height 2.25
    htext = gdstk.text("12345", 2.25, (0.25, 6))

    # Vertical text with height 1.5
    vtext = gdstk.text("ABC", 1.5, (10.5, 4), vertical=True)

    rect = gdstk.rectangle((0, 0), (10, 6), layer=10)
    draw(gdstk.Cell("text").add(*htext, *vtext, label, rect), path)

    # Boolean Operations
    # Create some text
    text = gdstk.text("GDSTK", 4, (0, 0))
    # Create a rectangle extending the text's bounding box by 1
    rect = gdstk.rectangle((-1, -1), (5 * 4 * 9 / 16 + 1, 4 + 1))

    # Subtract the text from the rectangle
    inv = gdstk.boolean(rect, text, "not")
    draw(gdstk.Cell("boolean_operations").add(*inv), path)

    # Slice Operation
    ring1 = gdstk.ellipse((-6, 0), 6, inner_radius=4)
    ring2 = gdstk.ellipse((0, 0), 6, inner_radius=4)
    ring3 = gdstk.ellipse((6, 0), 6, inner_radius=4)

    # Slice the first ring across x=-3, the second ring across x=-3
    # and x=3, and the third ring across x=3
    slices1 = gdstk.slice(ring1, -3, "x")
    slices2 = gdstk.slice(ring2, [-3, 3], "x")
    slices3 = gdstk.slice(ring3, 3, "x")

    slices = gdstk.Cell("SLICES")

    # Keep only the left side of slices1, the center part of slices2
    # and the right side of slices3
    slices.add(*slices1[0])
    slices.add(*slices2[1])
    slices.add(*slices3[1])
    # draw
    slices.name = "slice_operation"
    draw(slices, path)
    slices.name = "SLICES"

    # Offset Operation
    rect1 = gdstk.rectangle((-4, -4), (1, 1))
    rect2 = gdstk.rectangle((-1, -1), (4, 4))

    # Erosion: because we set `use_union=True`, the inner boundaries have no effect
    outer = gdstk.offset([rect1, rect2], -0.5, use_union=True, layer=1)
    draw(gdstk.Cell("offset_operation").add(rect1, rect2, *outer), path)

    # Fillet Operation
    flexpath = gdstk.FlexPath([(-8, -4), (0, -4), (0, 4), (8, 4)], 4)
    filleted_path = flexpath.to_polygons()[0]
    filleted_path.fillet(1.5)
    draw(gdstk.Cell("fillet_operation").add(filleted_path), path)
