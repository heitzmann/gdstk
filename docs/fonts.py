# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import pathlib
from tutorial_images import draw

import gdstk
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath


def render_text(text, size=None, position=(0, 0), font_prop=None, tolerance=0.1):
    tol = 0.1 * tolerance
    path = TextPath(position, text, size=size, prop=font_prop)
    polys = []
    xmax = position[0]
    for points, code in path.iter_segments():
        if code == path.MOVETO:
            c = gdstk.Curve(points, tolerance=tolerance)
        elif code == path.LINETO:
            c.segment(points.reshape(points.size // 2, 2))
        elif code == path.CURVE3:
            c.quadratic(points.reshape(points.size // 2, 2))
        elif code == path.CURVE4:
            c.cubic(points.reshape(points.size // 2, 2))
        elif code == path.CLOSEPOLY:
            poly = c.points()
            if poly.size > 0:
                if poly[:, 0].min() < xmax:
                    i = len(polys) - 1
                    while i >= 0:
                        if gdstk.inside(poly[:1], [polys[i]], precision=tol)[0]:
                            p = polys.pop(i)
                            b = gdstk.boolean([p], [poly], "xor", tol)
                            poly = b[0].points
                            break
                        elif gdstk.inside(polys[i][:1], [poly], precision=tol)[0]:
                            p = polys.pop(i)
                            b = gdstk.boolean([p], [poly], "xor", tol)
                            poly = b[0].points
                        i -= 1
                xmax = max(xmax, poly[:, 0].max())
                polys.append(poly)
    return polys


if __name__ == "__main__":
    cell = gdstk.Cell("fonts")
    fp = FontProperties(family="serif", style="italic")
    point_list = render_text("Text rendering", 10, font_prop=fp)
    cell.add(gdstk.Polygon(pts) for pts in point_list)
    path = pathlib.Path(__file__).parent.absolute() / "how-tos"
    draw(cell, path)
