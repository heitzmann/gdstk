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
    precision = 0.1 * tolerance
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
            pts = c.points()
            if pts.size > 0:
                poly = gdstk.Polygon(pts)
                if pts[:, 0].min() < xmax:
                    i = len(polys) - 1
                    while i >= 0:
                        if polys[i].contain_any(*poly.points):
                            p = polys.pop(i)
                            poly = gdstk.boolean(p, poly, "xor", precision)[0]
                            break
                        elif poly.contain_any(*polys[i].points):
                            p = polys.pop(i)
                            poly = gdstk.boolean(p, poly, "xor", precision)[0]
                        i -= 1
                xmax = max(xmax, poly.points[:, 0].max())
                polys.append(poly)
    return polys


if __name__ == "__main__":
    cell = gdstk.Cell("fonts")
    fp = FontProperties(family="serif", style="italic")
    polygons = render_text("Text rendering", 10, font_prop=fp)
    cell.add(*polygons)
    path = pathlib.Path(__file__).parent.absolute() / "how-tos"
    draw(cell, path)
