######################################################################
#                                                                    #
#  Copyright 2020-2020 Lucas Heitzmann Gabrielli.                    #
#  This file is part of gdstk, distributed under the terms of the    #
#  Boost Software License - Version 1.0.  See the accompanying       #
#  LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>            #
#                                                                    #
######################################################################

import pathlib
import gdstk
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath


def render_text(text, size=None, position=(0, 0), font_prop=None, tolerance=0.1):
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
                        if gdstk.inside(poly[:1], [polys[i]], precision=0.1 * tolerance)[0]:
                            p = polys.pop(i)
                            b = gdstk.boolean([p], [poly], "xor", precision=0.1 * tolerance)
                            poly = b[0].points
                            break
                        elif gdstk.inside(polys[i][:1], [poly], precision=0.1 * tolerance)[0]:
                            p = polys.pop(i)
                            b = gdstk.boolean([p], [poly], "xor", precision=0.1 * tolerance)
                            poly = b[0].points
                        i -= 1
                xmax = max(xmax, poly[:, 0].max())
                polys.append(poly)
    return polys


if __name__ == "__main__":
    fp = FontProperties(family="serif", style="italic")
    text = [gdstk.Polygon(pts, layer=1) for pts in render_text("Text rendering", 10, font_prop=fp)]
    lib = gdstk.Library()
    cell = lib.new_cell("TXT")
    cell.add(*text)
    lib.write_gds(pathlib.Path(__file__).parent / "fonts.gds")
