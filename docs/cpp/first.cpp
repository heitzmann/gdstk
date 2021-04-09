/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include "gdstk.h"

using namespace gdstk;

int main(int argc, char* argv[]) {
    char lib_name[] = "library";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    char cell_name[] = "FIRST";
    Cell cell = {.name = cell_name};
    lib.cell_array.append(&cell);

    Polygon rect = rectangle(Vec2{0, 0}, Vec2{2, 1}, 0, 0);
    cell.polygon_array.append(&rect);

    lib.write_gds("first.gds", 0, NULL);
    lib.write_oas("first.oas", 0, 6, OASIS_CONFIG_DETECT_ALL);

    StyleMap style = {0};
    StyleMap label_style = {0};
    cell.write_svg("first.svg", 10, style, label_style, "#222222", 5, true, NULL);

    return 0;
}
