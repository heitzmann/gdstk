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

    char main_cell_name[] = "Main";
    Cell main_cell = {.name = main_cell_name};
    lib.cell_array.append(&main_cell);

    Polygon square = regular_polygon(Vec2{0, 0}, 0.2, 4, 0, 0, 0);
    square.repetition = {RepetitionType::Rectangular, 3, 2, Vec2{1, 1}};
    main_cell.polygon_array.append(&square);

    Polygon triangle = regular_polygon(Vec2{0, 2.5}, 0.2, 3, 0, 0, 0);
    triangle.repetition = {RepetitionType::Regular, 3, 5, Vec2{0.4, -0.3}, Vec2{0.4, 0.2}};
    main_cell.polygon_array.append(&triangle);

    Polygon circle = ellipse(Vec2{3.5, 0}, 0.1, 0.1, 0, 0, 0, 0, 0.01, 0, 0);
    Vec2 offsets[] = {{0.5, 1}, {2, 0}, {1.5, 0.5}};
    circle.repetition.type = RepetitionType::Explicit;
    circle.repetition.offsets.count = COUNT(offsets);
    circle.repetition.offsets.items = offsets;
    main_cell.polygon_array.append(&circle);

    FlexPathElement velement = {};
    FlexPath vline = {.elements = &velement, .num_elements = 1, .simple_path = true};
    vline.init(Vec2{3, 2}, 0.1, 0, 0.01);
    vline.segment(Vec2{3, 3.5}, NULL, NULL, false);
    double vcoords[] = {0.2, 0.6, 1.4, 3.0};
    vline.repetition.type = RepetitionType::ExplicitX;
    vline.repetition.coords.count = COUNT(vcoords);
    vline.repetition.coords.items = vcoords;
    main_cell.flexpath_array.append(&vline);

    RobustPathElement helement = {.end_width = 0.05};
    RobustPath hline = {
        .end_point = {3, 2},
        .elements = &helement,
        .num_elements = 1,
        .tolerance = 0.01,
        .max_evals = 1000,
        .width_scale = 1,
        .offset_scale = 1,
        .trafo = {1, 0, 0, 0, 1, 0},
        .simple_path = true,
    };
    hline.segment(Vec2{6, 2}, NULL, NULL, false);
    double hcoords[] = {0.1, 0.3, 0.7, 1.5};
    hline.repetition.type = RepetitionType::ExplicitY;
    hline.repetition.coords.count = COUNT(hcoords);
    hline.repetition.coords.items = hcoords;
    main_cell.robustpath_array.append(&hline);

    lib.write_gds("repetitions.gds", 0, NULL);
    return 0;
}

