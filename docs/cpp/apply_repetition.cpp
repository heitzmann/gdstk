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

    FlexPathElement velement = {};
    FlexPath vline = {.elements = &velement, .num_elements = 1, .simple_path = true};
    vline.init(Vec2{3, 2}, 0.1, 0, 0.01);
    vline.segment(Vec2{3, 3.5}, NULL, NULL, false);
    double vcoords[] = {0.2, 0.6, 1.4, 3.0};
    vline.repetition.type = RepetitionType::ExplicitX;
    vline.repetition.coords.count = COUNT(vcoords);
    vline.repetition.coords.items = vcoords;

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

    Array<Polygon*> vlines = {0};
    vline.to_polygons(vlines);
    // Because we know there is only a single resulting polygon we dont need to
    // loop here.
    vlines[0]->apply_repetition(vlines);

    Array<Polygon*> hlines = {0};
    hline.to_polygons(hlines);
    hlines[0]->apply_repetition(vlines);

    Array<Polygon*> result = {0};
    boolean(vlines, hlines, Operation::Or, 1000, result);
    main_cell.polygon_array.extend(result);

    vlines.clear();
    hlines.clear();
    result.clear();

    lib.write_gds("apply_repetition.gds", 0, NULL);
    return 0;
}

