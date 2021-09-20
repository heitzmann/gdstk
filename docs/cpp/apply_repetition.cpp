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
    Library lib = {.unit = 1e-6, .precision = 1e-9};
    lib.name = copy_string("library", NULL);

    Cell main_cell = {0};
    main_cell.name = copy_string("Main", NULL);
    lib.cell_array.append(&main_cell);

    FlexPath vline = {.simple_path = true};
    vline.init(Vec2{3, 2}, 1, 0.1, 0, 0.01);
    vline.segment(Vec2{3, 3.5}, NULL, NULL, false);
    double vcoords[] = {0.2, 0.6, 1.4, 3.0};
    vline.repetition.type = RepetitionType::ExplicitX;
    vline.repetition.coords.extend({.count = COUNT(vcoords), .items = vcoords});

    Array<Polygon*> vlines = {0};
    vline.to_polygons(false, 0, vlines);
    vline.clear();

    // Because we know there is only a single resulting polygon we dont need to
    // loop here.
    vlines[0]->apply_repetition(vlines);

    RobustPath hline = {
        .end_point = {3, 2},
        .num_elements = 1,
        .tolerance = 0.01,
        .max_evals = 1000,
        .width_scale = 1,
        .offset_scale = 1,
        .trafo = {1, 0, 0, 0, 1, 0},
        .simple_path = true,
    };
    hline.elements =
        (RobustPathElement*)allocate_clear(sizeof(RobustPathElement) * hline.num_elements);
    hline.elements[0].end_width = 0.05;
    hline.segment(Vec2{6, 2}, NULL, NULL, false);
    double hcoords[] = {0.1, 0.3, 0.7, 1.5};
    hline.repetition.type = RepetitionType::ExplicitY;
    hline.repetition.coords.extend({.count = COUNT(hcoords), .items = hcoords});

    Array<Polygon*> hlines = {0};
    hline.to_polygons(false, 0, hlines);
    hline.clear();

    // Once again, no loop needed.
    hlines[0]->apply_repetition(vlines);

    Array<Polygon*> result = {0};
    boolean(vlines, hlines, Operation::Or, 1000, result);
    for (uint64_t i = 0; i < vlines.count; i++) {
        vlines[i]->clear();
        free_allocation(vlines[i]);
    }
    vlines.clear();
    for (uint64_t i = 0; i < hlines.count; i++) {
        hlines[i]->clear();
        free_allocation(hlines[i]);
    }
    hlines.clear();

    main_cell.polygon_array.extend(result);
    result.clear();

    lib.write_gds("apply_repetition.gds", 0, NULL);

    lib.clear();
    main_cell.free_all();
    return 0;
}
