/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include <gdstk/gdstk.hpp>

using namespace gdstk;

int main(int argc, char* argv[]) {
    Library lib = {};
    lib.init("library", 1e-6, 1e-9);

    Cell main_cell = {};
    main_cell.name = copy_string("Main", NULL);
    lib.cell_array.append(&main_cell);

    FlexPath vline = {};
    vline.init(Vec2{3, 2}, 1, 0.1, 0, 0.01, 0);
    vline.simple_path = true;
    vline.segment(Vec2{3, 3.5}, NULL, NULL, false);
    double vcoords[] = {0.2, 0.6, 1.4, 3.0};
    vline.repetition.type = RepetitionType::ExplicitX;
    vline.repetition.coords.extend({.capacity = 0, .count = COUNT(vcoords), .items = vcoords});

    Array<Polygon*> vlines = {};
    vline.to_polygons(false, 0, vlines);
    vline.clear();

    // Because we know there is only a single resulting polygon we dont need to
    // loop here.
    vlines[0]->apply_repetition(vlines);

    RobustPath hline = {};
    hline.init(Vec2{3, 2}, 1, 0.05, 0, 0.01, 1000, 0);
    hline.simple_path = true;
    hline.segment(Vec2{6, 2}, NULL, NULL, false);
    double hcoords[] = {0.1, 0.3, 0.7, 1.5};
    hline.repetition.type = RepetitionType::ExplicitY;
    hline.repetition.coords.extend({.capacity = 0, .count = COUNT(hcoords), .items = hcoords});

    Array<Polygon*> hlines = {};
    hline.to_polygons(false, 0, hlines);
    hline.clear();

    // Once again, no loop needed.
    hlines[0]->apply_repetition(vlines);

    Array<Polygon*> result = {};
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
