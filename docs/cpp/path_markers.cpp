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

    Cell* main_cell = (Cell*)allocate_clear(sizeof(Cell));
    main_cell->name = copy_string("Main", NULL);
    lib.cell_array.append(main_cell);

    RobustPath* path = (RobustPath*)allocate_clear(sizeof(RobustPath));
    path->init(Vec2{0, 0}, 1, 0.5, 0, 0.01, 1000, 0);
    path->segment(Vec2{8, 0}, NULL, NULL, false);

    Vec2 points[] = {{2, -4}, {-2, -6}, {-5, -8}, {-4, -12}};
    double angles[] = {0, 0, 0, 0, -M_PI / 4};
    bool angle_constraints[] = {true, false, false, false, true};
    Vec2 tension[] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    path->interpolation({.capacity = 0, .count = COUNT(points), .items = points}, angles,
                        angle_constraints, tension, 1, 1, false, NULL, NULL, true);

    path->segment(Vec2{3, -3}, NULL, NULL, true);
    main_cell->robustpath_array.append(path);

    Polygon* major = (Polygon*)allocate_clear(sizeof(Polygon));
    *major = regular_polygon(Vec2{0, 0}, 0.5, 6, 0, make_tag(1, 0));
    Polygon minor = rectangle(Vec2{-0.1, -0.5}, Vec2{0.1, 0.5}, make_tag(1, 0));

    // Reserve space for all markers in the main cell
    int64_t count = path->subpath_array.count;
    main_cell->polygon_array.ensure_slots(1 + 4 * count);

    for (int64_t i = 0; i < count; i++) {
        Polygon* poly = (Polygon*)allocate_clear(sizeof(Polygon));
        poly->copy_from(*major);
        poly->translate(path->position(i, true));
        main_cell->polygon_array.append(poly);
        for (int64_t j = 1; j < 4; j++) {
            poly = (Polygon*)allocate_clear(sizeof(Polygon));
            poly->copy_from(minor);
            double u = i + j / 4.0;
            poly->rotate(path->gradient(u, true).angle(), Vec2{0, 0});
            poly->translate(path->position(u, true));
            main_cell->polygon_array.append(poly);
        }
    }

    // Last marker: we use the original major marker
    major->translate(path->end_point);
    main_cell->polygon_array.append(major);
    minor.clear();

    lib.write_gds("path_markers.gds", 0, NULL);

    lib.free_all();
    return 0;
}
