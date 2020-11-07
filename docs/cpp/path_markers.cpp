/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <cstdio>

#include "gdstk.h"

using namespace gdstk;

int main(int argc, char* argv[]) {
    char lib_name[] = "library";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    char main_cell_name[] = "Main";
    Cell main_cell = {.name = main_cell_name};
    lib.cell_array.append(&main_cell);

    RobustPathElement element = {.end_width = 0.5};
    RobustPath path = {
        .elements = &element,
        .num_elements = 1,
        .tolerance = 0.01,
        .max_evals = 1000,
        .width_scale = 1,
        .offset_scale = 1,
        .trafo = {1, 0, 0, 0, 1, 0},
    };
    path.segment(Vec2{8, 0}, NULL, NULL, false);

    Vec2 points[] = {{2, -4}, {-2, -6}, {-5, -8}, {-4, -12}};
    double angles[] = {0, 0, 0, 0, -M_PI / 4};
    bool angle_constraints[] = {true, false, false, false, true};
    Vec2 tension[] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    path.interpolation({.size = COUNT(points), .items = points}, angles, angle_constraints, tension,
                       1, 1, false, NULL, NULL, true);

    path.segment(Vec2{3, -3}, NULL, NULL, true);
    main_cell.robustpath_array.append(&path);

    Polygon major = regular_polygon(Vec2{0, 0}, 0.5, 6, 0, 1, 0);
    Polygon minor = rectangle(Vec2{-0.1, -0.5}, Vec2{0.1, 0.5}, 1, 0);
    for (int64_t i = 0; i < path.subpath_array.size; i++) {
        Polygon* p_maj = (Polygon*)allocate_clear(sizeof(Polygon));
        p_maj->copy_from(major);
        p_maj->translate(path.position(i, true));
        main_cell.polygon_array.append(p_maj);
        for (int64_t j = 1; j < 4; j++) {
            Polygon* p_min = (Polygon*)allocate_clear(sizeof(Polygon));
            p_min->copy_from(minor);
            double u = i + j / 4.0;
            p_min->rotate(path.gradient(u, true).angle(), Vec2{0, 0});
            p_min->translate(path.position(u, true));
            main_cell.polygon_array.append(p_min);
        }
    }
    major.translate(path.end_point);
    main_cell.polygon_array.append(&major);

    lib.write_gds("path_markers.gds", 0, NULL);
    return 0;
}
