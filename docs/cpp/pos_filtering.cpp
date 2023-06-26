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
    char lib_name[] = "library";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    char unit_cell_name[] = "Unit";
    Cell unit_cell = {.name = unit_cell_name};

    Polygon cross_ = cross(Vec2{0, 0}, 1, 0.2, 0);
    unit_cell.polygon_array.append(&cross_);

    char main_cell_name[] = "Main";
    Cell main_cell = {.name = main_cell_name};
    lib.cell_array.append(&main_cell);

    double d = 2;
    Reference unit_refs[] = {
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .magnification = 1,
            .repetition = {RepetitionType::Rectangular, 11, 6, Vec2{d, d * sqrt(3)}},
        },
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .origin = Vec2{d / 2, d * sqrt(3) / 2},
            .magnification = 1,
            .repetition = {RepetitionType::Rectangular, 10, 5, Vec2{d, d * sqrt(3)}},
        },
    };
    main_cell.reference_array.append(unit_refs);
    main_cell.reference_array.append(unit_refs + 1);

    Array<Reference*> removed_references = {};
    main_cell.flatten(true, removed_references);
    removed_references.clear();

    Array<Polygon*> txt = {};
    text("PY", 8 * d, Vec2{0.5 * d, 0}, false, make_tag(1, 0), txt);
    for (uint64_t i = 0; i < main_cell.polygon_array.count; i++) {
        Polygon* poly = main_cell.polygon_array[i];
        if (any_inside(poly->point_array, txt)) {
            poly->clear();
            free_allocation(poly);
            main_cell.polygon_array.remove(i--);
        }
    }

    main_cell.polygon_array.extend(txt);
    txt.clear();

    lib.write_gds("pos_filtering.gds", 0, NULL);

    cross_.clear();
    for (uint64_t i = 0; i < main_cell.polygon_array.count; i++) {
        main_cell.polygon_array[i]->clear();
        free_allocation(main_cell.polygon_array[i]);
    }
    main_cell.reference_array.clear();
    main_cell.polygon_array.clear();
    unit_cell.polygon_array.clear();
    lib.cell_array.clear();
    return 0;
}
