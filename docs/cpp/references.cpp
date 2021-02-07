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
    int16_t layer_full_etch = 1;
    int16_t dt_full_etch = 3;
    int16_t layer_partial_etch = 2;
    int16_t dt_partial_etch = 3;
    int16_t layer_lift_off = 0;
    int16_t dt_lift_off = 7;

    char lib_name[] = "library";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    // CONTACT

    char contact_cell_name[] = "CONTACT";
    Cell contact_cell = {.name = contact_cell_name};
    lib.cell_array.append(&contact_cell);

    Polygon contact_poly[4];
    contact_poly[0] = rectangle(Vec2{-3, -3}, Vec2{3, 3}, layer_full_etch, dt_full_etch);
    contact_poly[1] = rectangle(Vec2{-5, -3}, Vec2{-3, 3}, layer_partial_etch, dt_partial_etch);
    contact_poly[2] = rectangle(Vec2{5, -3}, Vec2{3, 3}, layer_partial_etch, dt_partial_etch);
    contact_poly[3] = regular_polygon(Vec2{0, 0}, 2, 6, 0, layer_lift_off, dt_lift_off);

    Polygon* p[] = {contact_poly, contact_poly + 1, contact_poly + 2, contact_poly + 3};
    contact_cell.polygon_array.extend({.count = 4, .items = p});

    // DEVICE

    char device_cell_name[] = "DEVICE";
    Cell device_cell = {.name = device_cell_name};
    lib.cell_array.append(&device_cell);

    Vec2 cutout_points[] = {{0, 0}, {5, 0}, {5, 5}, {0, 5}, {0, 0},
                            {2, 2}, {2, 3}, {3, 3}, {3, 2}, {2, 2}};
    Polygon* cutout_poly = (Polygon*)allocate_clear(sizeof(Polygon));
    cutout_poly->point_array.extend({.count = COUNT(cutout_points), .items = cutout_points});
    device_cell.polygon_array.append(cutout_poly);

    Reference contact_ref1 = {
        .type = ReferenceType::Cell,
        .cell = &contact_cell,
        .origin = Vec2{3.5, 1},
        .magnification = 0.25,
    };
    device_cell.reference_array.append(&contact_ref1);

    Reference contact_ref2 = {
        .type = ReferenceType::Cell,
        .cell = &contact_cell,
        .origin = Vec2{1, 3.5},
        .rotation = M_PI / 2,
        .magnification = 0.25,
    };
    device_cell.reference_array.append(&contact_ref2);

    // MAIN

    char main_cell_name[] = "MAIN";
    Cell main_cell = {.name = main_cell_name};
    lib.cell_array.append(&main_cell);

    Reference device_ref = {
        .type = ReferenceType::Cell,
        .cell = &device_cell,
        .magnification = 1,
        .repetition = {RepetitionType::Rectangular, 3, 2, Vec2{6, 7}},
    };
    main_cell.reference_array.append(&device_ref);

    // Output

    lib.write_gds("references.gds", 0, NULL);

    return 0;
}
