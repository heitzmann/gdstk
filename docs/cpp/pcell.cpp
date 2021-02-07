/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include "gdstk.h"

using namespace gdstk;

Cell grating(double period, double fill_frac, double length, double width, int16_t layer,
             int16_t datatype, char* name) {
    double x = width / 2;
    double w = period * fill_frac;
    int64_t num = (int64_t)(length / period);

    Cell result = {.name = name};
    result.polygon_array.ensure_slots(num);
    Polygon* rect = (Polygon*)allocate(num * sizeof(Polygon));
    for (int64_t i = 0; i < num; i++) {
        double y = i * period;
        *rect = rectangle(Vec2{-x, y}, Vec2{x, y + w}, layer, datatype);
        result.polygon_array.append(rect++);
    }

    return result;
}

int main(int argc, char* argv[]) {
    char lib_name[] = "library";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    double length = 20;

    char grat1_name[] = "Grating 1";
    Cell grat1 = grating(3.5, 0.5, length, 25, 1, 0, grat1_name);
    lib.cell_array.append(&grat1);

    char grat2_name[] = "Grating 2";
    Cell grat2 = grating(3.0, 0.5, length, 25, 1, 0, grat2_name);
    lib.cell_array.append(&grat2);

    char main_name[] = "Main";
    Cell main = {.name = main_name};
    lib.cell_array.append(&main);

    Polygon rect = rectangle(Vec2{0, -10}, Vec2{150, 10}, 0, 0);
    main.polygon_array.append(&rect);

    Reference ref1 = {
        .type = ReferenceType::Cell,
        .cell = &grat1,
        .origin = Vec2{length, 0},
        .rotation = M_PI / 2,
        .magnification = 1,
    };
    main.reference_array.append(&ref1);

    Reference ref2 = {
        .type = ReferenceType::Cell,
        .cell = &grat2,
        .origin = Vec2{150 - length, 0},
        .rotation = -M_PI / 2,
        .magnification = 1,
    };
    main.reference_array.append(&ref2);

    lib.write_gds("pcell.gds", 0, NULL);

    return 0;
}
