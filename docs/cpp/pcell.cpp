/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include <gdstk/gdstk.hpp>

using namespace gdstk;

Cell* grating(double period, double fill_frac, double length, double width, Tag tag,
              const char* name) {
    double x = width / 2;
    double w = period * fill_frac;
    int64_t num = (int64_t)(length / period);

    Cell* result = (Cell*)allocate_clear(sizeof(Cell));
    result->name = copy_string(name, NULL);
    result->polygon_array.ensure_slots(num);
    for (int64_t i = 0; i < num; i++) {
        double y = i * period;
        Polygon* rect = (Polygon*)allocate(sizeof(Polygon));
        *rect = rectangle(Vec2{-x, y}, Vec2{x, y + w}, tag);
        result->polygon_array.append(rect);
    }

    return result;
}

int main(int argc, char* argv[]) {
    Library lib = {};
    lib.init("library", 1e-6, 1e-9);

    double length = 20;

    Cell* grat1 = grating(3.5, 0.5, length, 25, make_tag(1, 0), "Grating 1");
    lib.cell_array.append(grat1);

    Cell* grat2 = grating(3.0, 0.5, length, 25, make_tag(1, 0), "Grating 2");
    lib.cell_array.append(grat2);

    Cell* main_cell = (Cell*)allocate_clear(sizeof(Cell));
    main_cell->name = copy_string("Main", NULL);
    lib.cell_array.append(main_cell);

    Polygon* rect = (Polygon*)allocate(sizeof(Polygon));
    *rect = rectangle(Vec2{0, -10}, Vec2{150, 10}, 0);
    main_cell->polygon_array.append(rect);

    Reference* ref1 = (Reference*)allocate_clear(sizeof(Reference));
    ref1->init(grat1);
    ref1->origin = Vec2{length, 0};
    ref1->rotation = M_PI / 2;
    main_cell->reference_array.append(ref1);

    Reference* ref2 = (Reference*)allocate_clear(sizeof(Reference));
    ref2->type = ReferenceType::Cell, ref2->cell = grat2, ref2->origin = Vec2{150 - length, 0},
    ref2->rotation = -M_PI / 2, ref2->magnification = 1, main_cell->reference_array.append(ref2);

    lib.write_gds("pcell.gds", 0, NULL);

    lib.free_all();
    return 0;
}
