/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include "gdstk.h"

using namespace gdstk;

void make_first_lib(const char* filename) {
    char lib_name[] = "First";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    char main_cell_name[] = "Main";
    Cell main_cell = {.name = main_cell_name};
    lib.cell_array.append(&main_cell);

    text("First library", 10, Vec2{0, 0}, false, 0, 0, main_cell.polygon_array);

    char square_cell_name[] = "Square";
    Cell square_cell = {.name = square_cell_name};
    lib.cell_array.append(&square_cell);

    Polygon square = rectangle(Vec2{-15, 0}, Vec2{-5, 10}, 0, 0);
    square_cell.polygon_array.append(&square);

    Reference square_referece = {
        .type = ReferenceType::Cell,
        .cell = &square_cell,
        .magnification = 1,
    };
    main_cell.reference_array.append(&square_referece);

    char circle_cell_name[] = "Circle";
    Cell circle_cell = {.name = circle_cell_name};
    lib.cell_array.append(&circle_cell);

    Polygon circle = ellipse(Vec2{0, 0}, 4, 4, 0, 0, 0, 0, 0.01, 0, 0);
    circle_cell.polygon_array.append(&circle);

    Reference circle_referece = {
        .type = ReferenceType::Cell,
        .cell = &circle_cell,
        .origin = {-10, 5},
        .magnification = 1,
    };
    square_cell.reference_array.append(&circle_referece);

    lib.write_gds(filename, 0, NULL);
}

void make_second_lib(const char* filename) {
    char lib_name[] = "Second";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    char main_cell_name[] = "Main";
    Cell main_cell = {.name = main_cell_name};
    lib.cell_array.append(&main_cell);

    text("Second library", 10, Vec2{0, 0}, false, 0, 0, main_cell.polygon_array);

    char circle_cell_name[] = "Circle";
    Cell circle_cell = {.name = circle_cell_name};
    lib.cell_array.append(&circle_cell);

    Polygon circle = ellipse(Vec2{-10, 5}, 5, 5, 0, 0, 0, 0, 0.01, 0, 0);
    circle_cell.polygon_array.append(&circle);

    Reference circle_referece = {
        .type = ReferenceType::Cell,
        .cell = &circle_cell,
        .magnification = 1,
    };
    main_cell.reference_array.append(&circle_referece);

    lib.write_gds(filename, 0, NULL);
}

int main(int argc, char* argv[]) {
    make_first_lib("lib1.gds");
    make_second_lib("lib2.gds");

    Library lib1 = read_gds("lib1.gds", 0, 1e-2);
    Library lib2 = read_gds("lib2.gds", 0, 1e-2);

    // We could use a hash table to make this more efficient, but we're aiming
    // for simplicity.
    for (uint64_t i = 0; i < lib2.cell_array.count; i++) {
        Cell* cell = lib2.cell_array[i];
        for (uint64_t j = 0; j < lib1.cell_array.count; j++) {
            if (strcmp(cell->name, lib1.cell_array[j]->name) == 0) {
                uint64_t len = strlen(cell->name);
                cell->name = (char*)reallocate(cell->name, len + 5);
                strcpy(cell->name + len, "-lib2");
                // We should make sure the new name is also unique, but we are
                // skiping that.
                break;
            }
        }
        lib1.cell_array.append(cell);
    }

    lib1.write_gds("merging.gds", 0, NULL);

    return 0;
}
