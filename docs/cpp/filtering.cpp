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
    Library lib = read_gds("layout.gds", 0, 1e-2);
    if (lib.unit == 0) {
        // File not found
        exit(EXIT_FAILURE);
    }

    for (int64_t i = 0; i < lib.cell_array.count; i++) {
        Cell* cell = lib.cell_array[i];
        for (int64_t j = 0; j < cell->polygon_array.count; j++) {
            Polygon* poly = cell->polygon_array[j];
            // Decrement j so that we don't skip over the next polygon.
            if (poly->layer == 2) cell->polygon_array.remove(j--);
        }
        // Loaded libraries have no RobustPath elements
        for (int64_t j = 0; j < cell->flexpath_array.count; j++) {
            FlexPath* fp = cell->flexpath_array[j];
            // All paths in loaded libraries have only 1 element.
            // Decrement j so that we don't skip over the next path.
            if (fp->elements[0].layer == 10) cell->flexpath_array.remove(j--);
        }
    }

    lib.write_gds("filtered-layout.gds", 0, NULL);
    return 0;
}
