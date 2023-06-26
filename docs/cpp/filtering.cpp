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
    ErrorCode error_code = ErrorCode::NoError;
    Library lib = read_gds("layout.gds", 0, 1e-2, NULL, &error_code);
    if (error_code != ErrorCode::NoError) exit(EXIT_FAILURE);

    for (int64_t i = 0; i < lib.cell_array.count; i++) {
        Cell* cell = lib.cell_array[i];
        for (int64_t j = 0; j < cell->polygon_array.count; j++) {
            Polygon* poly = cell->polygon_array[j];
            // Decrement j so that we don't skip over the next polygon.
            if (get_layer(poly->tag) == 2) {
                cell->polygon_array.remove(j--);
                poly->clear();
                free_allocation(poly);
            }
        }
        // Loaded libraries have no RobustPath elements
        for (int64_t j = 0; j < cell->flexpath_array.count; j++) {
            FlexPath* fp = cell->flexpath_array[j];
            // All paths in loaded libraries have only 1 element.
            // Decrement j so that we don't skip over the next path.
            if (get_layer(fp->elements[0].tag) == 10) {
                cell->flexpath_array.remove(j--);
                fp->clear();
                free_allocation(fp);
            }
        }
    }

    lib.write_gds("filtered-layout.gds", 0, NULL);
    lib.free_all();
    return 0;
}
