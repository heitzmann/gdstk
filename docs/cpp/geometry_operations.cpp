/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include "gdstk.h"

using namespace gdstk;

void example_boolean(Cell& out_cell) {
    Array<Polygon*> txt = {0};
    text("GDSTK", 4, Vec2{0, 0}, false, 0, 0, txt);

    Polygon rect = rectangle(Vec2{-1, -1}, Vec2{5 * 4 * 9 / 16 + 1, 4 + 1}, 0, 0);
    Polygon* rectp = &rect;

    boolean({.count = 1, .items = &rectp}, txt, Operation::Not, 1000, out_cell.polygon_array);

    for (int i = 0; i < txt.count; i++) {
        txt[i]->clear();
        free_allocation(txt[i]);
    }
    txt.clear();
}

void example_slice(Cell& out_cell) {
    Polygon ring[3];
    ring[0] = ellipse(Vec2{-6, 0}, 6, 6, 4, 4, 0, 0, 0.01, 0, 0);
    ring[1] = ellipse(Vec2{0, 0}, 6, 6, 4, 4, 0, 0, 0.01, 0, 0);
    ring[2] = ellipse(Vec2{6, 0}, 6, 6, 4, 4, 0, 0, 0.01, 0, 0);

    double x[] = {-3, 3};
    Array<double> cuts = {.count = 1, .items = x};
    Array<Polygon*> result[3] = {0};
    slice(ring[0], cuts, true, 1000, result);
    out_cell.polygon_array.extend(result[0]);
    result[0].clear();
    result[1].clear();

    cuts.count = 2;
    slice(ring[1], cuts, true, 1000, result);
    out_cell.polygon_array.extend(result[1]);
    result[0].clear();
    result[1].clear();
    result[2].clear();

    cuts.count = 1;
    cuts.items = x + 1;
    slice(ring[2], cuts, true, 1000, result);
    out_cell.polygon_array.extend(result[1]);
    result[0].clear();
    result[1].clear();
}

void example_offset(Cell& out_cell) {
    Polygon* rect = (Polygon*)allocate_clear(2 * sizeof(Polygon));
    rect[0] = rectangle(Vec2{-4, -4}, Vec2{1, 1}, 0, 0);
    rect[1] = rectangle(Vec2{-1, -1}, Vec2{4, 4}, 0, 0);
    Polygon* rect_p[] = {rect, rect + 1};
    const Array<Polygon*> rect_array = {.count = 2, .items = rect_p};
    out_cell.polygon_array.extend(rect_array);
    uint64_t start = out_cell.polygon_array.count;
    offset(rect_array, -0.5, OffsetJoin::Miter, 2, 1000, true, out_cell.polygon_array);
    for (uint64_t i = start; i < out_cell.polygon_array.count; i++) {
        out_cell.polygon_array[i]->layer = 1;
    }
}

void example_fillet(Cell& out_cell) {
    FlexPath flexpath = {0};
    flexpath.init(Vec2{-8, -4}, 1, 4, 0, 0.01);
    Vec2 points[] = {{0, -4}, {0, 4}, {8, 4}};
    flexpath.segment({.count = COUNT(points), .items = points}, NULL, NULL, false);

    double r = 1.5;
    Array<Polygon*> poly_array = {0};
    flexpath.to_polygons(poly_array);
    for (int i = 0; i < poly_array.count; i++)
        poly_array[i]->fillet({.count = 1, .items = &r}, 0.01);

    out_cell.polygon_array.extend(poly_array);
    flexpath.clear();
    poly_array.clear();
}

int main(int argc, char* argv[]) {
    char lib_name[] = "library";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    char boolean_cell_name[] = "Boolean";
    Cell boolean_cell = {.name = boolean_cell_name};
    example_boolean(boolean_cell);
    lib.cell_array.append(&boolean_cell);

    char slice_cell_name[] = "Slice";
    Cell slice_cell = {.name = slice_cell_name};
    example_slice(slice_cell);
    lib.cell_array.append(&slice_cell);

    char offset_cell_name[] = "Offset";
    Cell offset_cell = {.name = offset_cell_name};
    example_offset(offset_cell);
    lib.cell_array.append(&offset_cell);

    char fillet_cell_name[] = "Fillet";
    Cell fillet_cell = {.name = fillet_cell_name};
    example_fillet(fillet_cell);
    lib.cell_array.append(&fillet_cell);

    lib.write_gds("geometry_operations.gds", 0, NULL);

    return 0;
}
