/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include <gdstk/gdstk.hpp>

using namespace gdstk;

Cell* example_boolean(const char* name) {
    Cell* out_cell = (Cell*)allocate_clear(sizeof(Cell));
    out_cell->name = copy_string(name, NULL);

    Array<Polygon*> txt = {};
    text("GDSTK", 4, Vec2{0, 0}, false, 0, txt);

    Polygon rect = rectangle(Vec2{-1, -1}, Vec2{5 * 4 * 9 / 16 + 1, 4 + 1}, 0);

    boolean(rect, txt, Operation::Not, 1000, out_cell->polygon_array);

    for (int i = 0; i < txt.count; i++) {
        txt[i]->clear();
        free_allocation(txt[i]);
    }
    txt.clear();
    rect.clear();

    return out_cell;
}

Cell* example_slice(const char* name) {
    Cell* out_cell = (Cell*)allocate_clear(sizeof(Cell));
    out_cell->name = copy_string(name, NULL);

    Polygon ring[3];
    ring[0] = ellipse(Vec2{-6, 0}, 6, 6, 4, 4, 0, 0, 0.01, 0);
    ring[1] = ellipse(Vec2{0, 0}, 6, 6, 4, 4, 0, 0, 0.01, 0);
    ring[2] = ellipse(Vec2{6, 0}, 6, 6, 4, 4, 0, 0, 0.01, 0);

    double x[] = {-3, 3};
    Array<double> cuts = {.capacity = 0, .count = 1, .items = x};
    Array<Polygon*> result[3] = {};
    slice(ring[0], cuts, true, 1000, result);
    out_cell->polygon_array.extend(result[0]);
    for (uint64_t i = 0; i < result[1].count; i++) {
        result[1][i]->clear();
        free_allocation(result[1][i]);
    }
    result[0].clear();
    result[1].clear();

    cuts.count = 2;
    slice(ring[1], cuts, true, 1000, result);
    out_cell->polygon_array.extend(result[1]);
    for (uint64_t i = 0; i < result[0].count; i++) {
        result[0][i]->clear();
        free_allocation(result[0][i]);
    }
    for (uint64_t i = 0; i < result[2].count; i++) {
        result[2][i]->clear();
        free_allocation(result[2][i]);
    }
    result[0].clear();
    result[1].clear();
    result[2].clear();

    cuts.count = 1;
    cuts.items = x + 1;
    slice(ring[2], cuts, true, 1000, result);
    out_cell->polygon_array.extend(result[1]);
    for (uint64_t i = 0; i < result[0].count; i++) {
        result[0][i]->clear();
        free_allocation(result[0][i]);
    }
    result[0].clear();
    result[1].clear();

    ring[0].clear();
    ring[1].clear();
    ring[2].clear();

    return out_cell;
}

Cell* example_offset(const char* name) {
    Cell* out_cell = (Cell*)allocate_clear(sizeof(Cell));
    out_cell->name = copy_string(name, NULL);

    Polygon* rect = (Polygon*)allocate(sizeof(Polygon));
    *rect = rectangle(Vec2{-4, -4}, Vec2{1, 1}, 0);
    out_cell->polygon_array.append(rect);

    rect = (Polygon*)allocate(sizeof(Polygon));
    *rect = rectangle(Vec2{-1, -1}, Vec2{4, 4}, 0);
    out_cell->polygon_array.append(rect);

    uint64_t start = out_cell->polygon_array.count;
    offset(out_cell->polygon_array, -0.5, OffsetJoin::Miter, 2, 1000, true,
           out_cell->polygon_array);
    for (uint64_t i = start; i < out_cell->polygon_array.count; i++) {
        out_cell->polygon_array[i]->tag = make_tag(1, 0);
    }

    return out_cell;
}

Cell* example_fillet(const char* name) {
    Cell* out_cell = (Cell*)allocate_clear(sizeof(Cell));
    out_cell->name = copy_string(name, NULL);

    FlexPath flexpath = {};
    flexpath.init(Vec2{-8, -4}, 1, 4, 0, 0.01, 0);
    Vec2 points[] = {{0, -4}, {0, 4}, {8, 4}};
    flexpath.segment({.capacity = 0, .count = COUNT(points), .items = points}, NULL, NULL, false);

    Array<Polygon*> poly_array = {};
    flexpath.to_polygons(false, 0, poly_array);
    flexpath.clear();

    double r = 1.5;
    for (int i = 0; i < poly_array.count; i++)
        poly_array[i]->fillet({.capacity = 0, .count = 1, .items = &r}, 0.01);

    out_cell->polygon_array.extend(poly_array);
    poly_array.clear();

    return out_cell;
}

int main(int argc, char* argv[]) {
    Library lib = {};
    lib.init("library", 1e-6, 1e-9);

    Cell* boolean_cell = example_boolean("Boolean");
    lib.cell_array.append(boolean_cell);

    Cell* slice_cell = example_slice("Slice");
    lib.cell_array.append(slice_cell);

    Cell* offset_cell = example_offset("Offset");
    lib.cell_array.append(offset_cell);

    Cell* fillet_cell = example_fillet("Fillet");
    lib.cell_array.append(fillet_cell);

    lib.write_gds("geometry_operations.gds", 0, NULL);

    lib.free_all();
    return 0;
}
