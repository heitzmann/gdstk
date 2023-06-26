/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include <gdstk/gdstk.hpp>

using namespace gdstk;

Cell* example_flexpath1(const char* name) {
    Cell* out_cell = (Cell*)allocate_clear(sizeof(Cell));
    out_cell->name = copy_string(name, NULL);

    FlexPath* fp = (FlexPath*)allocate_clear(sizeof(FlexPath));
    fp->init(Vec2{0, 0}, 1, 0.5, 0, 0.01, 0);
    fp->simple_path = true;

    Vec2 points1[] = {{3, 0}, {3, 2}, {5, 3}, {3, 4}, {0, 4}};
    fp->segment({.capacity = 0, .count = COUNT(points1), .items = points1}, NULL, NULL, false);

    out_cell->flexpath_array.append(fp);

    fp = (FlexPath*)allocate_clear(sizeof(FlexPath));
    const double widths[] = {0.3, 0.2, 0.4};
    const double offsets[] = {-0.5, 0, 0.5};
    const Tag tags[] = {0, 0, 0};
    fp->init(Vec2{12, 0}, 3, widths, offsets, 0.01, tags);

    fp->elements[0].end_type = EndType::HalfWidth;
    fp->elements[0].join_type = JoinType::Bevel;

    fp->elements[1].end_type = EndType::Flush;
    fp->elements[1].join_type = JoinType::Miter;

    fp->elements[2].end_type = EndType::Round;
    fp->elements[2].join_type = JoinType::Round;

    Vec2 points2[] = {{8, 0}, {8, 3}, {10, 2}};
    fp->segment({.capacity = 0, .count = COUNT(points2), .items = points2}, NULL, NULL, false);

    fp->arc(2, 2, -M_PI / 2, M_PI / 2, 0, NULL, NULL);
    fp->arc(1, 1, M_PI / 2, 1.5 * M_PI, 0, NULL, NULL);

    out_cell->flexpath_array.append(fp);
    return out_cell;
}

Cell* example_flexpath2(const char* name) {
    Cell* out_cell = (Cell*)allocate_clear(sizeof(Cell));
    out_cell->name = copy_string(name, NULL);

    Vec2 points[] = {{0, 10}, {20, 0}, {18, 15}, {8, 15}};

    for (uint64_t i = 0; i < 2; i++) {
        FlexPath* fp = (FlexPath*)allocate_clear(sizeof(FlexPath));
        fp->init(Vec2{0, 0}, 1, 0.5, 0, 0.01, 0);
        fp->simple_path = true;
        if (i == 0) {
            fp->elements[0].bend_type = BendType::Circular;
            fp->elements[0].bend_radius = 5;
        } else {
            fp->elements[0].tag = make_tag(1, 0);
        }
        fp->segment({.capacity = 0, .count = COUNT(points), .items = points}, NULL, NULL, false);
        out_cell->flexpath_array.append(fp);
    }

    return out_cell;
}

Cell* example_flexpath3(const char* name) {
    Cell* out_cell = (Cell*)allocate_clear(sizeof(Cell));
    out_cell->name = copy_string(name, NULL);

    double widths[] = {0.5, 0.5};
    double offsets[] = {-0.5, 0.5};
    Tag tags[] = {0, 0};
    FlexPath* fp = (FlexPath*)allocate_clear(sizeof(FlexPath));
    fp->init(Vec2{0, 0}, 2, widths, offsets, 0.01, tags);

    fp->horizontal(2, NULL, NULL, false);

    widths[0] = 0.8;
    widths[1] = 0.8;
    offsets[0] = -0.9;
    offsets[1] = 0.9;
    fp->horizontal(4, widths, offsets, false);

    fp->horizontal(6, NULL, NULL, false);

    out_cell->flexpath_array.append(fp);

    return out_cell;
}

int main(int argc, char* argv[]) {
    Library lib = {};
    lib.init("Paths", 1e-6, 1e-9);

    Cell* flexpath1_cell = example_flexpath1("FlexPath 1");
    lib.cell_array.append(flexpath1_cell);

    Cell* flexpath2_cell = example_flexpath2("FlexPath 2");
    lib.cell_array.append(flexpath2_cell);

    Cell* flexpath3_cell = example_flexpath3("FlexPath 3");
    lib.cell_array.append(flexpath3_cell);

    lib.write_gds("flexpaths.gds", 0, NULL);

    lib.free_all();
    return 0;
}
