/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include "gdstk.h"

using namespace gdstk;

void example_flexpath1(Cell& out_cell) {
    FlexPath* fp = (FlexPath*)allocate_clear(2 * sizeof(FlexPath));
    fp[0].simple_path = true;
    fp[0].init(Vec2{0, 0}, 1, 0.5, 0, 0.01);

    Vec2 points1[] = {{3, 0}, {3, 2}, {5, 3}, {3, 4}, {0, 4}};
    fp[0].segment({.count = COUNT(points1), .items = points1}, NULL, NULL, false);

    out_cell.flexpath_array.append(fp);

    const double widths[] = {0.3, 0.2, 0.4};
    const double offsets[] = {-0.5, 0, 0.5};
    fp[1].init(Vec2{12, 0}, 3, widths, offsets, 0.01);

    fp[1].elements[0].end_type = EndType::HalfWidth;
    fp[1].elements[0].join_type = JoinType::Bevel;

    fp[1].elements[1].end_type = EndType::Flush;
    fp[1].elements[1].join_type = JoinType::Miter;

    fp[1].elements[2].end_type = EndType::Round;
    fp[1].elements[2].join_type = JoinType::Round;

    Vec2 points2[] = {{8, 0}, {8, 3}, {10, 2}};
    fp[1].segment({.count = COUNT(points2), .items = points2}, NULL, NULL, false);

    fp[1].arc(2, 2, -M_PI / 2, M_PI / 2, 0, NULL, NULL);
    fp[1].arc(1, 1, M_PI / 2, 1.5 * M_PI, 0, NULL, NULL);

    out_cell.flexpath_array.append(fp + 1);
}

void example_flexpath2(Cell& out_cell) {
    Vec2 points[] = {{0, 10}, {20, 0}, {18, 15}, {8, 15}};

    FlexPath* flexpath = (FlexPath*)allocate_clear(2 * sizeof(FlexPath));

    for (FlexPath* fp = flexpath; fp < flexpath + 2; fp++) {
        fp->simple_path = true;
        fp->init(Vec2{0, 0}, 1, 0.5, 0, 0.01);
        fp->segment({.count = COUNT(points), .items = points}, NULL, NULL, false);
        out_cell.flexpath_array.append(fp);
    }

    flexpath[0].elements[0].bend_type = BendType::Circular;
    flexpath[0].elements[0].bend_radius = 5;
    flexpath[1].elements[0].layer = 1;
}

void example_flexpath3(Cell& out_cell) {
    double widths[] = {0.5, 0.5};
    double offsets[] = {-0.5, 0.5};
    FlexPath* fp = (FlexPath*)allocate_clear(sizeof(FlexPath));
    fp->init(Vec2{0, 0}, 2, widths, offsets, 0.01);

    fp->horizontal(2, NULL, NULL, false);

    widths[0] = 0.8;
    widths[1] = 0.8;
    offsets[0] = -0.9;
    offsets[1] = 0.9;
    fp->horizontal(4, widths, offsets, false);

    fp->horizontal(6, NULL, NULL, false);

    out_cell.flexpath_array.append(fp);
}

int main(int argc, char* argv[]) {
    char lib_name[] = "Paths";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    char flexpath1_cell_name[] = "FlexPath 1";
    Cell flexpath1_cell = {.name = flexpath1_cell_name};
    example_flexpath1(flexpath1_cell);
    lib.cell_array.append(&flexpath1_cell);

    char flexpath2_cell_name[] = "FlexPath 2";
    Cell flexpath2_cell = {.name = flexpath2_cell_name};
    example_flexpath2(flexpath2_cell);
    lib.cell_array.append(&flexpath2_cell);

    char flexpath3_cell_name[] = "FlexPath 3";
    Cell flexpath3_cell = {.name = flexpath3_cell_name};
    example_flexpath3(flexpath3_cell);
    lib.cell_array.append(&flexpath3_cell);

    lib.write_gds("flexpaths.gds", 0, NULL);

    return 0;
}
