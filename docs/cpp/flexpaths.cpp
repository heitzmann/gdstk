/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <cstdio>

#include "gdstk.h"

using namespace gdstk;

void example_flexpath1(Cell& out_cell) {
    FlexPath* fp = (FlexPath*)allocate_clear(2 * sizeof(FlexPath));
    fp[0].gdsii_path = true;
    fp[0].spine.tolerance = 0.01;
    fp[0].spine.append(Vec2{0, 0});
    fp[0].num_elements = 1;
    fp[0].elements = (FlexPathElement*)allocate_clear(sizeof(FlexPathElement));
    fp[0].elements[0].half_width_and_offset.append({0.5, 0});

    Vec2 points1[] = {{3, 0}, {3, 2}, {5, 3}, {3, 4}, {0, 4}};
    const Array<Vec2> point_array1 = {.size = COUNT(points1), .items = points1};
    fp[0].segment(point_array1, NULL, NULL, false);

    out_cell.flexpath_array.append(fp);

    fp[1].spine.tolerance = 0.01;
    fp[1].spine.append(Vec2{12, 0});
    fp[1].num_elements = 3;
    fp[1].elements = (FlexPathElement*)allocate_clear(3 * sizeof(FlexPathElement));

    fp[1].elements[0].half_width_and_offset.append({0.15, -0.5});
    fp[1].elements[0].end_type = EndType::HalfWidth;
    fp[1].elements[0].join_type = JoinType::Bevel;

    fp[1].elements[1].half_width_and_offset.append({0.1, 0});
    fp[1].elements[1].end_type = EndType::Flush;
    fp[1].elements[1].join_type = JoinType::Miter;

    fp[1].elements[2].half_width_and_offset.append({0.2, 0.5});
    fp[1].elements[2].end_type = EndType::Round;
    fp[1].elements[2].join_type = JoinType::Round;

    Vec2 points2[] = {{8, 0}, {8, 3}, {10, 2}};
    const Array<Vec2> point_array2 = {.size = COUNT(points2), .items = points2};
    fp[1].segment(point_array2, NULL, NULL, false);

    fp[1].arc(2, 2, -M_PI / 2, M_PI / 2, 0, NULL, NULL);
    fp[1].arc(1, 1, M_PI / 2, 1.5 * M_PI, 0, NULL, NULL);

    out_cell.flexpath_array.append(fp + 1);
}

void example_flexpath2(Cell& out_cell) {
    Vec2 points[] = {{0, 10}, {20, 0}, {18, 15}, {8, 15}};
    const Array<Vec2> point_array = {.size = COUNT(points), .items = points};

    FlexPath* flexpath = (FlexPath*)allocate_clear(2 * sizeof(FlexPath));

    for (FlexPath* fp = flexpath; fp < flexpath + 2; fp++) {
        fp->gdsii_path = true;
        fp->spine.tolerance = 0.01;
        fp->spine.append(Vec2{0, 0});
        fp->num_elements = 1;
        fp->elements = (FlexPathElement*)allocate_clear(sizeof(FlexPathElement));
        fp->elements[0].half_width_and_offset.append({0.25, 0});

        fp->segment(point_array, NULL, NULL, false);

        out_cell.flexpath_array.append(fp);
    }

    flexpath[0].elements[0].bend_type = BendType::Circular;
    flexpath[0].elements[0].bend_radius = 5;
    flexpath[1].elements[0].layer = 1;
}

void example_flexpath3(Cell& out_cell) {
    FlexPath* fp = (FlexPath*)allocate_clear(sizeof(FlexPath));

    fp->spine.tolerance = 0.01;
    fp->spine.append(Vec2{0, 0});
    fp->num_elements = 2;
    fp->elements = (FlexPathElement*)allocate_clear(2 * sizeof(FlexPathElement));
    fp->elements[0].half_width_and_offset.append({0.25, -0.5});
    fp->elements[1].half_width_and_offset.append({0.25, 0.5});

    double x = 2;
    fp->horizontal(&x, 1, NULL, NULL, false);

    x = 4;
    double width[] = {0.8, 0.8};
    double offset[] = {-0.9, 0.9};
    fp->horizontal(&x, 1, width, offset, false);

    x = 6;
    fp->horizontal(&x, 1, NULL, NULL, false);

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
