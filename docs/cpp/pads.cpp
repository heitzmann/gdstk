/*
Copyright 2022 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <math.h>
#include <stdio.h>

#include <gdstk/gdstk.hpp>

using namespace gdstk;

struct FilletedPadData {
    double pad_radius;
    double fillet_radius;
    double tolerance;
};

Array<Vec2> filleted_pad(const Vec2 p0, const Vec2 v0, const Vec2 p1, const Vec2 v1, void* data) {
    FilletedPadData* pad_data = (FilletedPadData*)data;
    double pad_radius = pad_data->pad_radius;
    double fillet_radius = pad_data->fillet_radius;

    Vec2 dp = p0 - p1;
    double half_trace_width = 0.5 * sqrt(dp.x * dp.x + dp.y * dp.y);
    double a = half_trace_width + fillet_radius;
    double c = pad_radius + fillet_radius;
    double b = sqrt(c * c + a * a);
    double alpha = acos(a / c);
    double gamma = atan2(v0.y, v0.x) + 0.5 * M_PI;

    Curve curve = {};
    curve.init(p0 - b * v0, pad_data->tolerance);
    if (fillet_radius > 0) {
        curve.arc(fillet_radius, fillet_radius, gamma, gamma - alpha, 0);
    }
    curve.arc(pad_radius, pad_radius, gamma - M_PI - alpha, gamma + alpha, 0);
    if (fillet_radius > 0) {
        curve.arc(fillet_radius, fillet_radius, gamma - M_PI + alpha, gamma - M_PI, 0);
    }

    return curve.point_array;
}

int main(int argc, char* argv[]) {
    Library lib = {};
    lib.init("library", 1e-6, 1e-9);

    Cell* main_cell = (Cell*)allocate_clear(sizeof(Cell));
    main_cell->name = copy_string("Main", NULL);
    lib.cell_array.append(main_cell);

    FlexPath* bus = (FlexPath*)allocate_clear(sizeof(FlexPath));
    bus->init(Vec2{0, 0}, 4, 3, 15, 0.01, 0);
    FilletedPadData data = {5, 3, 0.01};
    for (int64_t i = 0; i < 4; i++) {
        bus->elements[i].join_type = JoinType::Round;
        bus->elements[i].end_type = EndType::Function;
        bus->elements[i].end_function = filleted_pad;
        bus->elements[i].end_function_data = (void*)&data;
    }
    bus->segment(Vec2{10, 5}, NULL, NULL, false);
    double offsets1[] = {-9, -3, 3, 9};
    bus->segment(Vec2{20, 10}, NULL, offsets1, false);
    Vec2 points[] = {{40, 20}, {40, 50}, {80, 50}};
    const Array<Vec2> point_array = {.capacity = 0, .count = COUNT(points), .items = points};
    bus->segment(point_array, NULL, NULL, false);
    double offsets2[] = {-18, -6, 6, 18};
    bus->segment(Vec2{100, 50}, NULL, offsets2, false);

    main_cell->flexpath_array.append(bus);

    lib.write_gds("pads.gds", 0, NULL);

    lib.free_all();
    return 0;
}
