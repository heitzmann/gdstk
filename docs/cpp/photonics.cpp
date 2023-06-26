/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include <gdstk/gdstk.hpp>

using namespace gdstk;

Cell* alignment_mark() {
    Cell* cell = (Cell*)allocate_clear(sizeof(Cell));
    cell->name = copy_string("Alignment Mark", NULL);

    Polygon* cross_ = (Polygon*)allocate(sizeof(Polygon));
    *cross_ = cross(Vec2{0, 0}, 50, 3, make_tag(1, 0));
    cell->polygon_array.append(cross_);

    return cell;
}

Cell* directional_coupler() {
    Cell* cell = (Cell*)allocate_clear(sizeof(Cell));
    cell->name = copy_string("Directional Coupler", NULL);

    double widths[] = {0.5, 0.5};
    double offsets[] = {-1, 1};
    Tag tags[] = {make_tag(1, 0), make_tag(1, 0)};
    RobustPath* path = (RobustPath*)allocate_clear(sizeof(RobustPath));
    path->init(Vec2{0, 0}, 2, widths, offsets, 0.01, 1000, tags);

    Interpolation offset[4];
    offset[0].type = InterpolationType::Smooth;
    offset[0].initial_value = -1;
    offset[0].final_value = -0.3;
    offset[1].type = InterpolationType::Smooth;
    offset[1].initial_value = 1;
    offset[1].final_value = 0.3;
    offset[2].type = InterpolationType::Smooth;
    offset[2].initial_value = -0.3;
    offset[2].final_value = -1;
    offset[3].type = InterpolationType::Smooth;
    offset[3].initial_value = 0.3;
    offset[3].final_value = 1;

    path->segment(Vec2{0.1, 0}, NULL, NULL, true);
    path->segment(Vec2{2.2, 0}, NULL, offset, true);
    path->segment(Vec2{0.4, 0}, NULL, NULL, true);
    path->segment(Vec2{2.2, 0}, NULL, offset + 2, true);
    path->segment(Vec2{0.1, 0}, NULL, NULL, true);
    cell->robustpath_array.append(path);

    return cell;
}

Cell* mach_zenhder_interferometer(Cell* directional_coupler_cell) {
    Cell* cell = (Cell*)allocate_clear(sizeof(Cell));
    cell->name = copy_string("MZI", NULL);

    Reference* ref = (Reference*)allocate_clear(sizeof(Reference));
    ref->init(directional_coupler_cell);
    cell->reference_array.append(ref);

    ref = (Reference*)allocate_clear(sizeof(Reference));
    ref->init(directional_coupler_cell);
    ref->origin.x = 75;
    cell->reference_array.append(ref);

    const Vec2 starting_points[] = {{5, 1}, {5, -1}, {25, 20}, {25, -20}};
    FlexPath* path[4];
    for (int64_t i = 0; i < COUNT(path); i++) {
        path[i] = (FlexPath*)allocate_clear(sizeof(FlexPath));
        path[i]->simple_path = true;
        path[i]->init(starting_points[i], 1, i < 2 ? 0.5 : 2, 0, 0.01, make_tag(i < 2 ? 1 : 10, 0));
        path[i]->elements[0].bend_type = BendType::Circular;
        path[i]->elements[0].bend_radius = 15;
    }

    Vec2 arm_points[] = {{25, 1}, {25, 40}, {55, 40}, {55, 1}, {75, 1}};
    path[0]->segment({.capacity = 0, .count = COUNT(arm_points), .items = arm_points}, NULL, NULL,
                     false);

    for (int64_t i = 0; i < COUNT(arm_points); i++) arm_points[i].y = -arm_points[i].y;
    path[1]->segment({.capacity = 0, .count = COUNT(arm_points), .items = arm_points}, NULL, NULL,
                     false);

    Vec2 heater_points[] = {{25, 40}, {55, 40}, {55, 20}};
    path[2]->segment({.capacity = 0, .count = COUNT(heater_points), .items = heater_points}, NULL,
                     NULL, false);

    for (int64_t i = 0; i < COUNT(heater_points); i++) heater_points[i].y = -heater_points[i].y;
    path[3]->segment({.capacity = 0, .count = COUNT(heater_points), .items = heater_points}, NULL,
                     NULL, false);

    cell->flexpath_array.extend({.capacity = 0, .count = COUNT(path), .items = path});

    return cell;
}

int main(int argc, char* argv[]) {
    Library lib = {};
    lib.init("Photonics", 1e-6, 1e-9);
    lib.cell_array.append(alignment_mark());
    Cell* directional_coupler_cell = directional_coupler();
    lib.cell_array.append(directional_coupler_cell);
    lib.cell_array.append(mach_zenhder_interferometer(directional_coupler_cell));
    lib.write_gds("photonics.gds", 0, NULL);
    lib.free_all();
    return 0;
}
