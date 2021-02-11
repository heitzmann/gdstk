/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include "gdstk.h"

using namespace gdstk;

char alignment_mark_cell_name[] = "Alignment Mark";
char directional_coupler_cell_name[] = "Directional Coupler";
char mach_zehnder_interferometer_cell_name[] = "MZI";

void alignment_mark(Library& lib) {
    Cell* cell = (Cell*)allocate_clear(sizeof(Cell));
    cell->name = alignment_mark_cell_name;
    lib.cell_array.append(cell);

    Polygon* cross_ = (Polygon*)allocate(sizeof(Polygon));
    *cross_ = cross(Vec2{0, 0}, 50, 3, 1, 0);
    cell->polygon_array.append(cross_);
}

void directional_coupler(Library& lib) {
    Cell* cell = (Cell*)allocate_clear(sizeof(Cell));
    cell->name = directional_coupler_cell_name;
    lib.cell_array.append(cell);

    RobustPath* path = (RobustPath*)allocate_clear(sizeof(RobustPath));
    path->num_elements = 2;
    path->tolerance = 0.01;
    path->max_evals = 1000;
    path->width_scale = 1;
    path->offset_scale = 1;
    path->trafo[0] = 1;
    path->trafo[4] = 1;
    path->simple_path = true;

    path->elements = (RobustPathElement*)allocate_clear(2 * sizeof(RobustPathElement));
    path->elements[0].layer = 1;
    path->elements[0].end_width = 0.5;
    path->elements[0].end_offset = -1;
    path->elements[1].layer = 1;
    path->elements[1].end_width = 0.5;
    path->elements[1].end_offset = 1;

    Interpolation* offset = (Interpolation*)allocate_clear(4 * sizeof(Interpolation));
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
}

void mach_zenhder_interferometer(Library& lib) {
    Cell* cell = (Cell*)allocate_clear(sizeof(Cell));
    cell->name = mach_zehnder_interferometer_cell_name;
    lib.cell_array.append(cell);

    Reference* ref = (Reference*)allocate_clear(2 * sizeof(Reference));
    ref[0].type = ReferenceType::Name;
    ref[0].name = directional_coupler_cell_name;
    ref[0].magnification = 1;
    ref[1].type = ReferenceType::Name;
    ref[1].name = directional_coupler_cell_name;
    ref[1].origin.x = 75;
    ref[1].magnification = 1;
    cell->reference_array.append(ref);
    cell->reference_array.append(ref + 1);

    const Vec2 starting_points[] = {{5, 1}, {5, -1}, {25, 20}, {25, -20}};
    FlexPath* path = (FlexPath*)allocate_clear(4 * sizeof(FlexPath));
    FlexPathElement* element = (FlexPathElement*)allocate_clear(4 * sizeof(FlexPathElement));
    for (int64_t i = 0; i < 4; i++) {
        element[i].layer = i < 2 ? 1 : 10;
        element[i].bend_type = BendType::Circular;
        element[i].bend_radius = 15;
        path[i].num_elements = 1;
        path[i].elements = element + i;
        path[i].simple_path = true;
        path[i].init(starting_points[i], i < 2 ? 0.25 : 1, 0, 0.01);
    }

    Vec2 arm_points[] = {{25, 1}, {25, 40}, {55, 40}, {55, 1}, {75, 1}};
    path[0].segment({.count = COUNT(arm_points), .items = arm_points}, NULL, NULL, false);

    for (int64_t i = 0; i < COUNT(arm_points); i++) arm_points[i].y = -arm_points[i].y;
    path[1].segment({.count = COUNT(arm_points), .items = arm_points}, NULL, NULL, false);

    Vec2 heater_points[] = {{25, 40}, {55, 40}, {55, 20}};
    path[2].segment({.count = COUNT(heater_points), .items = heater_points}, NULL, NULL, false);

    for (int64_t i = 0; i < COUNT(heater_points); i++) heater_points[i].y = -heater_points[i].y;
    path[3].segment({.count = COUNT(heater_points), .items = heater_points}, NULL, NULL, false);

    FlexPath* path_p[] = {path, path + 1, path + 2, path + 3};
    cell->flexpath_array.extend({.count = 4, .items = path_p});
}

int main(int argc, char* argv[]) {
    char lib_name[] = "Photonics";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};
    alignment_mark(lib);
    directional_coupler(lib);
    mach_zenhder_interferometer(lib);
    lib.write_gds("photonics.gds", 0, NULL);
    return 0;
}
