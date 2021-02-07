/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include "gdstk.h"

using namespace gdstk;

double parametric1(double u, void*) { return 2 + 16 * u * (1 - u); }

double parametric2(double u, void*) { return 8 * u * (1 - u) * cos(12 * M_PI * u); }

double parametric3(double u, void*) { return -1 - 8 * u * (1 - u); }

double parametric4(double u, void*) { return 1 + 8 * u * (1 - u); }

double parametric5(double u, void*) { return -0.25 * cos(24 * M_PI * u); }

double parametric6(double u, void*) { return 0.25 * cos(24 * M_PI * u); }

Vec2 parametric7(double u, void*) { return Vec2{4 * sin(6 * M_PI * u), 45 * u}; }

int main(int argc, char* argv[]) {
    RobustPathElement rpe[] = {
        {.layer = 1, .end_width = 2, .end_offset = 0, .end_type = EndType::HalfWidth},
        {.layer = 0, .end_width = 0.5, .end_offset = 0, .end_type = EndType::Round},
        {.layer = 2, .end_width = 1, .end_offset = -1, .end_type = EndType::Flush},
        {.layer = 2, .end_width = 1, .end_offset = 1, .end_type = EndType::Flush},
    };
    RobustPath rp = {
        .end_point = Vec2{0, 50},
        .elements = rpe,
        .num_elements = 4,
        .tolerance = 0.01,
        .max_evals = 1000,
        .width_scale = 1,
        .offset_scale = 1,
        .trafo = {1, 0, 0, 0, 1, 0},
        .scale_width = true,
    };

    rp.segment(Vec2{0, 45}, NULL, NULL, false);

    Interpolation width1[] = {
        {.type = InterpolationType::Parametric},
        {.type = InterpolationType::Constant},
        {.type = InterpolationType::Constant},
        {.type = InterpolationType::Constant},
    };
    width1[0].function = parametric1;
    for (int i = 1; i < COUNT(width1); i++) {
        width1[i].value = rp.elements[i].end_width;
    }
    Interpolation offset1[] = {
        {.type = InterpolationType::Constant},
        {.type = InterpolationType::Parametric},
        {.type = InterpolationType::Parametric},
        {.type = InterpolationType::Parametric},
    };
    offset1[0].value = 0;
    offset1[1].function = parametric2;
    offset1[2].function = parametric3;
    offset1[3].function = parametric4;
    rp.segment(Vec2{0, 5}, width1, offset1, false);

    rp.segment(Vec2{0, 0}, NULL, NULL, false);

    Vec2 point = {15, 5};
    double angles[] = {0, M_PI / 2};
    bool angle_constraints[] = {true, true};
    Vec2 tension[] = {{1, 1}, {1, 1}};
    Interpolation width2[] = {
        {.type = InterpolationType::Linear},
        {.type = InterpolationType::Linear},
        {.type = InterpolationType::Linear},
        {.type = InterpolationType::Linear},
    };
    Interpolation offset2[] = {
        {.type = InterpolationType::Linear},
        {.type = InterpolationType::Linear},
        {.type = InterpolationType::Linear},
        {.type = InterpolationType::Linear},
    };
    for (int i = 0; i < COUNT(width2); i++) {
        width2[i].initial_value = rp.elements[i].end_width;
        width2[i].final_value = 0.5;
        offset2[i].initial_value = rp.elements[i].end_offset;
    }
    offset2[0].final_value = -0.25;
    offset2[1].final_value = 0.25;
    offset2[2].final_value = -0.75;
    offset2[3].final_value = 0.75;
    rp.interpolation({.count = 1, .items = &point}, angles, angle_constraints, tension, 1, 1, false,
                     width2, offset2, false);

    Interpolation offset3[] = {
        {.type = InterpolationType::Parametric},
        {.type = InterpolationType::Parametric},
        {.type = InterpolationType::Constant},
        {.type = InterpolationType::Constant},
    };
    offset3[0].function = parametric5;
    offset3[1].function = parametric6;
    offset3[2].value = -0.75;
    offset3[3].value = 0.75;
    rp.parametric(parametric7, NULL, NULL, NULL, NULL, offset3, true);

    char robustpath_cell_name[] = "RobustPath";
    Cell robustpath_cell = {.name = robustpath_cell_name};
    robustpath_cell.robustpath_array.append(&rp);

    char lib_name[] = "Paths";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};
    lib.cell_array.append(&robustpath_cell);

    lib.write_gds("robustpaths.gds", 0, NULL);

    return 0;
}
