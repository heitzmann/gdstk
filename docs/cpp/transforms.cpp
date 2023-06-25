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
    char lib_name[] = "library";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    int64_t n = 3;    // Unit cells around defect
    double d = 0.2;   // Unit cell size
    double r = 0.05;  // Circle radius
    double s = 1.5;   // Scaling factor

    char unit_cell_name[] = "Unit Cell";
    Cell unit_cell = {.name = unit_cell_name};
    lib.cell_array.append(&unit_cell);

    Polygon circle = ellipse(Vec2{0, 0}, r, r, 0, 0, 0, 0, 1e-3, 0);
    unit_cell.polygon_array.append(&circle);

    char resonator_cell_name[] = "Resonator";
    Cell resonator_cell = {.name = resonator_cell_name};
    lib.cell_array.append(&resonator_cell);

    Reference unit_refs[] = {
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .origin = Vec2{-n * d, -n * d},
            .magnification = 1,
            .repetition = {RepetitionType::Rectangular, (uint16_t)(2 * n + 1), (uint16_t)n,
                           Vec2{d, d}},
        },
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .origin = Vec2{-n * d, d},
            .magnification = 1,
            .repetition = {RepetitionType::Rectangular, (uint16_t)(2 * n + 1), (uint16_t)n,
                           Vec2{d, d}},
        },
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .origin = Vec2{-n * d, 0},
            .magnification = 1,
            .repetition = {RepetitionType::Rectangular, (uint16_t)n, 1, Vec2{d, d}},
        },
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .origin = Vec2{d, 0},
            .magnification = 1,
            .repetition = {RepetitionType::Rectangular, (uint16_t)n, 1, Vec2{d, d}},
        },
    };
    Reference* unit_refs_p[] = {unit_refs, unit_refs + 1, unit_refs + 2, unit_refs + 3};
    resonator_cell.reference_array.extend({.capacity = 0, .count = 4, .items = unit_refs_p});

    Polygon rect = rectangle(Vec2{-r / 2, -r / 2}, Vec2{r / 2, r / 2}, 0);
    resonator_cell.polygon_array.append(&rect);

    FlexPath path = {
        .simple_path = true,
        .scale_width = false,
    };
    path.init(Vec2{-n * d, 0}, 1, r, 0, 0.01, make_tag(1, 0));
    path.elements[0].end_type = EndType::Extended;
    path.elements[0].end_extensions = Vec2{r, r};
    path.segment(Vec2{n * d, 0}, NULL, NULL, false);
    resonator_cell.flexpath_array.append(&path);

    char main_cell_name[] = "Main";
    Cell main_cell = {.name = main_cell_name};
    lib.cell_array.append(&main_cell);

    Cell resonator_cell_copy = {};
    lib.cell_array.append(&resonator_cell_copy);
    resonator_cell_copy.copy_from(resonator_cell, "Resonator Copy", true);
    for (int64_t i = 0; i < resonator_cell_copy.polygon_array.count; i++) {
        Polygon* p = resonator_cell_copy.polygon_array[i];
        p->scale(Vec2{s, s}, Vec2{0, 0});
    }
    for (int64_t i = 0; i < resonator_cell_copy.flexpath_array.count; i++) {
        FlexPath* fp = resonator_cell_copy.flexpath_array[i];
        fp->scale(s, Vec2{0, 0});
    }
    for (int64_t i = 0; i < resonator_cell_copy.robustpath_array.count; i++) {
        RobustPath* rp = resonator_cell_copy.robustpath_array[i];
        rp->scale(s, Vec2{0, 0});
    }
    for (int64_t i = 0; i < resonator_cell_copy.reference_array.count; i++) {
        Reference* ref = resonator_cell_copy.reference_array[i];
        ref->transform(s, false, 0, Vec2{0, 0});
        ref->repetition.transform(s, false, 0);
    }

    Reference resonator_refs[] = {
        {
            .type = ReferenceType::Cell,
            .cell = &resonator_cell,
            .magnification = 1,
        },
        {
            .type = ReferenceType::Cell,
            .cell = &resonator_cell,
            .origin = Vec2{0, (1 + s) * (n + 1) * d},
            .magnification = s,
        },
        {
            .type = ReferenceType::Cell,
            .cell = &resonator_cell_copy,
            .origin = Vec2{0, (1 + 3 * s) * (n + 1) * d},
            .magnification = 1,
        },
    };
    Reference* resonator_refs_p[] = {resonator_refs, resonator_refs + 1, resonator_refs + 2};
    main_cell.reference_array.extend({.capacity = 0, .count = 3, .items = resonator_refs_p});

    Array<Polygon*> all_text = {};
    text("Original", d, Vec2{(n + 1) * d, -d / 2}, false, 0, all_text);
    text("Reference\nscaling", d, Vec2{s * (n + 1) * d, (1 + s) * (n + 1) * d}, false, 0, all_text);
    text("Cell copy\nscaling", d, Vec2{s * (n + 1) * d, (1 + 3 * s) * (n + 1) * d}, false, 0,
         all_text);
    main_cell.polygon_array.extend(all_text);

    lib.write_gds("transforms.gds", 0, NULL);

    for (uint16_t i = 0; i < all_text.count; i++) {
        all_text[i]->clear();
        free_allocation(all_text[i]);
    }
    all_text.clear();
    circle.clear();
    rect.clear();
    path.clear();
    unit_cell.polygon_array.clear();
    resonator_cell.reference_array.clear();
    resonator_cell.polygon_array.clear();
    resonator_cell.flexpath_array.clear();
    resonator_cell_copy.free_all();
    main_cell.reference_array.clear();
    main_cell.polygon_array.clear();
    lib.cell_array.clear();
    return 0;
}
