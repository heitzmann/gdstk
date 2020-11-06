/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <cstdio>

#include "gdstk.h"

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

    Polygon circle = ellipse(Vec2{0, 0}, r, r, 0, 0, 0, 0, 1e-3, 0, 0);
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
            .columns = (uint16_t)(2 * n + 1),
            .rows = (uint16_t)n,
            .spacing = Vec2{d, d},
        },
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .origin = Vec2{-n * d, d},
            .magnification = 1,
            .columns = (uint16_t)(2 * n + 1),
            .rows = (uint16_t)n,
            .spacing = Vec2{d, d},
        },
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .origin = Vec2{-n * d, 0},
            .magnification = 1,
            .columns = (uint16_t)n,
            .rows = 1,
            .spacing = Vec2{d, d},
        },
        {
            .type = ReferenceType::Cell,
            .cell = &unit_cell,
            .origin = Vec2{d, 0},
            .magnification = 1,
            .columns = (uint16_t)n,
            .rows = 1,
            .spacing = Vec2{d, d},
        },
    };
    Reference* unit_refs_p[] = {unit_refs, unit_refs + 1, unit_refs + 2, unit_refs + 3};
    resonator_cell.reference_array.extend({.size = 4, .items = unit_refs_p});

    Polygon rect = rectangle(Vec2{-r / 2, -r / 2}, Vec2{r / 2, r / 2}, 0, 0);
    resonator_cell.polygon_array.append(&rect);

    FlexPathElement element = {
        .layer = 1,
        .end_type = EndType::Extended,
        .end_extensions = Vec2{r, r},
    };
    FlexPath path = {
        .num_elements = 1,
        .elements = &element,
        .gdsii_path = true,
        .scale_width = false,
    };
    path.init(Vec2{-n * d, 0}, r, 0, 0.01);
    path.segment(Vec2{n * d, 0}, NULL, NULL, false);
    resonator_cell.flexpath_array.append(&path);

    char main_cell_name[] = "Main";
    Cell main_cell = {.name = main_cell_name};
    lib.cell_array.append(&main_cell);

    Cell resonator_cell_copy = {0};
    lib.cell_array.append(&resonator_cell_copy);
    resonator_cell_copy.copy_from(resonator_cell, "Resonator Copy", true);
    for (int64_t i = 0; i < resonator_cell_copy.polygon_array.size; i++) {
        Polygon* p = resonator_cell_copy.polygon_array[i];
        p->scale(Vec2{s, s}, Vec2{0, 0});
    }
    for (int64_t i = 0; i < resonator_cell_copy.flexpath_array.size; i++) {
        FlexPath* fp = resonator_cell_copy.flexpath_array[i];
        fp->scale(s, Vec2{0, 0});
    }
    for (int64_t i = 0; i < resonator_cell_copy.robustpath_array.size; i++) {
        RobustPath* rp = resonator_cell_copy.robustpath_array[i];
        rp->scale(s, Vec2{0, 0});
    }
    for (int64_t i = 0; i < resonator_cell_copy.reference_array.size; i++) {
        Reference* ref = resonator_cell_copy.reference_array[i];
        ref->transform(s, Vec2{0, 0}, false, 0, Vec2{0, 0});
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
    main_cell.reference_array.extend({.size = 3, .items = resonator_refs_p});

    Array<Polygon*> all_text = {0};
    text("Original", d, Vec2{(n + 1) * d, -d / 2}, false, 0, 0, all_text);
    text("Reference\nscaling", d, Vec2{s * (n + 1) * d, (1 + s) * (n + 1) * d}, false, 0, 0,
         all_text);
    text("Cell copy\nscaling", d, Vec2{s * (n + 1) * d, (1 + 3 * s) * (n + 1) * d}, false, 0, 0,
         all_text);
    main_cell.polygon_array.extend(all_text);

    lib.write_gds("transforms.gds", 0, NULL);
    return 0;
}
