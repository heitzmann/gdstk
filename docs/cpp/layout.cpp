/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include "gdstk.h"

using namespace gdstk;

// We redefine the grating function from pcell.cpp here instead of including
// the file because it contains a main function already.  In practice, the
// library source would only contain the relevant functions, but no main.
Cell grating(double period, double fill_frac, double length, double width, int16_t layer,
             int16_t datatype, char* name) {
    double x = width / 2;
    double w = period * fill_frac;
    int64_t num = (int64_t)(length / period);

    Cell result = {.name = name};
    result.polygon_array.ensure_slots(num);
    Polygon* rect = (Polygon*)allocate(num * sizeof(Polygon));
    for (int64_t i = 0; i < num; i++) {
        double y = i * period;
        *rect = rectangle(Vec2{-x, y}, Vec2{x, y + w}, layer, datatype);
        result.polygon_array.append(rect++);
    }

    return result;
}

int main(int argc, char* argv[]) {
    double unit = 0;
    double precision = 0;

    gds_units("photonics.gds", unit, precision);
    if (unit == 0) {
        // File not found.
        exit(EXIT_FAILURE);
    }

    printf("Using unit = %.3g, precision = %.3g\n", unit, precision);

    Map<RawCell*> pdk = read_rawcells("photonics.gds");

    char dev_cell_name[] = "Device";
    Cell dev_cell = {.name = dev_cell_name};

    Reference mzi_ref = {
        .type = ReferenceType::RawCell,
        .rawcell = pdk.get("MZI"),
        .origin = Vec2{-40, 0},
        .magnification = 1,
    };
    dev_cell.reference_array.append(&mzi_ref);

    char grating_cell_name[] = "Grating";
    Cell grating_cell = grating(0.62, 0.5, 20, 25, 2, 0, grating_cell_name);

    // We set type of these references to Regular so that we can apply the
    // rotation to the translation vectors v1 and v2 of the repetition. This
    // way, the GDSII writer will create and AREF element instead of multiple
    // SREFs. If x_reflection was set to true, that would also have to be
    // applied to v2 for an AREF to be created.
    Reference grating_ref1 = {
        .type = ReferenceType::Cell,
        .cell = &grating_cell,
        .origin = Vec2{-200, -150},
        .rotation = M_PI / 2,
        .magnification = 1,
        .repetition = {RepetitionType::Regular, 2, 1, Vec2{0, 300}, Vec2{1, 0}},
    };
    dev_cell.reference_array.append(&grating_ref1);

    Reference grating_ref2 = {
        .type = ReferenceType::Cell,
        .cell = &grating_cell,
        .origin = Vec2{200, 150},
        .rotation = -M_PI / 2,
        .magnification = 1,
        .repetition = {RepetitionType::Regular, 2, 1, Vec2{0, -300}, Vec2{1, 0}},
    };
    dev_cell.reference_array.append(&grating_ref2);

    FlexPathElement element = {.layer = 1, .bend_type = BendType::Circular, .bend_radius = 15};
    FlexPath waveguide = {.elements = &element, .num_elements = 1};
    waveguide.init(Vec2{-220, -150}, 20, 0, 0.01);

    waveguide.segment(Vec2{20, 0}, NULL, NULL, true);

    const double w = 0.5;
    waveguide.segment(Vec2{-100, -150}, &w, NULL, false);

    Vec2 p[] = {{-70, -150}, {-70, -1}, {-40, -1}};
    waveguide.segment({.count = COUNT(p), .items = p}, NULL, NULL, false);

    char wg_cell_name[] = "Waveguide";
    Cell wg_cell = {.name = wg_cell_name};
    wg_cell.flexpath_array.append(&waveguide);

    Reference wg_ref[] = {
        {
            .type = ReferenceType::Cell,
            .cell = &wg_cell,
            .magnification = 1,
        },
        {
            .type = ReferenceType::Cell,
            .cell = &wg_cell,
            .magnification = 1,
            .x_reflection = true,
        },
        {
            .type = ReferenceType::Cell,
            .cell = &wg_cell,
            .rotation = M_PI,
            .magnification = 1,
        },
        {
            .type = ReferenceType::Cell,
            .cell = &wg_cell,
            .rotation = M_PI,
            .magnification = 1,
            .x_reflection = true,
        },
    };
    Reference* wg_ref_p[] = {wg_ref, wg_ref + 1, wg_ref + 2, wg_ref + 3};
    dev_cell.reference_array.extend({.count = 4, .items = wg_ref_p});

    char main_cell_name[] = "Main";
    Cell main_cell = {.name = main_cell_name};

    Reference dev_ref[] = {
        {
            .type = ReferenceType::Cell,
            .cell = &dev_cell,
            .origin = Vec2{250, 250},
            .magnification = 1,
        },
        {
            .type = ReferenceType::Cell,
            .cell = &dev_cell,
            .origin = Vec2{250, 750},
            .magnification = 1,
        },
        {
            .type = ReferenceType::RawCell,
            .rawcell = pdk.get("Alignment Mark"),
            .magnification = 1,
            .repetition = {RepetitionType::Rectangular, 2, 3, Vec2{500, 500}},
        },
    };
    Reference* dev_ref_p[] = {dev_ref, dev_ref + 1, dev_ref + 2};
    main_cell.reference_array.extend({.count = 3, .items = dev_ref_p});

    char lib_name[] = "library";
    Library lib = {.name = lib_name, .unit = unit, .precision = precision};
    lib.cell_array.append(&main_cell);

    Map<Cell*> dependencies = {0};
    main_cell.get_dependencies(true, dependencies);
    dependencies.to_array(lib.cell_array);

    Map<RawCell*> raw_dependencies = {0};
    main_cell.get_raw_dependencies(true, raw_dependencies);
    raw_dependencies.to_array(lib.rawcell_array);

    lib.write_gds("layout.gds", 0, NULL);

    return 0;
}
