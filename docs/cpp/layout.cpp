/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include <stdio.h>

#include <gdstk/gdstk.hpp>

using namespace gdstk;

// We redefine the grating function from pcell.cpp here instead of including
// the file because it contains a main function already.  In practice, the
// library source would only contain the relevant functions, but no main.
Cell* grating(double period, double fill_frac, double length, double width, Tag tag,
              const char* name) {
    double x = width / 2;
    double w = period * fill_frac;
    int64_t num = (int64_t)(length / period);

    Cell* result = (Cell*)allocate_clear(sizeof(Cell));
    result->name = copy_string(name, NULL);
    result->polygon_array.ensure_slots(num);
    for (int64_t i = 0; i < num; i++) {
        double y = i * period;
        Polygon* rect = (Polygon*)allocate(sizeof(Polygon));
        *rect = rectangle(Vec2{-x, y}, Vec2{x, y + w}, tag);
        result->polygon_array.append(rect);
    }

    return result;
}

int main(int argc, char* argv[]) {
    double unit = 0;
    double precision = 0;

    ErrorCode error_code = gds_units("photonics.gds", unit, precision);
    if (error_code != ErrorCode::NoError) exit(EXIT_FAILURE);

    printf("Using unit = %.3g, precision = %.3g\n", unit, precision);

    Map<RawCell*> pdk = read_rawcells("photonics.gds", NULL);

    Cell* dev_cell = (Cell*)allocate_clear(sizeof(Cell));
    dev_cell->name = copy_string("Device", NULL);

    Reference* mzi_ref = (Reference*)allocate_clear(sizeof(Reference));
    mzi_ref->init(pdk.get("MZI"));
    mzi_ref->origin = Vec2{-40, 0};
    dev_cell->reference_array.append(mzi_ref);

    Cell* grating_cell = grating(0.62, 0.5, 20, 25, make_tag(2, 0), "Grating");

    // We set type of these references to Regular so that we can apply the
    // rotation to the translation vectors v1 and v2 of the repetition. This
    // way, the GDSII writer will create and AREF element instead of multiple
    // SREFs. If x_reflection was set to true, that would also have to be
    // applied to v2 for an AREF to be created.
    Reference* grating_ref1 = (Reference*)allocate_clear(sizeof(Reference));
    grating_ref1->init(grating_cell);
    grating_ref1->origin = Vec2{-200, -150};
    grating_ref1->rotation = M_PI / 2;
    grating_ref1->repetition.type = RepetitionType::Regular;
    grating_ref1->repetition.columns = 2;
    grating_ref1->repetition.rows = 1;
    grating_ref1->repetition.v1 = Vec2{0, 300};
    grating_ref1->repetition.v2 = Vec2{1, 0};
    dev_cell->reference_array.append(grating_ref1);

    Reference* grating_ref2 = (Reference*)allocate_clear(sizeof(Reference));
    grating_ref2->init(grating_cell);
    grating_ref2->origin = Vec2{200, 150};
    grating_ref2->rotation = -M_PI / 2;
    grating_ref2->repetition.type = RepetitionType::Regular;
    grating_ref2->repetition.columns = 2;
    grating_ref2->repetition.rows = 1;
    grating_ref2->repetition.v1 = Vec2{0, -300};
    grating_ref2->repetition.v2 = Vec2{1, 0};

    dev_cell->reference_array.append(grating_ref2);

    FlexPath* waveguide = (FlexPath*)allocate_clear(sizeof(FlexPath));
    waveguide->init(Vec2{-220, -150}, 1, 20, 0, 0.01, make_tag(1, 0));
    waveguide->elements[0].bend_type = BendType::Circular;
    waveguide->elements[0].bend_radius = 15;

    waveguide->segment(Vec2{20, 0}, NULL, NULL, true);

    const double w = 0.5;
    waveguide->segment(Vec2{-100, -150}, &w, NULL, false);

    Vec2 p[] = {{-70, -150}, {-70, -1}, {-40, -1}};
    waveguide->segment({.capacity = 0, .count = COUNT(p), .items = p}, NULL, NULL, false);

    Cell* wg_cell = (Cell*)allocate_clear(sizeof(Cell));
    wg_cell->name = copy_string("Waveguide", NULL);
    wg_cell->flexpath_array.append(waveguide);

    for (uint64_t i = 0; i < 4; i++) {
        Reference* wg_ref = (Reference*)allocate_clear(sizeof(Reference));
        wg_ref->init(wg_cell);
        if (i == 1) {
            wg_ref->x_reflection = true;
        } else if (i == 2) {
            wg_ref->rotation = M_PI;
        } else if (i == 3) {
            wg_ref->rotation = M_PI;
            wg_ref->x_reflection = true;
        }
        dev_cell->reference_array.append(wg_ref);
    }

    Cell* main_cell = (Cell*)allocate_clear(sizeof(Cell));
    main_cell->name = copy_string("Main", NULL);

    for (uint64_t i = 0; i < 2; i++) {
        Reference* dev_ref = (Reference*)allocate_clear(sizeof(Reference));
        dev_ref->init(dev_cell);
        dev_ref->origin = i == 0 ? Vec2{250, 250} : Vec2{250, 750};
        main_cell->reference_array.append(dev_ref);
    }

    Reference* align_ref = (Reference*)allocate_clear(sizeof(Reference));
    align_ref->init(pdk.get("Alignment Mark"));
    align_ref->repetition = {RepetitionType::Rectangular, 2, 3, Vec2{500, 500}};
    main_cell->reference_array.append(align_ref);

    Library lib = {};
    lib.init("library", unit, precision);
    lib.cell_array.append(main_cell);

    Map<Cell*> dependencies = {};
    main_cell->get_dependencies(true, dependencies);
    dependencies.to_array(lib.cell_array);
    dependencies.clear();

    Map<RawCell*> raw_dependencies = {};
    main_cell->get_raw_dependencies(true, raw_dependencies);
    raw_dependencies.to_array(lib.rawcell_array);
    raw_dependencies.clear();

    lib.write_gds("layout.gds", 0, NULL);

    lib.free_all();

    // Library::free_all does not free RawCells
    for (MapItem<RawCell*>* item = pdk.next(NULL); item; item = pdk.next(item)) {
        item->value->clear();
        free_allocation(item->value);
    }
    pdk.clear();

    return 0;
}
