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
    char lib_name[] = "Text";
    Library lib = {.name = lib_name, .unit = 1e-6, .precision = 1e-9};

    char text_cell_name[] = "Text";
    Cell text_cell = {.name = text_cell_name};
    lib.cell_array.append(&text_cell);

    char label_text[] = "Sample label";
    Label label = {
        .tag = make_tag(0, 2),
        .text = label_text,
        .origin = Vec2{5, 3},
        .magnification = 1,
    };
    text_cell.label_array.append(&label);

    Array<Polygon*> all_text = {};
    text("12345", 2.25, Vec2{0.25, 6}, false, 0, all_text);
    text("ABC", 1.5, Vec2{10.5, 4}, true, 0, all_text);
    text_cell.polygon_array.extend(all_text);

    Polygon rect = rectangle(Vec2{0, 0}, Vec2{10, 6}, make_tag(10, 0));
    text_cell.polygon_array.append(&rect);

    lib.write_gds("text.gds", 0, NULL);

    for (uint64_t i = 0; i < all_text.count; i++) {
        all_text[i]->clear();
        free_allocation(all_text[i]);
    }
    all_text.clear();
    rect.clear();
    text_cell.label_array.clear();
    text_cell.polygon_array.clear();
    lib.cell_array.clear();
    return 0;
}
