/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __CELL_H__
#define __CELL_H__

#include <cstdint>
#include <cstdio>
#include <ctime>

#include "array.h"
#include "flexpath.h"
#include "label.h"
#include "polygon.h"
#include "reference.h"
#include "robustpath.h"
#include "style.h"

namespace gdstk {

struct Cell {
    char* name;
    Array<Polygon*> polygon_array;
    Array<Reference*> reference_array;
    Array<FlexPath*> flexpath_array;
    Array<RobustPath*> robustpath_array;
    Array<Label*> label_array;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;

    void clear();

    void bounding_box(Vec2& min, Vec2& max) const;

    void copy_from(const Cell& cell, const char* new_name, bool deep_copy);

    Array<Polygon*> get_polygons(bool include_paths, int64_t depth) const;
    Array<FlexPath*> get_flexpaths(int64_t depth) const;
    Array<RobustPath*> get_robustpaths(int64_t depth) const;
    Array<Label*> get_labels(int64_t depth) const;

    void get_dependencies(bool recursive, Array<Cell*>& result) const;
    void get_raw_dependencies(bool recursive, Array<RawCell*>& result) const;

    // Return old references (caller is responsible for clearing)
    Array<Reference*> flatten();

    void to_gds(FILE* out, double scaling, int64_t max_points, double precision,
                const std::tm* timestamp) const;

    void to_svg(FILE* out, double scaling, const char* attributes) const;

    void write_svg(FILE* out, double scaling, StyleMap& style, StyleMap& label_style,
                   const char* background, double pad, bool pad_as_percentage) const;
};

}  // namespace gdstk

#endif
