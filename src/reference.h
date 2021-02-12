/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_REFERENCE
#define GDSTK_HEADER_REFERENCE

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include "flexpath.h"
#include "label.h"
#include "polygon.h"
#include "property.h"
#include "robustpath.h"
#include "vec.h"

namespace gdstk {

struct Cell;
struct RawCell;

enum struct ReferenceType { Cell = 0, RawCell, Name };

struct Reference {
    ReferenceType type;
    union {
        Cell* cell;
        RawCell* rawcell;
        char* name;
    };
    Vec2 origin;
    double rotation;  // in RADIANS
    double magnification;
    bool x_reflection;
    Repetition repetition;
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print() const;
    void clear();
    void copy_from(const Reference& reference);
    void bounding_box(Vec2& min, Vec2& max) const;
    // Arguments are in order of application to the coordinates
    void transform(double mag, bool x_refl, double rot, const Vec2 orig);
    void apply_repetition(Array<Reference*>& result);

    void polygons(bool apply_repetitions, bool include_paths, int64_t depth,
                  Array<Polygon*>& result) const;
    void flexpaths(bool apply_repetitions, int64_t depth, Array<FlexPath*>& result) const;
    void robustpaths(bool apply_repetitions, int64_t depth, Array<RobustPath*>& result) const;
    void labels(bool apply_repetitions, int64_t depth, Array<Label*>& result) const;
    void to_gds(FILE* out, double scaling) const;
    void to_svg(FILE* out, double scaling) const;
};

}  // namespace gdstk

#endif
