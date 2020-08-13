/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __REFERENCE_H__
#define __REFERENCE_H__

#include <cstdint>
#include <cstdio>

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
    uint16_t columns;
    uint16_t rows;
    Vec2 spacing;
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print() const;
    void clear();
    void copy_from(const Reference& reference);
    void bounding_box(Vec2& min, Vec2& max) const;
    void transform(double mag, const Vec2 trans, bool x_refl, double rot, const Vec2 orig);

    Array<Polygon*> polygons(bool include_paths, int64_t depth) const;
    Array<FlexPath*> flexpaths(int64_t depth) const;
    Array<RobustPath*> robustpaths(int64_t depth) const;
    Array<Label*> labels(int64_t depth) const;
    void to_gds(FILE* out, double scaling) const;
    void to_svg(FILE* out, double scaling) const;
};

}  // namespace gdstk

#endif
