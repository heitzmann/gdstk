/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __LABEL_H__
#define __LABEL_H__

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "allocator.h"
#include "property.h"
#include "repetition.h"
#include "vec.h"

namespace gdstk {

enum struct Anchor { NW = 0, N = 1, NE = 2, W = 4, O = 5, E = 6, SW = 8, S = 9, SE = 10 };

struct Label {
    int16_t layer;
    int16_t texttype;
    char* text;  // 0-terminated
    Vec2 origin;
    Anchor anchor;
    double rotation;  // in RADIANS
    double magnification;
    bool x_reflection;
    Repetition repetition;
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print();
    void clear();
    void copy_from(const Label& label);
    void transform(double mag, bool x_refl, double rot, const Vec2 orig);
    void apply_repetition(Array<Label*>& result);
    void to_gds(FILE* out, double scaling) const;
    void to_svg(FILE* out, double scaling) const;
};

}  // namespace gdstk

#endif
