/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_LABEL
#define GDSTK_HEADER_LABEL

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocator.h"
#include "property.h"
#include "repetition.h"
#include "vec.h"

namespace gdstk {

enum struct Anchor { NW = 0, N = 1, NE = 2, W = 4, O = 5, E = 6, SW = 8, S = 9, SE = 10 };

struct Label {
    uint32_t layer;
    uint32_t texttype;
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
