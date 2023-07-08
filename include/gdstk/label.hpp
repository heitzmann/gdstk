/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_LABEL
#define GDSTK_HEADER_LABEL

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocator.hpp"
#include "property.hpp"
#include "repetition.hpp"
#include "utils.hpp"
#include "vec.hpp"

namespace gdstk {

enum struct Anchor { NW = 0, N = 1, NE = 2, W = 4, O = 5, E = 6, SW = 8, S = 9, SE = 10 };

struct Label {
    Tag tag;
    char* text;  // NULL-terminated text string
    Vec2 origin;
    Anchor anchor;         // Text anchor (not supported by OASIS)
    double rotation;       // in radians (not supported by OASIS)
    double magnification;  // (not supported by OASIS)
    bool x_reflection;     // (not supported by OASIS)
    Repetition repetition;
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void init(const char* text_) {
        text = copy_string(text_, NULL);
        magnification = 1.0;
    }

    void print();

    void clear();

    // This label instance must be zeroed before copy_from
    void copy_from(const Label& label);

    // Bounding box corners are returned in min and max.  Repetitions are taken
    // into account for the calculation.
    void bounding_box(Vec2& min, Vec2& max) const;

    // Transformations are applied in the order of arguments, starting with
    // magnification and translating by origin at the end.  This is equivalent
    // to the transformation defined by a Reference with the same arguments.
    void transform(double mag, bool x_refl, double rot, const Vec2 orig);

    // Append the copies of this label defined by its repetition to result.
    void apply_repetition(Array<Label*>& result);

    // These functions output the label in the GDSII and SVG formats.  They are
    // not supposed to be called by the user.
    ErrorCode to_gds(FILE* out, double scaling) const;
    ErrorCode to_svg(FILE* out, double scaling, uint32_t precision) const;
};

}  // namespace gdstk

#endif
