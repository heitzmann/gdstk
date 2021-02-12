/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_REPETITION
#define GDSTK_HEADER_REPETITION

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include "array.h"
#include "property.h"
#include "vec.h"

namespace gdstk {

enum struct RepetitionType {
    None = 0,
    Rectangular,
    Regular,
    Explicit,
    ExplicitX,
    ExplicitY,
};

struct Repetition {
    RepetitionType type;
    union {
        struct {               // Rectangular (Oasis 1, 2, 3) or Regular (Oasis 8, 9)
            uint64_t columns;  // Along x or v1
            uint64_t rows;     // Along y or v2
            union {
                Vec2 spacing;
                Vec2 v1;
            };
            Vec2 v2;
        };
        // The original element should not be explicitly included.
        Array<Vec2> offsets;   // Explicit (10, 11)
        Array<double> coords;  // ExplicitX, ExplicitY (Oasis 4, 5, 6, 7)
    };

    void print() const;
    void clear();
    void copy_from(const Repetition repetition);
    uint64_t get_count() const;
    // NOTE: the coordinates for the original (0, 0) are includded as 1st element
    void get_offsets(Array<Vec2>& result) const;
    void get_extrema(Array<Vec2>& result) const;
    void transform(double magnification, bool x_reflection, double rotation);
};

}  // namespace gdstk

#endif
