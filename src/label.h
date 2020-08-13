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

#include "property.h"
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
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print();

    void clear() {
        if (text) {
            free(text);
            text = NULL;
        }
        properties_clear(properties);
        properties = NULL;
    }

    void copy_from(const Label& label) {
        layer = label.layer;
        texttype = label.texttype;
        text = (char*)malloc((strlen(label.text) + 1) * sizeof(char));
        strcpy(text, label.text);
        origin = label.origin;
        anchor = label.anchor;
        rotation = label.rotation;
        magnification = label.magnification;
        x_reflection = label.x_reflection;
        properties = properties_copy(label.properties);
    }

    void transform(double mag, const Vec2 trans, bool x_refl, double rot, const Vec2 orig) {
        const int r1 = x_refl ? -1 : 1;
        const double crot = cos(rot);
        const double srot = sin(rot);
        const double x = origin.x;
        const double y = origin.y;
        origin.x = orig.x + mag * (x * crot - r1 * y * srot) + trans.x * crot - r1 * trans.y * srot;
        origin.y = orig.y + mag * (x * srot + r1 * y * crot) + trans.x * srot + r1 * trans.y * crot;
        rotation = r1 * rotation + rot;
        magnification *= mag;
        x_reflection ^= x_refl;
    }

    void to_gds(FILE* out, double scaling) const;
    void to_svg(FILE* out, double scaling) const;
};

}  // namespace gdstk

#endif
