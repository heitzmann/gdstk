/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_POLYGON
#define GDSTK_HEADER_POLYGON

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include "array.h"
#include "oasis.h"
#include "property.h"
#include "repetition.h"
#include "vec.h"

namespace gdstk {

struct Polygon {
    uint32_t layer;
    uint32_t datatype;
    Array<Vec2> point_array;
    Repetition repetition;
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;
    void clear();
    void copy_from(const Polygon& polygon);
    double area() const;
    void bounding_box(Vec2& min, Vec2& max) const;
    void translate(const Vec2 v);
    void scale(const Vec2 scale, const Vec2 center);
    void mirror(const Vec2 p0, const Vec2 p1);
    void rotate(double angle, const Vec2 center);
    void transform(double magnification, bool x_reflection, double rotation, const Vec2 origin);
    void fillet(const Array<double> radii, double tol);
    void fracture(uint64_t max_points, double precision, Array<Polygon*>& result) const;
    void apply_repetition(Array<Polygon*>& result);
    void to_gds(FILE* out, double scaling) const;
    void to_oas(OasisStream& out, OasisState& state) const;
    void to_svg(FILE* out, double scaling) const;
};

Polygon rectangle(const Vec2 corner1, const Vec2 corner2, uint32_t layer, uint32_t datatype);
Polygon cross(const Vec2 center, double full_size, double arm_width, uint32_t layer,
              uint32_t datatype);
Polygon regular_polygon(const Vec2 center, double side_length, uint64_t sides, double rotation,
                        uint32_t layer, uint32_t datatype);
Polygon ellipse(const Vec2 center, double radius_x, double radius_y, double inner_radius_x,
                double inner_radius_y, double initial_angle, double final_angle, double tolerance,
                uint32_t layer, uint32_t datatype);
Polygon racetrack(const Vec2 center, double straight_length, double radius, double inner_radius,
                  bool vertical, double tolerance, uint32_t layer, uint32_t datatype);
void text(const char* s, double count, const Vec2 position, bool vertical, uint32_t layer,
          uint32_t datatype, Array<Polygon*>& result);

}  // namespace gdstk

#endif
