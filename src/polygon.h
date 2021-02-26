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

    // This polygon instance must be zeroed before copy_from
    void copy_from(const Polygon& polygon);

    // Total polygon area including any repetitions
    double area() const;

    // Bounding box corners are returned in min and max.  If the polygons has
    // no vertices, return min.x > max.x.  Repetitions are taken into account
    // for the calculation.
    void bounding_box(Vec2& min, Vec2& max) const;

    void translate(const Vec2 v);
    void scale(const Vec2 scale, const Vec2 center);
    void mirror(const Vec2 p0, const Vec2 p1);
    void rotate(double angle, const Vec2 center);

    // Transformations are applied in the order of arguments, starting with
    // magnification and translating by origin at the end.  This is equivalent
    // to the transformation defined by a Reference with the same arguments.
    void transform(double magnification, bool x_reflection, double rotation, const Vec2 origin);

    // Round the corners of this polygon.  Argument radii can include one
    // radius value for each polygon corner or less, in which case it will be
    // cycled.  If the desired fillet radius for a given corner is larger than
    // half the shortest edge adjacent to that corner, it is reduced to that
    // size.  The number of vertices used to approximate the circular arcs is
    // defined by tolerance.
    void fillet(const Array<double> radii, double tolerance);

    // Fracture the polygon horizontally and vertically until all pieces have
    // at most max_points vertices.  If max_points < 5, it doesn't do anything.
    // Resulting pieces are appended to result.
    void fracture(uint64_t max_points, double precision, Array<Polygon*>& result) const;

    // Append the copies of this polygon defined by its repetition to result.
    void apply_repetition(Array<Polygon*>& result);

    // These functions output the polygon in the GDSII, OASIS and SVG formats.
    // They are not supposed to be called by the user.
    void to_gds(FILE* out, double scaling) const;
    void to_oas(OasisStream& out, OasisState& state) const;
    void to_svg(FILE* out, double scaling) const;
};

Polygon rectangle(const Vec2 corner1, const Vec2 corner2, uint32_t layer, uint32_t datatype);

Polygon cross(const Vec2 center, double full_size, double arm_width, uint32_t layer,
              uint32_t datatype);

// The polygon is created with a horizontal lower edge when rotation is 0.
Polygon regular_polygon(const Vec2 center, double side_length, uint64_t sides, double rotation,
                        uint32_t layer, uint32_t datatype);

// Create circles, ellipses, rings, or sections of those.  The number of points
// used to approximate the arcs is such that the approximation error is less
// than tolerance.
Polygon ellipse(const Vec2 center, double radius_x, double radius_y, double inner_radius_x,
                double inner_radius_y, double initial_angle, double final_angle, double tolerance,
                uint32_t layer, uint32_t datatype);

Polygon racetrack(const Vec2 center, double straight_length, double radius, double inner_radius,
                  bool vertical, double tolerance, uint32_t layer, uint32_t datatype);

// Create a polygonal text form NULL-terminated string s.  Argument size
// defines the full height of the glyphs.  Polygons are appended to result.
// The character aspect ratio is 1:2.  For horizontal text, spacings between
// characters and between lines are 9/16 and 5/4 times the full height size,
// respectively.  For vertical text, characters and columns are respectively
// spaced by 9/8 and 1 times size.
void text(const char* s, double size, const Vec2 position, bool vertical, uint32_t layer,
          uint32_t datatype, Array<Polygon*>& result);

}  // namespace gdstk

#endif
