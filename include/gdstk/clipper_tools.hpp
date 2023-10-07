/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_CLIPPER_TOOLS
#define GDSTK_HEADER_CLIPPER_TOOLS

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>

#include "array.hpp"
#include "polygon.hpp"

namespace gdstk {

enum struct Operation { Or, And, Xor, Not };
enum struct OffsetJoin { Miter, Bevel, Round };

// The following operations are executed in an integer grid of vertices, so the
// geometry should be scaled by a large enough factor to garante a minimal
// precision level.  However, if the scaling factor is too large, it may cause
// overflow of coordinates.  Resulting polygons are appended to result.

// Boolean (clipping) operations on polygons
ErrorCode boolean(const Array<Polygon*>& polys1, const Array<Polygon*>& polys2, Operation operation,
                  double scaling, Array<Polygon*>& result);

inline ErrorCode boolean(const Polygon& poly1, const Array<Polygon*>& polys2, Operation operation,
                         double scaling, Array<Polygon*>& result) {
    const Polygon* p1 = &poly1;
    const Array<Polygon*> polys1 = {1, 1, (Polygon**)&p1};
    return boolean(polys1, polys2, operation, scaling, result);
}

inline ErrorCode boolean(const Array<Polygon*>& polys1, const Polygon& poly2, Operation operation,
                         double scaling, Array<Polygon*>& result) {
    const Polygon* p2 = &poly2;
    const Array<Polygon*> polys2 = {1, 1, (Polygon**)&p2};
    return boolean(polys1, polys2, operation, scaling, result);
}

inline ErrorCode boolean(const Polygon& poly1, const Polygon& poly2, Operation operation,
                         double scaling, Array<Polygon*>& result) {
    const Polygon* p1 = &poly1;
    const Polygon* p2 = &poly2;
    const Array<Polygon*> polys1 = {1, 1, (Polygon**)&p1};
    const Array<Polygon*> polys2 = {1, 1, (Polygon**)&p2};
    return boolean(polys1, polys2, operation, scaling, result);
}

// Shorthand for joining a set of polygons
inline ErrorCode merge(const Array<Polygon*>& polygons, double scaling, Array<Polygon*>& result) {
    const Array<Polygon*> empty = {};
    return boolean(polygons, empty, Operation::Or, scaling, result);
}

// Dilates or erodes polygons according to distance (negative distance results
// in erosion).  The effects of internal polygon edges (in polygons with holes,
// for example) can be suppressed by setting use_union to true.  Resulting
// polygons are appended to result.
ErrorCode offset(const Array<Polygon*>& polys, double distance, OffsetJoin join, double tolerance,
                 double scaling, bool use_union, Array<Polygon*>& result);

inline ErrorCode offset(const Polygon& poly, double distance, OffsetJoin join, double tolerance,
                        double scaling, bool use_union, Array<Polygon*>& result) {
    const Polygon* p = &poly;
    const Array<Polygon*> polys = {1, 1, (Polygon**)&p};
    return offset(polys, distance, join, tolerance, scaling, use_union, result);
}

// Slice the given polygon along the coordinates in positions.  Positions must be sorted.  Cuts are
// vertical (horizontal) when x_axis is set to true (false).  Argument result must be an array with
// length at least positions.count + 1.  The resulting slices are appended to the arrays in their
// respective bins.
ErrorCode slice(const Polygon& polygon, const Array<double>& positions, bool x_axis, double scaling,
                Array<Polygon*>* result);

}  // namespace gdstk

#endif
