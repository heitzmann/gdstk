/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_CLIPPER_TOOLS
#define GDSTK_HEADER_CLIPPER_TOOLS

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include <stdint.h>

#include "array.h"
#include "polygon.h"

namespace gdstk {

enum struct Operation { Or, And, Xor, Not };
enum struct OffsetJoin { Miter, Bevel, Round };
enum struct ShortCircuit { None, Any, All };

// The following operations are executed in an integer grid of vertices, so the
// geometry should be scaled by a large enough factor to garante a minimal
// precision level.  However, if the scaling factor is too large, it may cause
// overflow of coordinates.

// Resulting polygons are appended to result.
void boolean(const Array<Polygon*>& polys1, const Array<Polygon*>& polys2, Operation operation,
             double scaling, Array<Polygon*>& result);

// Dilates or erodes polygons acording to distance (negative distance results
// in erosion).  The effects of internal polygon edges (in polygons with holes,
// for example) can be suppressed by setting use_union to true.  Resulting
// polygons are appended to result.
void offset(const Array<Polygon*>& polys, double distance, OffsetJoin join, double tol,
            double scaling, bool use_union, Array<Polygon*>& result);

// Check whether the points in groups are inside or outside the set of
// polygons.  Checking within each group can the short-circuited to analyse if
// *any* point in the group is inside the polygon set, or if *all* of the
// points in the group are.  When no short-circuit is used, checks for each
// point are appended to results; otherwise, a single check per group.
void inside(const Array<Polygon*>& groups, const Array<Polygon*>& polygons,
            ShortCircuit short_circuit, double scaling, Array<bool>& result);

// Slice the given polygon along the coordinates in posiotions.  Cuts are
// vertical (horizontal) when x_axis is set to true (false).  Argument result
// must be an array with length at least positions.count + 1.  The resulting
// slices are appendend to the arrays in their respective bins.
void slice(const Polygon& polygon, const Array<double>& positions, bool x_axis, double scaling,
           Array<Polygon*>* result);

}  // namespace gdstk

#endif
