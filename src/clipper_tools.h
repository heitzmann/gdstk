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

void boolean(const Array<Polygon*>& polys1, const Array<Polygon*>& polys2, Operation operation,
             double scaling, Array<Polygon*>& result);
void offset(const Array<Polygon*>& polys, double distance, OffsetJoin join, double tol,
            double scaling, bool use_union, Array<Polygon*>& result);
void inside(const Array<Polygon*>& groups, const Array<Polygon*>& polygons,
            ShortCircuit short_circuit, double scaling, Array<bool>& result);
// result must have count at least positions.count + 1 to hold the arrays from each section.
void slice(const Polygon& polygon, const Array<double>& positions, bool x_axis, double scaling,
           Array<Polygon*>* result);

}  // namespace gdstk

#endif
