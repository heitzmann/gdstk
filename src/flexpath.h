/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __FLEXPATH_H__
#define __FLEXPATH_H__

#include <cstdint>
#include <cstring>

#include "array.h"
#include "curve.h"
#include "pathcommon.h"
#include "polygon.h"
#include "property.h"

namespace gdstk {

struct FlexPathElement {
    int16_t layer;
    int16_t datatype;
    Array<Vec2> half_width_and_offset;

    JoinType join_type;
    JoinFunction join_function;
    void* join_function_data;

    EndType end_type;
    Vec2 end_extensions;
    EndFunction end_function;
    void* end_function_data;

    BendType bend_type;
    double bend_radius;
    BendFunction bend_function;
    void* bend_function_data;
};

struct FlexPath {
    Curve spine;
    int64_t num_elements;
    FlexPathElement* elements;
    bool gdsii_path;
    bool scale_width;
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    // Note: width and offset must be NULL or arrays of size at least path.num_elements.
    void init(const Vec2 initial_position, const double* width, const double* offset);

    void print(bool all) const;
    void clear();
    void copy_from(const FlexPath& path);
    void translate(const Vec2 v);
    void scale(double scale, const Vec2 center);
    void mirror(const Vec2 p0, const Vec2 p1);
    void rotate(double angle, const Vec2 center);
    void transform(double magnification, const Vec2 translation, bool x_reflection, double rotation,
                   const Vec2 origin);

    // Note: width and offset must be NULL or arrays of size at least path.num_elements.
    void horizontal(const double* coord_x, int64_t size, const double* width, const double* offset,
                    bool relative);
    void vertical(const double* coord_y, int64_t size, const double* width, const double* offset,
                  bool relative);
    void segment(const Array<Vec2> point_array, const double* width, const double* offset,
                 bool relative);
    void cubic(const Array<Vec2> point_array, const double* width, const double* offset,
               bool relative);
    void cubic_smooth(const Array<Vec2> point_array, const double* width, const double* offset,
                      bool relative);
    void quadratic(const Array<Vec2> point_array, const double* width, const double* offset,
                   bool relative);
    void quadratic_smooth(const Array<Vec2> point_array, const double* width, const double* offset,
                          bool relative);
    void bezier(const Array<Vec2> point_array, const double* width, const double* offset,
                bool relative);
    void interpolation(const Array<Vec2> point_array, double* angles, bool* angle_constraints,
                       Vec2* tension, double initial_curl, double final_curl, bool cycle,
                       const double* width, const double* offset, bool relative);
    void arc(double radius_x, double radius_y, double initial_angle, double final_angle,
             double rotation, const double* width, const double* offset);
    void turn(double radius, double angle, const double* width, const double* offset);
    void parametric(ParametricVec2 curve_function, void* data, const double* width,
                    const double* offset, bool relative);

    // Return n = number of items processed.  If n < size, item n could not be parsed.  Width and
    // offset remain unchainged.
    int64_t commands(const CurveInstruction* items, int64_t size);

    Array<Polygon*> to_polygons();

    // Because fracturing occurs at cell_to_gds, the polygons must be checked there and, if needed,
    // fractured.  Therefore, to_gds should be used only when gdsii_path == true to produce
    // true GDSII path elements.
    void to_gds(FILE* out, double scaling);
    void to_svg(FILE* out, double scaling);

   private:
    void remove_overlapping_points();
    void fill_offsets_and_widths(const double* width, const double* offset);
};

}  // namespace gdstk

#endif

