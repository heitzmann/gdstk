/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_FLEXPATH
#define GDSTK_HEADER_FLEXPATH

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>

#include "array.hpp"
#include "curve.hpp"
#include "oasis.hpp"
#include "pathcommon.hpp"
#include "polygon.hpp"
#include "property.hpp"
#include "raithdata.hpp"
#include "repetition.hpp"
#include "utils.hpp"

namespace gdstk {

// A flexpath holds a spine and any number of elements.  The spine dictates the
// general shape of the path, but it is a simple curve, it doesn't have
// information about width.  The elements are the concrete paths that are
// created based on the spine, a width and, optionally, an offset from the
// spine.  Both width and offset can change along the spine.

struct FlexPathElement {
    Tag tag;

    // Array of widths and offsets for this path element.  The array count must
    // match the spine count.  Each Vec2 v holds the width of the path divided
    // by 2 in v.e[0] and the path offset in v.e[1] for the respective point
    // along the spine.
    Array<Vec2> half_width_and_offset;

    JoinType join_type;
    JoinFunction join_function;
    void* join_function_data;  // User data passed directly to join_function

    EndType end_type;
    Vec2 end_extensions;
    EndFunction end_function;
    void* end_function_data;  // User data passed directly to end_function

    BendType bend_type;
    double bend_radius;
    BendFunction bend_function;
    void* bend_function_data;  // User data passed directly to bend_function
};

struct FlexPath {
    Curve spine;
    FlexPathElement* elements;  // Array with count num_elements
    uint64_t num_elements;

    // If simple_path is true, all elements will be treated as if they have
    // constant widths (using the first elements in their respective
    // half_width_and_offset arrays) and saved as paths, not polygonal
    // boundaries.
    bool simple_path;

    // Flag indicating whether the width of the path elements should be scaled
    // when scaling the path (manually or through references).
    bool scale_width;

    Repetition repetition;
    Property* properties;

    RaithData raith_data;

    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    // These are initialization routines to facilitate the creation of new
    // flexpaths.  In versions with argument num_elements_, the elements array
    // will be dynamically allocated (and num_elements properly set).
    // Otherwise, num_elements and elements are expected to be already
    // allocated and set.  Arguments width, offset and tag can be single values
    // (which are applied to all elements) or arrays with count num_elements,
    // one value for each path element.  Argument separation is the desired
    // distance between adjacent elements.
    void init(const Vec2 initial_position, double width, double offset, double tolerance, Tag tag);
    void init(const Vec2 initial_position, const double* width, const double* offset,
              double tolerance, const Tag* tag);
    void init(const Vec2 initial_position, uint64_t num_elements_, double width, double separation,
              double tolerance, Tag tag);
    void init(const Vec2 initial_position, uint64_t num_elements_, const double* width,
              const double* offset, double tolerance, const Tag* tag);

    void print(bool all) const;

    void clear();

    // This path instance must be zeroed before copy_from
    void copy_from(const FlexPath& path);

    void translate(const Vec2 v);
    void scale(double scale, const Vec2 center);
    void mirror(const Vec2 p0, const Vec2 p1);
    void rotate(double angle, const Vec2 center);

    // Transformations are applied in the order of arguments, starting with
    // magnification and translating by origin at the end.  This is equivalent
    // to the transformation defined by a Reference with the same arguments.
    void transform(double magnification, bool x_reflection, double rotation, const Vec2 origin);

    // Append the copies of this path defined by its repetition to result.
    void apply_repetition(Array<FlexPath*>& result);

    // These functions are equivalent to those for curves (curve.h), with the
    // addition of width and offset, which can be NULL (no width or offset
    // changes) or arrays with count num_elements.
    void horizontal(double coord_x, const double* width, const double* offset, bool relative);
    void horizontal(const Array<double> coord_x, const double* width, const double* offset,
                    bool relative);
    void vertical(double coord_y, const double* width, const double* offset, bool relative);
    void vertical(const Array<double> coord_y, const double* width, const double* offset,
                  bool relative);
    void segment(Vec2 end_point, const double* width, const double* offset, bool relative);
    void segment(const Array<Vec2> point_array, const double* width, const double* offset,
                 bool relative);
    void cubic(const Array<Vec2> point_array, const double* width, const double* offset,
               bool relative);
    void cubic_smooth(const Array<Vec2> point_array, const double* width, const double* offset,
                      bool relative);
    void quadratic(const Array<Vec2> point_array, const double* width, const double* offset,
                   bool relative);
    void quadratic_smooth(Vec2 end_point, const double* width, const double* offset, bool relative);
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
    uint64_t commands(const CurveInstruction* items, uint64_t count);

    // Append the polygonal representation of this path to result.  If filter
    // is true, only elements with the indicated tag are processed.
    // Overlapping points are removed from the path before any processing is
    // executed.
    ErrorCode to_polygons(bool filter, Tag tag, Array<Polygon*>& result);

    // Calculate the center of an element of this path and append the resulting
    // curve to result.
    ErrorCode element_center(const FlexPathElement* el, Array<Vec2>& result);

    // These functions output the polygon in the GDSII, OASIS and SVG formats.
    // They are not supposed to be called by the user.  Because fracturing
    // occurs at cell_to_gds, the polygons must be checked there and, if
    // needed, fractured.  Therefore, to_gds should be used only when
    // simple_path == true to produce true GDSII path elements.  The same is
    // valid for to_oas, even though no fracturing ever occurs for OASIS files.
    ErrorCode to_gds(FILE* out, double scaling);
    ErrorCode to_oas(OasisStream& out, OasisState& state);
    ErrorCode to_svg(FILE* out, double scaling, uint32_t precision);

   private:
    void remove_overlapping_points();
    void fill_offsets_and_widths(const double* width, const double* offset);
};

}  // namespace gdstk

#endif
