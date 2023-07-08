/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_ROBUSTPATH
#define GDSTK_HEADER_ROBUSTPATH

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <string.h>

#include "array.hpp"
#include "curve.hpp"
#include "oasis.hpp"
#include "pathcommon.hpp"
#include "polygon.hpp"
#include "property.hpp"
#include "repetition.hpp"
#include "utils.hpp"

namespace gdstk {

// Robustpaths are similar to flexpaths in that they only hold a spine, that
// represents the main shape of the desired path, plus any number of elements
// that add widths and offsets on top of that spine to build the actual paths.
// The main difference from flexpaths is that, unlike those, the spine for
// robustpaths is represented by an array of parametric curves (subpaths),
// instead of an open polygonal curve.  Similarly, width and offset changes
// along the path are represented parametrically by interpolation functions.
// These properties make the process of finding the polygonal representation of
// robustpaths more numerically robust, but also more computationally
// intensive.

enum struct InterpolationType {
    Constant = 0,  // Step-change in join region
    Linear,        // LERP from past value to new
    Smooth,        // SERP from past value to new
    Parametric     // Uses function(…)
};

struct Interpolation {
    InterpolationType type;
    union {
        double value;  // Constant
        struct {       // Linear or smooth interpolation
            double initial_value;
            double final_value;
        };
        struct {
            ParametricDouble function;  // Parametric
            void* data;                 // User data
        };
    };
};

enum struct SubPathType {
    Segment,    // straight line segment
    Arc,        // elliptical arc
    Bezier,     // general Bézier
    Bezier2,    // quadratic Bézier
    Bezier3,    // cubic Bézier
    Parametric  // general parametric function
};

// Subpaths are not supposed to be created directly by the user, but through
// the construction functions of the RobustPath class instead.
struct SubPath {
    SubPathType type;
    union {
        struct {  // Segment
            Vec2 begin;
            Vec2 end;
        };
        struct {  // Arc
            // x = path.radius_x * cos(path.angle_i);
            // y = path.radius_y * sin(path.angle_i);
            // center = arc_start - Vec2{x * cos_rot - y * sin_rot, x * sin_rot + y * cos_rot};
            Vec2 center;
            double radius_x;
            double radius_y;
            double angle_i;  // initial_angle - rotation;
            double angle_f;  // final_angle - rotation;
            double cos_rot;
            double sin_rot;
        };
        struct {  // Bezier2, Bezier3
            Vec2 p0;
            Vec2 p1;
            Vec2 p2;
            Vec2 p3;  // Not used for Bezier2
        };
        Array<Vec2> ctrl;  // Bezier
        struct {           // Parametric
            ParametricVec2 path_function;
            ParametricVec2 path_gradient;
            Vec2 reference;
            void* func_data;
            union {
                void* grad_data;
                double step;
            };
        };
    };

    void print() const;
    Vec2 gradient(double u, const double* trafo) const;
    Vec2 eval(double u, const double* trafo) const;
};

struct RobustPathElement {
    Tag tag;

    // These arrays should have the same count as subpath_array
    Array<Interpolation> width_array;
    Array<Interpolation> offset_array;

    // Both end_width and end_offset should be initialized to the initial path
    // width and offset.  After that they will hold the last values used in the
    // construction functions.
    double end_width;
    double end_offset;

    EndType end_type;
    Vec2 end_extensions;
    EndFunction end_function;
    void* end_function_data;  // User data for end_function
};

struct RobustPath {
    // Last point on the path.  It should be initialized to the path origin on
    // the creation of a new path.
    Vec2 end_point;

    Array<SubPath> subpath_array;  // Path spine

    RobustPathElement* elements;  // Array with count num_elements
    uint64_t num_elements;

    // Numeric tolerance for intersection finding and curve approximation
    double tolerance;

    uint64_t max_evals;   // Maximal number of evaluations per function
    double width_scale;   // Width scale accumulated from path transforms
    double offset_scale;  // Offset scale accumulated from path transforms

    // Transformation matrix for this path.  It should be initialized to the
    // identity {1, 0, 0, 0, 1, 0}, unless you have a reason not to.  It
    // transforms a point (x, y) to (xt, yt) with:
    //   xt = x * trafo[0] + y * trafo[1] + trafo[2]
    //   yt = x * trafo[3] + y * trafo[4] + trafo[5]
    double trafo[6];

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
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    // These are initialization routines to facilitate the creation of new
    // robustpaths.  In versions with argument num_elements_, the elements
    // array will be dynamically allocated (and num_elements properly set).
    // Otherwise, num_elements and elements are expected to be already
    // allocated and set.  Arguments width, offset and tag can be single values
    // (which are applied to all elements) or arrays with count num_elements,
    // one value for each path element.  Argument separation is the desired
    // distance between adjacent elements.
    void init(const Vec2 initial_position, double width, double offset, double tolerance_,
              uint64_t max_evals_, Tag tag);
    void init(const Vec2 initial_position, const double* width, const double* offset,
              double tolerance_, uint64_t max_evals_, const Tag* tag);
    void init(const Vec2 initial_position, uint64_t num_elements_, double width, double separation,
              double tolerance_, uint64_t max_evals_, Tag tag);
    void init(const Vec2 initial_position, uint64_t num_elements_, const double* width,
              const double* offset, double tolerance_, uint64_t max_evals_, const Tag* tag);

    void print(bool all) const;

    void clear();

    // This path instance must be zeroed before copy_from
    void copy_from(const RobustPath& path);

    void translate(const Vec2 v);
    void scale(double scale, const Vec2 center);
    void mirror(const Vec2 p0, const Vec2 p1);
    void rotate(double angle, const Vec2 center);

    // Transformations are applied in the order of arguments, starting with
    // magnification and translating by origin at the end.  This is equivalent
    // to the transformation defined by a Reference with the same arguments.
    void transform(double magnification, bool x_reflection, double rotation, const Vec2 origin);

    // Append the copies of this path defined by its repetition to result.
    void apply_repetition(Array<RobustPath*>& result);

    // These functions are equivalent to those for curves (curve.h), with the
    // addition of width and offset, which can be NULL (no width or offset
    // changes) or arrays with count num_elements.
    void horizontal(double coord_x, const Interpolation* width, const Interpolation* offset,
                    bool relative);
    void vertical(double coord_y, const Interpolation* width, const Interpolation* offset,
                  bool relative);
    void segment(const Vec2 end_point, const Interpolation* width, const Interpolation* offset,
                 bool relative);
    void cubic(const Vec2 point1, const Vec2 point2, const Vec2 point3, const Interpolation* width,
               const Interpolation* offset, bool relative);
    void cubic_smooth(const Vec2 point2, const Vec2 point3, const Interpolation* width,
                      const Interpolation* offset, bool relative);
    void quadratic(const Vec2 point1, const Vec2 point2, const Interpolation* width,
                   const Interpolation* offset, bool relative);
    void quadratic_smooth(const Vec2 point2, const Interpolation* width,
                          const Interpolation* offset, bool relative);
    void bezier(const Array<Vec2> point_array, const Interpolation* width,
                const Interpolation* offset, bool relative);
    void interpolation(const Array<Vec2> point_array, double* angles, bool* angle_constraints,
                       Vec2* tension, double initial_curl, double final_curl, bool cycle,
                       const Interpolation* width, const Interpolation* offset, bool relative);
    void arc(double radius_x, double radius_y, double initial_angle, double final_angle,
             double rotation, const Interpolation* width, const Interpolation* offset);
    void turn(double radius, double angle, const Interpolation* width, const Interpolation* offset);
    void parametric(ParametricVec2 curve_function, void* func_data, ParametricVec2 curve_gradient,
                    void* grad_data, const Interpolation* width, const Interpolation* offset,
                    bool relative);
    uint64_t commands(const CurveInstruction* items, uint64_t count);

    // These functions retrieve the position and gradient of the path spine at
    // parametric value u, in which 0 <= u <= path.subpath_array.count.  For
    // integer values of u, uses the end position of the previous subpath or
    // the initial position of the next depending on whether from_below is true
    // or false, respectively.  Note that this functions does not calculate
    // intersections, it evaluates a single subpath.
    Vec2 position(double u, bool from_below) const;
    Vec2 gradient(double u, bool from_below) const;

    // These functions are similar to the ones above, except they retrieve the
    // width and offset of all path elements at the desired parametric value.
    // The results are written to the result array, which must have count at
    // least path.num_elements.
    void width(double u, bool from_below, double* result) const;
    void offset(double u, bool from_below, double* result) const;

    // Calculate the polygonal spine of this path and append the resulting
    // curve to result.
    ErrorCode spine(Array<Vec2>& result) const;

    // Calculate the center of an element of this path and append the resulting
    // curve to result.
    ErrorCode element_center(const RobustPathElement* el, Array<Vec2>& result) const;

    // Append the polygonal representation of this path to result.  If filter
    // is true, only elements with the indicated tag are processed.
    // Overlapping points are removed from the path before any processing is
    // executed.
    ErrorCode to_polygons(bool filter, Tag tag, Array<Polygon*>& result) const;

    // These functions output the polygon in the GDSII, OASIS and SVG formats.
    // They are not supposed to be called by the user.  Because fracturing
    // occurs at cell_to_gds, the polygons must be checked there and, if
    // needed, fractured.  Therefore, to_gds should be used only when
    // simple_path == true to produce true GDSII path elements.  The same is
    // valid for to_oas, even though no fracturing ever occurs for OASIS files.
    ErrorCode to_gds(FILE* out, double scaling) const;
    ErrorCode to_oas(OasisStream& out, OasisState& state) const;
    ErrorCode to_svg(FILE* out, double scaling, uint32_t precision) const;

   private:
    void simple_scale(double scale);
    void simple_rotate(double angle);
    void x_reflection();
    void fill_widths_and_offsets(const Interpolation* width, const Interpolation* offset);
    ErrorCode spine_intersection(const SubPath& sub0, const SubPath& sub1, double& u0,
                                 double& u1) const;
    ErrorCode center_intersection(const SubPath& sub0, const Interpolation& offset0,
                                  const SubPath& sub1, const Interpolation& offset1, double& u0,
                                  double& u1) const;
    ErrorCode left_intersection(const SubPath& sub0, const Interpolation& offset0,
                                const Interpolation& width0, const SubPath& sub1,
                                const Interpolation& offset1, const Interpolation& width1,
                                double& u0, double& u1) const;
    ErrorCode right_intersection(const SubPath& sub0, const Interpolation& offset0,
                                 const Interpolation& width0, const SubPath& sub1,
                                 const Interpolation& offset1, const Interpolation& width1,
                                 double& u0, double& u1) const;
    Vec2 spine_position(const SubPath& subpath, double u) const;
    Vec2 spine_gradient(const SubPath& subpath, double u) const;
    Vec2 center_position(const SubPath& subpath, const Interpolation& offset, double u) const;
    Vec2 center_gradient(const SubPath& subpath, const Interpolation& offset, double u) const;
    Vec2 left_position(const SubPath& subpath, const Interpolation& offset,
                       const Interpolation& width, double u) const;
    Vec2 left_gradient(const SubPath& subpath, const Interpolation& offset,
                       const Interpolation& width, double u) const;
    Vec2 right_position(const SubPath& subpath, const Interpolation& offset,
                        const Interpolation& width, double u) const;
    Vec2 right_gradient(const SubPath& subpath, const Interpolation& offset,
                        const Interpolation& width, double u) const;
    void spine_points(const SubPath& subpath, double u0, double u1, Array<Vec2>& result) const;
    void center_points(const SubPath& subpath, const Interpolation& offset, double u0, double u1,
                       Array<Vec2>& result) const;
    void left_points(const SubPath& subpath, const Interpolation& offset,
                     const Interpolation& width, double u0, double u1, Array<Vec2>& result) const;
    void right_points(const SubPath& subpath, const Interpolation& offset,
                      const Interpolation& width, double u0, double u1, Array<Vec2>& result) const;
};

}  // namespace gdstk

#endif
