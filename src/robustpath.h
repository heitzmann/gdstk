/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_ROBUSTPATH
#define GDSTK_HEADER_ROBUSTPATH

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <string.h>

#include "array.h"
#include "curve.h"
#include "oasis.h"
#include "pathcommon.h"
#include "polygon.h"
#include "property.h"
#include "repetition.h"

namespace gdstk {

enum struct InterpolationType {
    Constant = 0,  // Step-change in join region
    Linear,        // LERP from past value to new
    Smooth,        // SERP from past value to new
    Parametric     // Uses function(...)
};

struct Interpolation {
    InterpolationType type;
    union {
        double value;  // Constant
        struct {       // Linear, Smooth
            double initial_value;
            double final_value;
        };
        struct {
            ParametricDouble function;  // Parametric
            void* data;
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
    uint32_t layer;
    uint32_t datatype;
    Array<Interpolation> width_array;
    Array<Interpolation> offset_array;
    double end_width;
    double end_offset;
    EndType end_type;
    Vec2 end_extensions;
    EndFunction end_function;
    void* end_function_data;
};

struct RobustPath {
    Vec2 end_point;
    Array<SubPath> subpath_array;
    RobustPathElement* elements;
    uint64_t num_elements;
    double tolerance;
    uint64_t max_evals;
    double width_scale;
    double offset_scale;
    double trafo[6];  // Look at apply_transform for the meaning of each coefficient
    bool simple_path;
    bool scale_width;
    Repetition repetition;
    Property* properties;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;
    void clear();
    void copy_from(const RobustPath& path);
    void translate(const Vec2 v);
    void scale(double scale, const Vec2 center);
    void mirror(const Vec2 p0, const Vec2 p1);
    void rotate(double angle, const Vec2 center);
    void transform(double magnification, bool x_reflection, double rotation, const Vec2 origin);
    void apply_repetition(Array<RobustPath*>& result);

    // Note: width and offset must be NULL or arrays of count at least path.num_elements.
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
    // Return n = number of items processed.  If n < count, item n could not be parsed.
    // Width and offset remain unchainged.
    uint64_t commands(const CurveInstruction* items, uint64_t count);

    // 0 <= u <= path.subpath_array.count
    Vec2 position(double u, bool from_below) const;
    Vec2 gradient(double u, bool from_below) const;
    // Result must be an array with at least path.num_elements elements
    void width(double u, bool from_below, double* result) const;
    void offset(double u, bool from_below, double* result) const;

    void spine(Array<Vec2>& result) const;
    void element_center(const RobustPathElement* el, Array<Vec2>& result) const;
    void to_polygons(Array<Polygon*>& result) const;

    // Because fracturing occurs at cell_to_gds, the polygons must be checked there and, if needed,
    // fractured.  Therefore, to_gds should be used only when simple_path == true to produce true
    // GDSII path elements. The same is valid for to_oas, although no fracturing ever occurs for
    // OASIS files.
    void to_gds(FILE* out, double scaling) const;
    void to_oas(OasisStream& out, OasisState& state) const;
    void to_svg(FILE* out, double scaling) const;

   private:
    void simple_scale(double scale);
    void simple_rotate(double angle);
    void x_reflection();
    void fill_widths_and_offsets(const Interpolation* width, const Interpolation* offset);
    void spine_intersection(const SubPath& sub0, const SubPath& sub1, double& u0, double& u1) const;
    void center_intersection(const SubPath& sub0, const Interpolation& offset0, const SubPath& sub1,
                             const Interpolation& offset1, double& u0, double& u1) const;
    void left_intersection(const SubPath& sub0, const Interpolation& offset0,
                           const Interpolation& width0, const SubPath& sub1,
                           const Interpolation& offset1, const Interpolation& width1, double& u0,
                           double& u1) const;
    void right_intersection(const SubPath& sub0, const Interpolation& offset0,
                            const Interpolation& width0, const SubPath& sub1,
                            const Interpolation& offset1, const Interpolation& width1, double& u0,
                            double& u1) const;
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

