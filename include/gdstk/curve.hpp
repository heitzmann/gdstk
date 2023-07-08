/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_CURVE
#define GDSTK_HEADER_CURVE

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "array.hpp"
#include "utils.hpp"

namespace gdstk {

struct CurveInstruction {
    union {
        char command;
        double number;
    };
};

// Curves can used to build complex shapes by concatenating straight or curved
// sections.  Once complete, their point_array can be used to create a polygon.
// They are also used to build the FlexPath spine.
struct Curve {
    // Before appending any section to a curve, it must contain at least 1
    // point in its point_array
    Array<Vec2> point_array;

    // Tolerance for approximating curved sections with straight lines
    double tolerance;

    // Used internally.  Keep record of the last Bézier control point, which is
    // used for smooth continuations.
    Vec2 last_ctrl;

    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;

    // Point array must be in a valid state before initialization
    void init(const Vec2 initial_position, double tolerance_) {
        point_array.append(initial_position);
        tolerance = tolerance_;
    }

    // This curve instance must be zeroed before copy_from
    void copy_from(const Curve& curve) {
        point_array.copy_from(curve.point_array);
        tolerance = curve.tolerance;
        last_ctrl = curve.last_ctrl;
    }

    void clear() { point_array.clear(); }

    void append(const Vec2 v) { point_array.append(v); }

    void append_unsafe(const Vec2 v) { point_array.append_unsafe(v); }

    void remove(uint64_t index) { point_array.remove(index); }

    void ensure_slots(uint64_t free_slots) { point_array.ensure_slots(free_slots); }

    // A curve is considered closed if the distance between the first and last
    // points is less then of equal to tolerance
    bool closed() const {
        if (point_array.count < 2) return false;
        const Vec2 v = point_array[0] - point_array[point_array.count - 1];
        return v.length_sq() <= tolerance * tolerance;
    }

    // In the following functions, the flag relative indicates whether the
    // points are given relative to the last point in the curve.  If multiple
    // coordinates are used, all coordinates are relative to the same reference
    // (last curve point at the time of execution)
    void horizontal(double coord_x, bool relative);
    void horizontal(const Array<double> coord_x, bool relative);
    void vertical(double coord_y, bool relative);
    void vertical(const Array<double> coord_y, bool relative);
    void segment(Vec2 end_point, bool relative);
    void segment(const Array<Vec2> points, bool relative);

    // Every 3 points define a cubic Bézier section with 2 control points
    // followed by the section end point (starting point being the current last
    // point of the curve)
    void cubic(const Array<Vec2> points, bool relative);

    // Every 2 points define a cubic Bézier section with 1 control point
    // followed by the section end point (starting point being the current last
    // point of the curve and the first control calculated from the previous
    // curve section for continuity)
    void cubic_smooth(const Array<Vec2> points, bool relative);

    // Every 2 points define a quadratic Bézier section with 1 control point
    // followed by the section end point (starting point being the current last
    // point of the curve)
    void quadratic(const Array<Vec2> points, bool relative);

    // Quadratic Bézier section with control point calculated from the previous
    // curve section for continuity
    void quadratic_smooth(Vec2 end_point, bool relative);

    // Every point is used as the end point of a quadratic Bézier section (the
    // control point is calculated from the previous curve section for
    // continuity)
    void quadratic_smooth(const Array<Vec2> points, bool relative);

    // Single Bézier section defined by any number of control points
    void bezier(const Array<Vec2> points, bool relative);

    // Create a smooth interpolation curve through points using cubic Bézier
    // sections.  The angle at any point i can be constrained to the value
    // angles[i] if angle_constraints[i] is true.  The tension array controls
    // the input and output tensions of the curve at each point (the input at
    // the first point and output at the last point are only meaningful for
    // closed curves).  Because the first point of the interpolation is the
    // last point in the curve, the lengths of the arrays angles,
    // angle_constraints, and tension must be points.count + 1.  No argument
    // can be NULL.
    void interpolation(const Array<Vec2> points, double* angles, bool* angle_constraints,
                       Vec2* tension, double initial_curl, double final_curl, bool cycle,
                       bool relative);

    // Add an elliptical arc to the curve. Argument rotation is used to rotate
    // the axes of the ellipse.
    void arc(double radius_x, double radius_y, double initial_angle, double final_angle,
             double rotation);

    // Add a parametric curve section to the curve.  If relative is true,
    // curve_function(0, data) should be (0, 0) for the curve to be continuous.
    void parametric(ParametricVec2 curve_function, void* data, bool relative);

    // Short-hand function for appending several sections at once.  Array items
    // must be formed by a series of instruction characters followed by the
    // correct number of arguments for that instruction.  Instruction
    // characters and arguments are:
    // - Line segment:            'L', x, y
    // - Horizontal segment:      'H', x
    // - Vertical segment:        'V', y
    // - Cubic Bézier:            'C', x0, y0, x1, y1, x2, y2
    // - Smooth cubic Bézier:     'S', x0, y0, x1, y1
    // - Quadratic Bézier:        'Q', x0, y0, x1, y1
    // - Smooth quadratic Bézier: 'T', x, y
    // - Elliptical arc:          'E', rad0, rad1, angle0, angle1, rotation
    // - Circular arc:            'A', radius, angle0, angle1
    // - Circular turn:           'a', radius, angle
    // Coordinates in instructions L, H, V, C, S, Q and T are absolute.  Lower
    // case versions of those instructions can be used for relative coordinates
    // (in this case, each section will be relative to the previous).  Return
    // the number of items processed.  If n < count, item n could not be
    // parsed.
    uint64_t commands(const CurveInstruction* items, uint64_t count);

    // Add a circular arc to the curve ensuring continuity.  Positive
    // (negative) angles create counter-clockwise (clockwise) turns.
    void turn(double radius, double angle) {
        const Vec2 direction = point_array[point_array.count - 1] - last_ctrl;
        double initial_angle = direction.angle() + (angle < 0 ? 0.5 * M_PI : -0.5 * M_PI);
        arc(radius, radius, initial_angle, initial_angle + angle, 0);
    }

   private:
    void append_cubic(const Vec2 p0, const Vec2 p1, const Vec2 p2, const Vec2 p3);
    void append_quad(const Vec2 p0, const Vec2 p1, const Vec2 p2);
    void append_bezier(const Array<Vec2> ctrl);
};

}  // namespace gdstk

#endif
