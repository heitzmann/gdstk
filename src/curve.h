/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_CURVE
#define GDSTK_HEADER_CURVE

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "array.h"
#include "utils.h"

namespace gdstk {

struct CurveInstruction {
    union {
        char command;
        double number;
    };
};

struct Curve {
    Array<Vec2> point_array;
    double tolerance;
    Vec2 last_ctrl;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;

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

    bool closed() const {
        if (point_array.count < 2) return false;
        const Vec2 v = point_array[0] - point_array[point_array.count - 1];
        return v.length_sq() <= tolerance * tolerance;
    }

    void horizontal(double coord_x, bool relative);
    void horizontal(const Array<double> coord_x, bool relative);
    void vertical(double coord_y, bool relative);
    void vertical(const Array<double> coord_y, bool relative);
    void segment(Vec2 end_point, bool relative);
    void segment(const Array<Vec2> points, bool relative);
    void cubic(const Array<Vec2> points, bool relative);
    void cubic_smooth(const Array<Vec2> points, bool relative);
    void quadratic(const Array<Vec2> points, bool relative);
    void quadratic_smooth(Vec2 end_point, bool relative);
    void quadratic_smooth(const Array<Vec2> points, bool relative);
    void bezier(const Array<Vec2> points, bool relative);
    void interpolation(const Array<Vec2> points, double* angles, bool* angle_constraints,
                       Vec2* tension, double initial_curl, double final_curl, bool cycle,
                       bool relative);
    void arc(double radius_x, double radius_y, double initial_angle, double final_angle,
             double rotation);
    void parametric(ParametricVec2 curve_function, void* data, bool relative);

    // Return n = number of items processed.  If n < count, item n could not be
    // parsed.
    uint64_t commands(const CurveInstruction* items, uint64_t count);

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
