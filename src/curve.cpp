/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "curve.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "array.h"
#include "utils.h"

namespace gdstk {

void Curve::print(bool all) const {
    printf("Curve <%p>, size %" PRId64 ", tol %lg, last ctrl (%lg, %lg), owner <%p>:\n", this,
           point_array.size, tolerance, last_ctrl.x, last_ctrl.y, owner);
    if (all) {
        printf("Points: ");
        point_array.print(true);
    }
}

void Curve::append_cubic(const Vec2 p0, const Vec2 p1, const Vec2 p2, const Vec2 p3) {
    // Sampling based on curvature
    // dp* : 1st derivative
    const Vec2 dp0 = 3 * (p1 - p0);
    const Vec2 dp1 = 3 * (p2 - p1);
    const Vec2 dp2 = 3 * (p3 - p2);
    // d2p* : 2nd derivative
    const Vec2 d2p0 = 2 * (dp1 - dp0);
    const Vec2 d2p1 = 2 * (dp2 - dp1);

    const double tolerance_sq = tolerance * tolerance;
    Vec2 last = p0;
    double t = 0;
    while (t < 1) {
        Vec2 dc = eval_bezier2(t, dp0, dp1, dp2);
        Vec2 d2c = eval_line(t, d2p0, d2p1);
        double len_dc = dc.length();
        double dt = 0.5 / MIN_POINTS;
        if (len_dc > 0) {
            double curvature = fabs(dc.cross(d2c)) / (len_dc * len_dc * len_dc);
            if (curvature < PARALLEL_EPS) {
                dt = 1.0;
            } else {
                double angle = 2 * acos(1 - curvature * tolerance);
                dt = angle / (curvature * len_dc);
            }
        }
        if (t + dt > 1) dt = 1 - t;
        if (dt > 1.0 / MIN_POINTS) dt = 1.0 / MIN_POINTS;

        Vec2 next = eval_bezier3(t + dt, p0, p1, p2, p3);
        Vec2 mid = eval_bezier3(t + 0.5 * dt, p0, p1, p2, p3);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = eval_bezier3(t + dt / 3, p0, p1, p2, p3);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            dt *= 0.5;
            next = mid;
            mid = eval_bezier3(t + 0.5 * dt, p0, p1, p2, p3);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = eval_bezier3(t + dt / 3, p0, p1, p2, p3);
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }

        append(next);
        last = next;
        t += dt;
    }
}

void Curve::append_quad(const Vec2 p0, const Vec2 p1, const Vec2 p2) {
    // Sampling based on curvature
    // dp* : 1st derivative
    const Vec2 dp0 = 2 * (p1 - p0);
    const Vec2 dp1 = 2 * (p2 - p1);
    // d2p* : 2nd derivative
    const Vec2 d2c = dp1 - dp0;

    const double tolerance_sq = tolerance * tolerance;
    Vec2 last = p0;
    double t = 0;
    while (t < 1) {
        Vec2 dc = eval_line(t, dp0, dp1);
        double len_dc = dc.length();
        double dt = 0.5 / MIN_POINTS;
        if (len_dc > 0) {
            double curvature = fabs(dc.cross(d2c)) / (len_dc * len_dc * len_dc);
            if (curvature < PARALLEL_EPS) {
                dt = 1.0;
            } else {
                double angle = 2 * acos(1 - curvature * tolerance);
                dt = angle / (curvature * len_dc);
            }
        }
        if (t + dt > 1) dt = 1 - t;
        if (dt > 1.0 / MIN_POINTS) dt = 1.0 / MIN_POINTS;

        Vec2 next = eval_bezier2(t + dt, p0, p1, p2);
        Vec2 mid = eval_bezier2(t + 0.5 * dt, p0, p1, p2);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = eval_bezier2(t + 0.5 * dt, p0, p1, p2);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            dt *= 0.5;
            next = mid;
            mid = eval_bezier2(t + 0.5 * dt, p0, p1, p2);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = eval_bezier2(t + 0.5 * dt, p0, p1, p2);
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }

        append(next);
        last = next;
        t += dt;
    }
}

void Curve::append_bezier(const Array<Vec2> ctrl) {
    const int64_t size = ctrl.size;
    // Sampling based on curvature
    // dp : 1st derivative
    Array<Vec2> dp = {0};
    // d2p : 2nd derivative
    Array<Vec2> d2p = {0};

    dp.ensure_slots(size - 1 + size - 2);
    d2p.items = dp.items + size - 1;
    Vec2* dst0 = dp.items;
    Vec2* dst1 = d2p.items;
    const Vec2* src = ctrl.items;
    for (int64_t i = 0; i < size - 1; i++, src++, dst0++) {
        *dst0 = (size - 1) * (*(src + 1) - *src);
        if (i > 0) *dst1++ = (size - 2) * (*dst0 - *(dst0 - 1));
    }
    dp.size = size - 1;
    d2p.size = size - 2;

    const double tolerance_sq = tolerance * tolerance;
    const double dt_max = 1.0 / size;
    Vec2 last = ctrl[0];
    double t = 0;
    while (t < 1) {
        Vec2 dc = eval_bezier(t, dp.items, dp.size);
        Vec2 d2c = eval_bezier(t, d2p.items, d2p.size);
        double len_dc = dc.length();
        double dt = 0.5 * dt_max;
        if (len_dc > 0) {
            double curvature = fabs(dc.cross(d2c)) / (len_dc * len_dc * len_dc);
            if (curvature < PARALLEL_EPS) {
                dt = 1.0;
            } else {
                double angle = 2 * acos(1 - curvature * tolerance);
                dt = angle / (curvature * len_dc);
            }
        }
        if (t + dt > 1) dt = 1 - t;
        if (dt > dt_max) dt = dt_max;

        Vec2 next = eval_bezier(t + dt, ctrl.items, ctrl.size);
        Vec2 mid = eval_bezier(t + 0.5 * dt, ctrl.items, ctrl.size);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = eval_bezier(t + 0.5 * dt, ctrl.items, ctrl.size);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            dt *= 0.5;
            next = mid;
            mid = eval_bezier(t + 0.5 * dt, ctrl.items, ctrl.size);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = eval_bezier(t + 0.5 * dt, ctrl.items, ctrl.size);
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }

        append(next);
        last = next;
        t += dt;
    }
    dp.clear();
}

void Curve::horizontal(const double* coord_x, int64_t size, bool relative) {
    ensure_slots(size);
    Vec2* dst = point_array.items + point_array.size;
    const double* src_x = coord_x;
    if (relative) {
        const Vec2 ref = point_array[point_array.size - 1];
        for (int64_t i = 0; i < size; i++, dst++) {
            dst->x = ref.x + *src_x++;
            dst->y = ref.y;
        }
    } else {
        const double ref_y = point_array[point_array.size - 1].y;
        for (int64_t i = 0; i < size; i++, dst++) {
            dst->x = *src_x++;
            dst->y = ref_y;
        }
    }
    point_array.size += size;
    last_ctrl = point_array[point_array.size - 2];
}

void Curve::vertical(const double* coord_y, int64_t size, bool relative) {
    ensure_slots(size);
    Vec2* dst = point_array.items + point_array.size;
    const double* src_y = coord_y;
    if (relative) {
        const Vec2 ref = point_array[point_array.size - 1];
        for (int64_t i = 0; i < size; i++, dst++) {
            dst->x = ref.x;
            dst->y = ref.y + *src_y++;
        }
    } else {
        const double ref_x = point_array[point_array.size - 1].x;
        for (int64_t i = 0; i < size; i++, dst++) {
            dst->x = ref_x;
            dst->y = *src_y++;
        }
    }
    point_array.size += size;
    last_ctrl = point_array[point_array.size - 2];
}

void Curve::segment(const Array<Vec2> points, bool relative) {
    if (relative) {
        ensure_slots(points.size);
        const Vec2 ref = point_array[point_array.size - 1];
        const Vec2* src = points.items;
        Vec2* dst = point_array.items + point_array.size;
        for (int64_t i = 0; i < points.size; i++) *dst++ = ref + *src++;
        point_array.size += points.size;
    } else {
        point_array.extend(points);
    }
    last_ctrl = point_array[point_array.size - 2];
}

void Curve::cubic(const Array<Vec2> points, bool relative) {
    Vec2 last_point = point_array[point_array.size - 1];
    if (relative) {
        const Vec2 ref = point_array[point_array.size - 1];
        for (int64_t i = 0; i < points.size - 2; i += 3) {
            Vec2 first_point = last_point;
            last_point = ref + points[i + 2];
            append_cubic(first_point, ref + points[i], ref + points[i + 1], last_point);
        }
        last_ctrl = ref + points[points.size - 2];
    } else {
        for (int64_t i = 0; i < points.size - 2; i += 3) {
            Vec2 first_point = last_point;
            last_point = points[i + 2];
            append_cubic(first_point, points[i], points[i + 1], last_point);
        }
        last_ctrl = points[points.size - 2];
    }
}

void Curve::cubic_smooth(const Array<Vec2> points, bool relative) {
    Vec2 last_point = point_array[point_array.size - 1];
    const Vec2* point = points.items;
    if (relative) {
        const Vec2 ref = point_array[point_array.size - 1];
        for (int64_t i = 0; i < points.size - 1; i += 2) {
            Vec2 first_point = last_point;
            Vec2 smooth_ctrl = last_point * 2 - last_ctrl;
            last_ctrl = ref + *point++;
            last_point = ref + *point++;
            append_cubic(first_point, smooth_ctrl, last_ctrl, last_point);
        }
    } else {
        for (int64_t i = 0; i < points.size - 1; i += 2) {
            Vec2 first_point = last_point;
            Vec2 smooth_ctrl = last_point * 2 - last_ctrl;
            last_ctrl = *point++;
            last_point = *point++;
            append_cubic(first_point, smooth_ctrl, last_ctrl, last_point);
        }
    }
}

void Curve::quadratic(const Array<Vec2> points, bool relative) {
    Vec2 last_point = point_array[point_array.size - 1];
    if (relative) {
        const Vec2 ref = point_array[point_array.size - 1];
        for (int64_t i = 0; i < points.size - 1; i += 2) {
            Vec2 first_point = last_point;
            last_point = ref + points[i + 1];
            append_quad(first_point, ref + points[i], last_point);
        }
        last_ctrl = ref + points[points.size - 2];
    } else {
        for (int64_t i = 0; i < points.size - 1; i += 2) {
            Vec2 first_point = last_point;
            last_point = points[i + 1];
            append_quad(first_point, points[i], last_point);
        }
        last_ctrl = points[points.size - 2];
    }
}

void Curve::quadratic_smooth(const Array<Vec2> points, bool relative) {
    Vec2 last_point = point_array[point_array.size - 1];
    const Vec2* point = points.items;
    if (relative) {
        const Vec2 ref = point_array[point_array.size - 1];
        for (int64_t i = 0; i < points.size; i += 1) {
            Vec2 first_point = last_point;
            last_ctrl = last_point * 2 - last_ctrl;
            last_point = ref + *point++;
            append_quad(first_point, last_ctrl, last_point);
        }
    } else {
        for (int64_t i = 0; i < points.size; i += 1) {
            Vec2 first_point = last_point;
            last_ctrl = last_point * 2 - last_ctrl;
            last_point = *point++;
            append_quad(first_point, last_ctrl, last_point);
        }
    }
}

void Curve::bezier(const Array<Vec2> points, bool relative) {
    Array<Vec2> ctrl = {0};
    ctrl.ensure_slots(points.size + 1);
    if (relative) {
        const Vec2* point = points.items;
        const Vec2 ref = point_array[point_array.size - 1];
        ctrl[0] = ref;
        Vec2* dst = ctrl.items + 1;
        for (int64_t i = 0; i < points.size; i++, dst++) *dst = ref + *point++;
    } else {
        ctrl[0] = point_array[point_array.size - 1];
        memcpy(ctrl.items + 1, points.items, sizeof(Vec2) * points.size);
    }
    ctrl.size = points.size + 1;
    append_bezier(ctrl);
    last_ctrl = points[points.size - 2];
    ctrl.clear();
}

void Curve::interpolation(const Array<Vec2> points, double* angles, bool* angle_constraints,
                          Vec2* tension, double initial_curl, double final_curl, bool cycle,
                          bool relative) {
    Array<Vec2> hobby_vec = {0};
    hobby_vec.ensure_slots(3 * (points.size + 1) + 1);
    hobby_vec.size = 3 * (points.size + 1) + 1;
    const Vec2 ref = point_array[point_array.size - 1];
    const Vec2* src = points.items;
    Vec2* dst = hobby_vec.items + 3;
    hobby_vec[0] = ref;
    if (relative) {
        for (int64_t i = 0; i < points.size; i++, dst += 3) *dst = ref + *src++;
    } else {
        for (int64_t i = 0; i < points.size; i++, dst += 3) *dst = *src++;
    }
    hobby_interpolation(points.size + 1, hobby_vec.items, angles, angle_constraints, tension,
                        initial_curl, final_curl, cycle);
    Array<Vec2> tmp;
    tmp.items = hobby_vec.items + 1;
    if (cycle) {
        tmp.size = 3 * (points.size + 1);
        hobby_vec[3 * (points.size + 1)] = ref;
    } else {
        tmp.size = 3 * points.size;
    }
    cubic(tmp, false);
    hobby_vec.clear();
}

void Curve::arc(double radius_x, double radius_y, double initial_angle, double final_angle,
                double rotation) {
    const double full_angle = fabs(final_angle - initial_angle);
    const double max_radius = radius_x > radius_y ? radius_x : radius_y;
    int64_t num_points = 1 + arc_num_points(full_angle, max_radius, tolerance);
    if (num_points < MIN_POINTS) num_points = MIN_POINTS;

    initial_angle = elliptical_angle_transform(initial_angle - rotation, radius_x, radius_y);
    final_angle = elliptical_angle_transform(final_angle - rotation, radius_x, radius_y);
    const double cr = cos(rotation);
    const double sr = sin(rotation);
    double x = radius_x * cos(initial_angle);
    double y = radius_y * sin(initial_angle);
    const Vec2 point0 = {x * cr - y * sr, x * sr + y * cr};
    const Vec2 delta = point_array[point_array.size - 1] - point0;
    ensure_slots(num_points - 1);
    Vec2* dst = point_array.items + point_array.size;
    for (int64_t i = 1; i < num_points; i++) {
        double t = i / (num_points - 1.0);
        double angle = LERP(initial_angle, final_angle, t);
        x = radius_x * cos(angle);
        y = radius_y * sin(angle);
        Vec2 point = {x * cr - y * sr, x * sr + y * cr};
        *dst++ = point + delta;
    }
    point_array.size += num_points - 1;

    Vec2 v = point_array[point_array.size - 2] - point_array[point_array.size - 1];
    v *= 0.5 * (radius_x + radius_y) / v.length();
    last_ctrl = point_array[point_array.size - 1] + v;
}

void Curve::parametric(ParametricVec2 curve_function, void* data, bool relative) {
    const Vec2 last_curve_point = point_array[point_array.size - 1];
    const Vec2 ref = relative ? last_curve_point : Vec2{0, 0};
    const double tolerance_sq = tolerance * tolerance;
    double u = 0;
    Vec2 last = (*curve_function)(0, data) + ref;
    if ((last - last_curve_point).length_sq() > tolerance_sq) append(last);
    double du = 1.0 / MIN_POINTS;
    while (u < 1) {
        if (du > 1.0 / MIN_POINTS) du = 1.0 / MIN_POINTS;
        if (u + du > 1.0) du = 1.0 - u;
        Vec2 next = (*curve_function)(u + du, data) + ref;
        Vec2 mid = (*curve_function)(u + 0.5 * du, data) + ref;
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = (*curve_function)(u + du / 3, data) + ref;
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            du *= 0.5;
            next = mid;
            mid = (*curve_function)(u + 0.5 * du, data) + ref;
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = (*curve_function)(u + du / 3, data) + ref;
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }
        append(next);
        last = next;
        u += du;
        du *= 2;
    }
}

int64_t Curve::commands(const CurveInstruction* items, int64_t size) {
    const CurveInstruction* item = items;
    const CurveInstruction* end = items + size;
    Array<Vec2> points = {0};
    while (item < end) {
        const char instruction = (item++)->command;
        switch (instruction) {
            case 'h':
            case 'H':
                if (end - item < 1) return item - items - 1;
                horizontal((double*)item, 1, instruction == 'h');
                item += 1;
                break;
            case 'v':
            case 'V':
                if (end - item < 1) return item - items - 1;
                vertical((double*)item, 1, instruction == 'v');
                item += 1;
                break;
            case 'l':
            case 'L':
                if (end - item < 2) return item - items - 1;
                points.items = (Vec2*)item;
                points.size = 1;
                segment(points, instruction == 'l');
                item += 2;
                break;
            case 'c':
            case 'C':
                if (end - item < 6) return item - items - 1;
                points.items = (Vec2*)item;
                points.size = 3;
                cubic(points, instruction == 'c');
                item += 6;
                break;
            case 's':
            case 'S':
                if (end - item < 4) return item - items - 1;
                points.items = (Vec2*)item;
                points.size = 2;
                cubic_smooth(points, instruction == 's');
                item += 4;
                break;
            case 'q':
            case 'Q':
                if (end - item < 4) return item - items - 1;
                points.items = (Vec2*)item;
                points.size = 2;
                quadratic(points, instruction == 'q');
                item += 4;
                break;
            case 't':
            case 'T':
                if (end - item < 2) return item - items - 1;
                points.items = (Vec2*)item;
                points.size = 1;
                quadratic_smooth(points, instruction == 't');
                item += 2;
                break;
            case 'a':
                // Turn
                if (end - item < 2) return item - items - 1;
                turn(item->number, (item + 1)->number);
                item += 2;
                break;
            case 'A':
                // Circular arc
                if (end - item < 3) return item - items - 1;
                arc(item->number, item->number, (item + 1)->number, (item + 2)->number, 0);
                item += 3;
                break;
            case 'E':
                // Elliptical arc
                if (end - item < 5) return item - items - 1;
                arc(item->number, (item + 1)->number, (item + 2)->number, (item + 3)->number,
                    (item + 4)->number);
                item += 5;
                break;
            default:
                return item - items - 1;
        }
    }
    return size;
}

}  // namespace gdstk
