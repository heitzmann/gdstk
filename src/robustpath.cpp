/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <gdstk/allocator.hpp>
#include <gdstk/curve.hpp>
#include <gdstk/gdsii.hpp>
#include <gdstk/robustpath.hpp>
#include <gdstk/utils.hpp>

namespace gdstk {

static double interp(const Interpolation &interpolation, double u) {
    double result = 0;
    u = u < 0 ? 0 : (u > 1 ? 1 : u);
    switch (interpolation.type) {
        case InterpolationType::Constant:
            result = interpolation.value;
            break;
        case InterpolationType::Linear:
            result = LERP(interpolation.initial_value, interpolation.final_value, u);
            break;
        case InterpolationType::Smooth:
            result = SERP(interpolation.initial_value, interpolation.final_value, u);
            break;
        case InterpolationType::Parametric:
            result = (*interpolation.function)(u, interpolation.data);
    }
    return result;
}

void SubPath::print() const {
    switch (type) {
        case SubPathType::Segment:
            printf("Segment <%p>: (%lg, %lg) - (%lg, %lg)\n", this, begin.x, begin.y, end.x, end.y);
            break;
        case SubPathType::Arc:
            printf("Arc <%p>: center (%lg, %lg), radii %lg and %lg\n", this, center.x, center.y,
                   radius_x, radius_y);
            break;
        case SubPathType::Bezier:
            printf("Bezier <%p>: ", this);
            ctrl.print(true);
            break;
        case SubPathType::Bezier2:
            printf("Quadratic bezier <%p>: (%lg, %lg) - (%lg, %lg) - (%lg, %lg)\n", this, p0.x,
                   p0.y, p1.x, p1.y, p2.x, p2.y);
            break;
        case SubPathType::Bezier3:
            printf("Cubic bezier <%p>: (%lg, %lg) - (%lg, %lg) - (%lg, %lg) - (%lg, %lg)\n", this,
                   p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
            break;
        case SubPathType::Parametric:
            printf("Parametric <%p>: reference = (%lg, %lg), f <%p>, df <%p>, data <%p> and <%p>\n",
                   this, reference.x, reference.y, path_function, path_gradient, func_data,
                   grad_data);
            break;
    }
}

Vec2 SubPath::gradient(double u, const double *trafo) const {
    Vec2 grad;
    u = u < 0 ? 0 : (u > 1 ? 1 : u);
    switch (type) {
        case SubPathType::Segment:
            grad = end - begin;
            break;
        case SubPathType::Arc: {
            const double angle = LERP(angle_i, angle_f, u);
            const double dx = -radius_x * (angle_f - angle_i) * sin(angle);
            const double dy = radius_y * (angle_f - angle_i) * cos(angle);
            grad = Vec2{dx * cos_rot - dy * sin_rot, dx * sin_rot + dy * cos_rot};
        } break;
        case SubPathType::Bezier: {
            const uint64_t count = ctrl.count - 1;
            Vec2 *_ctrl = (Vec2 *)allocate(sizeof(Vec2) * count);
            Vec2 *dst = _ctrl;
            const Vec2 *src = ctrl.items;
            for (uint64_t i = 0; i < count; i++, src++, dst++)
                *dst = (double)count * (*(src + 1) - *src);
            grad = eval_bezier(u, _ctrl, count);
            free_allocation(_ctrl);
        } break;
        case SubPathType::Bezier2: {
            const Vec2 dp0 = 2 * (p1 - p0);
            const Vec2 dp1 = 2 * (p2 - p1);
            grad = eval_line(u, dp0, dp1);
        } break;
        case SubPathType::Bezier3: {
            const Vec2 dp0 = 3 * (p1 - p0);
            const Vec2 dp1 = 3 * (p2 - p1);
            const Vec2 dp2 = 3 * (p3 - p2);
            grad = eval_bezier2(u, dp0, dp1, dp2);
        } break;
        case SubPathType::Parametric:
            if (path_gradient == NULL) {
                const double u0 = u - step < 0 ? 0 : u - step;
                const double u1 = u + step > 1 ? 1 : u + step;
                grad =
                    ((*path_function)(u1, func_data) - (*path_function)(u0, func_data)) / (u1 - u0);
            } else {
                grad = (*path_gradient)(u, grad_data);
            }
            break;
        default:
            grad = Vec2{0, 0};
    }
    const double dx = grad.x;
    const double dy = grad.y;
    Vec2 result = Vec2{dx * trafo[0] + dy * trafo[1], dx * trafo[3] + dy * trafo[4]};
    return result;
}

Vec2 SubPath::eval(double u, const double *trafo) const {
    if (u < 0) {
        const Vec2 p = eval(0, trafo);
        const Vec2 v = gradient(0, trafo);
        return p + v * u;
    }

    if (u > 1) {
        const Vec2 p = eval(1, trafo);
        const Vec2 v = gradient(1, trafo);
        return p + v * (u - 1);
    }

    Vec2 point;
    switch (type) {
        case SubPathType::Segment:
            point = LERP(begin, end, u);
            break;
        case SubPathType::Arc: {
            const double angle = LERP(angle_i, angle_f, u);
            const double x = radius_x * cos(angle);
            const double y = radius_y * sin(angle);
            point = center + Vec2{x * cos_rot - y * sin_rot, x * sin_rot + y * cos_rot};
        } break;
        case SubPathType::Bezier: {
            point = eval_bezier(u, ctrl.items, ctrl.count);
        } break;
        case SubPathType::Bezier2: {
            point = eval_bezier2(u, p0, p1, p2);
        } break;
        case SubPathType::Bezier3: {
            point = eval_bezier3(u, p0, p1, p2, p3);
        } break;
        case SubPathType::Parametric:
            point = (*path_function)(u, func_data) + reference;
            break;
        default:
            point = Vec2{0, 0};
    }
    const double x = point.x;
    const double y = point.y;
    Vec2 result =
        Vec2{x * trafo[0] + y * trafo[1] + trafo[2], x * trafo[3] + y * trafo[4] + trafo[5]};
    return result;
}

Vec2 RobustPath::spine_position(const SubPath &subpath, double u) const {
    return subpath.eval(u, trafo);
}

Vec2 RobustPath::spine_gradient(const SubPath &subpath, double u) const {
    return subpath.gradient(u, trafo);
}

Vec2 RobustPath::center_position(const SubPath &subpath, const Interpolation &offset_,
                                 double u) const {
    const Vec2 sp_position = spine_position(subpath, u);
    const double offset_value = interp(offset_, u) * offset_scale;
    Vec2 spine_normal = subpath.gradient(u, trafo).ortho();
    spine_normal.normalize();
    Vec2 result = sp_position + offset_value * spine_normal;
    return result;
}

Vec2 RobustPath::center_gradient(const SubPath &subpath, const Interpolation &offset_,
                                 double u) const {
    const double step = 1.0 / (10.0 * max_evals);
    const double u0 = u - step < 0 ? 0 : u - step;
    const double u1 = u + step > 1 ? 1 : u + step;
    Vec2 result =
        (center_position(subpath, offset_, u1) - center_position(subpath, offset_, u0)) / (u1 - u0);
    return result;
}

Vec2 RobustPath::left_position(const SubPath &subpath, const Interpolation &offset_,
                               const Interpolation &width_, double u) const {
    const Vec2 ct_position = center_position(subpath, offset_, u);
    const double width_value = interp(width_, u) * width_scale;
    const Vec2 ct_gradient = center_gradient(subpath, offset_, u);
    Vec2 center_normal = ct_gradient.ortho();
    center_normal.normalize();
    Vec2 pos = ct_position + 0.5 * width_value * center_normal;
    return pos;
}

Vec2 RobustPath::left_gradient(const SubPath &subpath, const Interpolation &offset_,
                               const Interpolation &width_, double u) const {
    const double step = 1.0 / (10.0 * max_evals);
    const double u0 = u - step < 0 ? 0 : u - step;
    const double u1 = u + step > 1 ? 1 : u + step;
    Vec2 result = (left_position(subpath, offset_, width_, u1) -
                   left_position(subpath, offset_, width_, u0)) /
                  (u1 - u0);
    return result;
}

Vec2 RobustPath::right_position(const SubPath &subpath, const Interpolation &offset_,
                                const Interpolation &width_, double u) const {
    const Vec2 ct_position = center_position(subpath, offset_, u);
    const double width_value = interp(width_, u) * width_scale;
    const Vec2 ct_gradient = center_gradient(subpath, offset_, u);
    Vec2 center_normal = ct_gradient.ortho();
    center_normal.normalize();
    Vec2 pos = ct_position - 0.5 * width_value * center_normal;
    return pos;
}

Vec2 RobustPath::right_gradient(const SubPath &subpath, const Interpolation &offset_,
                                const Interpolation &width_, double u) const {
    const double step = 1.0 / (10.0 * max_evals);
    const double u0 = u - step < 0 ? 0 : u - step;
    const double u1 = u + step > 1 ? 1 : u + step;
    Vec2 result = (right_position(subpath, offset_, width_, u1) -
                   right_position(subpath, offset_, width_, u0)) /
                  (u1 - u0);
    return result;
}

// NOTE: Does NOT include the point at u0.
void RobustPath::spine_points(const SubPath &subpath, double u0, double u1,
                              Array<Vec2> &result) const {
    const double tolerance_sq = tolerance * tolerance;
    double u = u0;
    Vec2 last = spine_position(subpath, u0);
    uint64_t counter = max_evals - 1;
    double du = 1.0 / GDSTK_MIN_POINTS;
    while (u < u1 && counter-- > 0) {
        if (du > 1.0 / GDSTK_MIN_POINTS) du = 1.0 / GDSTK_MIN_POINTS;
        if (u + du > u1) du = u1 - u;
        Vec2 next = spine_position(subpath, u + du);
        Vec2 mid = spine_position(subpath, u + 0.5 * du);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = spine_position(subpath, u + du / 3);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            du *= 0.5;
            next = mid;
            mid = spine_position(subpath, u + 0.5 * du);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = spine_position(subpath, u + du / 3);
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }
        result.append(next);
        last = next;
        u += du;
        du *= 2;
    }
}

// NOTE: Does NOT include the point at u0.
void RobustPath::center_points(const SubPath &subpath, const Interpolation &offset_, double u0,
                               double u1, Array<Vec2> &result) const {
    const double tolerance_sq = tolerance * tolerance;
    double u = u0;
    Vec2 last = center_position(subpath, offset_, u0);
    uint64_t counter = max_evals - 1;
    double du = 1.0 / GDSTK_MIN_POINTS;
    while (u < u1 && counter-- > 0) {
        if (du > 1.0 / GDSTK_MIN_POINTS) du = 1.0 / GDSTK_MIN_POINTS;
        if (u + du > u1) du = u1 - u;
        Vec2 next = center_position(subpath, offset_, u + du);
        Vec2 mid = center_position(subpath, offset_, u + 0.5 * du);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = center_position(subpath, offset_, u + du / 3);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            du *= 0.5;
            next = mid;
            mid = center_position(subpath, offset_, u + 0.5 * du);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = center_position(subpath, offset_, u + du / 3);
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }
        result.append(next);
        last = next;
        u += du;
        du *= 2;
    }
}

// NOTE: Does NOT include the point at u0.
void RobustPath::left_points(const SubPath &subpath, const Interpolation &offset_,
                             const Interpolation &width_, double u0, double u1,
                             Array<Vec2> &result) const {
    const double tolerance_sq = tolerance * tolerance;
    double u = u0;
    Vec2 last = left_position(subpath, offset_, width_, u0);
    uint64_t counter = max_evals - 1;
    double du = 1.0 / GDSTK_MIN_POINTS;
    while (u < u1 && counter-- > 0) {
        if (du > 1.0 / GDSTK_MIN_POINTS) du = 1.0 / GDSTK_MIN_POINTS;
        if (u + du > u1) du = u1 - u;
        Vec2 next = left_position(subpath, offset_, width_, u + du);
        Vec2 mid = left_position(subpath, offset_, width_, u + 0.5 * du);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = left_position(subpath, offset_, width_, u + du / 3);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            du *= 0.5;
            next = mid;
            mid = left_position(subpath, offset_, width_, u + 0.5 * du);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = left_position(subpath, offset_, width_, u + du / 3);
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }
        result.append(next);
        last = next;
        u += du;
        du *= 2;
    }
}

// NOTE: Does NOT include the point at u0.
void RobustPath::right_points(const SubPath &subpath, const Interpolation &offset_,
                              const Interpolation &width_, double u0, double u1,
                              Array<Vec2> &result) const {
    const double tolerance_sq = tolerance * tolerance;
    double u = u0;
    Vec2 last = right_position(subpath, offset_, width_, u0);
    uint64_t counter = max_evals - 1;
    double du = 1.0 / GDSTK_MIN_POINTS;
    while (u < u1 && counter-- > 0) {
        if (du > 1.0 / GDSTK_MIN_POINTS) du = 1.0 / GDSTK_MIN_POINTS;
        if (u + du > u1) du = u1 - u;
        Vec2 next = right_position(subpath, offset_, width_, u + du);
        Vec2 mid = right_position(subpath, offset_, width_, u + 0.5 * du);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = right_position(subpath, offset_, width_, u + du / 3);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            du *= 0.5;
            next = mid;
            mid = right_position(subpath, offset_, width_, u + 0.5 * du);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = right_position(subpath, offset_, width_, u + du / 3);
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }
        result.append(next);
        last = next;
        u += du;
        du *= 2;
    }
}

static void interpolation_print(const Interpolation interp) {
    switch (interp.type) {
        case InterpolationType::Constant:
            printf("Constant interpolation to %lg\n", interp.value);
            break;
        case InterpolationType::Linear:
            printf("Linear interpolation from %lg to %lg\n", interp.initial_value,
                   interp.final_value);
            break;
        case InterpolationType::Smooth:
            printf("Smooth interpolation from %lg to %lg\n", interp.initial_value,
                   interp.final_value);
            break;
        case InterpolationType::Parametric:
            printf("Parametric interpolation (function <%p>, data <%p>)\n", interp.function,
                   interp.data);
            break;
    }
}

void RobustPath::init(const Vec2 initial_position, double width, double offset, double tolerance_,
                      uint64_t max_evals_, Tag tag) {
    tolerance = tolerance_;
    max_evals = max_evals_;
    width_scale = 1;
    offset_scale = 1;
    trafo[0] = 1;
    trafo[4] = 1;
    end_point = initial_position;
    for (uint64_t i = 0; i < num_elements; i++) {
        elements[i].end_width = width;
        elements[i].end_offset = offset;
        elements[i].tag = tag;
    }
}

void RobustPath::init(const Vec2 initial_position, const double *width, const double *offset,
                      double tolerance_, uint64_t max_evals_, const Tag *tag) {
    tolerance = tolerance_;
    max_evals = max_evals_;
    width_scale = 1;
    offset_scale = 1;
    trafo[0] = 1;
    trafo[4] = 1;
    end_point = initial_position;
    for (uint64_t i = 0; i < num_elements; i++) {
        elements[i].end_width = width[i];
        elements[i].end_offset = offset[i];
        elements[i].tag = tag[i];
    }
}

void RobustPath::init(const Vec2 initial_position, uint64_t num_elements_, double width,
                      double separation, double tolerance_, uint64_t max_evals_, Tag tag) {
    num_elements = num_elements_;
    elements = (RobustPathElement *)allocate_clear(num_elements * sizeof(RobustPathElement));
    tolerance = tolerance_;
    max_evals = max_evals_;
    width_scale = 1;
    offset_scale = 1;
    trafo[0] = 1;
    trafo[4] = 1;
    end_point = initial_position;
    double i0 = 0.5 * (num_elements - 1);
    for (uint64_t i = 0; i < num_elements; i++) {
        elements[i].end_width = width;
        elements[i].end_offset = separation * (i - i0);
        elements[i].tag = tag;
    }
}

void RobustPath::init(const Vec2 initial_position, uint64_t num_elements_, const double *width,
                      const double *offset, double tolerance_, uint64_t max_evals_,
                      const Tag *tag) {
    num_elements = num_elements_;
    elements = (RobustPathElement *)allocate_clear(num_elements * sizeof(RobustPathElement));
    tolerance = tolerance_;
    max_evals = max_evals_;
    width_scale = 1;
    offset_scale = 1;
    trafo[0] = 1;
    trafo[4] = 1;
    end_point = initial_position;
    for (uint64_t i = 0; i < num_elements; i++) {
        elements[i].end_width = width[i];
        elements[i].end_offset = offset[i];
        elements[i].tag = tag[i];
    }
}

void RobustPath::print(bool all) const {
    printf("RobustPath <%p> at (%lg, %lg), count %" PRIu64 ", %" PRIu64
           " elements, %s path,%s scaled widths, tolerance %lg, max_evals %" PRIu64
           ", properties <%p>, owner <%p>\n",
           this, end_point.x, end_point.y, subpath_array.count, num_elements,
           simple_path ? "GDSII" : "polygonal", scale_width ? "" : " no", tolerance, max_evals,
           properties, owner);
    printf("Transform: %lg,\t%lg,\t%lg\n           %lg,\t%lg,\t%lg\n", trafo[0], trafo[1], trafo[2],
           trafo[3], trafo[4], trafo[5]);
    if (all) {
        printf("Subpaths (count %" PRIu64 "/%" PRIu64 "):\n", subpath_array.count,
               subpath_array.capacity);
        for (uint64_t ns = 0; ns < subpath_array.count; ns++) {
            printf("Subpath %" PRIu64 ": ", ns);
            subpath_array[ns].print();
        }
        RobustPathElement *el = elements;
        for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
            printf("Element %" PRIu64 ", layer %" PRIu32 ", datatype %" PRIu32
                   ", end %s (function <%p>, data <%p>), end extensions (%lg, %lg)\n",
                   ne, get_layer(el->tag), get_type(el->tag), end_type_name(el->end_type),
                   el->end_function, el->end_function_data, el->end_extensions.u,
                   el->end_extensions.v);
            printf("Width interpolations (count %" PRIu64 "/%" PRIu64 "):\n", el->width_array.count,
                   el->width_array.capacity);
            Interpolation *interp = el->width_array.items;
            for (uint64_t ni = 0; ni < el->width_array.count; ni++, interp++) {
                printf("Width %" PRIu64 ": ", ni);
                interpolation_print(*interp);
            }
            printf("Offset interpolations (count %" PRIu64 "/%" PRIu64 "):\n",
                   el->offset_array.count, el->offset_array.capacity);
            interp = el->offset_array.items;
            for (uint64_t ni = 0; ni < el->offset_array.count; ni++, interp++) {
                printf("Offset %" PRIu64 ": ", ni);
                interpolation_print(*interp);
            }
        }
    }
    properties_print(properties);
    repetition.print();
}

void RobustPath::clear() {
    subpath_array.clear();
    RobustPathElement *el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        el->width_array.clear();
        el->offset_array.clear();
    }
    free_allocation(elements);
    elements = NULL;
    num_elements = 0;
    repetition.clear();
    properties_clear(properties);
}

void RobustPath::copy_from(const RobustPath &path) {
    properties = properties_copy(path.properties);
    repetition.copy_from(path.repetition);
    end_point = path.end_point;
    subpath_array.copy_from(path.subpath_array);
    num_elements = path.num_elements;
    elements = (RobustPathElement *)allocate_clear(num_elements * sizeof(RobustPathElement));
    tolerance = path.tolerance;
    max_evals = path.max_evals;
    width_scale = path.width_scale;
    offset_scale = path.offset_scale;
    memcpy(trafo, path.trafo, 6 * sizeof(double));
    scale_width = path.scale_width;
    simple_path = path.simple_path;

    RobustPathElement *src = path.elements;
    RobustPathElement *dst = elements;
    for (uint64_t ne = 0; ne < path.num_elements; ne++, src++, dst++) {
        dst->tag = src->tag;
        dst->end_width = src->end_width;
        dst->end_offset = src->end_offset;
        dst->end_type = src->end_type;
        dst->end_extensions = src->end_extensions;
        dst->end_function = src->end_function;
        dst->end_function_data = src->end_function_data;
        dst->width_array.copy_from(src->width_array);
        dst->offset_array.copy_from(src->offset_array);
    }
}

void RobustPath::translate(const Vec2 v) {
    trafo[2] += v.x;
    trafo[5] += v.y;
}

void RobustPath::simple_scale(double scale_factor) {
    trafo[0] *= scale_factor;
    trafo[1] *= scale_factor;
    trafo[2] *= scale_factor;
    trafo[3] *= scale_factor;
    trafo[4] *= scale_factor;
    trafo[5] *= scale_factor;
    offset_scale *= fabs(scale_factor);
    if (scale_width) width_scale *= fabs(scale_factor);
    RobustPathElement *el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        el->end_extensions *= scale_factor;
    }
}

void RobustPath::scale(double scale_factor, const Vec2 center) {
    const Vec2 delta = center * (1 - scale_factor);
    simple_scale(scale_factor);
    translate(delta);
}

void RobustPath::mirror(const Vec2 p0, const Vec2 p1) {
    Vec2 direction = p0 - p1;
    direction.normalize();
    translate(-p1);
    const double tr0 = trafo[0];
    const double tr1 = trafo[1];
    const double tr2 = trafo[2];
    const double tr3 = trafo[3];
    const double tr4 = trafo[4];
    const double tr5 = trafo[5];
    const double m0 = direction.x * direction.x - direction.y * direction.y;
    const double m1 = 2 * direction.x * direction.y;
    const double m3 = m1;
    const double m4 = -m0;
    trafo[0] = m0 * tr0 + m1 * tr3;
    trafo[1] = m0 * tr1 + m1 * tr4;
    trafo[2] = m0 * tr2 + m1 * tr5;
    trafo[3] = m3 * tr0 + m4 * tr3;
    trafo[4] = m3 * tr1 + m4 * tr4;
    trafo[5] = m3 * tr2 + m4 * tr5;
    translate(p1);
    offset_scale *= -1;
}

void RobustPath::simple_rotate(double angle) {
    const double c = cos(angle);
    const double s = sin(angle);
    const double tr0 = trafo[0];
    const double tr1 = trafo[1];
    const double tr2 = trafo[2];
    const double tr3 = trafo[3];
    const double tr4 = trafo[4];
    const double tr5 = trafo[5];
    trafo[0] = tr0 * c - tr3 * s;
    trafo[1] = tr1 * c - tr4 * s;
    trafo[2] = tr2 * c - tr5 * s;
    trafo[3] = tr0 * s + tr3 * c;
    trafo[4] = tr1 * s + tr4 * c;
    trafo[5] = tr2 * s + tr5 * c;
}

void RobustPath::rotate(double angle, const Vec2 center) {
    translate(-center);
    simple_rotate(angle);
    translate(center);
}

void RobustPath::x_reflection() {
    trafo[3] = -trafo[3];
    trafo[4] = -trafo[4];
    trafo[5] = -trafo[5];
    offset_scale *= -1;
}

void RobustPath::transform(double magnification, bool x_refl, double rotation, const Vec2 origin) {
    simple_scale(magnification);
    if (x_refl) x_reflection();
    simple_rotate(rotation);
    translate(origin);
}

void RobustPath::apply_repetition(Array<RobustPath *> &result) {
    if (repetition.type == RepetitionType::None) return;

    Array<Vec2> offsets = {};
    repetition.get_offsets(offsets);
    repetition.clear();

    // Skip first offset (0, 0)
    Vec2 *offset_p = offsets.items + 1;
    result.ensure_slots(offsets.count - 1);
    for (uint64_t offset_count = offsets.count - 1; offset_count > 0; offset_count--) {
        RobustPath *path = (RobustPath *)allocate_clear(sizeof(RobustPath));
        path->copy_from(*this);
        path->translate(*offset_p++);
        result.append_unsafe(path);
    }

    offsets.clear();
    return;
}

void RobustPath::fill_widths_and_offsets(const Interpolation *width_,
                                         const Interpolation *offset_) {
    if (width_ == NULL) {
        Interpolation interp = {InterpolationType::Constant};
        RobustPathElement *el = elements;
        for (uint64_t ne = num_elements; ne > 0; ne--, el++) {
            interp.value = el->end_width;
            el->width_array.append(interp);
        }
    } else {
        RobustPathElement *el = elements;
        for (uint64_t ne = num_elements; ne > 0; ne--, el++, width_++) {
            el->width_array.append(*width_);
            el->end_width = interp(*width_, 1);
        }
    }
    if (offset_ == NULL) {
        Interpolation interp = {InterpolationType::Constant};
        RobustPathElement *el = elements;
        for (uint64_t ne = num_elements; ne > 0; ne--, el++) {
            interp.value = el->end_offset;
            el->offset_array.append(interp);
        }
    } else {
        RobustPathElement *el = elements;
        for (uint64_t ne = num_elements; ne > 0; ne--, el++, offset_++) {
            el->offset_array.append(*offset_);
            el->end_offset = interp(*offset_, 1);
        }
    }
}

void RobustPath::horizontal(double coord_x, const Interpolation *width_,
                            const Interpolation *offset_, bool relative) {
    if (relative) coord_x += end_point.x;
    segment(Vec2{coord_x, end_point.y}, width_, offset_, false);
}

void RobustPath::vertical(double coord_y, const Interpolation *width_, const Interpolation *offset_,
                          bool relative) {
    if (relative) coord_y += end_point.y;
    segment(Vec2{end_point.x, coord_y}, width_, offset_, false);
}

void RobustPath::segment(const Vec2 end_pt, const Interpolation *width_,
                         const Interpolation *offset_, bool relative) {
    SubPath sub = {SubPathType::Segment};
    sub.begin = end_point;
    sub.end = end_pt;
    if (relative) sub.end += end_point;
    end_point = sub.end;
    subpath_array.append(sub);
    fill_widths_and_offsets(width_, offset_);
}

void RobustPath::cubic(const Vec2 point1, const Vec2 point2, const Vec2 point3,
                       const Interpolation *width_, const Interpolation *offset_, bool relative) {
    SubPath sub = {SubPathType::Bezier3};
    sub.p0 = end_point;
    sub.p1 = point1;
    sub.p2 = point2;
    sub.p3 = point3;
    if (relative) {
        sub.p1 += end_point;
        sub.p2 += end_point;
        sub.p3 += end_point;
    }
    end_point = sub.p3;
    subpath_array.append(sub);
    fill_widths_and_offsets(width_, offset_);
}

void RobustPath::cubic_smooth(const Vec2 point2, const Vec2 point3, const Interpolation *width_,
                              const Interpolation *offset_, bool relative) {
    SubPath sub = {SubPathType::Bezier3};
    sub.p0 = end_point;
    sub.p1 = end_point;
    if (subpath_array.count > 0)
        sub.p1 += subpath_array[subpath_array.count - 1].gradient(1, trafo) / 3;
    sub.p2 = point2;
    sub.p3 = point3;
    if (relative) {
        sub.p2 += end_point;
        sub.p3 += end_point;
    }
    end_point = sub.p3;
    subpath_array.append(sub);
    fill_widths_and_offsets(width_, offset_);
}

void RobustPath::quadratic(const Vec2 point1, const Vec2 point2, const Interpolation *width_,
                           const Interpolation *offset_, bool relative) {
    SubPath sub = {SubPathType::Bezier2};
    sub.p0 = end_point;
    sub.p1 = point1;
    sub.p2 = point2;
    if (relative) {
        sub.p1 += end_point;
        sub.p2 += end_point;
    }
    end_point = sub.p2;
    subpath_array.append(sub);
    fill_widths_and_offsets(width_, offset_);
}

void RobustPath::quadratic_smooth(const Vec2 point2, const Interpolation *width_,
                                  const Interpolation *offset_, bool relative) {
    SubPath sub = {SubPathType::Bezier2};
    sub.p0 = end_point;
    sub.p1 = end_point;
    if (subpath_array.count > 0)
        sub.p1 += subpath_array[subpath_array.count - 1].gradient(1, trafo) / 2;
    sub.p2 = point2;
    if (relative) sub.p2 += end_point;
    end_point = sub.p2;
    subpath_array.append(sub);
    fill_widths_and_offsets(width_, offset_);
}

void RobustPath::bezier(const Array<Vec2> point_array, const Interpolation *width_,
                        const Interpolation *offset_, bool relative) {
    SubPath sub = {SubPathType::Bezier};
    sub.ctrl.append(end_point);
    sub.ctrl.extend(point_array);
    if (relative) {
        for (uint64_t i = 1; i <= point_array.count; i++) sub.ctrl[i] += end_point;
    }
    end_point = sub.ctrl[sub.ctrl.count - 1];
    subpath_array.append(sub);
    fill_widths_and_offsets(width_, offset_);
}

void RobustPath::interpolation(const Array<Vec2> point_array, double *angles,
                               bool *angle_constraints, Vec2 *tension, double initial_curl,
                               double final_curl, bool cycle, const Interpolation *width_,
                               const Interpolation *offset_, bool relative) {
    Array<Vec2> hobby_vec = {};
    hobby_vec.ensure_slots(3 * (point_array.count + 1));
    hobby_vec.count = 3 * (point_array.count + 1);
    const Vec2 ref = end_point;
    const Vec2 *src = point_array.items;
    Vec2 *dst = hobby_vec.items + 3;
    hobby_vec[0] = ref;
    if (relative) {
        for (uint64_t i = 0; i < point_array.count; i++, dst += 3) *dst = ref + *src++;
    } else {
        for (uint64_t i = 0; i < point_array.count; i++, dst += 3) *dst = *src++;
    }
    hobby_interpolation(point_array.count + 1, hobby_vec.items, angles, angle_constraints, tension,
                        initial_curl, final_curl, cycle);
    dst = hobby_vec.items + 1;
    for (uint64_t i = 0; i < point_array.count; i++, dst += 3) {
        cubic(*dst, *(dst + 1), *(dst + 2), width_, offset_, false);
    }
    if (cycle) cubic(*dst, *(dst + 1), ref, width_, offset_, false);
    hobby_vec.clear();
}

void RobustPath::arc(double radius_x, double radius_y, double initial_angle, double final_angle,
                     double rotation, const Interpolation *width_, const Interpolation *offset_) {
    SubPath sub = {SubPathType::Arc};
    sub.radius_x = radius_x;
    sub.radius_y = radius_y;
    sub.angle_i = initial_angle - rotation;
    sub.angle_f = final_angle - rotation;
    sub.cos_rot = cos(rotation);
    sub.sin_rot = sin(rotation);
    double x = radius_x * cos(sub.angle_i);
    double y = radius_y * sin(sub.angle_i);
    sub.center =
        end_point - Vec2{x * sub.cos_rot - y * sub.sin_rot, x * sub.sin_rot + y * sub.cos_rot};
    x = radius_x * cos(sub.angle_f);
    y = radius_y * sin(sub.angle_f);
    end_point =
        sub.center + Vec2{x * sub.cos_rot - y * sub.sin_rot, x * sub.sin_rot + y * sub.cos_rot};
    subpath_array.append(sub);
    fill_widths_and_offsets(width_, offset_);
}

void RobustPath::turn(double radius, double angle, const Interpolation *width_,
                      const Interpolation *offset_) {
    Vec2 direction = Vec2{1, 0};
    if (subpath_array.count > 0)
        direction = subpath_array[subpath_array.count - 1].gradient(1, trafo);
    const double initial_angle = direction.angle() + (angle < 0 ? 0.5 * M_PI : -0.5 * M_PI);
    arc(radius, radius, initial_angle, initial_angle + angle, 0, width_, offset_);
}

void RobustPath::parametric(ParametricVec2 curve_function, void *func_data,
                            ParametricVec2 curve_gradient, void *grad_data,
                            const Interpolation *width_, const Interpolation *offset_,
                            bool relative) {
    SubPath sub = {SubPathType::Parametric};
    sub.path_function = curve_function;
    if (curve_gradient == NULL) {
        sub.step = 1.0 / (10.0 * max_evals);
    } else {
        sub.path_gradient = curve_gradient;
        sub.grad_data = grad_data;
    }
    sub.func_data = func_data;
    if (relative) sub.reference = end_point;
    end_point = sub.eval(1, trafo);
    subpath_array.append(sub);
    fill_widths_and_offsets(width_, offset_);
}

uint64_t RobustPath::commands(const CurveInstruction *items, uint64_t count) {
    const CurveInstruction *item = items;
    const CurveInstruction *end = items + count;
    while (item < end) {
        const char instruction = (item++)->command;
        switch (instruction) {
            case 'h':
            case 'H':
                if (end - item < 1) return item - items - 1;
                horizontal(item->number, NULL, NULL, instruction == 'h');
                item += 1;
                break;
            case 'v':
            case 'V':
                if (end - item < 1) return item - items - 1;
                vertical(item->number, NULL, NULL, instruction == 'v');
                item += 1;
                break;
            case 'l':
            case 'L':
                if (end - item < 2) return item - items - 1;
                segment(Vec2{item[0].number, item[1].number}, NULL, NULL, instruction == 'l');
                item += 2;
                break;
            case 'c':
            case 'C':
                if (end - item < 6) return item - items - 1;
                cubic(Vec2{item[0].number, item[1].number}, Vec2{item[2].number, item[3].number},
                      Vec2{item[4].number, item[4].number}, NULL, NULL, instruction == 'c');
                item += 6;
                break;
            case 's':
            case 'S':
                if (end - item < 4) return item - items - 1;
                cubic_smooth(Vec2{item[0].number, item[1].number},
                             Vec2{item[2].number, item[3].number}, NULL, NULL, instruction == 's');
                item += 4;
                break;
            case 'q':
            case 'Q':
                if (end - item < 4) return item - items - 1;
                quadratic(Vec2{item[0].number, item[1].number},
                          Vec2{item[2].number, item[3].number}, NULL, NULL, instruction == 'q');
                item += 4;
                break;
            case 't':
            case 'T':
                if (end - item < 2) return item - items - 1;
                quadratic_smooth(Vec2{item[0].number, item[1].number}, NULL, NULL,
                                 instruction == 't');
                item += 2;
                break;
            case 'a':
                // Turn
                if (end - item < 2) return item - items - 1;
                turn(item->number, (item + 1)->number, NULL, NULL);
                item += 2;
                break;
            case 'A':
                // Circular arc
                if (end - item < 3) return item - items - 1;
                arc(item->number, item->number, (item + 1)->number, (item + 2)->number, 0, NULL,
                    NULL);
                item += 3;
                break;
            case 'E':
                // Elliptical arc
                if (end - item < 5) return item - items - 1;
                arc(item->number, (item + 1)->number, (item + 2)->number, (item + 3)->number,
                    (item + 4)->number, NULL, NULL);
                item += 5;
                break;
            default:
                return item - items - 1;
        }
    }
    return count;
}

Vec2 RobustPath::position(double u, bool from_below) const {
    if (u >= subpath_array.count)
        u = (double)subpath_array.count;
    else if (u < 0)
        u = 0;
    uint64_t idx = (uint64_t)u;
    u -= idx;
    if (from_below && u == 0 && idx > 0) {
        idx--;
        u = 1;
    } else if (idx == subpath_array.count) {
        idx--;
        u = 1;
    }
    return spine_position(subpath_array[idx], u);
}

Vec2 RobustPath::gradient(double u, bool from_below) const {
    if (u >= subpath_array.count)
        u = (double)subpath_array.count;
    else if (u < 0)
        u = 0;
    uint64_t idx = (uint64_t)u;
    u -= idx;
    if (from_below && u == 0 && idx > 0) {
        idx--;
        u = 1;
    } else if (idx == subpath_array.count) {
        idx--;
        u = 1;
    }
    return spine_gradient(subpath_array[idx], u);
}

void RobustPath::width(double u, bool from_below, double *result) const {
    if (u >= subpath_array.count)
        u = (double)subpath_array.count;
    else if (u < 0)
        u = 0;
    uint64_t idx = (uint64_t)u;
    u -= idx;
    if (from_below && u == 0 && idx > 0) {
        idx--;
        u = 1;
    } else if (idx == subpath_array.count) {
        idx--;
        u = 1;
    }
    RobustPathElement *el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++)
        *result++ = interp(el->width_array[idx], u) * width_scale;
}

void RobustPath::offset(double u, bool from_below, double *result) const {
    if (u >= subpath_array.count)
        u = (double)subpath_array.count;
    else if (u < 0)
        u = 0;
    uint64_t idx = (uint64_t)u;
    u -= idx;
    if (from_below && u == 0 && idx > 0) {
        idx--;
        u = 1;
    } else if (idx == subpath_array.count) {
        idx--;
        u = 1;
    }
    RobustPathElement *el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++)
        *result++ = interp(el->offset_array[idx], u) * offset_scale;
}

ErrorCode RobustPath::spine_intersection(const SubPath &sub0, const SubPath &sub1, double &u0,
                                         double &u1) const {
    const double tolerance_sq = tolerance * tolerance;
    Vec2 p0 = spine_position(sub0, u0);
    Vec2 p1 = spine_position(sub1, u1);
    double err_sq = (p0 - p1).length_sq();

    if (err_sq <= tolerance_sq) return ErrorCode::NoError;

    Vec2 v0 = spine_gradient(sub0, u0);
    Vec2 v1 = spine_gradient(sub1, u1);
    double norm_v0 = v0.normalize();
    double norm_v1 = v1.normalize();
    double du0;
    double du1;
    segments_intersection(p0, v0, p1, v1, du0, du1);
    du0 /= norm_v0;
    du1 /= norm_v1;

    double step = 1;
    uint64_t evals = max_evals;
    const double step_min = 1.0 / (10.0 * max_evals);
    while (evals-- > 0 || fabs(step * du0) > step_min || fabs(step * du1) > step_min) {
        double new_u0 = u0 + step * du0;
        double new_u1 = u1 + step * du1;
        p0 = spine_position(sub0, new_u0);
        p1 = spine_position(sub1, new_u1);
        double new_err_sq = (p1 - p0).length_sq();
        if (new_err_sq >= err_sq)
            step *= 0.5;
        else {
            u0 = new_u0;
            u1 = new_u1;
            err_sq = new_err_sq;

            if (err_sq <= tolerance_sq) return ErrorCode::NoError;

            v0 = spine_gradient(sub0, u0);
            v1 = spine_gradient(sub1, u1);
            norm_v0 = v0.normalize();
            norm_v1 = v1.normalize();
            segments_intersection(p0, v0, p1, v1, du0, du1);
            du0 /= norm_v0;
            du1 /= norm_v1;
        }
    }
    if (error_logger)
        fprintf(
            error_logger,
            "[GDSTK] No intersection found in RobustPath spine construction around (%lg, %lg) and (%lg, %lg).\n",
            p0.x, p0.y, p1.x, p1.y);
    return ErrorCode::IntersectionNotFound;
}

ErrorCode RobustPath::center_intersection(const SubPath &sub0, const Interpolation &offset0,
                                          const SubPath &sub1, const Interpolation &offset1,
                                          double &u0, double &u1) const {
    const double tolerance_sq = tolerance * tolerance;
    Vec2 p0 = center_position(sub0, offset0, u0);
    Vec2 p1 = center_position(sub1, offset1, u1);
    double err_sq = (p0 - p1).length_sq();

    if (err_sq <= tolerance_sq) return ErrorCode::NoError;

    Vec2 v0 = center_gradient(sub0, offset0, u0);
    Vec2 v1 = center_gradient(sub1, offset1, u1);
    double norm_v0 = v0.normalize();
    double norm_v1 = v1.normalize();
    double du0;
    double du1;
    segments_intersection(p0, v0, p1, v1, du0, du1);
    du0 /= norm_v0;
    du1 /= norm_v1;

    double step = 1;
    uint64_t evals = max_evals;
    const double step_min = 1.0 / (10.0 * max_evals);
    while (evals-- > 0 || fabs(step * du0) > step_min || fabs(step * du1) > step_min) {
        double new_u0 = u0 + step * du0;
        double new_u1 = u1 + step * du1;
        p0 = center_position(sub0, offset0, new_u0);
        p1 = center_position(sub1, offset1, new_u1);
        double new_err_sq = (p1 - p0).length_sq();
        if (new_err_sq >= err_sq) {
            step *= 0.5;
        } else {
            u0 = new_u0;
            u1 = new_u1;
            err_sq = new_err_sq;

            if (err_sq <= tolerance_sq) return ErrorCode::NoError;

            v0 = center_gradient(sub0, offset0, u0);
            v1 = center_gradient(sub1, offset1, u1);
            norm_v0 = v0.normalize();
            norm_v1 = v1.normalize();
            segments_intersection(p0, v0, p1, v1, du0, du1);
            du0 /= norm_v0;
            du1 /= norm_v1;
        }
    }
    if (error_logger)
        fprintf(
            error_logger,
            "[GDSTK] No intersection found in RobustPath center construction around (%lg, %lg) and (%lg, %lg).\n",
            p0.x, p0.y, p1.x, p1.y);
    return ErrorCode::IntersectionNotFound;
}

ErrorCode RobustPath::left_intersection(const SubPath &sub0, const Interpolation &offset0,
                                        const Interpolation &width0, const SubPath &sub1,
                                        const Interpolation &offset1, const Interpolation &width1,
                                        double &u0, double &u1) const {
    const double tolerance_sq = tolerance * tolerance;
    Vec2 p0 = left_position(sub0, offset0, width0, u0);
    Vec2 p1 = left_position(sub1, offset1, width1, u1);
    double err_sq = (p0 - p1).length_sq();

    if (err_sq <= tolerance_sq) return ErrorCode::NoError;

    Vec2 v0 = left_gradient(sub0, offset0, width0, u0);
    Vec2 v1 = left_gradient(sub1, offset1, width1, u1);
    double norm_v0 = v0.normalize();
    double norm_v1 = v1.normalize();
    double du0;
    double du1;
    segments_intersection(p0, v0, p1, v1, du0, du1);
    du0 /= norm_v0;
    du1 /= norm_v1;

    double step = 1;
    uint64_t evals = max_evals;
    const double step_min = 1.0 / (10.0 * max_evals);
    while (evals-- > 0 || fabs(step * du0) > step_min || fabs(step * du1) > step_min) {
        double new_u0 = u0 + step * du0;
        double new_u1 = u1 + step * du1;
        p0 = left_position(sub0, offset0, width0, new_u0);
        p1 = left_position(sub1, offset1, width1, new_u1);
        double new_err_sq = (p1 - p0).length_sq();
        if (new_err_sq >= err_sq) {
            step *= 0.5;
        } else {
            u0 = new_u0;
            u1 = new_u1;
            err_sq = new_err_sq;

            if (err_sq <= tolerance_sq) return ErrorCode::NoError;

            v0 = left_gradient(sub0, offset0, width0, u0);
            v1 = left_gradient(sub1, offset1, width1, u1);
            norm_v0 = v0.normalize();
            norm_v1 = v1.normalize();
            segments_intersection(p0, v0, p1, v1, du0, du1);
            du0 /= norm_v0;
            du1 /= norm_v1;
        }
    }
    if (error_logger)
        fprintf(
            error_logger,
            "[GDSTK] No intersection found in RobustPath left side construction around (%lg, %lg) and (%lg, %lg).\n",
            p0.x, p0.y, p1.x, p1.y);
    return ErrorCode::IntersectionNotFound;
}

ErrorCode RobustPath::right_intersection(const SubPath &sub0, const Interpolation &offset0,
                                         const Interpolation &width0, const SubPath &sub1,
                                         const Interpolation &offset1, const Interpolation &width1,
                                         double &u0, double &u1) const {
    const double tolerance_sq = tolerance * tolerance;
    Vec2 p0 = right_position(sub0, offset0, width0, u0);
    Vec2 p1 = right_position(sub1, offset1, width1, u1);
    double err_sq = (p0 - p1).length_sq();

    if (err_sq <= tolerance_sq) return ErrorCode::NoError;

    Vec2 v0 = right_gradient(sub0, offset0, width0, u0);
    Vec2 v1 = right_gradient(sub1, offset1, width1, u1);
    double norm_v0 = v0.normalize();
    double norm_v1 = v1.normalize();
    double du0;
    double du1;
    segments_intersection(p0, v0, p1, v1, du0, du1);
    du0 /= norm_v0;
    du1 /= norm_v1;

    double step = 1;
    uint64_t evals = max_evals;
    const double step_min = 1.0 / (10.0 * max_evals);
    while (evals-- > 0 || fabs(step * du0) > step_min || fabs(step * du1) > step_min) {
        double new_u0 = u0 + step * du0;
        double new_u1 = u1 + step * du1;
        p0 = right_position(sub0, offset0, width0, new_u0);
        p1 = right_position(sub1, offset1, width1, new_u1);
        double new_err_sq = (p1 - p0).length_sq();
        if (new_err_sq >= err_sq)
            step *= 0.5;
        else {
            u0 = new_u0;
            u1 = new_u1;
            err_sq = new_err_sq;

            if (err_sq <= tolerance_sq) return ErrorCode::NoError;

            v0 = right_gradient(sub0, offset0, width0, u0);
            v1 = right_gradient(sub1, offset1, width1, u1);
            norm_v0 = v0.normalize();
            norm_v1 = v1.normalize();
            segments_intersection(p0, v0, p1, v1, du0, du1);
            du0 /= norm_v0;
            du1 /= norm_v1;
        }
    }
    if (error_logger)
        fprintf(
            error_logger,
            "[GDSTK] No intersection found in RobustPath right side construction around (%lg, %lg) and (%lg, %lg).\n",
            p0.x, p0.y, p1.x, p1.y);
    return ErrorCode::IntersectionNotFound;
}

ErrorCode RobustPath::spine(Array<Vec2> &result) const {
    ErrorCode error_code = ErrorCode::NoError;
    if (subpath_array.count == 0) return error_code;
    result.ensure_slots(subpath_array.count + 1);
    double u0 = 0;
    SubPath *sub0 = subpath_array.items;
    SubPath *sub1 = sub0 + 1;
    result.append(spine_position(*sub0, 0));
    for (uint64_t ns = 1; ns < subpath_array.count; ns++, sub1++) {
        double u1 = 1;
        double u2 = 0;
        ErrorCode err = spine_intersection(*sub0, *sub1, u1, u2);
        if (err != ErrorCode::NoError) error_code = err;
        if (u2 < 1) {
            if (u1 > u0) spine_points(*sub0, u0, u1, result);
            u0 = u2;
            sub0 = sub1;
        }
    }
    spine_points(*sub0, u0, 1, result);
    return error_code;
}

ErrorCode RobustPath::to_polygons(bool filter, Tag tag, Array<Polygon *> &result) const {
    ErrorCode error_code = ErrorCode::NoError;
    if (num_elements == 0 || subpath_array.count == 0) return error_code;

    const double tolerance_sq = tolerance * tolerance;
    RobustPathElement *el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        if (filter && el->tag != tag) continue;

        Array<Vec2> left_side = {};
        Array<Vec2> right_side = {};
        left_side.ensure_slots(subpath_array.count);
        right_side.ensure_slots(subpath_array.count);
        Curve initial_cap = {};
        Curve final_cap = {};
        initial_cap.tolerance = tolerance;
        final_cap.tolerance = tolerance;

        {  // Initial cap
            const Vec2 cap_l =
                left_position(subpath_array[0], el->offset_array[0], el->width_array[0], 0);
            const Vec2 cap_r =
                right_position(subpath_array[0], el->offset_array[0], el->width_array[0], 0);
            if (el->end_type == EndType::Flush) {
                initial_cap.append(cap_l);
                if ((cap_l - cap_r).length_sq() > tolerance_sq) initial_cap.append(cap_r);
            } else if (el->end_type == EndType::HalfWidth || el->end_type == EndType::Extended) {
                Vec2 direction = center_gradient(subpath_array[0], el->offset_array[0], 0);
                direction.normalize();
                const double half_width = 0.5 * interp(el->width_array[0], 0) * width_scale;
                const double extension =
                    el->end_type == EndType::Extended ? el->end_extensions.u : half_width;
                if (extension > 0) initial_cap.append(cap_l);
                initial_cap.append(cap_l - extension * direction);
                if (half_width != 0) initial_cap.append(cap_r - extension * direction);
                if (extension > 0) initial_cap.append(cap_r);
            } else if (el->end_type == EndType::Round) {
                initial_cap.append(cap_l);
                const Vec2 direction = center_gradient(subpath_array[0], el->offset_array[0], 0);
                const double initial_angle = direction.angle() + 0.5 * M_PI;
                const double half_width = 0.5 * interp(el->width_array[0], 0) * width_scale;
                initial_cap.arc(half_width, half_width, initial_angle, initial_angle + M_PI, 0);
            } else if (el->end_type == EndType::Smooth) {
                initial_cap.append(cap_l);
                Array<Vec2> point_array = {};
                point_array.items = (Vec2 *)&cap_r;
                point_array.count = 1;
                bool angle_constraints[2] = {true, true};
                const Vec2 grad_l =
                    left_gradient(subpath_array[0], el->offset_array[0], el->width_array[0], 0);
                const Vec2 grad_r =
                    right_gradient(subpath_array[0], el->offset_array[0], el->width_array[0], 0);
                double angles[2] = {(-grad_l).angle(), grad_r.angle()};
                Vec2 tension[2] = {Vec2{1, 1}, Vec2{1, 1}};
                initial_cap.interpolation(point_array, angles, angle_constraints, tension, 1, 1,
                                          false, false);
            } else if (el->end_type == EndType::Function) {
                Vec2 dir_l =
                    -left_gradient(subpath_array[0], el->offset_array[0], el->width_array[0], 0);
                Vec2 dir_r =
                    right_gradient(subpath_array[0], el->offset_array[0], el->width_array[0], 0);
                dir_l.normalize();
                dir_r.normalize();
                Array<Vec2> point_array =
                    (*el->end_function)(cap_l, dir_l, cap_r, dir_r, el->end_function_data);
                initial_cap.segment(point_array, false);
                point_array.clear();
            }
        }

        {  // Left side
            double u0 = 0;
            SubPath *sub0 = subpath_array.items;
            SubPath *sub1 = sub0 + 1;
            Interpolation *offset0 = el->offset_array.items;
            Interpolation *offset1 = offset0 + 1;
            Interpolation *width0 = el->width_array.items;
            Interpolation *width1 = width0 + 1;
            for (uint64_t ns = 1; ns < subpath_array.count; ns++, sub1++, offset1++, width1++) {
                double u1 = 1;
                double u2 = 0;
                ErrorCode err =
                    left_intersection(*sub0, *offset0, *width0, *sub1, *offset1, *width1, u1, u2);
                if (err != ErrorCode::NoError) error_code = err;
                if (u2 < 1) {
                    if (u1 > u0) left_points(*sub0, *offset0, *width0, u0, u1, left_side);
                    u0 = u2;
                    sub0 = sub1;
                    offset0 = offset1;
                    width0 = width1;
                }
            }
            left_points(*sub0, *offset0, *width0, u0, 1, left_side);
        }

        {  // Right side
            double u0 = 0;
            SubPath *sub0 = subpath_array.items;
            SubPath *sub1 = sub0 + 1;
            Interpolation *offset0 = el->offset_array.items;
            Interpolation *offset1 = offset0 + 1;
            Interpolation *width0 = el->width_array.items;
            Interpolation *width1 = width0 + 1;
            for (uint64_t ns = 1; ns < subpath_array.count; ns++, sub1++, offset1++, width1++) {
                double u1 = 1;
                double u2 = 0;
                ErrorCode err =
                    right_intersection(*sub0, *offset0, *width0, *sub1, *offset1, *width1, u1, u2);
                if (err != ErrorCode::NoError) error_code = err;
                if (u2 < 1) {
                    if (u1 > u0) right_points(*sub0, *offset0, *width0, u0, u1, right_side);
                    u0 = u2;
                    sub0 = sub1;
                    offset0 = offset1;
                    width0 = width1;
                }
            }
            right_points(*sub0, *offset0, *width0, u0, 1, right_side);
        }

        {  // End cap
            const uint64_t last = subpath_array.count - 1;
            const Vec2 cap_l = left_position(subpath_array[last], el->offset_array[last],
                                             el->width_array[last], 1);
            const Vec2 cap_r = right_position(subpath_array[last], el->offset_array[last],
                                              el->width_array[last], 1);
            if (el->end_type == EndType::Flush) {
                final_cap.append(cap_r);
                if ((cap_l - cap_r).length_sq() > tolerance_sq) final_cap.append(cap_l);
            } else if (el->end_type == EndType::HalfWidth || el->end_type == EndType::Extended) {
                Vec2 direction = center_gradient(subpath_array[last], el->offset_array[last], 1);
                direction.normalize();
                const double half_width = 0.5 * interp(el->width_array[last], 1) * width_scale;
                const double extension =
                    el->end_type == EndType::Extended ? el->end_extensions.v : half_width;
                if (extension > 0) final_cap.append(cap_r);
                final_cap.append(cap_r + extension * direction);
                if (half_width != 0) final_cap.append(cap_l + extension * direction);
                if (extension > 0) final_cap.append(cap_l);
            } else if (el->end_type == EndType::Round) {
                final_cap.append(cap_r);
                const Vec2 direction =
                    center_gradient(subpath_array[last], el->offset_array[last], 1);
                const double initial_angle = direction.angle() - 0.5 * M_PI;
                const double half_width = 0.5 * interp(el->width_array[last], 1) * width_scale;
                final_cap.arc(half_width, half_width, initial_angle, initial_angle + M_PI, 0);
            } else if (el->end_type == EndType::Smooth) {
                final_cap.append(cap_r);
                Array<Vec2> point_array = {};
                point_array.items = (Vec2 *)&cap_l;
                point_array.count = 1;
                bool angle_constraints[2] = {true, true};
                const Vec2 grad_l = left_gradient(subpath_array[last], el->offset_array[last],
                                                  el->width_array[last], 1);
                const Vec2 grad_r = right_gradient(subpath_array[last], el->offset_array[last],
                                                   el->width_array[last], 1);
                double angles[2] = {grad_r.angle(), (-grad_l).angle()};
                Vec2 tension[2] = {Vec2{1, 1}, Vec2{1, 1}};
                final_cap.interpolation(point_array, angles, angle_constraints, tension, 1, 1,
                                        false, false);
            } else if (el->end_type == EndType::Function) {
                Vec2 dir_l = -left_gradient(subpath_array[last], el->offset_array[last],
                                            el->width_array[last], 1);
                Vec2 dir_r = right_gradient(subpath_array[last], el->offset_array[last],
                                            el->width_array[last], 1);
                dir_l.normalize();
                dir_r.normalize();
                Array<Vec2> point_array =
                    (*el->end_function)(cap_r, dir_r, cap_l, dir_l, el->end_function_data);
                final_cap.segment(point_array, false);
                point_array.clear();
            }
        }

        uint64_t num =
            left_side.count + initial_cap.point_array.count + final_cap.point_array.count - 2;
        right_side.ensure_slots(num);
        Vec2 *dst = right_side.items + right_side.count - 1;

        memcpy(dst, final_cap.point_array.items, sizeof(Vec2) * final_cap.point_array.count);
        dst += final_cap.point_array.count;
        final_cap.clear();

        Vec2 *src = left_side.items + left_side.count - 2;
        for (uint64_t i = left_side.count - 1; i > 0; i--) *dst++ = *src--;
        left_side.clear();

        memcpy(dst, initial_cap.point_array.items, sizeof(Vec2) * initial_cap.point_array.count);
        initial_cap.clear();
        right_side.count += num;

        Polygon *result_polygon = (Polygon *)allocate_clear(sizeof(Polygon));
        result_polygon->tag = el->tag;
        result_polygon->point_array = right_side;
        result_polygon->repetition.copy_from(repetition);
        result_polygon->properties = properties_copy(properties);
        result.append(result_polygon);
    }
    return error_code;
}

ErrorCode RobustPath::element_center(const RobustPathElement *el, Array<Vec2> &result) const {
    ErrorCode error_code = ErrorCode::NoError;
    if (subpath_array.count == 0) return error_code;
    double u0 = 0;
    SubPath *sub0 = subpath_array.items;
    SubPath *sub1 = sub0 + 1;
    Interpolation *offset0 = el->offset_array.items;
    Interpolation *offset1 = offset0 + 1;
    result.append(center_position(*sub0, *offset0, 0));
    for (uint64_t ns = 1; ns < subpath_array.count; ns++, sub1++, offset1++) {
        double u1 = 1;
        double u2 = 0;
        ErrorCode err = center_intersection(*sub0, *offset0, *sub1, *offset1, u1, u2);
        if (err != ErrorCode::NoError) error_code = err;
        if (u2 < 1) {
            if (u1 > u0) center_points(*sub0, *offset0, u0, u1, result);
            u0 = u2;
            sub0 = sub1;
            offset0 = offset1;
        }
    }
    center_points(*sub0, *offset0, u0, 1, result);
    return error_code;
}

ErrorCode RobustPath::to_gds(FILE *out, double scaling) const {
    ErrorCode error_code = ErrorCode::NoError;
    if (num_elements == 0 || subpath_array.count == 0) return error_code;

    uint16_t buffer_end[] = {4, 0x1100};
    big_endian_swap16(buffer_end, COUNT(buffer_end));

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }

    Array<int32_t> coords = {};
    Array<Vec2> point_array = {};
    point_array.ensure_slots(subpath_array.count * GDSTK_MIN_POINTS);

    RobustPathElement *el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        uint16_t end_type;
        switch (el->end_type) {
            case EndType::HalfWidth:
                end_type = 2;
                break;
            case EndType::Extended:
                end_type = 4;
                break;
            case EndType::Round:
                end_type = 1;
                break;
            case EndType::Smooth:
                end_type = 1;
                break;
            default:
                end_type = 0;
        }

        uint16_t buffer0[] = {4, 0x0900};
        uint16_t buffer1[] = {6, 0x2102, end_type, 8, 0x0F03};
        int32_t width_ = (scale_width ? 1 : -1) *
                         (int32_t)lround(interp(el->width_array[0], 0) * width_scale * scaling);
        big_endian_swap16(buffer0, COUNT(buffer0));
        big_endian_swap16(buffer1, COUNT(buffer1));
        big_endian_swap32((uint32_t *)&width_, 1);

        uint16_t buffer_ext1[] = {8, 0x3003};
        uint16_t buffer_ext2[] = {8, 0x3103};
        int32_t ext_size[] = {0, 0};
        if (end_type == 4) {
            ext_size[0] = (int32_t)lround(el->end_extensions.u * scaling);
            ext_size[1] = (int32_t)lround(el->end_extensions.v * scaling);
            big_endian_swap16(buffer_ext1, COUNT(buffer_ext1));
            big_endian_swap16(buffer_ext2, COUNT(buffer_ext2));
            big_endian_swap32((uint32_t *)ext_size, COUNT(ext_size));
        }

        ErrorCode err = element_center(el, point_array);
        if (err != ErrorCode::NoError) error_code = err;

        coords.ensure_slots(point_array.count * 2);
        coords.count = point_array.count * 2;

        double *offset_p = (double *)offsets.items;
        for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
            fwrite(buffer0, sizeof(uint16_t), COUNT(buffer0), out);
            tag_to_gds(out, el->tag, GdsiiRecord::DATATYPE);
            fwrite(buffer1, sizeof(uint16_t), COUNT(buffer1), out);
            fwrite(&width_, sizeof(int32_t), 1, out);
            if (end_type == 4) {
                fwrite(buffer_ext1, sizeof(uint16_t), COUNT(buffer_ext1), out);
                fwrite(ext_size, sizeof(int32_t), 1, out);
                fwrite(buffer_ext2, sizeof(uint16_t), COUNT(buffer_ext2), out);
                fwrite(ext_size + 1, sizeof(int32_t), 1, out);
            }

            int32_t *c = coords.items;
            double *p = (double *)point_array.items;
            double offset_x = *offset_p++;
            double offset_y = *offset_p++;
            for (uint64_t i = point_array.count; i > 0; i--) {
                *c++ = (int32_t)lround((*p++ + offset_x) * scaling);
                *c++ = (int32_t)lround((*p++ + offset_y) * scaling);
            }
            big_endian_swap32((uint32_t *)coords.items, coords.count);

            uint64_t total = point_array.count;
            uint64_t i0 = 0;
            while (i0 < total) {
                uint64_t i1 = total < i0 + 8190 ? total : i0 + 8190;
                uint16_t buffer_pts[] = {(uint16_t)(4 + 8 * (i1 - i0)), 0x1003};
                big_endian_swap16(buffer_pts, COUNT(buffer_pts));
                fwrite(buffer_pts, sizeof(uint16_t), COUNT(buffer_pts), out);
                fwrite(coords.items + 2 * i0, sizeof(int32_t), 2 * (i1 - i0), out);
                i0 = i1;
            }

            err = properties_to_gds(properties, out);
            if (err != ErrorCode::NoError) error_code = err;

            fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
        }

        point_array.count = 0;
        coords.count = 0;
    }

    coords.clear();
    point_array.clear();
    if (repetition.type != RepetitionType::None) offsets.clear();
    return error_code;
}

ErrorCode RobustPath::to_oas(OasisStream &out, OasisState &state) const {
    ErrorCode error_code = ErrorCode::NoError;
    if (num_elements == 0 || subpath_array.count == 0) return error_code;

    bool has_repetition = repetition.get_count() > 1;

    Array<Vec2> point_array = {};
    point_array.ensure_slots(subpath_array.count * GDSTK_MIN_POINTS);

    RobustPathElement *el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        uint8_t info = 0xFB;
        if (has_repetition) info |= 0x04;

        oasis_putc((int)OasisRecord::PATH, out);
        oasis_putc(info, out);
        oasis_write_unsigned_integer(out, get_layer(el->tag));
        oasis_write_unsigned_integer(out, get_type(el->tag));
        uint64_t half_width =
            (uint64_t)llround(interp(el->width_array[0], 0) * width_scale * state.scaling);
        oasis_write_unsigned_integer(out, half_width);

        switch (el->end_type) {
            case EndType::Extended: {
                uint8_t extension_scheme = 0;
                int64_t start_extension = (int64_t)llround(el->end_extensions.u * state.scaling);
                int64_t end_extension = (int64_t)llround(el->end_extensions.v * state.scaling);
                if (start_extension == 0) {
                    extension_scheme |= 0x04;
                } else if (start_extension > 0 && (uint64_t)start_extension == half_width) {
                    extension_scheme |= 0x08;
                    start_extension = 0;
                } else {
                    extension_scheme |= 0x0C;
                }
                if (end_extension == 0) {
                    extension_scheme |= 0x01;
                } else if (end_extension > 0 && (uint64_t)end_extension == half_width) {
                    extension_scheme |= 0x02;
                    end_extension = 0;
                } else {
                    extension_scheme |= 0x03;
                }
                oasis_putc(extension_scheme, out);
                if (start_extension != 0) oasis_write_integer(out, start_extension);
                if (end_extension != 0) oasis_write_integer(out, end_extension);
            } break;
            case EndType::HalfWidth:
                oasis_putc(0x0A, out);
                break;
            default:  // Flush
                oasis_putc(0x05, out);
        }

        ErrorCode err = element_center(el, point_array);
        if (err != ErrorCode::NoError) error_code = err;

        oasis_write_point_list(out, point_array, state.scaling, false);
        oasis_write_integer(out, (int64_t)llround(point_array[0].x * state.scaling));
        oasis_write_integer(out, (int64_t)llround(point_array[0].y * state.scaling));
        if (has_repetition) oasis_write_repetition(out, repetition, state.scaling);
        err = properties_to_oas(properties, out, state);
        if (err != ErrorCode::NoError) error_code = err;

        point_array.count = 0;
    }
    point_array.clear();
    return error_code;
}

ErrorCode RobustPath::to_svg(FILE *out, double scaling, uint32_t precision) const {
    Array<Polygon *> array = {};
    ErrorCode error_code = to_polygons(false, 0, array);
    for (uint64_t i = 0; i < array.count; i++) {
        ErrorCode err = array[i]->to_svg(out, scaling, precision);
        if (err != ErrorCode::NoError) error_code = err;
        array[i]->clear();
        free_allocation(array[i]);
    }
    array.clear();
    return error_code;
}

}  // namespace gdstk
