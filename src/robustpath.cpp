/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "robustpath.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "curve.h"
#include "utils.h"

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
            printf("Bezier <%p>:", this);
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
            printf("Parametric <%p>: f = %p, df = %p, reference = (%lg, %lg), data = %p %p\n", this,
                   path_function, path_gradient, reference.x, reference.y, func_data, grad_data);
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
            const int64_t size = ctrl.size - 1;
            Vec2 *_ctrl = (Vec2 *)malloc(sizeof(Vec2) * size);
            Vec2 *dst = _ctrl;
            const Vec2 *src = ctrl.items;
            for (int64_t i = 0; i < size; i++, src++, dst++) *dst = size * (*(src + 1) - *src);
            grad = eval_bezier(u, _ctrl, size);
            free(_ctrl);
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
            point = eval_bezier(u, ctrl.items, ctrl.size);
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

Vec2 RobustPath::center_position(const SubPath &subpath, const Interpolation &offset,
                                 double u) const {
    const Vec2 sp_position = spine_position(subpath, u);
    const double offset_value = interp(offset, u) * offset_scale;
    Vec2 spine_normal = subpath.gradient(u, trafo).ortho();
    spine_normal.normalize();
    Vec2 result = sp_position + offset_value * spine_normal;
    return result;
}

Vec2 RobustPath::center_gradient(const SubPath &subpath, const Interpolation &offset,
                                 double u) const {
    const double step = 1.0 / (10.0 * max_evals);
    const double u0 = u - step < 0 ? 0 : u - step;
    const double u1 = u + step > 1 ? 1 : u + step;
    Vec2 result =
        (center_position(subpath, offset, u1) - center_position(subpath, offset, u0)) / (u1 - u0);
    return result;
}

Vec2 RobustPath::left_position(const SubPath &subpath, const Interpolation &offset,
                               const Interpolation &width, double u) const {
    const Vec2 ct_position = center_position(subpath, offset, u);
    const double width_value = interp(width, u) * width_scale;
    const Vec2 ct_gradient = center_gradient(subpath, offset, u);
    Vec2 center_normal = ct_gradient.ortho();
    center_normal.normalize();
    Vec2 position = ct_position + 0.5 * width_value * center_normal;
    return position;
}

Vec2 RobustPath::left_gradient(const SubPath &subpath, const Interpolation &offset,
                               const Interpolation &width, double u) const {
    const double step = 1.0 / (10.0 * max_evals);
    const double u0 = u - step < 0 ? 0 : u - step;
    const double u1 = u + step > 1 ? 1 : u + step;
    Vec2 result =
        (left_position(subpath, offset, width, u1) - left_position(subpath, offset, width, u0)) /
        (u1 - u0);
    return result;
}

Vec2 RobustPath::right_position(const SubPath &subpath, const Interpolation &offset,
                                const Interpolation &width, double u) const {
    const Vec2 ct_position = center_position(subpath, offset, u);
    const double width_value = interp(width, u) * width_scale;
    const Vec2 ct_gradient = center_gradient(subpath, offset, u);
    Vec2 center_normal = ct_gradient.ortho();
    center_normal.normalize();
    Vec2 position = ct_position - 0.5 * width_value * center_normal;
    return position;
}

Vec2 RobustPath::right_gradient(const SubPath &subpath, const Interpolation &offset,
                                const Interpolation &width, double u) const {
    const double step = 1.0 / (10.0 * max_evals);
    const double u0 = u - step < 0 ? 0 : u - step;
    const double u1 = u + step > 1 ? 1 : u + step;
    Vec2 result =
        (right_position(subpath, offset, width, u1) - right_position(subpath, offset, width, u0)) /
        (u1 - u0);
    return result;
}

// NOTE: Does NOT include the point at u0.
void RobustPath::spine_points(const SubPath &subpath, double u0, double u1,
                              Array<Vec2> &result) const {
    const double tolerance_sq = tolerance * tolerance;
    double u = u0;
    Vec2 last = spine_position(subpath, u0);
    int64_t counter = max_evals - 1;
    double du = 1.0 / MIN_POINTS;
    while (u < u1 && counter-- > 0) {
        if (du > 1.0 / MIN_POINTS) du = 1.0 / MIN_POINTS;
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
void RobustPath::center_points(const SubPath &subpath, const Interpolation &offset, double u0,
                               double u1, Array<Vec2> &result) const {
    const double tolerance_sq = tolerance * tolerance;
    double u = u0;
    Vec2 last = center_position(subpath, offset, u0);
    int64_t counter = max_evals - 1;
    double du = 1.0 / MIN_POINTS;
    while (u < u1 && counter-- > 0) {
        if (du > 1.0 / MIN_POINTS) du = 1.0 / MIN_POINTS;
        if (u + du > u1) du = u1 - u;
        Vec2 next = center_position(subpath, offset, u + du);
        Vec2 mid = center_position(subpath, offset, u + 0.5 * du);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = center_position(subpath, offset, u + du / 3);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            du *= 0.5;
            next = mid;
            mid = center_position(subpath, offset, u + 0.5 * du);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = center_position(subpath, offset, u + du / 3);
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
void RobustPath::left_points(const SubPath &subpath, const Interpolation &offset,
                             const Interpolation &width, double u0, double u1,
                             Array<Vec2> &result) const {
    const double tolerance_sq = tolerance * tolerance;
    double u = u0;
    Vec2 last = left_position(subpath, offset, width, u0);
    int64_t counter = max_evals - 1;
    double du = 1.0 / MIN_POINTS;
    while (u < u1 && counter-- > 0) {
        if (du > 1.0 / MIN_POINTS) du = 1.0 / MIN_POINTS;
        if (u + du > u1) du = u1 - u;
        Vec2 next = left_position(subpath, offset, width, u + du);
        Vec2 mid = left_position(subpath, offset, width, u + 0.5 * du);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = left_position(subpath, offset, width, u + du / 3);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            du *= 0.5;
            next = mid;
            mid = left_position(subpath, offset, width, u + 0.5 * du);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = left_position(subpath, offset, width, u + du / 3);
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
void RobustPath::right_points(const SubPath &subpath, const Interpolation &offset,
                              const Interpolation &width, double u0, double u1,
                              Array<Vec2> &result) const {
    const double tolerance_sq = tolerance * tolerance;
    double u = u0;
    Vec2 last = right_position(subpath, offset, width, u0);
    int64_t counter = max_evals - 1;
    double du = 1.0 / MIN_POINTS;
    while (u < u1 && counter-- > 0) {
        if (du > 1.0 / MIN_POINTS) du = 1.0 / MIN_POINTS;
        if (u + du > u1) du = u1 - u;
        Vec2 next = right_position(subpath, offset, width, u + du);
        Vec2 mid = right_position(subpath, offset, width, u + 0.5 * du);
        double err_sq = distance_to_line_sq(mid, last, next);
        if (err_sq <= tolerance_sq) {
            const Vec2 extra = right_position(subpath, offset, width, u + du / 3);
            err_sq = distance_to_line_sq(extra, last, next);
        }
        while (err_sq > tolerance_sq) {
            du *= 0.5;
            next = mid;
            mid = right_position(subpath, offset, width, u + 0.5 * du);
            err_sq = distance_to_line_sq(mid, last, next);
            if (err_sq <= tolerance_sq) {
                const Vec2 extra = right_position(subpath, offset, width, u + du / 3);
                err_sq = distance_to_line_sq(extra, last, next);
            }
        }
        result.append(next);
        last = next;
        u += du;
        du *= 2;
    }
}

void RobustPath::print(bool all) const {
    printf("RobustPath <%p> at (%lg, %lg), size %" PRId64 ", %" PRId64
           " elements, tol %lg, max_evals %" PRId64 ", properties <%p>, owner <%p>\n",
           this, end_point.x, end_point.y, subpath_array.size, num_elements, tolerance, max_evals,
           properties, owner);
    if (all) {
        for (int64_t ns = 0; ns < subpath_array.size; ns++) {
            printf("(%" PRId64 ") ", ns);
            subpath_array[ns].print();
        }
        RobustPathElement *el = elements;
        for (int64_t ne = 0; ne < num_elements; ne++, el++)
            printf("Element %" PRId64 ", layer %d, datatype %d, end %d (%lg, %lg)\n", ne, el->layer,
                   el->datatype, (int)el->end_type, el->end_extensions.u, el->end_extensions.v);
    }
}

void RobustPath::clear() {
    subpath_array.clear();
    RobustPathElement *el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++) {
        el->width_array.clear();
        el->offset_array.clear();
    }
    free(elements);
    elements = NULL;
    num_elements = 0;
    properties_clear(properties);
    properties = NULL;
}

void RobustPath::copy_from(const RobustPath &path) {
    properties = properties_copy(path.properties);
    end_point = path.end_point;
    subpath_array.copy_from(path.subpath_array);
    num_elements = path.num_elements;
    elements = (RobustPathElement *)calloc(num_elements, sizeof(RobustPathElement));
    tolerance = path.tolerance;
    max_evals = path.max_evals;
    width_scale = path.width_scale;
    offset_scale = path.offset_scale;
    memcpy(trafo, path.trafo, 6 * sizeof(double));
    scale_width = path.scale_width;
    gdsii_path = path.gdsii_path;

    RobustPathElement *src = path.elements;
    RobustPathElement *dst = elements;
    for (int64_t ne = 0; ne < path.num_elements; ne++, src++, dst++) {
        dst->layer = src->layer;
        dst->datatype = src->datatype;
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

void RobustPath::simple_scale(double scale) {
    trafo[0] *= scale;
    trafo[1] *= scale;
    trafo[2] *= scale;
    trafo[3] *= scale;
    trafo[4] *= scale;
    trafo[5] *= scale;
    offset_scale *= fabs(scale);
    if (scale_width) width_scale *= fabs(scale);
}

void RobustPath::scale(double scale, const Vec2 center) {
    const Vec2 delta = center * (1 - scale);
    simple_scale(scale);
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

void RobustPath::transform(double magnification, const Vec2 translation, bool x_refl,
                           double rotation, const Vec2 origin) {
    simple_scale(magnification);
    translate(translation);
    if (x_refl) x_reflection();
    simple_rotate(rotation);
    translate(origin);
}

void RobustPath::fill_widths_and_offsets(const Interpolation *width, const Interpolation *offset) {
    if (width == NULL) {
        Interpolation interpolation = {InterpolationType::Constant};
        RobustPathElement *el = elements;
        for (int64_t ne = num_elements; ne > 0; ne--, el++) {
            interpolation.value = el->end_width;
            el->width_array.append(interpolation);
        }
    } else {
        RobustPathElement *el = elements;
        for (int64_t ne = num_elements; ne > 0; ne--, el++, width++) {
            el->width_array.append(*width);
            el->end_width = interp(*width, 1);
        }
    }
    if (offset == NULL) {
        Interpolation interpolation = {InterpolationType::Constant};
        RobustPathElement *el = elements;
        for (int64_t ne = num_elements; ne > 0; ne--, el++) {
            interpolation.value = el->end_offset;
            el->offset_array.append(interpolation);
        }
    } else {
        RobustPathElement *el = elements;
        for (int64_t ne = num_elements; ne > 0; ne--, el++, offset++) {
            el->offset_array.append(*offset);
            el->end_offset = interp(*offset, 1);
        }
    }
}

void RobustPath::horizontal(double coord_x, const Interpolation *width, const Interpolation *offset,
                            bool relative) {
    if (relative) coord_x += end_point.x;
    segment(Vec2{coord_x, end_point.y}, width, offset, false);
}

void RobustPath::vertical(double coord_y, const Interpolation *width, const Interpolation *offset,
                          bool relative) {
    if (relative) coord_y += end_point.y;
    segment(Vec2{end_point.x, coord_y}, width, offset, false);
}

void RobustPath::segment(const Vec2 end_pt, const Interpolation *width, const Interpolation *offset,
                         bool relative) {
    SubPath sub = {SubPathType::Segment};
    sub.begin = end_point;
    sub.end = end_pt;
    if (relative) sub.end += end_point;
    end_point = sub.end;
    subpath_array.append(sub);
    fill_widths_and_offsets(width, offset);
}

void RobustPath::cubic(const Vec2 point1, const Vec2 point2, const Vec2 point3,
                       const Interpolation *width, const Interpolation *offset, bool relative) {
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
    fill_widths_and_offsets(width, offset);
}

void RobustPath::cubic_smooth(const Vec2 point2, const Vec2 point3, const Interpolation *width,
                              const Interpolation *offset, bool relative) {
    SubPath sub = {SubPathType::Bezier3};
    sub.p0 = end_point;
    sub.p1 = end_point;
    if (subpath_array.size > 0)
        sub.p1 += subpath_array[subpath_array.size - 1].gradient(1, trafo) / 3;
    sub.p2 = point2;
    sub.p3 = point3;
    if (relative) {
        sub.p2 += end_point;
        sub.p3 += end_point;
    }
    end_point = sub.p3;
    subpath_array.append(sub);
    fill_widths_and_offsets(width, offset);
}

void RobustPath::quadratic(const Vec2 point1, const Vec2 point2, const Interpolation *width,
                           const Interpolation *offset, bool relative) {
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
    fill_widths_and_offsets(width, offset);
}

void RobustPath::quadratic_smooth(const Vec2 point2, const Interpolation *width,
                                  const Interpolation *offset, bool relative) {
    SubPath sub = {SubPathType::Bezier2};
    sub.p0 = end_point;
    sub.p1 = end_point;
    if (subpath_array.size > 0)
        sub.p1 += subpath_array[subpath_array.size - 1].gradient(1, trafo) / 2;
    sub.p2 = point2;
    if (relative) sub.p2 += end_point;
    end_point = sub.p2;
    subpath_array.append(sub);
    fill_widths_and_offsets(width, offset);
}

void RobustPath::bezier(const Array<Vec2> point_array, const Interpolation *width,
                        const Interpolation *offset, bool relative) {
    SubPath sub = {SubPathType::Bezier};
    sub.ctrl.append(end_point);
    sub.ctrl.extend(point_array);
    if (relative)
        for (int64_t i = 1; i <= point_array.size; i++) sub.ctrl[i] += end_point;
    end_point = sub.ctrl[sub.ctrl.size - 1];
    subpath_array.append(sub);
    fill_widths_and_offsets(width, offset);
}

void RobustPath::interpolation(const Array<Vec2> point_array, double *angles,
                               bool *angle_constraints, Vec2 *tension, double initial_curl,
                               double final_curl, bool cycle, const Interpolation *width,
                               const Interpolation *offset, bool relative) {
    Array<Vec2> hobby_vec = {0};
    hobby_vec.ensure_slots(3 * (point_array.size + 1));
    hobby_vec.size = 3 * (point_array.size + 1);
    const Vec2 ref = end_point;
    const Vec2 *src = point_array.items;
    Vec2 *dst = hobby_vec.items + 3;
    hobby_vec[0] = ref;
    if (relative)
        for (int64_t i = 0; i < point_array.size; i++, dst += 3) *dst = ref + *src++;
    else
        for (int64_t i = 0; i < point_array.size; i++, dst += 3) *dst = *src++;
    hobby_interpolation(point_array.size + 1, hobby_vec.items, angles, angle_constraints, tension,
                        initial_curl, final_curl, cycle);
    dst = hobby_vec.items + 1;
    for (int64_t i = 0; i < point_array.size; i++, dst += 3)
        cubic(*dst, *(dst + 1), *(dst + 2), width, offset, false);
    if (cycle) cubic(*dst, *(dst + 1), ref, width, offset, false);
    hobby_vec.clear();
}

void RobustPath::arc(double radius_x, double radius_y, double initial_angle, double final_angle,
                     double rotation, const Interpolation *width, const Interpolation *offset) {
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
    fill_widths_and_offsets(width, offset);
}

void RobustPath::turn(double radius, double angle, const Interpolation *width,
                      const Interpolation *offset) {
    Vec2 direction = Vec2{1, 0};
    if (subpath_array.size > 0)
        direction = subpath_array[subpath_array.size - 1].gradient(1, trafo);
    const double initial_angle = direction.angle() + (angle < 0 ? 0.5 * M_PI : -0.5 * M_PI);
    arc(radius, radius, initial_angle, initial_angle + angle, 0, width, offset);
}

void RobustPath::parametric(ParametricVec2 curve_function, void *func_data,
                            ParametricVec2 curve_gradient, void *grad_data,
                            const Interpolation *width, const Interpolation *offset,
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
    fill_widths_and_offsets(width, offset);
}

int64_t RobustPath::commands(const CurveInstruction *items, int64_t size) {
    const CurveInstruction *item = items;
    const CurveInstruction *end = items + size;
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
    return size;
}

Vec2 RobustPath::position(double u, bool from_below) const {
    if (u >= subpath_array.size)
        u = subpath_array.size;
    else if (u < 0)
        u = 0;
    int64_t idx = (int64_t)u;
    u -= idx;
    if (from_below && u == 0 && idx > 0) {
        idx--;
        u = 1;
    } else if (idx == subpath_array.size) {
        idx--;
        u = 1;
    }
    return spine_position(subpath_array[idx], u);
}

Vec2 RobustPath::gradient(double u, bool from_below) const {
    if (u >= subpath_array.size)
        u = subpath_array.size;
    else if (u < 0)
        u = 0;
    int64_t idx = (int64_t)u;
    u -= idx;
    if (from_below && u == 0 && idx > 0) {
        idx--;
        u = 1;
    } else if (idx == subpath_array.size) {
        idx--;
        u = 1;
    }
    return spine_gradient(subpath_array[idx], u);
}

void RobustPath::width(double u, bool from_below, double *result) const {
    if (u >= subpath_array.size)
        u = subpath_array.size;
    else if (u < 0)
        u = 0;
    int64_t idx = (int64_t)u;
    u -= idx;
    if (from_below && u == 0 && idx > 0) {
        idx--;
        u = 1;
    } else if (idx == subpath_array.size) {
        idx--;
        u = 1;
    }
    RobustPathElement *el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++)
        *result++ = interp(el->width_array[idx], u) * width_scale;
}

void RobustPath::offset(double u, bool from_below, double *result) const {
    if (u >= subpath_array.size)
        u = subpath_array.size;
    else if (u < 0)
        u = 0;
    int64_t idx = (int64_t)u;
    u -= idx;
    if (from_below && u == 0 && idx > 0) {
        idx--;
        u = 1;
    } else if (idx == subpath_array.size) {
        idx--;
        u = 1;
    }
    RobustPathElement *el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++)
        *result++ = interp(el->offset_array[idx], u) * offset_scale;
}

void RobustPath::spine_intersection(const SubPath &sub0, const SubPath &sub1, double &u0,
                                    double &u1) const {
    const double tolerance_sq = tolerance * tolerance;
    Vec2 p0 = spine_position(sub0, u0);
    Vec2 p1 = spine_position(sub1, u1);
    double err_sq = (p0 - p1).length_sq();

    if (err_sq <= tolerance_sq) return;

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
    int64_t count = max_evals;
    const double step_min = 1.0 / (10.0 * max_evals);
    while (count-- > 0 || fabs(step * du0) > step_min || fabs(step * du1) > step_min) {
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

            if (err_sq <= tolerance_sq) return;

            v0 = spine_gradient(sub0, u0);
            v1 = spine_gradient(sub1, u1);
            norm_v0 = v0.normalize();
            norm_v1 = v1.normalize();
            segments_intersection(p0, v0, p1, v1, du0, du1);
            du0 /= norm_v0;
            du1 /= norm_v1;
        }
    }
    fputs("[GDSTK] No intersection found in RobustPath spine construction.\n", stderr);
}

void RobustPath::center_intersection(const SubPath &sub0, const Interpolation &offset0,
                                     const SubPath &sub1, const Interpolation &offset1, double &u0,
                                     double &u1) const {
    const double tolerance_sq = tolerance * tolerance;
    Vec2 p0 = center_position(sub0, offset0, u0);
    Vec2 p1 = center_position(sub1, offset1, u1);
    double err_sq = (p0 - p1).length_sq();

    if (err_sq <= tolerance_sq) return;

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
    int64_t count = max_evals;
    const double step_min = 1.0 / (10.0 * max_evals);
    while (count-- > 0 || fabs(step * du0) > step_min || fabs(step * du1) > step_min) {
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

            if (err_sq <= tolerance_sq) return;

            v0 = center_gradient(sub0, offset0, u0);
            v1 = center_gradient(sub1, offset1, u1);
            norm_v0 = v0.normalize();
            norm_v1 = v1.normalize();
            segments_intersection(p0, v0, p1, v1, du0, du1);
            du0 /= norm_v0;
            du1 /= norm_v1;
        }
    }
    fputs("[GDSTK] No intersection found in RobustPath center construction.\n", stderr);
}

void RobustPath::left_intersection(const SubPath &sub0, const Interpolation &offset0,
                                   const Interpolation &width0, const SubPath &sub1,
                                   const Interpolation &offset1, const Interpolation &width1,
                                   double &u0, double &u1) const {
    const double tolerance_sq = tolerance * tolerance;
    Vec2 p0 = left_position(sub0, offset0, width0, u0);
    Vec2 p1 = left_position(sub1, offset1, width1, u1);
    double err_sq = (p0 - p1).length_sq();

    if (err_sq <= tolerance_sq) return;

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
    int64_t count = max_evals;
    const double step_min = 1.0 / (10.0 * max_evals);
    while (count-- > 0 || fabs(step * du0) > step_min || fabs(step * du1) > step_min) {
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

            if (err_sq <= tolerance_sq) return;

            v0 = left_gradient(sub0, offset0, width0, u0);
            v1 = left_gradient(sub1, offset1, width1, u1);
            norm_v0 = v0.normalize();
            norm_v1 = v1.normalize();
            segments_intersection(p0, v0, p1, v1, du0, du1);
            du0 /= norm_v0;
            du1 /= norm_v1;
        }
    }
    fputs("[GDSTK] No intersection found in RobustPath left side construction.\n", stderr);
}

void RobustPath::right_intersection(const SubPath &sub0, const Interpolation &offset0,
                                    const Interpolation &width0, const SubPath &sub1,
                                    const Interpolation &offset1, const Interpolation &width1,
                                    double &u0, double &u1) const {
    const double tolerance_sq = tolerance * tolerance;
    Vec2 p0 = right_position(sub0, offset0, width0, u0);
    Vec2 p1 = right_position(sub1, offset1, width1, u1);
    double err_sq = (p0 - p1).length_sq();

    if (err_sq <= tolerance_sq) return;

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
    int64_t count = max_evals;
    const double step_min = 1.0 / (10.0 * max_evals);
    while (count-- > 0 || fabs(step * du0) > step_min || fabs(step * du1) > step_min) {
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

            if (err_sq <= tolerance_sq) return;

            v0 = right_gradient(sub0, offset0, width0, u0);
            v1 = right_gradient(sub1, offset1, width1, u1);
            norm_v0 = v0.normalize();
            norm_v1 = v1.normalize();
            segments_intersection(p0, v0, p1, v1, du0, du1);
            du0 /= norm_v0;
            du1 /= norm_v1;
        }
    }
    fputs("[GDSTK] No intersection found in RobustPath right side construction.\n", stderr);
}

Array<Vec2> RobustPath::spine() const {
    Array<Vec2> result = {0};
    if (subpath_array.size == 0) return result;
    result.ensure_slots(subpath_array.size + 1);
    double u0 = 0;
    SubPath *sub0 = subpath_array.items;
    SubPath *sub1 = sub0 + 1;
    result.append(spine_position(*sub0, 0));
    for (int64_t ns = 1; ns < subpath_array.size; ns++, sub1++) {
        double u1 = 1;
        double u2 = 0;
        spine_intersection(*sub0, *sub1, u1, u2);
        if (u2 < 1) {
            if (u1 > u0) spine_points(*sub0, u0, u1, result);
            u0 = u2;
            sub0 = sub1;
        }
    }
    spine_points(*sub0, u0, 1, result);
    return result;
}

Array<Polygon *> RobustPath::to_polygons() const {
    Array<Polygon *> result = {0};
    if (num_elements == 0 || subpath_array.size == 0) return result;

    const double tolerance_sq = tolerance * tolerance;
    result.ensure_slots(num_elements);
    RobustPathElement *el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++) {
        Array<Vec2> left_side = {0};
        Array<Vec2> right_side = {0};
        left_side.ensure_slots(subpath_array.size);
        right_side.ensure_slots(subpath_array.size);
        Curve initial_cap = {0};
        Curve final_cap = {0};
        initial_cap.tolerance = tolerance;
        final_cap.tolerance = tolerance;
        initial_cap.ensure_slots(2);
        final_cap.ensure_slots(2);

        {  // Initial cap
            const Vec2 cap_l =
                left_position(subpath_array[0], el->offset_array[0], el->width_array[0], 0);
            const Vec2 cap_r =
                right_position(subpath_array[0], el->offset_array[0], el->width_array[0], 0);
            if (el->end_type == EndType::Flush) {
                initial_cap.append(cap_l);
                if ((cap_l - cap_r).length_sq() > tolerance_sq) initial_cap.append(cap_r);
            } else if (el->end_type == EndType::Extended) {
                initial_cap.append(cap_l);
                Vec2 direction = center_gradient(subpath_array[0], el->offset_array[0], 0);
                direction.normalize();
                const double half_width = 0.5 * interp(el->width_array[0], 0) * width_scale;
                const double extension =
                    el->end_extensions.u >= 0 ? el->end_extensions.u : half_width;
                initial_cap.append(cap_l - extension * direction);
                if (half_width != 0) initial_cap.append(cap_r - extension * direction);
                initial_cap.append(cap_r);
            } else if (el->end_type == EndType::Round) {
                initial_cap.append(cap_l);
                const Vec2 direction = center_gradient(subpath_array[0], el->offset_array[0], 0);
                const double initial_angle = direction.angle() + 0.5 * M_PI;
                const double half_width = 0.5 * interp(el->width_array[0], 0) * width_scale;
                initial_cap.arc(half_width, half_width, initial_angle, initial_angle + M_PI, 0);
            } else if (el->end_type == EndType::Smooth) {
                initial_cap.append(cap_l);
                Array<Vec2> point_array = {0};
                point_array.items = (Vec2 *)&cap_r;
                point_array.size = 1;
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
            for (int64_t ns = 1; ns < subpath_array.size; ns++, sub1++, offset1++, width1++) {
                double u1 = 1;
                double u2 = 0;
                left_intersection(*sub0, *offset0, *width0, *sub1, *offset1, *width1, u1, u2);
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
            for (int64_t ns = 1; ns < subpath_array.size; ns++, sub1++, offset1++, width1++) {
                double u1 = 1;
                double u2 = 0;
                right_intersection(*sub0, *offset0, *width0, *sub1, *offset1, *width1, u1, u2);
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
            const int64_t last = subpath_array.size - 1;
            const Vec2 cap_l = left_position(subpath_array[last], el->offset_array[last],
                                             el->width_array[last], 1);
            const Vec2 cap_r = right_position(subpath_array[last], el->offset_array[last],
                                              el->width_array[last], 1);
            if (el->end_type == EndType::Flush) {
                final_cap.append(cap_r);
                if ((cap_l - cap_r).length_sq() > tolerance_sq) final_cap.append(cap_l);
            } else if (el->end_type == EndType::Extended) {
                final_cap.append(cap_r);
                Vec2 direction = center_gradient(subpath_array[last], el->offset_array[last], 1);
                direction.normalize();
                const double half_width = 0.5 * interp(el->width_array[last], 1) * width_scale;
                const double extension =
                    el->end_extensions.v >= 0 ? el->end_extensions.v : half_width;
                final_cap.append(cap_r + extension * direction);
                if (half_width != 0) final_cap.append(cap_l + extension * direction);
                final_cap.append(cap_l);
            } else if (el->end_type == EndType::Round) {
                final_cap.append(cap_r);
                const Vec2 direction =
                    center_gradient(subpath_array[last], el->offset_array[last], 1);
                const double initial_angle = direction.angle() - 0.5 * M_PI;
                const double half_width = 0.5 * interp(el->width_array[last], 1) * width_scale;
                final_cap.arc(half_width, half_width, initial_angle, initial_angle + M_PI, 0);
            } else if (el->end_type == EndType::Smooth) {
                final_cap.append(cap_r);
                Array<Vec2> point_array = {0};
                point_array.items = (Vec2 *)&cap_l;
                point_array.size = 1;
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

        int64_t num =
            left_side.size + initial_cap.point_array.size + final_cap.point_array.size - 2;
        right_side.ensure_slots(num);
        Vec2 *dst = right_side.items + right_side.size;

        memcpy(dst, final_cap.point_array.items + 1,
               sizeof(Vec2) * (final_cap.point_array.size - 2));
        dst += final_cap.point_array.size - 2;
        final_cap.clear();

        Vec2 *src = left_side.items + left_side.size - 1;
        for (int64_t i = left_side.size - 1; i >= 0; i--) *dst++ = *src--;
        left_side.clear();

        memcpy(dst, initial_cap.point_array.items, sizeof(Vec2) * initial_cap.point_array.size);
        initial_cap.clear();
        right_side.size += num;

        Polygon *result_polygon = (Polygon *)malloc(sizeof(Polygon));
        result_polygon->layer = el->layer;
        result_polygon->datatype = el->datatype;
        result_polygon->point_array = right_side;
        result_polygon->properties = properties_copy(properties);
        result.append(result_polygon);
    }
    return result;
}

void RobustPath::to_gds(FILE *out, double scaling) const {
    if (num_elements == 0 || subpath_array.size == 0) return;

    Array<int32_t> coords = {0};
    Array<Vec2> point_array = {0};
    point_array.ensure_slots(subpath_array.size * MIN_POINTS);

    RobustPathElement *el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++) {
        uint16_t end_type;
        switch (el->end_type) {
            case EndType::Extended:
                if (el->end_extensions.u >= 0 || el->end_extensions.v >= 0)
                    end_type = 4;
                else
                    end_type = 2;
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
        uint16_t buffer_start[] = {
            4,      0x0900,   6, 0x0D02, (uint16_t)el->layer, 6, 0x0E02, (uint16_t)el->datatype, 6,
            0x2102, end_type, 8, 0x0F03};
        int32_t width = (scale_width ? 1 : -1) *
                        (int32_t)lround(interp(el->width_array[0], 0) * width_scale * scaling);
        swap16(buffer_start, COUNT(buffer_start));
        swap32((uint32_t *)&width, 1);
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
        fwrite(&width, sizeof(int32_t), 1, out);

        if (end_type == 4) {
            uint16_t buffer_ext[] = {8, 0x3003};
            width = (int32_t)lround(el->end_extensions.u * scaling);
            swap16(buffer_ext, COUNT(buffer_ext));
            swap32((uint32_t *)&width, 1);
            fwrite(buffer_ext, sizeof(uint16_t), COUNT(buffer_ext), out);
            fwrite(&width, sizeof(int32_t), 1, out);
            buffer_ext[1] = 0x3103;
            width = (int32_t)lround(el->end_extensions.v * scaling);
            swap16(buffer_ext + 1, 1);
            swap32((uint32_t *)&width, 1);
            fwrite(buffer_ext, sizeof(uint16_t), COUNT(buffer_ext), out);
            fwrite(&width, sizeof(int32_t), 1, out);
        }

        {  // Calculate path coordinates (analogous to RobustPath::to_polygons)
            double u0 = 0;
            SubPath *sub0 = subpath_array.items;
            SubPath *sub1 = sub0 + 1;
            Interpolation *offset0 = el->offset_array.items;
            Interpolation *offset1 = offset0 + 1;
            point_array.append(center_position(*sub0, *offset0, 0));
            for (int64_t ns = 1; ns < subpath_array.size; ns++, sub1++, offset1++) {
                double u1 = 1;
                double u2 = 0;
                center_intersection(*sub0, *offset0, *sub1, *offset1, u1, u2);
                if (u2 < 1) {
                    if (u1 > u0) center_points(*sub0, *offset0, u0, u1, point_array);
                    u0 = u2;
                    sub0 = sub1;
                    offset0 = offset1;
                }
            }
            center_points(*sub0, *offset0, u0, 1, point_array);
        }

        coords.ensure_slots(point_array.size * 2);
        coords.size = point_array.size * 2;
        int32_t *c = coords.items;
        double *p = (double *)point_array.items;
        for (int64_t i = coords.size; i > 0; i--) *c++ = (int32_t)lround((*p++) * scaling);
        swap32((uint32_t *)coords.items, coords.size);

        int64_t total = point_array.size;
        int64_t i0 = 0;
        while (i0 < total) {
            int64_t i1 = total < i0 + 8190 ? total : i0 + 8190;
            uint16_t buffer_pts[] = {(uint16_t)(4 + 8 * (i1 - i0)), 0x1003};
            swap16(buffer_pts, COUNT(buffer_pts));
            fwrite(buffer_pts, sizeof(uint16_t), COUNT(buffer_pts), out);
            fwrite(coords.items + 2 * i0, sizeof(int32_t), 2 * (i1 - i0), out);
            i0 = i1;
        }
        point_array.size = 0;
        coords.size = 0;

        properties_to_gds(properties, out);

        uint16_t buffer_end[] = {4, 0x1100};
        swap16(buffer_end, COUNT(buffer_end));
        fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
    }
    coords.clear();
    point_array.clear();
}

void RobustPath::to_svg(FILE *out, double scaling) const {
    Array<Polygon *> array = to_polygons();
    for (int64_t i = 0; i < array.size; i++) {
        array[i]->to_svg(out, scaling);
        array[i]->clear();
        free(array[i]);
    }
    array.clear();
}

}  // namespace gdstk
