/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "flexpath.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "curve.h"
#include "utils.h"

namespace gdstk {

void FlexPath::init(const Vec2 initial_position, const double* width, const double* offset) {
    spine.append(initial_position);
    for (int64_t i = 0; i < num_elements; i++)
        elements[i].half_width_and_offset.append(Vec2{0.5 * width[i], offset[i]});
}

void FlexPath::print(bool all) const {
    printf("FlexPath <%p>, size %" PRId64 ", %" PRId64
           " elements, gdsii %d, width scaling %d, properties <%p>, owner <%p>\n",
           this, spine.point_array.size, num_elements, gdsii_path, scale_width, properties, owner);
    if (all) {
        FlexPathElement* el = elements;
        for (int64_t ne = 0; ne < num_elements; ne++, el++) {
            printf("Element %" PRId64
                   ", layer %d, datatype %d, join %d, end %d (%lg, %lg), bend %d (%lg)\n",
                   ne, el->layer, el->datatype, (int)el->join_type, (int)el->end_type,
                   el->end_extensions.u, el->end_extensions.v, (int)el->bend_type, el->bend_radius);
        }
        printf("Spine: ");
        spine.print(true);
    }
}

void FlexPath::clear() {
    spine.clear();
    FlexPathElement* el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++) el->half_width_and_offset.clear();
    free(elements);
    elements = NULL;
    num_elements = 0;
    properties_clear(properties);
    properties = NULL;
}

void FlexPath::copy_from(const FlexPath& path) {
    spine.copy_from(path.spine);
    properties = properties_copy(path.properties);
    scale_width = path.scale_width;
    gdsii_path = path.gdsii_path;
    num_elements = path.num_elements;
    elements = (FlexPathElement*)calloc(num_elements, sizeof(FlexPathElement));

    FlexPathElement* src = path.elements;
    FlexPathElement* dst = elements;
    for (int64_t ne = 0; ne < path.num_elements; ne++, src++, dst++) {
        dst->half_width_and_offset.copy_from(src->half_width_and_offset);
        dst->layer = src->layer;
        dst->datatype = src->datatype;
        dst->join_type = src->join_type;
        dst->join_function = src->join_function;
        dst->join_function_data = src->join_function_data;
        dst->end_type = src->end_type;
        dst->end_extensions = src->end_extensions;
        dst->end_function = src->end_function;
        dst->end_function_data = src->end_function_data;
        dst->bend_type = src->bend_type;
        dst->bend_radius = src->bend_radius;
        dst->bend_function = src->bend_function;
        dst->bend_function_data = src->bend_function_data;
    }
}

void FlexPath::translate(const Vec2 v) {
    Vec2* p = spine.point_array.items;
    for (int64_t num = spine.point_array.size; num > 0; num--) *p++ += v;
}

void FlexPath::scale(double scale, const Vec2 center) {
    Vec2* p = spine.point_array.items;
    for (int64_t num = spine.point_array.size; num > 0; num--, p++)
        *p = (*p - center) * scale + center;
    Vec2 wo_scale = {1, fabs(scale)};
    if (scale_width) wo_scale.u = wo_scale.v;
    FlexPathElement* el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++) {
        Vec2* wo = el->half_width_and_offset.items;
        for (int64_t num = spine.point_array.size; num > 0; num--) *wo++ *= wo_scale;
    }
}

void FlexPath::mirror(const Vec2 p0, const Vec2 p1) {
    Vec2 v = p1 - p0;
    double tmp = v.length_sq();
    if (tmp == 0) return;
    Vec2 r = v * (2 / tmp);
    Vec2 p2 = p0 * 2;
    Vec2* p = spine.point_array.items;
    for (int64_t num = spine.point_array.size; num > 0; num--, p++)
        *p = v * (*p - p0).inner(r) - *p + p2;
    FlexPathElement* el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++) {
        Vec2* wo = el->half_width_and_offset.items;
        for (int64_t num = spine.point_array.size; num > 0; num--, wo++) wo->v = -wo->v;
    }
}

void FlexPath::rotate(double angle, const Vec2 center) {
    double ca = cos(angle);
    double sa = sin(angle);
    Vec2* p = spine.point_array.items;
    for (int64_t num = spine.point_array.size; num > 0; num--, p++) {
        Vec2 q = *p - center;
        p->x = q.x * ca - q.y * sa + center.x;
        p->y = q.x * sa + q.y * ca + center.y;
    }
}

void FlexPath::transform(double magnification, const Vec2 translation, bool x_reflection,
                         double rotation, const Vec2 origin) {
    double ca = cos(rotation);
    double sa = sin(rotation);
    Vec2* p = spine.point_array.items;
    for (int64_t num = spine.point_array.size; num > 0; num--, p++) {
        Vec2 q = *p * magnification + translation;
        if (x_reflection) q.y = -q.y;
        p->x = q.x * ca - q.y * sa + origin.x;
        p->y = q.x * sa + q.y * ca + origin.y;
    }
    Vec2 wo_scale = {magnification, 1};
    if (scale_width) wo_scale.y = magnification;
    FlexPathElement* el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++) {
        Vec2* wo = el->half_width_and_offset.items;
        for (int64_t num = spine.point_array.size; num > 0; num--) *wo++ *= wo_scale;
    }
}

void FlexPath::remove_overlapping_points() {
    const double tol_sq = spine.tolerance * spine.tolerance;
    Array<Vec2>* array = &spine.point_array;
    for (int64_t i = 1; i < array->size;) {
        if (((*array)[i] - (*array)[i - 1]).length_sq() <= tol_sq) {
            array->remove(i);
            FlexPathElement* el = elements;
            for (int64_t ne = 0; ne < num_elements; ne++, el++) el->half_width_and_offset.remove(i);
        } else {
            i++;
        }
    }
}

Array<Polygon*> FlexPath::to_polygons() {
    Array<Polygon*> result = {0};

    remove_overlapping_points();
    if (spine.point_array.size < 2) return result;

    const Array<Vec2> spine_points = spine.point_array;
    int64_t curve_size_guess = spine_points.size * 2 + 4;

    result.ensure_slots(num_elements);
    FlexPathElement* el = elements;
    for (int64_t ne = 0; ne < num_elements; ne++, el++) {
        const double* half_widths = (double*)el->half_width_and_offset.items;
        const double* offsets = half_widths + 1;
        const JoinType join_type = el->join_type;
        const BendType bend_type = el->bend_type;
        const double bend_radius = el->bend_radius;

        Curve right_curve = {0};
        Curve left_curve = {0};
        right_curve.tolerance = spine.tolerance;
        left_curve.tolerance = spine.tolerance;
        right_curve.ensure_slots(curve_size_guess);
        left_curve.ensure_slots(curve_size_guess / 2);

        // Normal to spine segment
        Vec2 spine_normal = (spine_points[1] - spine_points[0]).ortho();
        spine_normal.normalize();
        // First points
        Vec2 p0 = spine_points[0] + spine_normal * offsets[2 * 0];
        Vec2 p1 = spine_points[1] + spine_normal * offsets[2 * 1];
        // Tangent unit vector and segment length
        Vec2 t0 = p1 - p0;
        t0.normalize();
        // Normal to segment
        Vec2 n0 = t0.ortho();

        {  // Initial cap
            const Vec2 cap_l = p0 + n0 * half_widths[2 * 0];
            const Vec2 cap_r = p0 - n0 * half_widths[2 * 0];
            if (el->end_type == EndType::Flush) {
                right_curve.append(cap_l);
                if (half_widths[2 * 0] != 0) right_curve.append(cap_r);
            } else if (el->end_type == EndType::Extended) {
                right_curve.append(cap_l);
                const double extension =
                    el->end_extensions.u >= 0 ? el->end_extensions.u : half_widths[2 * 0];
                right_curve.append(cap_l - extension * t0);
                if (half_widths[2 * 0] != 0) right_curve.append(cap_r - extension * t0);
                right_curve.append(cap_r);
            } else if (el->end_type == EndType::Round) {
                right_curve.append(cap_l);
                double initial_angle = n0.angle();
                right_curve.arc(half_widths[2 * 0], half_widths[2 * 0], initial_angle,
                                initial_angle + M_PI, 0);
            } else if (el->end_type == EndType::Smooth) {
                right_curve.append(cap_l);
                const Vec2 p1_l = p1 + n0 * half_widths[2 * 1];
                const Vec2 p1_r = p1 - n0 * half_widths[2 * 1];
                Array<Vec2> point_array = {0};
                point_array.items = (Vec2*)&cap_r;
                point_array.size = 1;
                bool angle_constraints[2] = {true, true};
                double angles[2] = {(cap_l - p1_l).angle(), (p1_r - cap_r).angle()};
                Vec2 tension[2] = {Vec2{1, 1}, Vec2{1, 1}};
                right_curve.interpolation(point_array, angles, angle_constraints, tension, 1, 1,
                                          false, false);
            } else if (el->end_type == EndType::Function) {
                Vec2 dir_l = cap_l - (p1 + n0 * half_widths[2 * 1]);
                dir_l.normalize();
                Vec2 dir_r = (p1 - n0 * half_widths[2 * 1]) - cap_r;
                dir_r.normalize();
                Array<Vec2> point_array =
                    (*el->end_function)(cap_l, dir_l, cap_r, dir_r, el->end_function_data);
                right_curve.segment(point_array, false);
                point_array.clear();
            }
        }

        if (spine_points.size > 2) {
            spine_normal = (spine_points[2] - spine_points[1]).ortho();
            spine_normal.normalize();
            Vec2 p2 = spine_points[1] + spine_normal * offsets[2 * 1];
            Vec2 p3 = spine_points[2] + spine_normal * offsets[2 * 2];
            Vec2 t1 = p3 - p2;
            t1.normalize();
            Vec2 n1 = t1.ortho();
            double u0, u1;
            segments_intersection(p1, t0, p2, t1, u0, u1);
            Vec2 p_next = 0.5 * (p1 + u0 * t0 + p2 + u1 * t1);
            Vec2 p = p0;
            double len_sq_next = (p_next - p).length_sq();

            // Right side: -n
            Vec2 r2 = p - n0 * half_widths[2 * 0];
            Vec2 r3 = p_next - n0 * half_widths[2 * 1];
            Vec2 tr1 = r3 - r2;
            tr1.normalize();

            // Left side: +n
            Vec2 l2 = p + n0 * half_widths[2 * 0];
            Vec2 l3 = p_next + n0 * half_widths[2 * 1];
            Vec2 tl1 = l3 - l2;
            tl1.normalize();

            for (int64_t i = 1; i < spine_points.size - 1; i++) {
                Vec2 t2, n2;
                Vec2 r1 = r3;
                Vec2 tr0 = tr1;
                Vec2 l1 = l3;
                Vec2 tl0 = tl1;
                p0 = p2;
                p1 = p3;
                p = p_next;

                if (i + 2 == spine_points.size) {
                    // Last point: no need to find an intersection
                    p_next = p1;
                    t2 = {0, 0};
                    n2 = {0, 0};
                } else {
                    spine_normal = (spine_points[i + 2] - spine_points[i + 1]).ortho();
                    spine_normal.normalize();
                    p2 = spine_points[i + 1] + spine_normal * offsets[2 * (i + 1)];
                    p3 = spine_points[i + 2] + spine_normal * offsets[2 * (i + 2)];
                    t2 = p3 - p2;
                    t2.normalize();
                    n2 = t2.ortho();
                    segments_intersection(p1, t1, p2, t2, u0, u1);
                    p_next = 0.5 * (p1 + u0 * t1 + p2 + u1 * t2);
                }

                r2 = p - n1 * half_widths[2 * i];
                r3 = p_next - n1 * half_widths[2 * (i + 1)];
                tr1 = r3 - r2;
                tr1.normalize();

                l2 = p + n1 * half_widths[2 * i];
                l3 = p_next + n1 * half_widths[2 * (i + 1)];
                tl1 = l3 - l2;
                tl1.normalize();

                // Check whether there is enough room for the bend
                int8_t bend_dir = 0;
                double center_radius = 0;
                if (bend_type != BendType::None) {
                    const double len_sq_prev = len_sq_next;
                    len_sq_next = (p_next - p).length_sq();
                    if (t0.cross(t1) > 0) {
                        center_radius = bend_radius - offsets[2 * i];
                        const double min_len_sq = 4 * center_radius * center_radius;
                        if ((len_sq_prev >= min_len_sq ||
                             (i == 1 && len_sq_prev >= min_len_sq / 4)) &&
                            (len_sq_next >= min_len_sq ||
                             (i == spine_points.size - 2 && len_sq_next >= min_len_sq / 4)))
                            bend_dir = 1;  // Left
                    } else {
                        center_radius = bend_radius + offsets[2 * i];
                        const double min_len_sq = 4 * center_radius * center_radius;
                        if ((len_sq_prev >= min_len_sq ||
                             (i == 1 && len_sq_prev >= min_len_sq / 4)) &&
                            (len_sq_next >= min_len_sq ||
                             (i == spine_points.size - 2 && len_sq_next >= min_len_sq / 4)))
                            bend_dir = -1;  // Right
                    }
                }

                if (bend_dir < 0) {
                    const Vec2 sum_t = t0 + t1;
                    const double d = (fabs(sum_t.x) > fabs(sum_t.y)) ? (n1.x - n0.x) / sum_t.x
                                                                     : (n1.y - n0.y) / sum_t.y;
                    const double initial_angle = n0.angle();
                    double final_angle = n1.angle();
                    if (final_angle > initial_angle) final_angle -= 2 * M_PI;
                    const Vec2 center = p - 0.5 * center_radius * (n0 + n1 + d * (t0 - t1));

                    // Right: inner side of the bend
                    double radius = center_radius - half_widths[2 * i];
                    if (bend_type == BendType::Circular) {
                        const Vec2 arc_start = center + n0 * radius;
                        right_curve.append(arc_start);
                        right_curve.arc(radius, radius, initial_angle, final_angle, 0);
                    } else if (bend_type == BendType::Function) {
                        Array<Vec2> point_array = (*el->bend_function)(
                            radius, initial_angle, final_angle, center, el->bend_function_data);
                        right_curve.segment(point_array, false);
                        point_array.clear();
                    }

                    // Left: outer side of the bend
                    radius = center_radius + half_widths[2 * i];
                    if (bend_type == BendType::Circular) {
                        const Vec2 arc_start = center + n0 * radius;
                        left_curve.append(arc_start);
                        left_curve.arc(radius, radius, initial_angle, final_angle, 0);
                    } else if (bend_type == BendType::Function) {
                        Array<Vec2> point_array = (*el->bend_function)(
                            radius, initial_angle, final_angle, center, el->bend_function_data);
                        left_curve.segment(point_array, false);
                        point_array.clear();
                    }
                } else if (bend_dir > 0) {
                    const Vec2 sum_t = t0 + t1;
                    const double d = (fabs(sum_t.x) > fabs(sum_t.y)) ? (n0.x - n1.x) / sum_t.x
                                                                     : (n0.y - n1.y) / sum_t.y;
                    const double initial_angle = (-n0).angle();
                    double final_angle = (-n1).angle();
                    if (final_angle < initial_angle) final_angle += 2 * M_PI;
                    Vec2 center = p + 0.5 * center_radius * (n0 + n1 + d * (t1 - t0));

                    // Right: outer side of the bend
                    double radius = center_radius + half_widths[2 * i];
                    if (bend_type == BendType::Circular) {
                        const Vec2 arc_start = center - n0 * radius;
                        right_curve.append(arc_start);
                        right_curve.arc(radius, radius, initial_angle, final_angle, 0);
                    } else if (bend_type == BendType::Function) {
                        Array<Vec2> point_array = (*el->bend_function)(
                            radius, initial_angle, final_angle, center, el->bend_function_data);
                        right_curve.segment(point_array, false);
                        point_array.clear();
                    }

                    // Left: inner side of the bend
                    radius = center_radius - half_widths[2 * i];
                    if (bend_type == BendType::Circular) {
                        const Vec2 arc_start = center - n0 * radius;
                        left_curve.append(arc_start);
                        left_curve.arc(radius, radius, initial_angle, final_angle, 0);
                    } else if (bend_type == BendType::Function) {
                        Array<Vec2> point_array = (*el->bend_function)(
                            radius, initial_angle, final_angle, center, el->bend_function_data);
                        left_curve.segment(point_array, false);
                        point_array.clear();
                    }
                } else {
                    if (tr0.cross(tr1) < 0) {
                        // Right: inner side of the bend
                        segments_intersection(r1, tr0, r2, tr1, u0, u1);
                        const Vec2 ri = 0.5 * (r1 + u0 * tr0 + r2 + u1 * tr1);
                        right_curve.append(ri);
                    } else {
                        // Right: outer side of the bend
                        if (join_type == JoinType::Bevel) {
                            right_curve.append(r1);
                            right_curve.append(r2);
                        } else if (join_type == JoinType::Miter) {
                            segments_intersection(r1, tr0, r2, tr1, u0, u1);
                            const Vec2 ri = 0.5 * (r1 + u0 * tr0 + r2 + u1 * tr1);
                            right_curve.append(ri);
                        } else if (join_type == JoinType::Natural) {
                            segments_intersection(r1, tr0, r2, tr1, u0, u1);
                            const double half_width = half_widths[2 * i];
                            u1 = -u1;
                            if (u0 <= half_width && u1 <= half_width) {
                                const Vec2 ri = 0.5 * (r1 + u0 * tr0 + r2 - u1 * tr1);
                                right_curve.append(ri);
                            } else {
                                const Vec2 ri0 = r1 + (u0 > half_width ? half_width : u0) * tr0;
                                right_curve.append(ri0);
                                const Vec2 ri1 = r2 - (u1 > half_width ? half_width : u1) * tr1;
                                right_curve.append(ri1);
                            }
                        } else if (join_type == JoinType::Round) {
                            right_curve.append(r1);
                            const double initial_angle = (-n0).angle();
                            double final_angle = (-n1).angle();
                            if (final_angle < initial_angle) final_angle += 2 * M_PI;
                            right_curve.arc(half_widths[2 * i], half_widths[2 * i], initial_angle,
                                            final_angle, 0);
                        } else if (join_type == JoinType::Smooth) {
                            right_curve.append(r1);
                            Array<Vec2> point_array = {0};
                            point_array.items = (Vec2*)&r2;
                            point_array.size = 1;
                            bool angle_constraints[2] = {true, true};
                            double angles[2] = {tr0.angle(), tr1.angle()};
                            Vec2 tension[2] = {Vec2{1, 1}, Vec2{1, 1}};
                            right_curve.interpolation(point_array, angles, angle_constraints,
                                                      tension, 1, 1, false, false);
                        } else if (join_type == JoinType::Function) {
                            Array<Vec2> point_array =
                                (*el->join_function)(r1, tr0, r2, tr1, p, half_widths[2 * i] * 2,
                                                     el->join_function_data);
                            right_curve.segment(point_array, false);
                            point_array.clear();
                        }
                    }

                    if (tl0.cross(tl1) > 0) {
                        // Left: inner side of the bend
                        segments_intersection(l1, tl0, l2, tl1, u0, u1);
                        const Vec2 li = 0.5 * (l1 + u0 * tl0 + l2 + u1 * tl1);
                        left_curve.append(li);
                    } else {
                        // Left: outer side of the bend
                        if (join_type == JoinType::Bevel) {
                            left_curve.append(l1);
                            left_curve.append(l2);
                        } else if (join_type == JoinType::Miter) {
                            segments_intersection(l1, tl0, l2, tl1, u0, u1);
                            const Vec2 li = 0.5 * (l1 + u0 * tl0 + l2 + u1 * tl1);
                            left_curve.append(li);
                        } else if (join_type == JoinType::Natural) {
                            segments_intersection(l1, tl0, l2, tl1, u0, u1);
                            const double half_width = half_widths[2 * i];
                            u1 = -u1;
                            if (u0 <= half_width && u1 <= half_width) {
                                const Vec2 li = 0.5 * (l1 + u0 * tl0 + l2 - u1 * tl1);
                                left_curve.append(li);
                            } else {
                                const Vec2 li0 = l1 + (u0 > half_width ? half_width : u0) * tl0;
                                left_curve.append(li0);
                                const Vec2 li1 = l2 - (u1 > half_width ? half_width : u1) * tl1;
                                left_curve.append(li1);
                            }
                        } else if (join_type == JoinType::Round) {
                            left_curve.append(l1);
                            const double initial_angle = n0.angle();
                            double final_angle = n1.angle();
                            if (final_angle > initial_angle) final_angle -= 2 * M_PI;
                            left_curve.arc(half_widths[2 * i], half_widths[2 * i], initial_angle,
                                           final_angle, 0);
                        } else if (join_type == JoinType::Smooth) {
                            left_curve.append(l1);
                            Array<Vec2> point_array = {0};
                            point_array.items = (Vec2*)&l2;
                            point_array.size = 1;
                            bool angle_constraints[2] = {true, true};
                            double angles[2] = {tl0.angle(), tl1.angle()};
                            Vec2 tension[2] = {Vec2{1, 1}, Vec2{1, 1}};
                            left_curve.interpolation(point_array, angles, angle_constraints,
                                                     tension, 1, 1, false, false);
                        } else if (join_type == JoinType::Function) {
                            Array<Vec2> point_array =
                                (*el->join_function)(l1, tl0, l2, tl1, p, half_widths[2 * i] * 2,
                                                     el->join_function_data);
                            left_curve.segment(point_array, false);
                            point_array.clear();
                        }
                    }
                }

                t0 = t1;
                n0 = n1;
                t1 = t2;
                n1 = n2;
            }
        }

        {  // End cap
            const int64_t last = spine_points.size - 1;
            const Vec2 cap_l = p1 + n0 * half_widths[2 * (last)];
            const Vec2 cap_r = p1 - n0 * half_widths[2 * (last)];
            if (el->end_type == EndType::Flush) {
                left_curve.append(cap_l);
                if (half_widths[2 * (last)] != 0) left_curve.append(cap_r);
            } else if (el->end_type == EndType::Extended) {
                left_curve.append(cap_l);
                const double extension =
                    el->end_extensions.v >= 0 ? el->end_extensions.v : half_widths[2 * (last)];
                left_curve.append(cap_l + extension * t0);
                if (half_widths[2 * (last)] != 0) left_curve.append(cap_r + extension * t0);
                left_curve.append(cap_r);
            } else if (el->end_type == EndType::Round) {
                left_curve.append(cap_l);
                double initial_angle = n0.angle();
                left_curve.arc(half_widths[2 * (last)], half_widths[2 * (last)], initial_angle,
                               initial_angle - M_PI, 0);
            } else if (el->end_type == EndType::Smooth) {
                left_curve.append(cap_l);
                const Vec2 p0_l = p0 + n0 * half_widths[2 * (last - 1)];
                const Vec2 p0_r = p0 - n0 * half_widths[2 * (last - 1)];
                Array<Vec2> point_array = {0};
                point_array.items = (Vec2*)&cap_r;
                point_array.size = 1;
                bool angle_constraints[2] = {true, true};
                double angles[2] = {(cap_l - p0_l).angle(), (p0_r - cap_r).angle()};
                Vec2 tension[2] = {Vec2{1, 1}, Vec2{1, 1}};
                left_curve.interpolation(point_array, angles, angle_constraints, tension, 1, 1,
                                         false, false);
            } else if (el->end_type == EndType::Function) {
                Vec2 dir_r = cap_r - (p0 - n0 * half_widths[2 * (last - 1)]);
                dir_r.normalize();
                Vec2 dir_l = (p0 + n0 * half_widths[2 * (last - 1)]) - cap_l;
                dir_l.normalize();
                Array<Vec2> point_array =
                    (*el->end_function)(cap_r, dir_r, cap_l, dir_l, el->end_function_data);
                const int64_t size = point_array.size;
                for (int64_t j = 0; j < size / 2; j++) {
                    Vec2 tmp = point_array[size - 1 - j];
                    point_array[size - 1 - j] = point_array[j];
                    point_array[j] = tmp;
                }
                left_curve.segment(point_array, false);
                point_array.clear();
            }
        }

        right_curve.ensure_slots(left_curve.point_array.size);
        Vec2* dst = right_curve.point_array.items + right_curve.point_array.size;
        Vec2* src = left_curve.point_array.items + left_curve.point_array.size - 1;
        for (int64_t i = left_curve.point_array.size - 1; i >= 0; i--) *dst++ = *src--;
        right_curve.point_array.size += left_curve.point_array.size;
        left_curve.clear();

        curve_size_guess = right_curve.point_array.size * 6 / 5;

        Polygon* result_polygon = (Polygon*)malloc(sizeof(Polygon));
        result_polygon->layer = el->layer;
        result_polygon->datatype = el->datatype;
        result_polygon->point_array = right_curve.point_array;
        result_polygon->properties = properties_copy(properties);
        result.append(result_polygon);
    }
    return result;
}

void FlexPath::to_gds(FILE* out, double scaling) {
    remove_overlapping_points();
    if (spine.point_array.size < 2) return;

    const Array<Vec2> spine_points = spine.point_array;
    Array<int32_t> coords = {0};
    coords.ensure_slots(spine_points.size);

    FlexPathElement* el = elements;
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
        int32_t width =
            (scale_width ? 1 : -1) * (int32_t)lround(2 * el->half_width_and_offset[0].u * scaling);
        swap16(buffer_start, COUNT(buffer_start));
        swap32((uint32_t*)&width, 1);
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
        fwrite(&width, sizeof(int32_t), 1, out);

        if (end_type == 4) {
            uint16_t buffer_ext[] = {8, 0x3003};
            width = (int32_t)lround(el->end_extensions.u * scaling);
            swap16(buffer_ext, COUNT(buffer_ext));
            swap32((uint32_t*)&width, 1);
            fwrite(buffer_ext, sizeof(uint16_t), COUNT(buffer_ext), out);
            fwrite(&width, sizeof(int32_t), 1, out);
            buffer_ext[1] = 0x3103;
            width = (int32_t)lround(el->end_extensions.v * scaling);
            swap16(buffer_ext + 1, 1);
            swap32((uint32_t*)&width, 1);
            fwrite(buffer_ext, sizeof(uint16_t), COUNT(buffer_ext), out);
            fwrite(&width, sizeof(int32_t), 1, out);
        }

        {  // Calculate path coordinates (analogous to to_polygons)
            const BendType bend_type = el->bend_type;
            const double bend_radius = el->bend_radius;
            const double* offsets = ((double*)el->half_width_and_offset.items) + 1;
            Vec2 spine_normal = (spine_points[1] - spine_points[0]).ortho();
            spine_normal.normalize();
            Vec2 p0 = spine_points[0] + spine_normal * offsets[2 * 0];
            Vec2 p1 = spine_points[1] + spine_normal * offsets[2 * 1];
            Vec2 t0 = p1 - p0;
            t0.normalize();
            Vec2 n0 = t0.ortho();
            coords.append((int32_t)lround(p0.x * scaling));
            coords.append((int32_t)lround(p0.y * scaling));

            if (spine_points.size > 2) {
                Curve arc = {0};
                arc.tolerance = spine.tolerance;
                spine_normal = (spine_points[2] - spine_points[1]).ortho();
                spine_normal.normalize();
                Vec2 p2 = spine_points[1] + spine_normal * offsets[2 * 1];
                Vec2 p3 = spine_points[2] + spine_normal * offsets[2 * 2];
                Vec2 t1 = p3 - p2;
                t1.normalize();
                Vec2 n1 = t1.ortho();
                double u0, u1;
                segments_intersection(p1, t0, p2, t1, u0, u1);
                Vec2 p_next = 0.5 * (p1 + u0 * t0 + p2 + u1 * t1);
                Vec2 p = p0;
                double len_sq_next = (p_next - p).length_sq();

                for (int64_t i = 1; i < spine_points.size - 1; i++) {
                    Vec2 t2, n2;
                    p0 = p2;
                    p1 = p3;
                    p = p_next;

                    if (i + 2 == spine_points.size) {
                        // Last point: no need to find an intersection
                        p_next = p1;
                        t2 = {0, 0};
                        n2 = {0, 0};
                    } else {
                        spine_normal = (spine_points[i + 2] - spine_points[i + 1]).ortho();
                        spine_normal.normalize();
                        p2 = spine_points[i + 1] + spine_normal * offsets[2 * (i + 1)];
                        p3 = spine_points[i + 2] + spine_normal * offsets[2 * (i + 2)];
                        t2 = p3 - p2;
                        t2.normalize();
                        n2 = t2.ortho();
                        segments_intersection(p1, t1, p2, t2, u0, u1);
                        p_next = 0.5 * (p1 + u0 * t1 + p2 + u1 * t2);
                    }

                    int8_t bend_dir = 0;
                    double radius = 0;
                    if (bend_type != BendType::None) {
                        const double len_sq_prev = len_sq_next;
                        len_sq_next = (p_next - p).length_sq();
                        if (t0.cross(t1) > 0) {
                            radius = bend_radius - offsets[2 * i];
                            const double min_len_sq = 4 * radius * radius;
                            if ((len_sq_prev >= min_len_sq ||
                                 (i == 1 && len_sq_prev >= min_len_sq / 4)) &&
                                (len_sq_next >= min_len_sq ||
                                 (i == spine_points.size - 2 && len_sq_next >= min_len_sq / 4)))
                                bend_dir = 1;  // Left
                        } else {
                            radius = bend_radius + offsets[2 * i];
                            const double min_len_sq = 4 * radius * radius;
                            if ((len_sq_prev >= min_len_sq ||
                                 (i == 1 && len_sq_prev >= min_len_sq / 4)) &&
                                (len_sq_next >= min_len_sq ||
                                 (i == spine_points.size - 2 && len_sq_next >= min_len_sq / 4)))
                                bend_dir = -1;  // Right
                        }
                    }

                    if (bend_dir < 0) {
                        const Vec2 sum_t = t0 + t1;
                        const double d = (fabs(sum_t.x) > fabs(sum_t.y)) ? (n1.x - n0.x) / sum_t.x
                                                                         : (n1.y - n0.y) / sum_t.y;
                        const double initial_angle = n0.angle();
                        double final_angle = n1.angle();
                        if (final_angle > initial_angle) final_angle -= 2 * M_PI;
                        const Vec2 center = p - 0.5 * radius * (n0 + n1 + d * (t0 - t1));
                        if (bend_type == BendType::Circular) {
                            const Vec2 arc_start = center + n0 * radius;
                            arc.append(arc_start);
                            arc.arc(radius, radius, initial_angle, final_angle, 0);
                            double* c = (double*)arc.point_array.items;
                            for (; arc.point_array.size > 0; arc.point_array.size--) {
                                coords.append((int32_t)lround(*c++ * scaling));
                                coords.append((int32_t)lround(*c++ * scaling));
                            }
                        } else if (bend_type == BendType::Function) {
                            Array<Vec2> point_array = (*el->bend_function)(
                                radius, initial_angle, final_angle, center, el->bend_function_data);
                            double* c = (double*)point_array.items;
                            for (; point_array.size > 0; point_array.size--) {
                                coords.append((int32_t)lround(*c++ * scaling));
                                coords.append((int32_t)lround(*c++ * scaling));
                            }
                            point_array.clear();
                        }
                    } else if (bend_dir > 0) {
                        const Vec2 sum_t = t0 + t1;
                        const double d = (fabs(sum_t.x) > fabs(sum_t.y)) ? (n0.x - n1.x) / sum_t.x
                                                                         : (n0.y - n1.y) / sum_t.y;
                        const double initial_angle = (-n0).angle();
                        double final_angle = (-n1).angle();
                        if (final_angle < initial_angle) final_angle += 2 * M_PI;
                        Vec2 center = p + 0.5 * radius * (n0 + n1 + d * (t1 - t0));
                        if (bend_type == BendType::Circular) {
                            const Vec2 arc_start = center - n0 * radius;
                            arc.append(arc_start);
                            arc.arc(radius, radius, initial_angle, final_angle, 0);
                            double* c = (double*)arc.point_array.items;
                            for (; arc.point_array.size > 0; arc.point_array.size--) {
                                coords.append((int32_t)lround(*c++ * scaling));
                                coords.append((int32_t)lround(*c++ * scaling));
                            }
                        } else if (bend_type == BendType::Function) {
                            Array<Vec2> point_array = (*el->bend_function)(
                                radius, initial_angle, final_angle, center, el->bend_function_data);
                            double* c = (double*)point_array.items;
                            for (; point_array.size > 0; point_array.size--) {
                                coords.append((int32_t)lround(*c++ * scaling));
                                coords.append((int32_t)lround(*c++ * scaling));
                            }
                            point_array.clear();
                        }
                    } else {
                        coords.append((int32_t)lround(p.x * scaling));
                        coords.append((int32_t)lround(p.y * scaling));
                    }

                    t0 = t1;
                    n0 = n1;
                    t1 = t2;
                    n1 = n2;
                }
                arc.clear();
            }
            coords.append((int32_t)lround(p1.x * scaling));
            coords.append((int32_t)lround(p1.y * scaling));
        }

        swap32((uint32_t*)coords.items, coords.size);
        int64_t total = coords.size / 2;
        int64_t i0 = 0;
        while (i0 < total) {
            int64_t i1 = total < i0 + 8190 ? total : i0 + 8190;
            uint16_t buffer_pts[] = {(uint16_t)(4 + 8 * (i1 - i0)), 0x1003};
            swap16(buffer_pts, COUNT(buffer_pts));
            fwrite(buffer_pts, sizeof(uint16_t), COUNT(buffer_pts), out);
            fwrite(coords.items + 2 * i0, sizeof(int32_t), 2 * (i1 - i0), out);
            i0 = i1;
        }
        coords.size = 0;

        properties_to_gds(properties, out);

        uint16_t buffer_end[] = {4, 0x1100};
        swap16(buffer_end, COUNT(buffer_end));
        fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
    }
    coords.clear();
}

void FlexPath::to_svg(FILE* out, double scaling) {
    Array<Polygon*> array = to_polygons();
    for (int64_t i = 0; i < array.size; i++) {
        array[i]->to_svg(out, scaling);
        array[i]->clear();
        free(array[i]);
    }
    array.clear();
}

void FlexPath::fill_offsets_and_widths(const double* width, const double* offset) {
    if (num_elements < 1) return;
    const int64_t num_pts = spine.point_array.size - elements[0].half_width_and_offset.size;
    for (int64_t ne = 0; ne < num_elements; ne++) {
        Array<Vec2>* half_width_and_offset = &elements[ne].half_width_and_offset;
        const Vec2 initial_widoff = (*half_width_and_offset)[half_width_and_offset->size - 1];
        const double wid = width == NULL ? 0 : 0.5 * *width++ - initial_widoff.u;
        const double off = offset == NULL ? 0 : *offset++ - initial_widoff.v;
        const Vec2 widoff_change = Vec2{wid, off};
        half_width_and_offset->ensure_slots(num_pts);
        for (int64_t i = 1; i <= num_pts; i++)
            half_width_and_offset->append(initial_widoff + widoff_change * ((double)i / num_pts));
    }
}

void FlexPath::horizontal(const double* coord_x, int64_t size, const double* width,
                          const double* offset, bool relative) {
    spine.horizontal(coord_x, size, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::vertical(const double* coord_y, int64_t size, const double* width,
                        const double* offset, bool relative) {
    spine.vertical(coord_y, size, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::segment(const Array<Vec2> point_array, const double* width, const double* offset,
                       bool relative) {
    spine.segment(point_array, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::cubic(const Array<Vec2> point_array, const double* width, const double* offset,
                     bool relative) {
    spine.cubic(point_array, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::cubic_smooth(const Array<Vec2> point_array, const double* width,
                            const double* offset, bool relative) {
    spine.cubic_smooth(point_array, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::quadratic(const Array<Vec2> point_array, const double* width, const double* offset,
                         bool relative) {
    spine.quadratic(point_array, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::quadratic_smooth(const Array<Vec2> point_array, const double* width,
                                const double* offset, bool relative) {
    spine.quadratic_smooth(point_array, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::bezier(const Array<Vec2> point_array, const double* width, const double* offset,
                      bool relative) {
    spine.bezier(point_array, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::interpolation(const Array<Vec2> point_array, double* angles, bool* angle_constraints,
                             Vec2* tension, double initial_curl, double final_curl, bool cycle,
                             const double* width, const double* offset, bool relative) {
    spine.interpolation(point_array, angles, angle_constraints, tension, initial_curl, final_curl,
                        cycle, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::arc(double radius_x, double radius_y, double initial_angle, double final_angle,
                   double rotation, const double* width, const double* offset) {
    spine.arc(radius_x, radius_y, initial_angle, final_angle, rotation);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::turn(double radius, double angle, const double* width, const double* offset) {
    spine.turn(radius, angle);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::parametric(ParametricVec2 curve_function, void* data, const double* width,
                          const double* offset, bool relative) {
    spine.parametric(curve_function, data, relative);
    fill_offsets_and_widths(width, offset);
}

int64_t FlexPath::commands(const CurveInstruction* items, int64_t size) {
    int64_t result = spine.commands(items, size);
    fill_offsets_and_widths(NULL, NULL);
    return result;
}

}  // namespace gdstk
