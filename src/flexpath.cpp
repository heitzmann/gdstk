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

#include <cstddef>
#include <gdstk/allocator.hpp>
#include <gdstk/curve.hpp>
#include <gdstk/gdsii.hpp>
#include <gdstk/flexpath.hpp>
#include <gdstk/utils.hpp>

namespace gdstk {

void FlexPath::init(const Vec2 initial_position, double width, double offset, double tolerance,
                    Tag tag) {
    spine.tolerance = tolerance;
    spine.append(initial_position);
    width /= 2;
    for (uint64_t i = 0; i < num_elements; i++) {
        elements[i].half_width_and_offset.append(Vec2{width, offset});
        elements[i].tag = tag;
    }
}

void FlexPath::init(const Vec2 initial_position, const double* width, const double* offset,
                    double tolerance, const Tag* tag) {
    spine.tolerance = tolerance;
    spine.append(initial_position);
    for (uint64_t i = 0; i < num_elements; i++) {
        elements[i].half_width_and_offset.append(Vec2{0.5 * width[i], offset[i]});
        elements[i].tag = tag[i];
    }
}

void FlexPath::init(const Vec2 initial_position, uint64_t num_elements_, double width,
                    double separation, double tolerance, Tag tag) {
    num_elements = num_elements_;
    elements = (FlexPathElement*)allocate_clear(num_elements * sizeof(FlexPathElement));
    spine.tolerance = tolerance;
    spine.append(initial_position);
    width /= 2;
    double i0 = 0.5 * (num_elements - 1);
    for (uint64_t i = 0; i < num_elements; i++) {
        elements[i].half_width_and_offset.append(Vec2{width, separation * (i - i0)});
        elements[i].tag = tag;
    }
}

void FlexPath::init(const Vec2 initial_position, uint64_t num_elements_, const double* width,
                    const double* offset, double tolerance, const Tag* tag) {
    num_elements = num_elements_;
    elements = (FlexPathElement*)allocate_clear(num_elements * sizeof(FlexPathElement));
    spine.tolerance = tolerance;
    spine.append(initial_position);
    for (uint64_t i = 0; i < num_elements; i++) {
        elements[i].half_width_and_offset.append(Vec2{0.5 * width[i], offset[i]});
        elements[i].tag = tag[i];
    }
}

void FlexPath::print(bool all) const {
    printf("FlexPath <%p>, %" PRIu64
           " elements, %s path,%s scaled widths, properties <%p>, owner <%p>\nSpine: ",
           this, num_elements, simple_path ? "GDSII" : "polygonal", scale_width ? "" : " no",
           properties, owner);
    if (all) {
        printf("Spine: ");
        spine.print(true);
        FlexPathElement* el = elements;
        for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
            printf(
                "Element %" PRIu64 ", layer %" PRIu32 ", datatype %" PRIu32
                ", join %s (function <%p>, data <%p>), end %s (function <%p>, data <%p>), end extensions (%lg, %lg), bend %s (function <%p>, data <%p>), bend radius %lg\n",
                ne, get_layer(el->tag), get_type(el->tag), join_type_name(el->join_type),
                el->join_function, el->join_function_data, end_type_name(el->end_type),
                el->end_function, el->end_function_data, el->end_extensions.u, el->end_extensions.v,
                bend_type_name(el->bend_type), el->bend_function, el->bend_function_data,
                el->bend_radius);
        }
    }
    properties_print(properties);
    repetition.print();
}

void FlexPath::clear() {
    spine.clear();
    raith_data.clear();
    FlexPathElement* el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) el->half_width_and_offset.clear();
    free_allocation(elements);
    elements = NULL;
    num_elements = 0;
    repetition.clear();
    properties_clear(properties);
}

void FlexPath::copy_from(const FlexPath& path) {
    spine.copy_from(path.spine);
    properties = properties_copy(path.properties);
    repetition.copy_from(path.repetition);
    scale_width = path.scale_width;
    simple_path = path.simple_path;
    num_elements = path.num_elements;
    raith_data.copy_from(path.raith_data);
    elements = (FlexPathElement*)allocate_clear(num_elements * sizeof(FlexPathElement));

    FlexPathElement* src = path.elements;
    FlexPathElement* dst = elements;
    for (uint64_t ne = 0; ne < path.num_elements; ne++, src++, dst++) {
        dst->half_width_and_offset.copy_from(src->half_width_and_offset);
        dst->tag = src->tag;
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
    for (uint64_t num = spine.point_array.count; num > 0; num--) *p++ += v;
}

void FlexPath::scale(double scael_factor, const Vec2 center) {
    Vec2* p = spine.point_array.items;
    for (uint64_t num = spine.point_array.count; num > 0; num--, p++)
        *p = (*p - center) * scael_factor + center;
    Vec2 wo_scale = {1, fabs(scael_factor)};
    if (scale_width) wo_scale.u = wo_scale.v;
    FlexPathElement* el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        el->end_extensions *= scael_factor;
        Vec2* wo = el->half_width_and_offset.items;
        for (uint64_t num = spine.point_array.count; num > 0; num--) *wo++ *= wo_scale;
    }
}

void FlexPath::mirror(const Vec2 p0, const Vec2 p1) {
    Vec2 v = p1 - p0;
    double tmp = v.length_sq();
    if (tmp == 0) return;
    Vec2 r = v * (2 / tmp);
    Vec2 p2 = p0 * 2;
    Vec2* p = spine.point_array.items;
    for (uint64_t num = spine.point_array.count; num > 0; num--, p++)
        *p = v * (*p - p0).inner(r) - *p + p2;
    FlexPathElement* el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        Vec2* wo = el->half_width_and_offset.items;
        for (uint64_t num = spine.point_array.count; num > 0; num--, wo++) wo->v = -wo->v;
    }
}

void FlexPath::rotate(double angle, const Vec2 center) {
    double ca = cos(angle);
    double sa = sin(angle);
    Vec2* p = spine.point_array.items;
    for (uint64_t num = spine.point_array.count; num > 0; num--, p++) {
        Vec2 q = *p - center;
        p->x = q.x * ca - q.y * sa + center.x;
        p->y = q.x * sa + q.y * ca + center.y;
    }
}

void FlexPath::apply_repetition(Array<FlexPath*>& result) {
    if (repetition.type == RepetitionType::None) return;

    Array<Vec2> offsets = {};
    repetition.get_offsets(offsets);
    repetition.clear();

    // Skip first offset (0, 0)
    Vec2* offset_p = offsets.items + 1;
    result.ensure_slots(offsets.count - 1);
    for (uint64_t offset_count = offsets.count - 1; offset_count > 0; offset_count--) {
        FlexPath* path = (FlexPath*)allocate_clear(sizeof(FlexPath));
        path->copy_from(*this);
        path->translate(*offset_p++);
        result.append_unsafe(path);
    }

    offsets.clear();
    return;
}

void FlexPath::transform(double magnification, bool x_reflection, double rotation,
                         const Vec2 origin) {
    double ca = cos(rotation);
    double sa = sin(rotation);
    Vec2* p = spine.point_array.items;
    for (uint64_t num = spine.point_array.count; num > 0; num--, p++) {
        Vec2 q = *p * magnification;
        if (x_reflection) q.y = -q.y;
        p->x = q.x * ca - q.y * sa + origin.x;
        p->y = q.x * sa + q.y * ca + origin.y;
    }
    Vec2 wo_scale = {1, magnification};
    if (scale_width) wo_scale.x = magnification;
    FlexPathElement* el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        el->end_extensions *= magnification;
        Vec2* wo = el->half_width_and_offset.items;
        for (uint64_t num = spine.point_array.count; num > 0; num--) *wo++ *= wo_scale;
    }
}

void FlexPath::remove_overlapping_points() {
    const double tol_sq = spine.tolerance * spine.tolerance;
    Array<Vec2>* array = &spine.point_array;
    for (uint64_t i = 1; i < array->count;) {
        if (((*array)[i] - (*array)[i - 1]).length_sq() < tol_sq) {
            array->remove(i);
            FlexPathElement* el = elements;
            for (uint64_t ne = 0; ne < num_elements; ne++, el++)
                el->half_width_and_offset.remove(i);
        } else {
            i++;
        }
    }
}

ErrorCode FlexPath::to_polygons(bool filter, Tag tag, Array<Polygon*>& result) {
    remove_overlapping_points();
    if (spine.point_array.count < 2) return ErrorCode::EmptyPath;

    const Array<Vec2> spine_points = spine.point_array;
    uint64_t curve_size_guess = spine_points.count * 2 + 4;

    FlexPathElement* el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        if (filter && el->tag != tag) continue;

        const double* half_widths = (double*)el->half_width_and_offset.items;
        const double* offsets = half_widths + 1;
        const JoinType join_type = el->join_type;
        const BendType bend_type = el->bend_type;
        const double bend_radius = el->bend_radius;

        Curve right_curve = {};
        Curve left_curve = {};
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
            } else if (el->end_type == EndType::HalfWidth || el->end_type == EndType::Extended) {
                const double extension =
                    el->end_type == EndType::Extended ? el->end_extensions.u : half_widths[2 * 0];
                if (extension > 0) right_curve.append(cap_l);
                right_curve.append(cap_l - extension * t0);
                if (half_widths[2 * 0] != 0) right_curve.append(cap_r - extension * t0);
                if (extension > 0) right_curve.append(cap_r);
            } else if (el->end_type == EndType::Round) {
                right_curve.append(cap_l);
                double initial_angle = n0.angle();
                right_curve.arc(half_widths[2 * 0], half_widths[2 * 0], initial_angle,
                                initial_angle + M_PI, 0);
            } else if (el->end_type == EndType::Smooth) {
                right_curve.append(cap_l);
                const Vec2 p1_l = p1 + n0 * half_widths[2 * 1];
                const Vec2 p1_r = p1 - n0 * half_widths[2 * 1];
                Array<Vec2> point_array = {};
                point_array.items = (Vec2*)&cap_r;
                point_array.count = 1;
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

        if (spine_points.count > 2) {
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
            double len_next = (p_next - p).length();

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

            for (uint64_t i = 1; i < spine_points.count - 1; i++) {
                Vec2 t2, n2;
                Vec2 r1 = r3;
                Vec2 tr0 = tr1;
                Vec2 l1 = l3;
                Vec2 tl0 = tl1;
                p0 = p2;
                p1 = p3;
                p = p_next;

                if (i + 2 == spine_points.count) {
                    // Last point: no need to find an intersection
                    p_next = p1;
                    t2 = Vec2{0, 0};
                    n2 = Vec2{0, 0};
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
                double bend_dir = 0;
                double len_factor = 0;
                double center_radius = 0;
                if (bend_type != BendType::None) {
                    bend_dir = t0.cross(t1) < 0 ? -1 : 1;
                    const Vec2 sum_t = t0 + t1;
                    const double len_prev = len_next;
                    len_next = (p_next - p).length();
                    len_factor = (fabs(sum_t.x) > fabs(sum_t.y))
                                     ? bend_dir * (n0.x - n1.x) / sum_t.x
                                     : bend_dir * (n0.y - n1.y) / sum_t.y;
                    center_radius = bend_radius - bend_dir * offsets[2 * i];
                    const double len_required = len_factor * center_radius;
                    if (len_required > len_prev || len_required > len_next ||
                        center_radius <= half_widths[2 * i]) {
                        // Not enough room for the bend
                        bend_dir = 0;
                    } else {
                        len_next -= len_required;
                    }
                }

                if (bend_dir < 0) {
                    const double initial_angle = n0.angle();
                    double final_angle = n1.angle();
                    if (final_angle > initial_angle) final_angle -= 2 * M_PI;
                    const Vec2 center =
                        p - 0.5 * center_radius * (n0 + n1 + len_factor * (t0 - t1));

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
                    const double initial_angle = (-n0).angle();
                    double final_angle = (-n1).angle();
                    if (final_angle < initial_angle) final_angle += 2 * M_PI;
                    Vec2 center = p + 0.5 * center_radius * (n0 + n1 + len_factor * (t1 - t0));

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
                            Array<Vec2> point_array = {};
                            point_array.items = (Vec2*)&r2;
                            point_array.count = 1;
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
                            Array<Vec2> point_array = {};
                            point_array.items = (Vec2*)&l2;
                            point_array.count = 1;
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
            const uint64_t last = spine_points.count - 1;
            const Vec2 cap_l = p1 + n0 * half_widths[2 * (last)];
            const Vec2 cap_r = p1 - n0 * half_widths[2 * (last)];
            if (el->end_type == EndType::Flush) {
                left_curve.append(cap_l);
                if (half_widths[2 * (last)] != 0) left_curve.append(cap_r);
            } else if (el->end_type == EndType::HalfWidth || el->end_type == EndType::Extended) {
                const double extension = el->end_type == EndType::Extended
                                             ? el->end_extensions.v
                                             : half_widths[2 * (last)];
                if (extension > 0) left_curve.append(cap_l);
                left_curve.append(cap_l + extension * t0);
                if (half_widths[2 * (last)] != 0) left_curve.append(cap_r + extension * t0);
                if (extension > 0) left_curve.append(cap_r);
            } else if (el->end_type == EndType::Round) {
                left_curve.append(cap_l);
                double initial_angle = n0.angle();
                left_curve.arc(half_widths[2 * (last)], half_widths[2 * (last)], initial_angle,
                               initial_angle - M_PI, 0);
            } else if (el->end_type == EndType::Smooth) {
                left_curve.append(cap_l);
                const Vec2 p0_l = p0 + n0 * half_widths[2 * (last - 1)];
                const Vec2 p0_r = p0 - n0 * half_widths[2 * (last - 1)];
                Array<Vec2> point_array = {};
                point_array.items = (Vec2*)&cap_r;
                point_array.count = 1;
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
                const uint64_t count = point_array.count;
                for (uint64_t j = 0; j < count / 2; j++) {
                    Vec2 tmp = point_array[count - 1 - j];
                    point_array[count - 1 - j] = point_array[j];
                    point_array[j] = tmp;
                }
                left_curve.segment(point_array, false);
                point_array.clear();
            }
        }

        right_curve.ensure_slots(left_curve.point_array.count);
        Vec2* dst = right_curve.point_array.items + right_curve.point_array.count;
        Vec2* src = left_curve.point_array.items + left_curve.point_array.count - 1;
        for (uint64_t i = left_curve.point_array.count; i > 0; i--) *dst++ = *src--;
        right_curve.point_array.count += left_curve.point_array.count;
        left_curve.clear();

        curve_size_guess = right_curve.point_array.count * 6 / 5;

        Polygon* result_polygon = (Polygon*)allocate_clear(sizeof(Polygon));
        result_polygon->tag = el->tag;
        result_polygon->point_array = right_curve.point_array;
        result_polygon->repetition.copy_from(repetition);
        result_polygon->properties = properties_copy(properties);
        result.append(result_polygon);
    }
    return ErrorCode::NoError;
}

ErrorCode FlexPath::element_center(const FlexPathElement* el, Array<Vec2>& result) {
    const Array<Vec2> spine_points = spine.point_array;
    const BendType bend_type = el->bend_type;
    const double bend_radius = el->bend_radius;
    const double* path_half_widths = (double*)el->half_width_and_offset.items;
    const double* path_offsets = ((double*)el->half_width_and_offset.items) + 1;
    Vec2 spine_normal = (spine_points[1] - spine_points[0]).ortho();
    spine_normal.normalize();
    Vec2 p0 = spine_points[0] + spine_normal * path_offsets[2 * 0];
    Vec2 p1 = spine_points[1] + spine_normal * path_offsets[2 * 1];
    Vec2 t0 = p1 - p0;
    t0.normalize();
    Vec2 n0 = t0.ortho();
    result.append(p0);

    if (spine_points.count > 2) {
        Curve arc_curve = {};
        arc_curve.tolerance = spine.tolerance;
        spine_normal = (spine_points[2] - spine_points[1]).ortho();
        spine_normal.normalize();
        Vec2 p2 = spine_points[1] + spine_normal * path_offsets[2 * 1];
        Vec2 p3 = spine_points[2] + spine_normal * path_offsets[2 * 2];
        Vec2 t1 = p3 - p2;
        t1.normalize();
        Vec2 n1 = t1.ortho();
        double u0, u1;
        segments_intersection(p1, t0, p2, t1, u0, u1);
        Vec2 p_next = 0.5 * (p1 + u0 * t0 + p2 + u1 * t1);
        Vec2 p = p0;
        double len_next = (p_next - p).length();

        for (uint64_t i = 1; i < spine_points.count - 1; i++) {
            Vec2 t2, n2;
            p0 = p2;
            p1 = p3;
            p = p_next;

            if (i + 2 == spine_points.count) {
                // Last point: no need to find an intersection
                p_next = p1;
                t2 = Vec2{0, 0};
                n2 = Vec2{0, 0};
            } else {
                spine_normal = (spine_points[i + 2] - spine_points[i + 1]).ortho();
                spine_normal.normalize();
                p2 = spine_points[i + 1] + spine_normal * path_offsets[2 * (i + 1)];
                p3 = spine_points[i + 2] + spine_normal * path_offsets[2 * (i + 2)];
                t2 = p3 - p2;
                t2.normalize();
                n2 = t2.ortho();
                segments_intersection(p1, t1, p2, t2, u0, u1);
                p_next = 0.5 * (p1 + u0 * t1 + p2 + u1 * t2);
            }

            if (bend_type != BendType::None) {
                const double bend_dir = t0.cross(t1) < 0 ? -1 : 1;
                const double len_prev = len_next;
                len_next = (p_next - p).length();
                const Vec2 sum_t = t0 + t1;
                const double len_factor = (fabs(sum_t.x) > fabs(sum_t.y))
                                              ? bend_dir * (n0.x - n1.x) / sum_t.x
                                              : bend_dir * (n0.y - n1.y) / sum_t.y;
                const double radius = bend_radius - bend_dir * path_offsets[2 * i];
                const double len_required = len_factor * radius;
                if (len_required > len_prev || len_required > len_next ||
                    radius <= path_half_widths[2 * 1]) {
                    // Not enough room for the bend
                    result.append(p);
                } else {
                    len_next -= len_required;
                    if (bend_dir < 0) {  // Right turn
                        const double initial_angle = n0.angle();
                        double final_angle = n1.angle();
                        if (final_angle > initial_angle) final_angle -= 2 * M_PI;
                        const Vec2 center = p - 0.5 * radius * (n0 + n1 + len_factor * (t0 - t1));
                        if (bend_type == BendType::Circular) {
                            const Vec2 arc_start = center + n0 * radius;
                            arc_curve.append(arc_start);
                            arc_curve.arc(radius, radius, initial_angle, final_angle, 0);
                            result.extend(arc_curve.point_array);
                            arc_curve.point_array.count = 0;
                        } else if (bend_type == BendType::Function) {
                            Array<Vec2> bend_array = (*el->bend_function)(
                                radius, initial_angle, final_angle, center, el->bend_function_data);
                            result.extend(bend_array);
                            bend_array.clear();
                        }
                    } else {  // Left turn
                        const double initial_angle = (-n0).angle();
                        double final_angle = (-n1).angle();
                        if (final_angle < initial_angle) final_angle += 2 * M_PI;
                        Vec2 center = p + 0.5 * radius * (n0 + n1 + len_factor * (t1 - t0));
                        if (bend_type == BendType::Circular) {
                            const Vec2 arc_start = center - n0 * radius;
                            arc_curve.append(arc_start);
                            arc_curve.arc(radius, radius, initial_angle, final_angle, 0);
                            result.extend(arc_curve.point_array);
                            arc_curve.point_array.count = 0;
                        } else if (bend_type == BendType::Function) {
                            Array<Vec2> bend_array = (*el->bend_function)(
                                radius, initial_angle, final_angle, center, el->bend_function_data);
                            result.extend(bend_array);
                            bend_array.clear();
                        }
                    }
                }
            } else {
                result.append(p);
            }

            t0 = t1;
            n0 = n1;
            t1 = t2;
            n1 = n2;
        }
        arc_curve.clear();
    }
    result.append(p1);
    return ErrorCode::NoError;
}

ErrorCode FlexPath::to_gds(FILE* out, double scaling) {
    ErrorCode error_code = ErrorCode::NoError;

    remove_overlapping_points();

    if (spine.point_array.count < 2) return ErrorCode::EmptyPath;

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
    point_array.ensure_slots(spine.point_array.count);

    FlexPathElement* el = elements;
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

        uint16_t path_type = raith_data.base_cell_name ? 0x5A00 : 0x0900;
        uint16_t buffer0[] = {4, path_type};
        uint16_t buffer1[] = {6, 0x2102, end_type, 8, 0x0F03};

        PXXData pxxdata = raith_data.to_pxxdata(scaling);
        pxxdata.little_endian_swap();

        uint64_t len = raith_data.base_cell_name ? strlen(raith_data.base_cell_name) : 0;
        if (len % 2) len++;
        uint16_t sname_start[] = {(uint16_t)(4 + len), 0x1206};
        big_endian_swap16(sname_start, COUNT(sname_start));

        int32_t width =
            (scale_width ? 1 : -1) * (int32_t)lround(2 * el->half_width_and_offset[0].u * scaling);
        big_endian_swap16(buffer0, COUNT(buffer0));
        big_endian_swap16(buffer1, COUNT(buffer1));
        big_endian_swap32((uint32_t*)&width, 1);

        uint16_t buffer_ext1[] = {8, 0x3003};
        uint16_t buffer_ext2[] = {8, 0x3103};
        int32_t ext_size[] = {0, 0};
        if (end_type == 4) {
            ext_size[0] = (int32_t)lround(el->end_extensions.u * scaling);
            ext_size[1] = (int32_t)lround(el->end_extensions.v * scaling);
            big_endian_swap16(buffer_ext1, COUNT(buffer_ext1));
            big_endian_swap16(buffer_ext2, COUNT(buffer_ext2));
            big_endian_swap32((uint32_t*)ext_size, COUNT(ext_size));
        }

        ErrorCode err = element_center(el, point_array);
        if (err != ErrorCode::NoError) error_code = err;
        coords.ensure_slots(point_array.count * 2);
        coords.count = point_array.count * 2;

        double* offset_p = (double*)offsets.items;
        for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
            fwrite(buffer0, sizeof(uint16_t), COUNT(buffer0), out);
            tag_to_gds(out, el->tag, GdsiiRecord::DATATYPE);
            fwrite(buffer1, sizeof(uint16_t), COUNT(buffer1), out);
            fwrite(&width, sizeof(int32_t), 1, out);
            if (raith_data.base_cell_name) {
                fwrite(sname_start, sizeof(uint16_t), COUNT(sname_start), out);
                fwrite(raith_data.base_cell_name, 1, len, out);
                uint16_t buffer_pxx[] = {(uint16_t)(4 + sizeof(PXXData)), 0x6206};
                big_endian_swap16(buffer_pxx, COUNT(buffer_pxx));
                fwrite(buffer_pxx, sizeof(uint16_t), COUNT(buffer_pxx), out);
                fwrite(&pxxdata, 1, sizeof(PXXData), out);
            }

            if (end_type == 4) {
                fwrite(buffer_ext1, sizeof(uint16_t), COUNT(buffer_ext1), out);
                fwrite(ext_size, sizeof(int32_t), 1, out);
                fwrite(buffer_ext2, sizeof(uint16_t), COUNT(buffer_ext2), out);
                fwrite(ext_size + 1, sizeof(int32_t), 1, out);
            }

            int32_t* c = coords.items;
            double* p = (double*)point_array.items;
            double offset_x = *offset_p++;
            double offset_y = *offset_p++;
            for (uint64_t i = point_array.count; i > 0; i--) {
                *c++ = (int32_t)lround((*p++ + offset_x) * scaling);
                *c++ = (int32_t)lround((*p++ + offset_y) * scaling);
            }
            big_endian_swap32((uint32_t*)coords.items, coords.count);

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

ErrorCode FlexPath::to_oas(OasisStream& out, OasisState& state) {
    ErrorCode error_code = ErrorCode::NoError;

    remove_overlapping_points();

    if (spine.point_array.count < 2) return ErrorCode::EmptyPath;

    bool has_repetition = repetition.get_count() > 1;

    Array<Vec2> point_array = {};
    point_array.ensure_slots(spine.point_array.count);

    FlexPathElement* el = elements;
    for (uint64_t ne = 0; ne < num_elements; ne++, el++) {
        uint8_t info = 0xFB;
        if (has_repetition) info |= 0x04;

        oasis_putc((int)OasisRecord::PATH, out);
        oasis_putc(info, out);
        oasis_write_unsigned_integer(out, get_layer(el->tag));
        oasis_write_unsigned_integer(out, get_type(el->tag));
        uint64_t half_width = (uint64_t)llround(el->half_width_and_offset[0].u * state.scaling);
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

ErrorCode FlexPath::to_svg(FILE* out, double scaling, uint32_t precision) {
    Array<Polygon*> array = {};
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

void FlexPath::fill_offsets_and_widths(const double* width, const double* offset) {
    if (num_elements < 1) return;
    const uint64_t num_pts = spine.point_array.count - elements[0].half_width_and_offset.count;
    for (uint64_t ne = 0; ne < num_elements; ne++) {
        Array<Vec2>* half_width_and_offset = &elements[ne].half_width_and_offset;
        const Vec2 initial_widoff = (*half_width_and_offset)[half_width_and_offset->count - 1];
        const double wid = width == NULL ? 0 : 0.5 * *width++ - initial_widoff.u;
        const double off = offset == NULL ? 0 : *offset++ - initial_widoff.v;
        const Vec2 widoff_change = Vec2{wid, off};
        half_width_and_offset->ensure_slots(num_pts);
        for (uint64_t i = 1; i <= num_pts; i++)
            half_width_and_offset->append_unsafe(initial_widoff +
                                                 widoff_change * ((double)i / num_pts));
    }
}

void FlexPath::horizontal(double coord_x, const double* width, const double* offset,
                          bool relative) {
    spine.horizontal(coord_x, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::horizontal(const Array<double> coord_x, const double* width, const double* offset,
                          bool relative) {
    spine.horizontal(coord_x, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::vertical(double coord_y, const double* width, const double* offset, bool relative) {
    spine.vertical(coord_y, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::vertical(Array<double> coord_y, const double* width, const double* offset,
                        bool relative) {
    spine.vertical(coord_y, relative);
    fill_offsets_and_widths(width, offset);
}

void FlexPath::segment(Vec2 end_point, const double* width, const double* offset, bool relative) {
    spine.segment(end_point, relative);
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

void FlexPath::quadratic_smooth(Vec2 end_point, const double* width, const double* offset,
                                bool relative) {
    spine.quadratic_smooth(end_point, relative);
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

uint64_t FlexPath::commands(const CurveInstruction* items, uint64_t count) {
    uint64_t result = spine.commands(items, count);
    fill_offsets_and_widths(NULL, NULL);
    return result;
}

}  // namespace gdstk
