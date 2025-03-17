/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <gdstk/allocator.hpp>
#include <gdstk/clipper_tools.hpp>
#include <gdstk/font.hpp>
#include <gdstk/gdsii.hpp>
#include <gdstk/polygon.hpp>
#include <gdstk/repetition.hpp>
#include <gdstk/sort.hpp>
#include <gdstk/utils.hpp>
#include <gdstk/vec.hpp>

namespace gdstk {

void Polygon::print(bool all) const {
    printf("Polygon <%p>, count %" PRIu64 ", layer %" PRIu32 ", datatype %" PRIu32
           ", properties <%p>, owner <%p>\n",
           this, point_array.count, get_layer(tag), get_type(tag), properties, owner);
    if (all) {
        printf("Points: ");
        point_array.print(true);
    }
    properties_print(properties);
    repetition.print();
}

void Polygon::clear() {
    point_array.clear();
    repetition.clear();
    properties_clear(properties);
}

void Polygon::copy_from(const Polygon& polygon) {
    tag = polygon.tag;
    point_array.copy_from(polygon.point_array);
    repetition.copy_from(polygon.repetition);
    properties = properties_copy(polygon.properties);
}

double Polygon::area() const {
    if (point_array.count < 3) return 0;
    double result = 0;
    Vec2* p = point_array.items;
    Vec2 v0 = *p++;
    Vec2 v1 = *p++ - v0;
    for (uint64_t num = point_array.count - 2; num > 0; num--) {
        Vec2 v2 = *p++ - v0;
        result += v1.cross(v2);
        v1 = v2;
    }
    if (repetition.type != RepetitionType::None) result *= repetition.get_count();
    return 0.5 * fabs(result);
}

double Polygon::signed_area() const {
    if (point_array.count < 3) return 0;
    double result = 0;
    Vec2* p = point_array.items;
    Vec2 v0 = *p++;
    Vec2 v1 = *p++ - v0;
    for (uint64_t num = point_array.count - 2; num > 0; num--) {
        Vec2 v2 = *p++ - v0;
        result += v1.cross(v2);
        v1 = v2;
    }
    return 0.5 * result;
}

double Polygon::perimeter() const {
    if (point_array.count < 3) return 0;
    double result = 0;
    Vec2* p = point_array.items;
    Vec2 v0 = *p++;

    for (uint64_t num = point_array.count - 1; num > 0; num--) {
        Vec2 v1 = *p++ - v0;
        result += v1.length();
        v0 += v1;
    }
    result += (point_array.items[0] - point_array.items[point_array.count - 1]).length();
    if (repetition.type != RepetitionType::None) result *= repetition.get_count();
    return result;
}

// Based on algorithm 7 from: Kai Hormann, Alexander Agathos, “The point in
// polygon problem for arbitrary polygons,” Computational Geometry, Volume 20,
// Issue 3, 2001, Pages 131-144, ISSN 0925-7721.
// https://doi.org/10.1016/S0925-7721(01)00012-8
bool Polygon::contain(const Vec2 point) const {
    if (point_array.count == 0) {
        return false;
    }

    Vec2 p0 = point_array[point_array.count - 1];
    if (point == p0) {
        return true;
    }

    int64_t winding = 0;
    Vec2* v = point_array.items;
    for (uint64_t i = point_array.count; i > 0; i--, v++) {
        Vec2 p1 = *v;
        if (p1.y == point.y &&
            (p1.x == point.x || (p0.y == point.y && (p1.x > point.x) == (p0.x < point.x)))) {
            return true;
        }
        if ((p0.y < point.y) != (p1.y < point.y)) {
            if (p0.x >= point.x) {
                if (p1.x > point.x) {
                    winding += p1.y > p0.y ? 1 : -1;
                } else {
                    double det = (p0 - point).cross(p1 - point);
                    if (det == 0) {
                        return true;
                    }
                    if ((det > 0) == (p1.y > p0.y)) {
                        winding += p1.y > p0.y ? 1 : -1;
                    }
                }
            } else if (p1.x > point.x) {
                double det = (p0 - point).cross(p1 - point);
                if (det == 0) {
                    return true;
                }
                if ((det > 0) == (p1.y > p0.y)) {
                    winding += p1.y > p0.y ? 1 : -1;
                }
            }
        }
        p0 = p1;
    }
    return winding != 0;
}

bool Polygon::contain_all(const Array<Vec2>& points) const {
    Vec2 min, max;
    bounding_box(min, max);
    for (uint64_t i = 0; i < points.count; i++) {
        Vec2 point = points[i];
        if (point.x < min.x || point.x > max.x || point.y < min.y || point.x > max.x) return false;
    }
    for (uint64_t i = 0; i < points.count; i++) {
        if (!contain(points[i])) return false;
    }
    return true;
}

bool Polygon::contain_any(const Array<Vec2>& points) const {
    Vec2 min, max;
    bounding_box(min, max);
    for (uint64_t i = 0; i < points.count; i++) {
        Vec2 point = points[i];
        if (point.x >= min.x && point.x <= max.x && point.y >= min.y && point.x <= max.x &&
            contain(point))
            return true;
    }
    return false;
}

void Polygon::bounding_box(Vec2& min, Vec2& max) const {
    min.x = min.y = DBL_MAX;
    max.x = max.y = -DBL_MAX;
    Vec2* p = point_array.items;
    for (uint64_t num = point_array.count; num > 0; num--, p++) {
        if (p->x < min.x) min.x = p->x;
        if (p->x > max.x) max.x = p->x;
        if (p->y < min.y) min.y = p->y;
        if (p->y > max.y) max.y = p->y;
    }
    if (repetition.type != RepetitionType::None) {
        Array<Vec2> offsets = {};
        repetition.get_extrema(offsets);
        Vec2* off = offsets.items;
        Vec2 min0 = min;
        Vec2 max0 = max;
        for (uint64_t i = offsets.count; i > 0; i--, off++) {
            if (min0.x + off->x < min.x) min.x = min0.x + off->x;
            if (max0.x + off->x > max.x) max.x = max0.x + off->x;
            if (min0.y + off->y < min.y) min.y = min0.y + off->y;
            if (max0.y + off->y > max.y) max.y = max0.y + off->y;
        }
        offsets.clear();
    }
}

void Polygon::translate(const Vec2 v) {
    Vec2* p = point_array.items;
    for (uint64_t num = point_array.count; num > 0; num--) *p++ += v;
}

void Polygon::scale(const Vec2 scale_factor, const Vec2 center) {
    Vec2* p = point_array.items;
    for (uint64_t num = point_array.count; num > 0; num--, p++)
        *p = (*p - center) * scale_factor + center;
}

void Polygon::mirror(const Vec2 p0, const Vec2 p1) {
    Vec2 v = p1 - p0;
    double tmp = v.length_sq();
    if (tmp == 0) return;
    Vec2 r = v * (2 / tmp);
    Vec2 p2 = p0 * 2;
    Vec2* p = point_array.items;
    for (uint64_t num = point_array.count; num > 0; num--, p++)
        *p = v * (*p - p0).inner(r) - *p + p2;
}

void Polygon::rotate(double angle, const Vec2 center) {
    double ca = cos(angle);
    double sa = sin(angle);
    Vec2* p = point_array.items;
    for (uint64_t num = point_array.count; num > 0; num--, p++) {
        Vec2 q = *p - center;
        p->x = q.x * ca - q.y * sa + center.x;
        p->y = q.x * sa + q.y * ca + center.y;
    }
}

void Polygon::transform(double magnification, bool x_reflection, double rotation,
                        const Vec2 origin) {
    double ca = cos(rotation);
    double sa = sin(rotation);
    Vec2* p = point_array.items;
    for (uint64_t num = point_array.count; num > 0; num--, p++) {
        Vec2 q = *p * magnification;
        if (x_reflection) q.y = -q.y;
        p->x = q.x * ca - q.y * sa + origin.x;
        p->y = q.x * sa + q.y * ca + origin.y;
    }
}

void Polygon::fillet(const Array<double> radii, double tolerance) {
    if (point_array.count < 3) return;

    Array<Vec2> old_pts;
    old_pts.copy_from(point_array);
    point_array.count = 0;

    const uint64_t old_size = old_pts.count;
    uint64_t j = 0;
    if (old_pts[old_size - 1] == old_pts[0]) {
        j = old_size - 1;
        while (old_pts[j - 1] == old_pts[j]) j -= 1;
    }
    const uint64_t last = j;

    uint64_t i = j == 0 ? old_size - 1 : j - 1;
    Vec2 p0 = old_pts[i];
    Vec2 p1 = old_pts[j];
    Vec2 v0 = p1 - p0;
    double len0 = v0.normalize();

    uint64_t k = last + 1;
    while (k != last) {
        k = j == old_size - 1 ? 0 : j + 1;
        while (old_pts[k] == old_pts[j]) k += 1;

        const Vec2 p2 = old_pts[k];
        Vec2 v1 = p2 - p1;
        const double len1 = v1.normalize();

        const double theta = acos(v0.inner(v1));
        if (theta > 1e-12) {
            const double tant = tan(0.5 * theta);
            const double cost = cos(0.5 * theta);
            Vec2 dv = v1 - v0;
            const double fac = 1 / (cost * dv.length());
            dv *= fac;

            double radius = radii[j % radii.count];
            double max_len = radius * tant;
            if (max_len > 0.5 * (len0 - tolerance)) {
                max_len = 0.5 * (len0 - tolerance);
                radius = max_len / tant;
            }
            if (max_len > 0.5 * (len1 - tolerance)) {
                max_len = 0.5 * (len1 - tolerance);
                radius = max_len / tant;
            }

            double a0 = (v0 * -tant - dv).angle();
            double a1 = (v1 * tant - dv).angle();
            if (a1 - a0 > M_PI)
                a1 -= 2 * M_PI;
            else if (a1 - a0 < -M_PI)
                a1 += 2 * M_PI;

            uint64_t n = 1;
            if (radius > 0) {
                n = arc_num_points(fabs(a1 - a0), radius, tolerance);
                if (n == 0) n = 1;
            }

            point_array.ensure_slots(n);
            if (n == 1) {
                point_array.append_unsafe(p1);
            } else {
                for (uint64_t l = 0; l < n; l++) {
                    const double a = a0 + l * (a1 - a0) / (n - 1.0);
                    Vec2 cosi = {cos(a), sin(a)};
                    point_array.append_unsafe(p1 + (dv + cosi) * radius);
                }
            }
        } else {
            point_array.append(p1);
        }

        // i = j;
        j = k;
        p0 = p1;
        p1 = p2;
        v0 = v1;
        len0 = len1;
    }

    old_pts.clear();
}

void Polygon::fracture(uint64_t max_points, double precision, Array<Polygon*>& result) const {
    if (max_points <= 4) return;
    Polygon* poly = (Polygon*)allocate_clear(sizeof(Polygon));
    poly->point_array.copy_from(point_array);
    result.append(poly);

    double scaling = 1.0 / precision;
    for (uint64_t i = 0; i < result.count;) {
        Polygon* subj = result[i];
        uint64_t num_points = subj->point_array.count;
        if (num_points <= max_points) {
            i++;
            continue;
        }

        Vec2 min;
        Vec2 max;
        subj->bounding_box(min, max);

        const uint64_t num_cuts = num_points / max_points;
        bool x_axis;
        double* coords = (double*)allocate(sizeof(double) * num_points);
        if (max.x - min.x > max.y - min.y) {
            x_axis = true;
            double* x = coords;
            Vec2* pt = subj->point_array.items;
            for (uint64_t j = 0; j < num_points; j++) (*x++) = (pt++)->x;
        } else {
            x_axis = false;
            double* y = coords;
            Vec2* pt = subj->point_array.items;
            for (uint64_t j = 0; j < num_points; j++) (*y++) = (pt++)->y;
        }
        sort(coords, num_points);
        Array<double> interior_coords = {0, 0, coords};
        while (interior_coords.items[0] == coords[0]) ++interior_coords.items;
        interior_coords.count = num_points - (interior_coords.items - coords);
        while (interior_coords.count > 0 &&
               interior_coords.items[interior_coords.count - 1] == coords[num_points - 1])
            --interior_coords.count;

        Array<double> cuts = {};
        if (interior_coords.count == 0) {
            cuts.append((coords[0] + coords[num_points - 1]) * 0.5);
        } else if (interior_coords.count <= num_cuts) {
            cuts.extend(interior_coords);
        } else {
            cuts.ensure_slots(num_cuts);
            const double frac = interior_coords.count / (num_cuts + 1.0);
            for (uint64_t j = 1; j <= num_cuts; j++)
                cuts.append(interior_coords[(uint64_t)(j * frac)]);
        }
        free_allocation(coords);

        Array<Polygon*>* chopped =
            (Array<Polygon*>*)allocate_clear((cuts.count + 1) * sizeof(Array<Polygon*>));
        slice(*subj, cuts, x_axis, scaling, chopped);

        subj->point_array.clear();
        result.remove_unordered(i);
        free_allocation(subj);

        uint64_t total = 0;
        for (uint64_t j = 0; j <= cuts.count; j++) total += chopped[j].count;
        result.ensure_slots(total);

        for (uint64_t j = 0; j <= cuts.count; j++) {
            result.extend(chopped[j]);
            chopped[j].clear();
        }

        cuts.clear();
        free_allocation(chopped);
    }

    for (uint64_t i = 0; i < result.count; i++) {
        poly = result[i];
        poly->tag = tag;
        poly->repetition.copy_from(repetition);
        poly->properties = properties_copy(properties);
    }
}

void Polygon::apply_repetition(Array<Polygon*>& result) {
    if (repetition.type == RepetitionType::None) return;

    Array<Vec2> offsets = {};
    repetition.get_offsets(offsets);
    repetition.clear();

    // Skip first offset (0, 0)
    Vec2* offset_p = offsets.items + 1;
    result.ensure_slots(offsets.count - 1);
    for (uint64_t offset_count = offsets.count - 1; offset_count > 0; offset_count--) {
        Polygon* poly = (Polygon*)allocate_clear(sizeof(Polygon));
        poly->copy_from(*this);
        poly->translate(*offset_p++);
        result.append_unsafe(poly);
    }

    offsets.clear();
    return;
}

ErrorCode Polygon::to_gds(FILE* out, double scaling) const {
    ErrorCode error_code = ErrorCode::NoError;
    if (point_array.count < 3) return error_code;

    uint16_t buffer_start[] = {4, 0x0800};
    uint16_t buffer_end[] = {4, 0x1100};
    big_endian_swap16(buffer_start, COUNT(buffer_start));
    big_endian_swap16(buffer_end, COUNT(buffer_end));

    uint64_t total = point_array.count + 1;
    if (total > 8190) {
        if (error_logger)
            fputs(
                "[GDSTK] Polygons with more than 8190 are not supported by the official GDSII specification. This GDSII file might not be compatible with all readers.\n",
                error_logger);
        error_code = ErrorCode::UnofficialSpecification;
    }
    Array<int32_t> coords = {};
    coords.ensure_slots(2 * total);
    coords.count = 2 * total;

    Vec2 zero = {0, 0};
    Array<Vec2> offsets = {};
    if (repetition.type != RepetitionType::None) {
        repetition.get_offsets(offsets);
    } else {
        offsets.count = 1;
        offsets.items = &zero;
    }

    double* offset_p = (double*)offsets.items;
    for (uint64_t offset_count = offsets.count; offset_count > 0; offset_count--) {
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
        tag_to_gds(out, tag, GdsiiRecord::DATATYPE);

        double offset_x = *offset_p++;
        double offset_y = *offset_p++;
        int32_t* c = coords.items;
        Vec2* p = point_array.items;
        for (uint64_t j = point_array.count; j > 0; j--) {
            *c++ = (int32_t)lround((offset_x + p->x) * scaling);
            *c++ = (int32_t)lround((offset_y + p->y) * scaling);
            p++;
        }
        *c++ = coords[0];
        *c++ = coords[1];
        big_endian_swap32((uint32_t*)coords.items, coords.count);

        uint64_t i0 = 0;
        while (i0 < total) {
            uint64_t i1 = total < i0 + 8190 ? total : i0 + 8190;
            uint16_t buffer_pts[] = {(uint16_t)(4 + 8 * (i1 - i0)), 0x1003};
            big_endian_swap16(buffer_pts, COUNT(buffer_pts));
            fwrite(buffer_pts, sizeof(uint16_t), COUNT(buffer_pts), out);
            fwrite(coords.items + 2 * i0, sizeof(int32_t), 2 * (i1 - i0), out);
            i0 = i1;
        }

        ErrorCode err = properties_to_gds(properties, out);
        if (err != ErrorCode::NoError) error_code = err;

        fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
    }

    if (repetition.type != RepetitionType::None) offsets.clear();
    coords.clear();
    return error_code;
}

static bool is_rectangle(const Array<IntVec2> points, IntVec2& corner, IntVec2& size) {
    if (points.count == 4 && ((points[0].x == points[1].x && points[1].y == points[2].y &&
                               points[2].x == points[3].x && points[3].y == points[0].y) ||
                              (points[0].y == points[1].y && points[1].x == points[2].x &&
                               points[2].y == points[3].y && points[3].x == points[0].x))) {
        for (uint64_t i = 0; i < 2; i++) {
            int64_t e0 = points[0].e[i];
            int64_t e2 = points[2].e[i];
            if (e0 < e2) {
                size.e[i] = e2 - e0;
                corner.e[i] = e0;
            } else {
                size.e[i] = e0 - e2;
                corner.e[i] = e2;
            }
        }
        return true;
    }
    return false;
}

// 0 <= type < 26 matches the CTRAPEZOID types, 26 is for horizontal TRAPEZOID, 27 for vertical
static bool is_trapezoid(const Array<IntVec2> points, uint8_t& type, IntVec2& corner, IntVec2& size,
                         int64_t& delta_a, int64_t& delta_b) {
    if (points.count == 4) {
        IntVec2 p, q, r, s;
        if (points[0].x == points[1].x && points[2].x == points[3].x) {
            if (points[0].x < points[2].x) {
                if (points[0].y < points[1].y || points[3].y < points[2].y) {
                    p = points[0];
                    q = points[1];
                    r = points[3];
                    s = points[2];
                } else {
                    p = points[1];
                    q = points[0];
                    r = points[2];
                    s = points[3];
                }
            } else {
                if (points[0].y < points[1].y || points[3].y < points[2].y) {
                    p = points[3];
                    q = points[2];
                    r = points[0];
                    s = points[1];
                } else {
                    p = points[2];
                    q = points[3];
                    r = points[1];
                    s = points[0];
                }
            }
            type = 27;
        } else if (points[3].x == points[0].x && points[1].x == points[2].x) {
            if (points[3].x < points[1].x) {
                if (points[3].y < points[0].y || points[2].y < points[1].y) {
                    p = points[3];
                    q = points[0];
                    r = points[2];
                    s = points[1];
                } else {
                    p = points[0];
                    q = points[3];
                    r = points[1];
                    s = points[2];
                }
            } else {
                if (points[3].y < points[0].y || points[2].y < points[1].y) {
                    p = points[2];
                    q = points[1];
                    r = points[3];
                    s = points[0];
                } else {
                    p = points[1];
                    q = points[2];
                    r = points[0];
                    s = points[3];
                }
            }
            type = 27;
        } else if (points[0].y == points[1].y && points[2].y == points[3].y) {
            if (points[0].y < points[2].y) {
                if (points[0].x < points[1].x || points[3].x < points[2].x) {
                    p = points[3];
                    q = points[2];
                    r = points[0];
                    s = points[1];
                } else {
                    p = points[2];
                    q = points[3];
                    r = points[1];
                    s = points[0];
                }
            } else {
                if (points[0].x < points[1].x || points[3].x < points[2].x) {
                    p = points[0];
                    q = points[1];
                    r = points[3];
                    s = points[2];
                } else {
                    p = points[1];
                    q = points[0];
                    r = points[2];
                    s = points[3];
                }
            }
            type = 26;
        } else if (points[3].y == points[0].y && points[1].y == points[2].y) {
            if (points[3].y < points[1].y) {
                if (points[3].x < points[0].x || points[2].x < points[1].x) {
                    p = points[2];
                    q = points[1];
                    r = points[3];
                    s = points[0];
                } else {
                    p = points[1];
                    q = points[2];
                    r = points[0];
                    s = points[3];
                }
            } else {
                if (points[3].x < points[0].x || points[2].x < points[1].x) {
                    p = points[3];
                    q = points[0];
                    r = points[2];
                    s = points[1];
                } else {
                    p = points[0];
                    q = points[3];
                    r = points[1];
                    s = points[2];
                }
            }
            type = 26;
        } else {
            return false;
        }

        if (type == 26) {
            corner.x = p.x < r.x ? p.x : r.x;
            size.x = (s.x > q.x ? s.x : q.x) - corner.x;
            delta_a = p.x - r.x;
            delta_b = q.x - s.x;
            corner.y = r.y;
            size.y = p.y - r.y;
            if (delta_a == 0) {
                if (delta_b == 0) {
                    type = size.x == size.y ? 25 : 24;
                } else if (delta_b == -size.y) {
                    type = 0;
                } else if (delta_b == size.y) {
                    type = 1;
                }
            } else if (delta_a == -size.y) {
                if (delta_b == 0) {
                    type = 3;
                } else if (delta_b == -size.y) {
                    type = 7;
                } else if (delta_b == size.y) {
                    type = 5;
                }
            } else if (delta_a == size.y) {
                if (delta_b == 0) {
                    type = 2;
                } else if (delta_b == -size.y) {
                    type = 4;
                } else if (delta_b == size.y) {
                    type = 6;
                }
            }
        } else {
            corner.y = p.y < r.y ? p.y : r.y;
            size.y = (s.y > q.y ? s.y : q.y) - corner.y;
            delta_a = p.y - r.y;
            delta_b = q.y - s.y;
            corner.x = p.x;
            size.x = r.x - p.x;
            if (delta_a == 0) {
                if (delta_b == 0) {
                    type = size.x == size.y ? 25 : 24;
                } else if (delta_b == -size.x) {
                    type = 9;
                } else if (delta_b == size.x) {
                    type = 8;
                }
            } else if (delta_a == -size.x) {
                if (delta_b == 0) {
                    type = 10;
                } else if (delta_b == -size.x) {
                    type = 14;
                } else if (delta_b == size.x) {
                    type = 12;
                }
            } else if (delta_a == size.x) {
                if (delta_b == 0) {
                    type = 11;
                } else if (delta_b == -size.x) {
                    type = 13;
                } else if (delta_b == size.x) {
                    type = 15;
                }
            }
        }
        return true;
    } else if (points.count == 3) {
        IntVec2 p, q, r;
        // Sort p < q < r
        if (points[0] < points[1]) {
            if (points[0] < points[2]) {
                p = points[0];
                if (points[1] < points[2]) {
                    q = points[1];
                    r = points[2];
                } else {
                    q = points[2];
                    r = points[1];
                }
            } else {
                p = points[2];
                q = points[0];
                r = points[1];
            }
        } else {
            if (points[1] < points[2]) {
                p = points[1];
                if (points[0] < points[2]) {
                    q = points[0];
                    r = points[2];
                } else {
                    q = points[2];
                    r = points[0];
                }
            } else {
                p = points[2];
                q = points[1];
                r = points[0];
            }
        }
        corner.x = p.x;
        size.x = r.x - p.x;
        corner.y = p.y < q.y ? (p.y < r.y ? p.y : r.y) : (q.y < r.y ? q.y : r.y);
        size.y = (p.y > q.y ? (p.y > r.y ? p.y : r.y) : (q.y > r.y ? q.y : r.y)) - corner.y;
        if (size.x == size.y) {
            if (q.x == p.x) {
                if (r.y == p.y) {
                    type = 16;
                    return true;
                } else if (r.y == q.y) {
                    type = 17;
                    return true;
                }
                return false;
            } else if (r.x == q.x) {
                if (p.y == q.y) {
                    type = 18;
                    return true;
                } else if (p.y == r.y) {
                    type = 19;
                    return true;
                }
                return false;
            }
            return false;
        } else if (size.x == 2 * size.y && p.y == r.y && q.x == corner.x + size.y) {
            type = q.y > p.y ? 20 : 21;
            return true;
        } else if (size.y == 2 * size.x) {
            if (p.x == q.x && r.y == corner.y + size.x) {
                type = 22;
                return true;
            } else if (q.x == r.x && p.y == corner.y + size.x) {
                type = 23;
                return true;
            }
            return false;
        }
        return false;
    }
    return false;
}

#define CIRCLE_DETECTION_LSQ_COEFFICIENTS 4
static bool is_circle(const Array<Vec2> point_array, double tolerance, Vec2& center,
                      double& radius) {
    if (point_array.count <= CIRCLE_DETECTION_LSQ_COEFFICIENTS) return false;

    double coef_a = 0;
    double coef_b = 0;
    double coef_m = 0;
    double res_a = 0;
    double res_b = 0;
    Vec2 ref = point_array[0];
    double ref_length_sq = ref.length_sq();
    for (uint64_t i = 1; i <= CIRCLE_DETECTION_LSQ_COEFFICIENTS; i++) {
        uint64_t j = i * (point_array.count - 1) / CIRCLE_DETECTION_LSQ_COEFFICIENTS;
        Vec2 ab = 2 * (point_array[j] - ref);
        double r = point_array[j].length_sq() - ref_length_sq;
        coef_a += ab.x * ab.x;
        coef_b += ab.y * ab.y;
        coef_m += ab.x * ab.y;
        res_a += ab.x * r;
        res_b += ab.y * r;
    }
    double den = coef_a * coef_b - coef_m * coef_m;
    if (fabs(den) < GDSTK_PARALLEL_EPS) return false;
    center.x = (coef_b * res_a - coef_m * res_b) / den;
    center.y = (coef_a * res_b - coef_m * res_a) / den;
    // printf("Center: (%lf, %lf)\n", center.x, center.y);

    radius = 0;
    for (uint64_t i = 0; i <= CIRCLE_DETECTION_LSQ_COEFFICIENTS; i++) {
        uint64_t j = i * (point_array.count - 1) / CIRCLE_DETECTION_LSQ_COEFFICIENTS;
        radius += (point_array[j] - center).length();
    }
    radius /= 1 + CIRCLE_DETECTION_LSQ_COEFFICIENTS;
    // printf("Radius: %lf\n", radius);

    if (point_array.count < arc_num_points(2 * M_PI, radius, tolerance)) return false;

    double radius_sq = radius * radius;
    double neighbor_distance_sq = tolerance + 2 * sqrt(2 * tolerance * (radius - tolerance));
    neighbor_distance_sq *= neighbor_distance_sq;
    Vec2* pt = point_array.items;
    Vec2* last = point_array.items + point_array.count - 1;
    for (uint64_t i = point_array.count; i > 0; i--) {
        if (fabs((*pt - center).length_sq() - radius_sq) >= tolerance ||
            (*pt - *last).length_sq() >= neighbor_distance_sq) {
            return false;
        }
        last = pt++;
    }

    return true;
}

ErrorCode Polygon::to_oas(OasisStream& out, OasisState& state) const {
    ErrorCode error_code = ErrorCode::NoError;
    Vec2 center;
    double radius;
    IntVec2 corner, size;
    int64_t delta_a, delta_b;
    uint8_t type;
    bool has_repetition = repetition.get_count() > 1;
    Array<IntVec2> points = {};
    scale_and_round_array(point_array, state.scaling, points);

    if ((state.config_flags & OASIS_CONFIG_DETECT_RECTANGLES) &&
        is_rectangle(points, corner, size)) {
        bool is_square = size.x == size.y;
        uint8_t info;
        if (is_square) {
            info = 0xDB;
        } else {
            info = 0x7B;
        }
        if (has_repetition) info |= 0x04;
        oasis_putc((int)OasisRecord::RECTANGLE, out);
        oasis_putc(info, out);
        oasis_write_unsigned_integer(out, get_layer(tag));
        oasis_write_unsigned_integer(out, get_type(tag));
        oasis_write_unsigned_integer(out, size.x);
        if (!is_square) oasis_write_unsigned_integer(out, size.y);
        oasis_write_integer(out, corner.x);
        oasis_write_integer(out, corner.y);
        // if (is_square)
        //     printf("SQUARE @ (%ld, %ld) w %ld\n", corner.x, corner.y, size.x);
        // else
        //     printf("RECTANGLE @ (%ld, %ld) w %ld, h %ld\n", corner.x, corner.y, size.x,
        //     size.y);
    } else if ((state.config_flags & OASIS_CONFIG_DETECT_TRAPEZOIDS) &&
               is_trapezoid(points, type, corner, size, delta_a, delta_b)) {
        if (type > 25) {
            uint8_t info = type == 26 ? 0x7B : 0xFB;
            if (has_repetition) info |= 0x04;
            if (delta_a == 0) {
                oasis_putc((int)OasisRecord::TRAPEZOID_B, out);
            } else if (delta_b == 0) {
                oasis_putc((int)OasisRecord::TRAPEZOID_A, out);
            } else {
                oasis_putc((int)OasisRecord::TRAPEZOID_AB, out);
            }
            oasis_putc(info, out);
            oasis_write_unsigned_integer(out, get_layer(tag));
            oasis_write_unsigned_integer(out, get_type(tag));
            oasis_write_unsigned_integer(out, size.x);
            oasis_write_unsigned_integer(out, size.y);
            if (delta_a == 0) {
                oasis_write_1delta(out, delta_b);
                // printf("TRAPEZOID_B %s @ (%ld, %ld) w %ld, h %ld, db %ld\n",
                //        type == 26 ? "hor" : "ver", corner.x, corner.y, size.x, size.y,
                //        delta_b);
            } else if (delta_b == 0) {
                oasis_write_1delta(out, delta_a);
                // printf("TRAPEZOID_A %s @ (%ld, %ld) w %ld, h %ld, da %ld\n",
                //        type == 26 ? "hor" : "ver", corner.x, corner.y, size.x, size.y,
                //        delta_a);
            } else {
                oasis_write_1delta(out, delta_a);
                oasis_write_1delta(out, delta_b);
                // printf("TRAPEZOID_AB %s @ (%ld, %ld) w %ld, h %ld, da %ld, db %ld\n",
                //        type == 26 ? "hor" : "ver", corner.x, corner.y, size.x, size.y,
                //        delta_a, delta_b);
            }
        } else {
            uint8_t info = 0x9B;
            bool use_h = type < 16 || type == 20 || type == 21 || type == 24;
            bool use_w = type != 20 && type != 21;
            if (use_h) info |= 0x20;
            if (use_w) info |= 0x40;
            if (has_repetition) info |= 0x04;
            oasis_putc((int)OasisRecord::CTRAPEZOID, out);
            oasis_putc(info, out);
            oasis_write_unsigned_integer(out, get_layer(tag));
            oasis_write_unsigned_integer(out, get_type(tag));
            oasis_putc(type, out);
            if (use_w) oasis_write_unsigned_integer(out, size.x);
            if (use_h) oasis_write_unsigned_integer(out, size.y);
            // if (use_w && use_h)
            //     printf("CTRAPEZOID %hu @ (%ld, %ld) w  %ld, h %ld\n", type, corner.x,
            //     corner.y,
            //            size.x, size.y);
            // else if (use_w)
            //     printf("CTRAPEZOID %hu @ (%ld, %ld) w %ld\n", type, corner.x, corner.y,
            //     size.x);
            // else
            //     printf("CTRAPEZOID %hu @ (%ld, %ld) h %ld\n", type, corner.x, corner.y,
            //     size.y);
        }
        oasis_write_integer(out, corner.x);
        oasis_write_integer(out, corner.y);
    } else if (state.circle_tolerance > 0 &&
               is_circle(point_array, state.circle_tolerance, center, radius)) {
        uint8_t info = 0x3B;
        if (has_repetition) info |= 0x04;
        oasis_putc((int)OasisRecord::CIRCLE, out);
        oasis_putc(info, out);
        oasis_write_unsigned_integer(out, get_layer(tag));
        oasis_write_unsigned_integer(out, get_type(tag));
        oasis_write_unsigned_integer(out, (uint64_t)llround(radius * state.scaling));
        oasis_write_integer(out, (int64_t)llround(center.x * state.scaling));
        oasis_write_integer(out, (int64_t)llround(center.y * state.scaling));
        // printf("CIRCLE @ (%lf, %lf) r %lf\n", center.x, center.y, radius);
    } else {
        uint8_t info = 0x3B;
        if (has_repetition) info |= 0x04;
        oasis_putc((int)OasisRecord::POLYGON, out);
        oasis_putc(info, out);
        oasis_write_unsigned_integer(out, get_layer(tag));
        oasis_write_unsigned_integer(out, get_type(tag));
        oasis_write_point_list(out, points, true);
        oasis_write_integer(out, points[0].x);
        oasis_write_integer(out, points[0].y);
        // printf("POLYGON @ (%ld, %ld)\n", points[0].x, points[0].y);
    }
    if (has_repetition) oasis_write_repetition(out, repetition, state.scaling);
    ErrorCode err = properties_to_oas(properties, out, state);
    if (err != ErrorCode::NoError) error_code = err;

    points.clear();
    return error_code;
}

ErrorCode Polygon::to_svg(FILE* out, double scaling, uint32_t precision) const {
    if (point_array.count < 3) return ErrorCode::NoError;
    char double_buffer[GDSTK_DOUBLE_BUFFER_COUNT];
    fprintf(out, "<polygon id=\"%p\" class=\"l%" PRIu32 "d%" PRIu32 "\" points=\"", this,
            get_layer(tag), get_type(tag));
    Vec2* p = point_array.items;
    for (uint64_t j = 0; j < point_array.count - 1; j++) {
        fputs(double_print(p->x * scaling, precision, double_buffer, COUNT(double_buffer)), out);
        fputc(',', out);
        fputs(double_print(p->y * scaling, precision, double_buffer, COUNT(double_buffer)), out);
        fputc(' ', out);
        p++;
    }
    fputs(double_print(p->x * scaling, precision, double_buffer, COUNT(double_buffer)), out);
    fputc(',', out);
    fputs(double_print(p->y * scaling, precision, double_buffer, COUNT(double_buffer)), out);
    fputs("\"/>\n", out);
    if (repetition.type != RepetitionType::None) {
        Array<Vec2> offsets = {};
        repetition.get_offsets(offsets);
        double* offset_p = (double*)(offsets.items + 1);
        for (uint64_t offset_count = offsets.count - 1; offset_count > 0; offset_count--) {
            double offset_x = *offset_p++;
            double offset_y = *offset_p++;
            fprintf(out, "<use href=\"#%p\" x=\"", this);
            fputs(double_print(offset_x * scaling, precision, double_buffer, COUNT(double_buffer)),
                  out);
            fputs("\" y=\"", out);
            fputs(double_print(offset_y * scaling, precision, double_buffer, COUNT(double_buffer)),
                  out);
            fputs("\"/>\n", out);
        }
        offsets.clear();
    }
    return ErrorCode::NoError;
}

Polygon rectangle(const Vec2 corner1, const Vec2 corner2, Tag tag) {
    Polygon result = {};
    result.tag = tag;
    result.point_array.ensure_slots(4);
    result.point_array.count = 4;
    result.point_array[0] = corner1;
    result.point_array[1] = Vec2{corner2.x, corner1.y};
    result.point_array[2] = corner2;
    result.point_array[3] = Vec2{corner1.x, corner2.y};
    return result;
};

Polygon cross(const Vec2 center, double full_size, double arm_width, Tag tag) {
    const double len = full_size / 2;
    const double half_width = arm_width / 2;
    Polygon result = {};
    result.tag = tag;
    result.point_array.ensure_slots(12);
    result.point_array.count = 12;
    result.point_array[0] = center + Vec2{len, half_width};
    result.point_array[1] = center + Vec2{half_width, half_width};
    result.point_array[2] = center + Vec2{half_width, len};
    result.point_array[3] = center + Vec2{-half_width, len};
    result.point_array[4] = center + Vec2{-half_width, half_width};
    result.point_array[5] = center + Vec2{-len, half_width};
    result.point_array[6] = center + Vec2{-len, -half_width};
    result.point_array[7] = center + Vec2{-half_width, -half_width};
    result.point_array[8] = center + Vec2{-half_width, -len};
    result.point_array[9] = center + Vec2{half_width, -len};
    result.point_array[10] = center + Vec2{half_width, -half_width};
    result.point_array[11] = center + Vec2{len, -half_width};
    return result;
};

Polygon regular_polygon(const Vec2 center, double side_length, uint64_t sides, double rotation,
                        Tag tag) {
    Polygon result = {};
    result.tag = tag;
    result.point_array.ensure_slots(sides);
    result.point_array.count = sides;
    rotation += M_PI / sides - 0.5 * M_PI;
    const double radius = side_length / (2 * sin(M_PI / sides));
    Vec2* v = result.point_array.items;
    for (uint64_t i = 0; i < sides; i++) {
        const double angle = rotation + i * 2 * M_PI / sides;
        *v++ = center + Vec2{radius * cos(angle), radius * sin(angle)};
    }
    return result;
};

Polygon ellipse(const Vec2 center, double radius_x, double radius_y, double inner_radius_x,
                double inner_radius_y, double initial_angle, double final_angle, double tolerance,
                Tag tag) {
    Polygon result = {};
    result.tag = tag;
    const double full_angle =
        (final_angle == initial_angle) ? 2 * M_PI : fabs(final_angle - initial_angle);
    if (inner_radius_x > 0 && inner_radius_y > 0) {
        uint64_t num_points1 =
            1 + arc_num_points(full_angle, radius_x > radius_y ? radius_x : radius_y, tolerance);
        if (num_points1 < GDSTK_MIN_POINTS) num_points1 = GDSTK_MIN_POINTS;
        uint64_t num_points2 =
            1 + arc_num_points(full_angle,
                               inner_radius_x > inner_radius_y ? inner_radius_x : inner_radius_y,
                               tolerance);
        if (num_points2 < GDSTK_MIN_POINTS) num_points2 = GDSTK_MIN_POINTS;

        result.point_array.ensure_slots(num_points1 + num_points2);
        result.point_array.count = num_points1 + num_points2;
        Vec2* v = result.point_array.items;
        if (full_angle == 2 * M_PI) {
            // Ring
            for (uint64_t i = 0; i < num_points1; i++) {
                const double angle = i * 2 * M_PI / (num_points1 - 1);
                *v++ = center + Vec2{radius_x * cos(angle), radius_y * sin(angle)};
            }
            for (uint64_t i = num_points2; i > 0; i--) {
                const double angle = (i - 1) * 2 * M_PI / (num_points2 - 1);
                *v++ = center + Vec2{inner_radius_x * cos(angle), inner_radius_y * sin(angle)};
            }
        } else {
            // Ring slice
            double initial_ell_angle =
                elliptical_angle_transform(initial_angle, radius_x, radius_y);
            double final_ell_angle = elliptical_angle_transform(final_angle, radius_x, radius_y);
            for (uint64_t i = 0; i < num_points1; i++) {
                const double angle =
                    LERP(initial_ell_angle, final_ell_angle, (double)i / (double)(num_points1 - 1));
                *v++ = center + Vec2{radius_x * cos(angle), radius_y * sin(angle)};
            }
            initial_ell_angle =
                elliptical_angle_transform(initial_angle, inner_radius_x, inner_radius_y);
            final_ell_angle =
                elliptical_angle_transform(final_angle, inner_radius_x, inner_radius_y);
            for (uint64_t i = num_points2; i > 0; i--) {
                const double angle = LERP(initial_ell_angle, final_ell_angle,
                                          (double)(i - 1) / (double)(num_points2 - 1));
                *v++ = center + Vec2{inner_radius_x * cos(angle), inner_radius_y * sin(angle)};
            }
        }
    } else {
        uint64_t num_points =
            1 + arc_num_points(full_angle, radius_x > radius_y ? radius_x : radius_y, tolerance);
        if (num_points < GDSTK_MIN_POINTS) num_points = GDSTK_MIN_POINTS;
        if (full_angle == 2 * M_PI) {
            // Full ellipse
            result.point_array.ensure_slots(num_points);
            result.point_array.count = num_points;
            Vec2* v = result.point_array.items;
            for (uint64_t i = 0; i < num_points; i++) {
                const double angle = i * 2 * M_PI / num_points;
                *v++ = center + Vec2{radius_x * cos(angle), radius_y * sin(angle)};
            }
        } else {
            // Slice
            const double initial_ell_angle =
                elliptical_angle_transform(initial_angle, radius_x, radius_y);
            const double final_ell_angle =
                elliptical_angle_transform(final_angle, radius_x, radius_y);
            result.point_array.ensure_slots(num_points + 1);
            result.point_array.count = num_points + 1;
            Vec2* v = result.point_array.items;
            *v++ = center;
            for (uint64_t i = 0; i < num_points; i++) {
                const double angle =
                    LERP(initial_ell_angle, final_ell_angle, (double)i / (num_points - 1.0));
                *v++ = center + Vec2{radius_x * cos(angle), radius_y * sin(angle)};
            }
        }
    }
    return result;
}

Polygon racetrack(const Vec2 center, double straight_length, double radius, double inner_radius,
                  bool vertical, double tolerance, Tag tag) {
    Polygon result = {};
    result.tag = tag;

    double initial_angle;
    Vec2 direction = {};
    if (vertical) {
        direction.y = straight_length / 2;
        initial_angle = 0;
    } else {
        direction.x = straight_length / 2;
        initial_angle = -M_PI / 2;
    }

    const Vec2 c1 = center + direction;
    const Vec2 c2 = center - direction;
    uint64_t num_points = 1 + arc_num_points(M_PI, radius, tolerance);
    if (num_points < GDSTK_MIN_POINTS) num_points = GDSTK_MIN_POINTS;
    result.point_array.ensure_slots(2 * num_points);
    result.point_array.count = 2 * num_points;
    Vec2* v1 = result.point_array.items;
    Vec2* v2 = result.point_array.items + num_points;
    for (uint64_t i = 0; i < num_points; i++) {
        const double angle = initial_angle + i * M_PI / (num_points - 1);
        const Vec2 rad_vec = {radius * cos(angle), radius * sin(angle)};
        *v1++ = c1 + rad_vec;
        *v2++ = c2 - rad_vec;
    }
    if (inner_radius > 0) {
        num_points = 1 + arc_num_points(M_PI, inner_radius, tolerance);
        if (num_points < GDSTK_MIN_POINTS) num_points = GDSTK_MIN_POINTS;
        result.point_array.ensure_slots(2 * num_points + 2);
        v2 = result.point_array.items + result.point_array.count;
        result.point_array.count += 2 * num_points + 2;
        *v2++ = result.point_array[0];
        *v2++ = c1 + Vec2{inner_radius * cos(initial_angle), inner_radius * sin(initial_angle)};
        v1 = v2 + num_points;
        for (uint64_t i = num_points; i > 0; i--) {
            const double angle = initial_angle + (i - 1) * M_PI / (num_points - 1);
            const Vec2 rad_vec = {inner_radius * cos(angle), inner_radius * sin(angle)};
            *v1++ = c1 + rad_vec;
            *v2++ = c2 - rad_vec;
        }
    }
    return result;
}

void text(const char* s, double size, const Vec2 position, bool vertical, Tag tag,
          Array<Polygon*>& result) {
    size /= 16;
    Vec2 cursor = position;
    for (; *s != 0; s++) {
        switch (*s) {
            case 0x20:  // Space
                if (vertical)
                    cursor.y -= size * VERTICAL_STEP;
                else
                    cursor.x += size * HORIZONTAL_STEP;
                break;
            case 0x09:  // Horizontal tab
                if (vertical)
                    cursor.y += size * VERTICAL_TAB;
                else
                    cursor.x += size * HORIZONTAL_TAB;
                break;
            case 0x0A:  // Carriage return
                if (vertical) {
                    cursor.y = position.y;
                    cursor.x += size * VERTICAL_LINESKIP;
                } else {
                    cursor.x = position.x;
                    cursor.y -= size * HORIZONTAL_LINESKIP;
                }
                break;
            default: {
                const int32_t index = *s - FIRST_CODEPOINT;
                if (index >= 0 && index < (int32_t)(COUNT(_first_poly))) {
                    uint16_t p_idx = _first_poly[index];
                    for (uint16_t i = _num_polys[index]; i > 0; i--, p_idx++) {
                        Polygon* p = (Polygon*)allocate_clear(sizeof(Polygon));
                        p->tag = tag;
                        p->point_array.ensure_slots(_num_coords[p_idx]);
                        uint16_t c_idx = _first_coord[p_idx];
                        for (uint16_t j = _num_coords[p_idx]; j > 0; j--, c_idx++) {
                            p->point_array.append_unsafe(cursor + size * _all_coords[c_idx]);
                        }
                        result.append(p);
                    }
                    if (vertical)
                        cursor.y -= size * VERTICAL_STEP;
                    else
                        cursor.x += size * HORIZONTAL_STEP;
                }
            }
        }
    }
}

enum ContourDirection { O = 0, S, W, N, E };

#ifndef NDEBUG
const char DEBUG_DIR[] = "OSWNE";
const char* DEBUG_CASE[] = {" ?", " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7",
                            " 8", " 9", "10", "11", "12", "13", "14", "15", " X"};
#endif

enum ContourState {
    UNINITIALIZED = 0,
    S0000,
    S0001,
    S0010,
    S0011,
    S0100,
    S0101,
    S0110,
    S0111,
    S1000,
    S1001,
    S1010,
    S1011,
    S1100,
    S1101,
    S1110,
    S1111,
    TERMINATED
};

static inline void append_contour_point(Array<Vec2>* points, Vec2 v, const double tolerance) {
    // Simplify polygon
    uint64_t count = points->count;
    if (count < 2) {
        points->append(v);
    } else {
        Vec2 v0 = points->items[count - 1] - points->items[count - 2];
        Vec2 v1 = v - points->items[count - 2];
        double v1_length = v1.length();
        if (fabs(v0.cross(v1)) > tolerance * v1_length) {
            points->append(v);
        } else {
            // DEBUG_PRINT("(%g, %g)×(%g, %g) [%g ≤ %g] ", v0.x, v0.y, v1.x, v1.y,
            //     fabs(v0.cross(v1)), v1_length * tolerance);
            if (v1_length > tolerance) {
                // DEBUG_PRINT("Substitute: (%g, %g) -- (%g, %g) → (%g, %g)\n",
                //             points->items[count - 2].x, points->items[count - 2].y,
                //             points->items[count - 1].x, points->items[count - 1].y, v.x,
                //             v.y);
                points->items[count - 1] = v;
            } else {
                // DEBUG_PRINT("Remove: (%g, %g) ← (%g, %g) + (%g, %g)  [%g ≤ %g]\n",
                //             points->items[count - 2].x, points->items[count - 2].y,
                //             points->items[count - 1].x, points->items[count - 1].y, v.x, v.y,
                //             v1_length, tolerance);
                points->count--;
            }
        }
    }
}

static Polygon* get_polygon(const int64_t start_row, const int64_t start_col, const double* field,
                            const double level, uint8_t* state, const int64_t state_rows,
                            const int64_t state_cols, const double tolerance) {
    // DEBUG_PRINT("Start polygon\n");

    const ContourDirection direction_lookup[] = {O, O, S, E, E, W, S, W, E,
                                                 N, S, N, N, W, S, W, O, O};

    Polygon* result = (Polygon*)allocate_clear(sizeof(Polygon));
    Array<Vec2>* points = &result->point_array;

    const double* f0 = field + start_col + start_row * (state_cols + 1);
    uint8_t* s = state + start_col + start_row * state_cols;
    int64_t row = start_row;
    int64_t col = start_col;
    const ContourDirection start_from = direction_lookup[*s];
    ContourDirection from = start_from;

    do {
        // for (int r = state_rows; r >= 0; r--) {
        //     if (r < state_rows) {
        //         for (int c = 0; c < state_cols; c++) {
        //             DEBUG_PRINT("  | %s", DEBUG_CASE[state[c + r * state_cols]]);
        //         }
        //         DEBUG_PRINT("  |\n");
        //     }
        //     for (int c = 0; c <= state_cols; c++) {
        //         if (c) DEBUG_PRINT("=");
        //         DEBUG_PRINT("%+4.2f", field[c + r * (state_cols + 1)]);
        //     }
        //     DEBUG_PRINT("\n");
        // }

        if (row == -1 && col < state_cols) {
            // S edge
            // DEBUG_PRINT("[%ld, %ld] ", col, row);

            const double* fa = f0 + 1 + state_cols + 1;
            while (col < state_cols && *fa >= level) {
                fa++;
                col++;
            }
            if (col < state_cols) {
                const double fb = *(fa - 1);
                Vec2 v = {col + (level - fb) / (*fa - fb), 0};
                append_contour_point(points, v, tolerance);
                row++;
                s = state + col + row * state_cols;
                from = S;
                // DEBUG_PRINT("→ [%ld, %ld] (%g, %g)\n", col, row, points->items[points->count -
                // 1].x,
                //             points->items[points->count - 1].y);
            } else {
                Vec2 v = {(double)state_cols, 0};
                append_contour_point(points, v, tolerance);
                // DEBUG_PRINT("→ Corner [%ld, %ld] (%g, %g)\n", col, row,
                //             points->items[points->count - 1].x, points->items[points->count -
                //             1].y);
            }
            f0 = field + col + row * (state_cols + 1);
        } else if (col == state_cols && row < state_rows) {
            // E edge
            // DEBUG_PRINT("[%ld, %ld] ", col, row);

            const double* fa = f0 + state_cols + 1;
            while (row < state_rows && *fa >= level) {
                fa += state_cols + 1;
                row++;
            }
            if (row < state_rows) {
                const double fb = *(fa - (state_cols + 1));
                Vec2 v = {(double)state_cols, row + (level - fb) / (*fa - fb)};
                append_contour_point(points, v, tolerance);
                col--;
                s = state + col + row * state_cols;
                from = E;
                // DEBUG_PRINT("→ [%ld, %ld] (%g, %g)\n", col, row, points->items[points->count -
                // 1].x,
                //             points->items[points->count - 1].y);
            } else {
                Vec2 v = {(double)state_cols, (double)state_rows};
                append_contour_point(points, v, tolerance);
                // DEBUG_PRINT("→ Corner [%ld, %ld] (%g, %g)\n", col, row,
                //             points->items[points->count - 1].x, points->items[points->count -
                //             1].y);
            }
            f0 = field + col + row * (state_cols + 1);
        } else if (row == state_rows && col >= 0) {
            // N edge
            // DEBUG_PRINT("[%ld, %ld] ", col, row);

            const double* fa = f0;
            while (col >= 0 && *fa >= level) {
                fa--;
                col--;
            }
            if (col >= 0) {
                const double fb = *(fa + 1);
                Vec2 v = {col + (level - *fa) / (fb - *fa), (double)state_rows};
                append_contour_point(points, v, tolerance);
                row--;
                s = state + col + row * state_cols;
                from = N;
                // DEBUG_PRINT("→ [%ld, %ld] (%g, %g)\n", col, row, points->items[points->count -
                // 1].x,
                //             points->items[points->count - 1].y);
            } else {
                Vec2 v = {0, (double)state_rows};
                append_contour_point(points, v, tolerance);
                // DEBUG_PRINT("→ Corner [%ld, %ld] (%g, %g)\n", col, row,
                //             points->items[points->count - 1].x, points->items[points->count -
                //             1].y);
            }
            f0 = field + col + row * (state_cols + 1);
        } else if (col == -1 && row >= 0) {
            // W edge
            // DEBUG_PRINT("[%ld, %ld] ", col, row);

            const double* fa = f0 + 1;
            while (row >= 0 && *fa >= level) {
                fa -= state_cols + 1;
                row--;
            }
            if (row >= 0) {
                const double fb = *(fa + (state_cols + 1));
                Vec2 v = {0, row + (level - *fa) / (fb - *fa)};
                append_contour_point(points, v, tolerance);
                col++;
                s = state + col + row * state_cols;
                from = W;
                // DEBUG_PRINT("→ [%ld, %ld] (%g, %g)\n", col, row, points->items[points->count -
                // 1].x,
                //             points->items[points->count - 1].y);
            } else {
                Vec2 v = {0, 0};
                append_contour_point(points, v, tolerance);
                // DEBUG_PRINT("→ Corner [%ld, %ld] (%g, %g)\n", col, row,
                //             points->items[points->count - 1].x, points->items[points->count -
                //             1].y);
            }
            f0 = field + col + row * (state_cols + 1);
        } else {
            const double* f1 = f0 + 1;
            const double* f2 = f0 + state_cols + 1;
            const double* f3 = f1 + state_cols + 1;
            Vec2 v = {(double)col, (double)row};

            if (*s == UNINITIALIZED) {
                *s = (*f3 >= level) * 8 + (*f2 >= level) * 4 + (*f1 >= level) * 2 + (*f0 >= level) +
                     1;
            }
            // DEBUG_PRINT("[%ld, %ld] CASE %s from %c", col, row, DEBUG_CASE[*s], DEBUG_DIR[from]);

            switch (*s) {
                // Exit N
                case S0100:
                case S0101:
                case S0111:
                    *s = TERMINATED;
                    v.x += (level - *f2) / (*f3 - *f2);
                    v.y += 1;
                    row++;
                    f0 += state_cols + 1;
                    s += state_cols;
                    from = S;
                    break;
                // Exit E
                case S1000:
                case S1100:
                case S1101:
                    *s = TERMINATED;
                    v.x += 1;
                    v.y += (level - *f1) / (*f3 - *f1);
                    col++;
                    f0++;
                    s++;
                    from = W;
                    break;
                // Exit W
                case S0001:
                case S0011:
                case S1011:
                    *s = TERMINATED;
                    v.y += (level - *f0) / (*f2 - *f0);
                    col--;
                    f0--;
                    s--;
                    from = E;
                    break;
                // Exit S
                case S0010:
                case S1010:
                case S1110:
                    *s = TERMINATED;
                    v.x += (level - *f0) / (*f1 - *f0);
                    row--;
                    f0 -= state_cols + 1;
                    s -= state_cols;
                    from = N;
                    break;
                case S0110:
                    if ((0.25 * (*f0 + *f1 + *f2 + *f3) >= level) ^ (from == W)) {
                        // Exit N
                        *s = from == W ? S0010 : S1110;
                        v.x += (level - *f2) / (*f3 - *f2);
                        v.y += 1;
                        row++;
                        f0 += state_cols + 1;
                        s += state_cols;
                        from = S;
                    } else {
                        // Exit S
                        *s = from == W ? S0111 : S0100;
                        v.x += (level - *f0) / (*f1 - *f0);
                        row--;
                        f0 -= state_cols + 1;
                        s -= state_cols;
                        from = N;
                    }
                    break;
                case S1001:
                    if ((0.25 * (*f0 + *f1 + *f2 + *f3) >= level) ^ (from == S)) {
                        // Exit W
                        *s = from == S ? S1000 : S1101;
                        v.y += (level - *f0) / (*f2 - *f0);
                        col--;
                        f0--;
                        s--;
                        from = E;
                    } else {
                        // Exit E
                        *s = from == S ? S1011 : S0001;
                        v.x += 1;
                        v.y += (level - *f1) / (*f3 - *f1);
                        col++;
                        f0++;
                        s++;
                        from = W;
                    }
                    break;
                default:
                    assert(false);
            }
            append_contour_point(points, v, tolerance);
            // DEBUG_PRINT(" → (%g, %g) → [%ld, %ld]\n", v.x, v.y, col, row);
        }
    } while (row != start_row || col != start_col || from != start_from);

    // DEBUG_PRINT("Finish polygon: %lu points, area = %g\n", result->point_array.count,
    //             result->area());
    return result;
}

ErrorCode contour(const double* data, uint64_t rows, uint64_t cols, double level, double scaling,
                  Array<Polygon*>& result) {
    if (rows == 0 || cols == 0) return ErrorCode::NoError;
    if (rows >= UINT64_MAX - 2 || cols >= UINT64_MAX - 2) return ErrorCode::Overflow;
    ErrorCode error_code = ErrorCode::NoError;

    const uint64_t state_rows = rows - 1;
    const uint64_t state_cols = cols - 1;
    const double tolerance = 0.5 / scaling;
    uint8_t* state = (uint8_t*)allocate_clear(sizeof(uint8_t) * state_rows * state_cols);

    Array<Polygon*> islands = {};
    Array<Polygon*> holes = {};
    Array<double> island_areas = {};
    Array<double> hole_areas = {};

    for (uint64_t row = 0; row < state_rows; row++) {
        for (uint64_t col = 0; col < state_cols; col++) {
            const double* f0 = data + col + row * cols;
            const double* f1 = f0 + 1;
            const double* f2 = f0 + cols;
            const double* f3 = f1 + cols;
            uint8_t* s = state + col + row * state_cols;
            if (*s == UNINITIALIZED) {
                *s = (*f3 >= level) * 8 + (*f2 >= level) * 4 + (*f1 >= level) * 2 + (*f0 >= level) +
                     1;
            }
            // DEBUG_PRINT("Check [%lu, %lu]: %hu\n", col, row, *s - 1);
            // Saddle points must be visited twice, that why we use a while here.
            while (*s > S0000 && *s < S1111) {
                Polygon* poly =
                    get_polygon(row, col, data, level, state, state_rows, state_cols, tolerance);
                double area = poly->signed_area();
                if (area > 0) {
                    // islands are ordered by increasing area
                    uint64_t i = 0;
                    while (i < island_areas.count && area >= island_areas[i]) i++;
                    island_areas.insert(i, area);
                    islands.insert(i, poly);
                } else {
                    area = -area;
                    // holes are ordered by decreasing area
                    uint64_t i = 0;
                    while (i < hole_areas.count && area <= hole_areas[i]) i++;
                    hole_areas.insert(i, area);
                    holes.insert(i, poly);
                }
            }
        }
    }

    if ((island_areas.count == 0 && data[0] >= level) ||
        (island_areas.count > 0 && hole_areas.count > 0 &&
         island_areas[island_areas.count - 1] <= hole_areas[0])) {
        // The whole data edge is above level
        Polygon* poly = (Polygon*)allocate(sizeof(Polygon));
        *poly = rectangle(Vec2{0, 0}, Vec2{(double)state_cols, (double)state_rows}, 0);
        islands.append(poly);
        island_areas.append((double)state_cols * (double)state_rows);
        // DEBUG_PRINT("Appending full rectangle: island[%" PRIu64 "], area = %g\n",
        // islands.count - 1, island_areas[islands.count - 1]);
    }

    // Associate each hole to its island
    Array<Polygon*>* islands_holes =
        (Array<Polygon*>*)allocate_clear(sizeof(Array<Polygon*>) * islands.count);
    for (uint64_t h = 0; h < holes.count; h++) {
        Polygon* hole = holes[h];
        double hole_area = hole_areas[h];
        bool found = false;
        for (uint64_t i = 0; i < islands.count && !found; i++) {
            if (hole_area < island_areas[i] && islands[i]->contain_all(hole->point_array)) {
                islands_holes[i].append(hole);
                found = true;
            }
        }
        if (!found) {
            if (error_logger)
                fprintf(error_logger, "[GDSTK] Unable to process polygon hole in contour.\n");
            error_code = ErrorCode::BooleanError;
            hole->clear();
            free_allocation(hole);
        }
    }

    holes.clear();
    hole_areas.clear();
    island_areas.clear();

    for (uint64_t i = 0; i < islands.count; i++) {
        Polygon* island = islands[i];
        Array<Polygon*>* island_holes = islands_holes + i;
        if (island_holes->count > 0) {
            ErrorCode err = boolean(*island, *island_holes, Operation::Not, scaling, result);
            if (err != ErrorCode::NoError) error_code = err;
            for (uint64_t h = 0; h < island_holes->count; h++) {
                Polygon* hole = island_holes->items[h];
                hole->clear();
                free_allocation(hole);
            }
            island_holes->clear();
            island->clear();
            free_allocation(island);
        } else {
            result.append(island);
        }
    }

    islands.clear();
    free_allocation(islands_holes);
    free_allocation(state);

    return error_code;
}

void inside(const Array<Vec2>& points, const Array<Polygon*>& polygons, bool* result) {
    Vec2 min = {DBL_MAX, DBL_MAX};
    Vec2 max = {-DBL_MAX, -DBL_MAX};
    for (uint64_t j = 0; j < polygons.count; j++) {
        Vec2 a, b;
        polygons[j]->bounding_box(a, b);
        if (a.x < min.x) min.x = a.x;
        if (a.y < min.y) min.y = a.y;
        if (b.x > max.x) max.x = b.x;
        if (b.y > max.y) max.y = b.y;
    }
    for (uint64_t i = 0; i < points.count; i++) {
        Vec2 point = points[i];
        result[i] = false;
        if (point.x >= min.x && point.x <= max.x && point.y >= min.y && point.x <= max.x) {
            for (uint64_t j = 0; j < polygons.count; j++) {
                if (polygons[j]->contain(point)) {
                    result[i] = true;
                    break;
                }
            }
        }
    }
}

bool all_inside(const Array<Vec2>& points, const Array<Polygon*>& polygons) {
    Vec2 min = {DBL_MAX, DBL_MAX};
    Vec2 max = {-DBL_MAX, -DBL_MAX};
    for (uint64_t j = 0; j < polygons.count; j++) {
        Vec2 a, b;
        polygons[j]->bounding_box(a, b);
        if (a.x < min.x) min.x = a.x;
        if (a.y < min.y) min.y = a.y;
        if (b.x > max.x) max.x = b.x;
        if (b.y > max.y) max.y = b.y;
    }
    for (uint64_t i = 0; i < points.count; i++) {
        Vec2 point = points[i];
        if (point.x < min.x || point.x > max.x || point.y < min.y || point.x > max.x) return false;
    }
    for (uint64_t i = 0; i < points.count; i++) {
        Vec2 point = points[i];
        bool inside = false;
        for (uint64_t j = 0; j < polygons.count; j++) {
            if (polygons[j]->contain(point)) {
                inside = true;
                break;
            }
        }
        if (!inside) return false;
    }
    return true;
}

bool any_inside(const Array<Vec2>& points, const Array<Polygon*>& polygons) {
    Vec2 min = {DBL_MAX, DBL_MAX};
    Vec2 max = {-DBL_MAX, -DBL_MAX};
    for (uint64_t j = 0; j < polygons.count; j++) {
        Vec2 a, b;
        polygons[j]->bounding_box(a, b);
        if (a.x < min.x) min.x = a.x;
        if (a.y < min.y) min.y = a.y;
        if (b.x > max.x) max.x = b.x;
        if (b.y > max.y) max.y = b.y;
    }
    for (uint64_t i = 0; i < points.count; i++) {
        Vec2 point = points[i];
        if (point.x >= min.x && point.x <= max.x && point.y >= min.y && point.x <= max.x) {
            for (uint64_t j = 0; j < polygons.count; j++) {
                if (polygons[j]->contain(point)) return true;
            }
        }
    }
    return false;
}

}  // namespace gdstk
