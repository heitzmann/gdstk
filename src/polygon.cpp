/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "polygon.h"

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "clipper_tools.h"
#include "font.h"
#include "utils.h"
#include "vec.h"

namespace gdstk {

void Polygon::print(bool all) const {
    printf("Polygon <%p>, size %" PRId64 ", layer %hd, datatype %hd, properties <%p>, owner <%p>\n",
           this, point_array.size, layer, datatype, properties, owner);
    if (all) {
        printf("Points: ");
        point_array.print(true);
    }
}

void Polygon::clear() {
    point_array.clear();
    properties_clear(properties);
    properties = NULL;
}

void Polygon::copy_from(const Polygon& polygon) {
    layer = polygon.layer;
    datatype = polygon.datatype;
    point_array.copy_from(polygon.point_array);
    properties = properties_copy(polygon.properties);
}

double Polygon::area() const {
    if (point_array.size < 3) return 0;
    double result = 0;
    Vec2* p = point_array.items;
    Vec2 v0 = *p++;
    Vec2 v1 = *p++ - v0;
    for (int64_t num = point_array.size - 2; num > 0; num--) {
        Vec2 v2 = *p++ - v0;
        result += v1.cross(v2);
        v1 = v2;
    }
    return 0.5 * fabs(result);
}

void Polygon::bounding_box(Vec2& min, Vec2& max) const {
    min.x = min.y = DBL_MAX;
    max.x = max.y = -DBL_MAX;
    Vec2* p = point_array.items;
    for (int64_t num = point_array.size; num > 0; num--, p++) {
        if (p->x < min.x) min.x = p->x;
        if (p->x > max.x) max.x = p->x;
        if (p->y < min.y) min.y = p->y;
        if (p->y > max.y) max.y = p->y;
    }
}

void Polygon::translate(const Vec2 v) {
    Vec2* p = point_array.items;
    for (int64_t num = point_array.size; num > 0; num--) *p++ += v;
}

void Polygon::scale(const Vec2 scale, const Vec2 center) {
    Vec2* p = point_array.items;
    for (int64_t num = point_array.size; num > 0; num--, p++) *p = (*p - center) * scale + center;
}

void Polygon::mirror(const Vec2 p0, const Vec2 p1) {
    Vec2 v = p1 - p0;
    double tmp = v.length_sq();
    if (tmp == 0) return;
    Vec2 r = v * (2 / tmp);
    Vec2 p2 = p0 * 2;
    Vec2* p = point_array.items;
    for (int64_t num = point_array.size; num > 0; num--, p++) *p = v * (*p - p0).inner(r) - *p + p2;
}

void Polygon::rotate(double angle, const Vec2 center) {
    double ca = cos(angle);
    double sa = sin(angle);
    Vec2* p = point_array.items;
    for (int64_t num = point_array.size; num > 0; num--, p++) {
        Vec2 q = *p - center;
        p->x = q.x * ca - q.y * sa + center.x;
        p->y = q.x * sa + q.y * ca + center.y;
    }
}

void Polygon::transform(double magnification, const Vec2 translation, bool x_reflection,
                        double rotation, const Vec2 origin) {
    double ca = cos(rotation);
    double sa = sin(rotation);
    Vec2* p = point_array.items;
    for (int64_t num = point_array.size; num > 0; num--, p++) {
        Vec2 q = *p * magnification + translation;
        if (x_reflection) q.y = -q.y;
        p->x = q.x * ca - q.y * sa + origin.x;
        p->y = q.x * sa + q.y * ca + origin.y;
    }
}

// radii must be an array of length polygon.size
void Polygon::fillet(const double* radii, double tol) {
    if (point_array.size < 3) return;

    Array<Vec2> old_pts;
    old_pts.copy_from(point_array);
    point_array.size = 0;

    const int64_t old_size = old_pts.size;
    int64_t j = 0;
    if (old_pts[old_size - 1] == old_pts[0]) {
        j = old_size - 1;
        while (old_pts[j - 1] == old_pts[j]) j -= 1;
    }
    const int64_t last = j;

    int64_t i = j == 0 ? old_size - 1 : j - 1;
    Vec2 p0 = old_pts[i];
    Vec2 p1 = old_pts[j];
    Vec2 v0 = p1 - p0;
    double len0 = v0.normalize();

    int64_t k = last + 1;
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

            double radius = radii[j];
            double max_len = radius * tant;
            if (max_len > 0.5 * (len0 - tol)) {
                max_len = 0.5 * (len0 - tol);
                radius = max_len / tant;
            }
            if (max_len > 0.5 * (len1 - tol)) {
                max_len = 0.5 * (len1 - tol);
                radius = max_len / tant;
            }

            double a0 = (v0 * -tant - dv).angle();
            double a1 = (v1 * tant - dv).angle();
            if (a1 - a0 > M_PI)
                a1 -= 2 * M_PI;
            else if (a1 - a0 < -M_PI)
                a1 += 2 * M_PI;
            int64_t n = arc_num_points(fabs(a1 - a0), radius, tol);
            if (n < 1) n = 1;

            point_array.ensure_slots(n);
            if (n == 1) {
                point_array.append(p1);
            } else {
                for (int64_t l = 0; l < n; l++) {
                    const double a = a0 + l * (a1 - a0) / (n - 1.0);
                    Vec2 cosi = {cos(a), sin(a)};
                    point_array.append(p1 + (dv + cosi) * radius);
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

Array<Polygon*> Polygon::fracture(int64_t max_points, double precision) const {
    Array<Polygon*> result = {0};

    if (max_points <= 4) return result;
    Polygon* poly = (Polygon*)calloc(1, sizeof(Polygon));
    poly->point_array.copy_from(point_array);
    result.append(poly);

    double scaling = 1.0 / precision;
    for (int64_t i = 0; i < result.size;) {
        Polygon* subj = result[i];
        int64_t num_points = subj->point_array.size;
        if (num_points <= max_points) {
            i++;
            continue;
        }

        Vec2 min;
        Vec2 max;
        subj->bounding_box(min, max);

        const int64_t num_cuts = num_points / max_points;
        const double frac = num_points / (num_cuts + 1.0);
        Array<double> cuts = {0};
        cuts.ensure_slots(num_cuts);
        cuts.size = num_cuts;
        bool x_axis;
        double* coords = (double*)malloc(sizeof(double) * num_points);
        if (max.x - min.x > max.y - min.y) {
            double* x = coords;
            double* px = x;
            Vec2* pt = subj->point_array.items;
            for (int64_t j = 0; j < num_points; j++) (*px++) = (pt++)->x;
            std::sort(x, x + num_points);
            x_axis = true;
            px = cuts.items;
            for (int64_t j = 0; j < num_cuts; j++) (*px++) = x[(int64_t)((j + 1.0) * frac + 0.5)];
        } else {
            double* y = coords;
            double* py = y;
            Vec2* pt = subj->point_array.items;
            for (int64_t j = 0; j < num_points; j++) (*py++) = (pt++)->y;
            std::sort(y, y + num_points);
            x_axis = false;
            py = cuts.items;
            for (int64_t j = 0; j < num_cuts; j++) (*py++) = y[(int64_t)((j + 1.0) * frac + 0.5)];
        }
        free(coords);

        Array<Polygon*>* chopped = slice(*subj, cuts, x_axis, scaling);
        cuts.clear();

        subj->point_array.clear();
        result.remove_unordered(i);
        free(subj);

        int64_t total = 0;
        for (int64_t j = 0; j <= num_cuts; j++) total += chopped[j].size;
        result.ensure_slots(total);

        for (int64_t j = 0; j <= num_cuts; j++) {
            result.extend(chopped[j]);
            chopped[j].clear();
        }

        free(chopped);
    }

    for (int64_t i = 0; i < result.size; i++) {
        poly = result[i];
        poly->layer = layer;
        poly->datatype = datatype;
        poly->properties = properties_copy(properties);
    }

    return result;
}

void Polygon::to_gds(FILE* out, double scaling) const {
    if (point_array.size < 3) return;
    uint16_t buffer_start[] = {
        4, 0x0800, 6, 0x0D02, (uint16_t)layer, 6, 0x0E02, (uint16_t)datatype};
    swap16(buffer_start, COUNT(buffer_start));
    fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);

    int64_t total = point_array.size + 1;
    Array<int32_t> coords = {0};
    coords.ensure_slots(2 * total);
    coords.size = 2 * total;
    int32_t* c = coords.items;
    Vec2* p = point_array.items;
    for (int64_t j = point_array.size; j > 0; j--) {
        *c++ = (int32_t)lround(p->x * scaling);
        *c++ = (int32_t)lround(p->y * scaling);
        p++;
    }
    *c++ = coords[0];
    *c++ = coords[1];
    swap32((uint32_t*)coords.items, coords.size);

    if (total > 8190)
        fputs(
            "[GDSTK] Polygons with more than 8190 are not supported by the official GDSII specification.  This GDSII file might not be compatible with all readers.\n", stderr);

    int64_t i0 = 0;
    while (i0 < total) {
        int64_t i1 = total < i0 + 8190 ? total : i0 + 8190;
        uint16_t buffer_pts[] = {(uint16_t)(4 + 8 * (i1 - i0)), 0x1003};
        swap16(buffer_pts, COUNT(buffer_pts));
        fwrite(buffer_pts, sizeof(uint16_t), COUNT(buffer_pts), out);
        fwrite(coords.items + 2 * i0, sizeof(int32_t), 2 * (i1 - i0), out);
        i0 = i1;
    }
    coords.clear();

    properties_to_gds(properties, out);

    uint16_t buffer_end[] = {4, 0x1100};
    swap16(buffer_end, COUNT(buffer_end));
    fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
}

void Polygon::to_svg(FILE* out, double scaling) const {
    if (point_array.size < 3) return;
    fprintf(out, "<polygon class=\"l%hdd%hd\" points=\"", layer, datatype);
    Vec2* p = point_array.items;
    for (int64_t j = 0; j < point_array.size - 1; j++) {
        fprintf(out, "%lf,%lf ", p->x * scaling, p->y * scaling);
        p++;
    }
    fprintf(out, "%lf,%lf\"/>\n", p->x * scaling, p->y * scaling);
}

Polygon rectangle(const Vec2 corner1, const Vec2 corner2, int16_t layer, int16_t datatype) {
    Polygon result = {0};
    result.layer = layer;
    result.datatype = datatype;
    result.point_array.ensure_slots(4);
    result.point_array.size = 4;
    result.point_array[0] = corner1;
    result.point_array[1] = Vec2{corner2.x, corner1.y};
    result.point_array[2] = corner2;
    result.point_array[3] = Vec2{corner1.x, corner2.y};
    return result;
};

Polygon cross(const Vec2 center, double full_size, double arm_width, int16_t layer,
              int16_t datatype) {
    const double len = full_size / 2;
    const double half_width = arm_width / 2;
    Polygon result = {0};
    result.layer = layer;
    result.datatype = datatype;
    result.point_array.ensure_slots(12);
    result.point_array.size = 12;
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

Polygon regular_polygon(const Vec2 center, double side_length, int64_t sides, double rotation,
                        int16_t layer, int16_t datatype) {
    Polygon result = {0};
    result.layer = layer;
    result.datatype = datatype;
    result.point_array.ensure_slots(sides);
    result.point_array.size = sides;
    rotation += M_PI / sides - 0.5 * M_PI;
    const double radius = side_length / (2 * sin(M_PI / sides));
    Vec2* v = result.point_array.items;
    for (int64_t i = 0; i < sides; i++) {
        const double angle = rotation + i * 2 * M_PI / sides;
        *v++ = center + Vec2{radius * cos(angle), radius * sin(angle)};
    }
    return result;
};

Polygon ellipse(const Vec2 center, double radius_x, double radius_y, double inner_radius_x,
                double inner_radius_y, double initial_angle, double final_angle, double tolerance,
                int16_t layer, int16_t datatype) {
    Polygon result = {0};
    result.layer = layer;
    result.datatype = datatype;
    const double full_angle =
        (final_angle == initial_angle) ? 2 * M_PI : fabs(final_angle - initial_angle);
    if (inner_radius_x > 0 && inner_radius_y > 0) {
        int64_t num_points1 =
            1 + arc_num_points(full_angle, radius_x > radius_y ? radius_x : radius_y, tolerance);
        if (num_points1 < MIN_POINTS) num_points1 = MIN_POINTS;
        int64_t num_points2 =
            1 + arc_num_points(full_angle,
                               inner_radius_x > inner_radius_y ? inner_radius_x : inner_radius_y,
                               tolerance);
        if (num_points2 < MIN_POINTS) num_points2 = MIN_POINTS;

        result.point_array.ensure_slots(num_points1 + num_points2);
        result.point_array.size = num_points1 + num_points2;
        Vec2* v = result.point_array.items;
        if (full_angle == 2 * M_PI) {
            // Ring
            for (int64_t i = 0; i < num_points1; i++) {
                const double angle = i * 2 * M_PI / (num_points1 - 1);
                *v++ = center + Vec2{radius_x * cos(angle), radius_y * sin(angle)};
            }
            for (int64_t i = num_points2 - 1; i >= 0; i--) {
                const double angle = i * 2 * M_PI / (num_points2 - 1);
                *v++ = center + Vec2{inner_radius_x * cos(angle), inner_radius_y * sin(angle)};
            }
        } else {
            // Ring slice
            double initial_ell_angle =
                elliptical_angle_transform(initial_angle, radius_x, radius_y);
            double final_ell_angle = elliptical_angle_transform(final_angle, radius_x, radius_y);
            for (int64_t i = 0; i < num_points1; i++) {
                const double angle =
                    LERP(initial_ell_angle, final_ell_angle, (double)i / (num_points1 - 1.0));
                *v++ = center + Vec2{radius_x * cos(angle), radius_y * sin(angle)};
            }
            initial_ell_angle =
                elliptical_angle_transform(initial_angle, inner_radius_x, inner_radius_y);
            final_ell_angle =
                elliptical_angle_transform(final_angle, inner_radius_x, inner_radius_y);
            for (int64_t i = num_points2 - 1; i >= 0; i--) {
                const double angle =
                    LERP(initial_ell_angle, final_ell_angle, (double)i / (num_points2 - 1.0));
                *v++ = center + Vec2{inner_radius_x * cos(angle), inner_radius_y * sin(angle)};
            }
        }
    } else {
        int64_t num_points =
            1 + arc_num_points(full_angle, radius_x > radius_y ? radius_x : radius_y, tolerance);
        if (num_points < MIN_POINTS) num_points = MIN_POINTS;
        if (full_angle == 2 * M_PI) {
            // Full ellipse
            result.point_array.ensure_slots(num_points);
            result.point_array.size = num_points;
            Vec2* v = result.point_array.items;
            for (int64_t i = 0; i < num_points; i++) {
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
            result.point_array.size = num_points + 1;
            Vec2* v = result.point_array.items;
            *v++ = center;
            for (int64_t i = 0; i < num_points; i++) {
                const double angle =
                    LERP(initial_ell_angle, final_ell_angle, (double)i / (num_points - 1.0));
                *v++ = center + Vec2{radius_x * cos(angle), radius_y * sin(angle)};
            }
        }
    }
    return result;
}

Polygon racetrack(const Vec2 center, double straight_length, double radius, double inner_radius,
                  bool vertical, double tolerance, int16_t layer, int16_t datatype) {
    Polygon result = {0};
    result.layer = layer;
    result.datatype = datatype;

    double initial_angle;
    Vec2 direction = {0};
    if (vertical) {
        direction.y = straight_length / 2;
        initial_angle = 0;
    } else {
        direction.x = straight_length / 2;
        initial_angle = -M_PI / 2;
    }

    const Vec2 c1 = center + direction;
    const Vec2 c2 = center - direction;
    int64_t num_points = 1 + arc_num_points(M_PI, radius, tolerance);
    if (num_points < MIN_POINTS) num_points = MIN_POINTS;
    result.point_array.ensure_slots(2 * num_points);
    result.point_array.size = 2 * num_points;
    Vec2* v1 = result.point_array.items;
    Vec2* v2 = result.point_array.items + num_points;
    for (int64_t i = 0; i < num_points; i++) {
        const double angle = initial_angle + i * M_PI / num_points;
        const Vec2 rad_vec = {radius * cos(angle), radius * sin(angle)};
        *v1++ = c1 + rad_vec;
        *v2++ = c2 - rad_vec;
    }
    if (inner_radius > 0) {
        num_points = 1 + arc_num_points(M_PI, inner_radius, tolerance);
        if (num_points < MIN_POINTS) num_points = MIN_POINTS;
        result.point_array.ensure_slots(2 * num_points + 2);
        v2 = result.point_array.items + result.point_array.size;
        result.point_array.size += 2 * num_points + 2;
        *v2++ = result.point_array[0];
        *v2++ = c1 + Vec2{inner_radius * cos(initial_angle), inner_radius * sin(initial_angle)};
        v1 = v2 + num_points;
        for (int64_t i = num_points - 1; i >= 0; i--) {
            const double angle = initial_angle + i * M_PI / num_points;
            const Vec2 rad_vec = {inner_radius * cos(angle), inner_radius * sin(angle)};
            *v1++ = c1 + rad_vec;
            *v2++ = c2 - rad_vec;
        }
    }
    return result;
}

// NOTE: s must be 0-terminated
Array<Polygon*> text(const char* s, double size, const Vec2 position, bool vertical, int16_t layer,
                     int16_t datatype) {
    Array<Polygon*> result = {0};
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
                const int64_t index = *s - FIRST_CODEPOINT;
                if (index >= 0 && index < (int64_t)(COUNT(_first_poly))) {
                    int16_t p_idx = _first_poly[index];
                    for (int16_t i = _num_polys[index] - 1; i >= 0; i--, p_idx++) {
                        Polygon* p = (Polygon*)calloc(1, sizeof(Polygon));
                        p->layer = layer;
                        p->datatype = datatype;
                        p->point_array.ensure_slots(_num_coords[p_idx]);
                        int16_t c_idx = _first_coord[p_idx];
                        for (int16_t j = _num_coords[p_idx] - 1; j >= 0; j--, c_idx++)
                            p->point_array.append(cursor + size * _all_coords[c_idx]);
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
    return result;
}

}  // namespace gdstk
