/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <gdstk/allocator.hpp>
#include <gdstk/gdsii.hpp>
#include <gdstk/utils.hpp>
#include <gdstk/vec.hpp>

// Qhull
#include <libqhull_r/qhull_ra.h>

namespace gdstk {

FILE* error_logger = stderr;

void set_error_logger(FILE* log) { error_logger = log; }

void tag_to_gds(FILE* out, Tag tag, GdsiiRecord type_record) {
    uint32_t layer = get_layer(tag);
    if (layer > UINT16_MAX) {
        uint16_t buffer_start[] = {8, 0x0D03};
        big_endian_swap16(buffer_start, COUNT(buffer_start));
        big_endian_swap32(&layer, 1);
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
        fwrite(&layer, sizeof(uint32_t), 1, out);
    } else {
        uint16_t buffer_start[] = {6, 0x0D02, (uint16_t)layer};
        big_endian_swap16(buffer_start, COUNT(buffer_start));
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
    }

    uint32_t type = get_type(tag);
    if (type > UINT16_MAX) {
        uint16_t buffer_start[] = {8, 0x03};
        buffer_start[1] |= (uint16_t)type_record << 8;
        big_endian_swap16(buffer_start, COUNT(buffer_start));
        big_endian_swap32(&type, 1);
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
        fwrite(&type, sizeof(uint32_t), 1, out);
    } else {
        uint16_t buffer_start[] = {6, 0x02, (uint16_t)type};
        buffer_start[1] |= (uint16_t)type_record << 8;
        big_endian_swap16(buffer_start, COUNT(buffer_start));
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
    }
}

char* copy_string(const char* str, uint64_t* len) {
    uint64_t size = 1 + strlen(str);
    char* result = (char*)allocate(size);
    memcpy(result, str, size);
    if (len) *len = size;
    return result;
}

bool is_multiple_of_pi_over_2(double angle, int64_t& m) {
    if (angle == 0) {
        m = 0;
        return true;
    } else if (angle == 0.5 * M_PI) {
        m = 1;
        return true;
    } else if (angle == -0.5 * M_PI) {
        m = -1;
        return true;
    } else if (angle == M_PI) {
        m = 2;
        return true;
    } else if (angle == -M_PI) {
        m = -2;
        return true;
    } else if (angle == 1.5 * M_PI) {
        m = 3;
        return true;
    } else if (angle == -1.5 * M_PI) {
        m = -3;
        return true;
    } else if (angle == 2 * M_PI) {
        m = 4;
        return true;
    } else if (angle == -2 * M_PI) {
        m = -4;
        return true;
    }
    m = (int64_t)llround(angle / (M_PI / 2));
    if (fabs(m * (M_PI / 2) - angle) < 1e-16) return true;
    return false;
}

uint64_t arc_num_points(double angle, double radius, double tolerance) {
    assert(radius > 0);
    assert(tolerance > 0);
    double c = 1 - tolerance / radius;
    double a = c < -1 ? M_PI : acos(c);
    return (uint64_t)(0.5 + 0.5 * fabs(angle) / a);
}

static double modulo(double x, double y) {
    double m = fmod(x, y);
    return m < 0 ? m + y : m;
}

double elliptical_angle_transform(double angle, double radius_x, double radius_y) {
    if (angle == 0 || angle == M_PI || radius_x == radius_y) return angle;
    double frac = angle - (modulo(angle + M_PI, 2 * M_PI) - M_PI);
    double ell_angle = frac + atan2(radius_x * sin(angle), radius_y * cos(angle));
    return ell_angle;
}

double distance_to_line_sq(const Vec2 p, const Vec2 p1, const Vec2 p2) {
    const Vec2 v_line = p2 - p1;
    const Vec2 v_point = p - p1;
    const double c = v_point.cross(v_line);
    return c * c / v_line.length_sq();
}

double distance_to_line(const Vec2 p, const Vec2 p1, const Vec2 p2) {
    const Vec2 v_line = p2 - p1;
    const Vec2 v_point = p - p1;
    return fabs(v_point.cross(v_line)) / v_line.length();
}

void segments_intersection(const Vec2 p0, const Vec2 ut0, const Vec2 p1, const Vec2 ut1, double& u0,
                           double& u1) {
    const double den = ut0.cross(ut1);
    u0 = 0;
    u1 = 0;
    if (den >= GDSTK_PARALLEL_EPS || den <= -GDSTK_PARALLEL_EPS) {
        const Vec2 delta_p = p1 - p0;
        u0 = delta_p.cross(ut1) / den;
        u1 = delta_p.cross(ut0) / den;
    }
}

void scale_and_round_array(const Array<Vec2> points, double scaling,
                           Array<IntVec2>& scaled_points) {
    scaled_points.ensure_slots(points.count);
    scaled_points.count = points.count;
    int64_t* s = (int64_t*)scaled_points.items;
    double* p = (double*)points.items;
    for (uint64_t i = 2 * points.count; i > 0; i--) {
        *s++ = (int64_t)llround((*p++) * scaling);
    }
}

void big_endian_swap16(uint16_t* buffer, uint64_t n) {
    if (IS_BIG_ENDIAN) return;
    for (; n > 0; n--) {
        uint16_t b = *buffer;
        *buffer++ = (b << 8) | (b >> 8);
    }
}

void big_endian_swap32(uint32_t* buffer, uint64_t n) {
    if (IS_BIG_ENDIAN) return;
    for (; n > 0; n--) {
        uint32_t b = *buffer;
        *buffer++ = (b << 24) | ((b & 0x0000FF00) << 8) | ((b & 0x00FF0000) >> 8) | (b >> 24);
    }
}

void big_endian_swap64(uint64_t* buffer, uint64_t n) {
    if (IS_BIG_ENDIAN) return;
    for (; n > 0; n--) {
        uint64_t b = *buffer;
        *buffer++ = (b << 56) | ((b & 0x000000000000FF00) << 40) |
                    ((b & 0x0000000000FF0000) << 24) | ((b & 0x00000000FF000000) << 8) |
                    ((b & 0x000000FF00000000) >> 8) | ((b & 0x0000FF0000000000) >> 24) |
                    ((b & 0x00FF000000000000) >> 40) | (b >> 56);
    }
}

void little_endian_swap16(uint16_t* buffer, uint64_t n) {
    if (!IS_BIG_ENDIAN) return;
    for (; n > 0; n--) {
        uint16_t b = *buffer;
        *buffer++ = (b << 8) | (b >> 8);
    }
}

void little_endian_swap32(uint32_t* buffer, uint64_t n) {
    if (!IS_BIG_ENDIAN) return;
    for (; n > 0; n--) {
        uint32_t b = *buffer;
        *buffer++ = (b << 24) | ((b & 0x0000FF00) << 8) | ((b & 0x00FF0000) >> 8) | (b >> 24);
    }
}

void little_endian_swap64(uint64_t* buffer, uint64_t n) {
    if (!IS_BIG_ENDIAN) return;
    for (; n > 0; n--) {
        uint64_t b = *buffer;
        *buffer++ = (b << 56) | ((b & 0x000000000000FF00) << 40) |
                    ((b & 0x0000000000FF0000) << 24) | ((b & 0x00000000FF000000) << 8) |
                    ((b & 0x000000FF00000000) >> 8) | ((b & 0x0000FF0000000000) >> 24) |
                    ((b & 0x00FF000000000000) >> 40) | (b >> 56);
    }
}

uint32_t checksum32(uint32_t checksum, const uint8_t* bytes, uint64_t count) {
    uint64_t c = checksum;
    while (count-- > 0) c = (c + *bytes++) & 0xFFFFFFFF;
    return (uint32_t)c;
}

Vec2 eval_line(double t, const Vec2 p0, const Vec2 p1) { return LERP(p0, p1, t); }

Vec2 eval_bezier2(double t, const Vec2 p0, const Vec2 p1, const Vec2 p2) {
    const double t2 = t * t;
    const double r = 1 - t;
    const double r2 = r * r;
    Vec2 result = r2 * p0 + 2 * r * t * p1 + t2 * p2;
    return result;
}

Vec2 eval_bezier3(double t, const Vec2 p0, const Vec2 p1, const Vec2 p2, const Vec2 p3) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double r = 1 - t;
    const double r2 = r * r;
    const double r3 = r2 * r;
    Vec2 result = r3 * p0 + 3 * r2 * t * p1 + 3 * r * t2 * p2 + t3 * p3;
    return result;
}

Vec2 eval_bezier(double t, const Vec2* ctrl, uint64_t count) {
    Vec2 result;
    Vec2* p = (Vec2*)allocate(sizeof(Vec2) * count);
    memcpy(p, ctrl, sizeof(Vec2) * count);
    const double r = 1 - t;
    for (uint64_t j = count - 1; j > 0; j--)
        for (uint64_t i = 0; i < j; i++) p[i] = r * p[i] + t * p[i + 1];
    result = p[0];
    free_allocation(p);
    return result;
}

#ifndef NDEBUG

// NOTE: m[rows * cols] is assumed to be stored in row-major order
void print_matrix(double* m, uint64_t rows, uint64_t cols) {
    for (uint64_t r = 0; r < rows; r++) {
        printf("[");
        for (uint64_t c = 0; c < cols; c++) {
            if (c) printf("\t");
            printf("%.3g", *m++);
        }
        printf("]\n");
    }
}

#endif

// NOTE: m[rows * cols] must be stored in row-major order
// NOTE: pivots[rows] returns the pivoting order
uint64_t gauss_jordan_elimination(double* m, uint64_t* pivots, uint64_t rows, uint64_t cols) {
    assert(cols >= rows);
    uint64_t result = 0;

    uint64_t* p = pivots;
    for (uint64_t i = 0; i < rows; ++i) {
        *p++ = i;
    }

    for (uint64_t i = 0; i < rows; ++i) {
        // Select pivot: row with largest absolute value at column i
        double pivot_value = fabs(m[pivots[i] * cols + i]);
        uint64_t pivot_row = i;
        for (uint64_t j = i + 1; j < rows; ++j) {
            double candidate = fabs(m[pivots[j] * cols + i]);
            if (candidate > pivot_value) {
                pivot_value = candidate;
                pivot_row = j;
            }
        }
        if (pivot_value == 0) {
            result += 1;
            continue;
        }

        uint64_t row = pivots[pivot_row];
        pivots[pivot_row] = pivots[i];
        pivots[i] = row;

        // Scale row
        double* element = m + (row * cols + i);
        double factor = 1.0 / *element;
        for (uint64_t j = i; j < cols; ++j) {
            *element++ *= factor;
        }

        // Zero i-th column from other rows
        for (uint64_t r = 0; r < rows; ++r) {
            if (r == row) continue;
            element = m + row * cols;
            double* other_row = m + r * cols;
            factor = other_row[i];
            for (uint64_t j = 0; j < cols; ++j) {
                *other_row++ -= factor * *element++;
            }
        }
    }
    return result;
}

// NOTE: Matrix stored in column-major order!
void hobby_interpolation(uint64_t count, Vec2* points, double* angles, bool* angle_constraints,
                         Vec2* tension, double initial_curl, double final_curl, bool cycle) {
    const double A = sqrt(2.0);
    const double B = 1.0 / 16.0;
    const double C = 0.5 * (3.0 - sqrt(5.0));

    double* m = (double*)allocate(sizeof(double) * (2 * count * (2 * count + 1)));
    uint64_t* pivots = (uint64_t*)allocate(sizeof(uint64_t) * (2 * count));

    Vec2* pts = points;
    Vec2* tens = tension;
    double* ang = angles;
    bool* ang_c = angle_constraints;
    uint64_t points_size = count;

    uint64_t rotate = 0;
    if (cycle) {
        while (rotate < count && !angle_constraints[rotate]) rotate++;
        if (rotate == count) {
            // No angle constraints
            const uint64_t rows = 2 * count;
            const uint64_t cols = rows + 1;
            Vec2 v = points[3] - points[0];
            Vec2 v_prev = points[0] - points[3 * (count - 1)];
            double length_v = v.length();
            double delta_prev = v_prev.angle();
            memset(m, 0, sizeof(double) * rows * cols);
            for (uint64_t i = 0; i < count; i++) {
                const uint64_t i_1 = i == 0 ? count - 1 : i - 1;
                const uint64_t i1 = i == count - 1 ? 0 : i + 1;
                const uint64_t i2 = i < count - 2 ? i + 2 : (i == count - 2 ? 0 : 1);
                const uint64_t j = count + i;
                const uint64_t j_1 = count + i_1;
                const uint64_t j1 = count + i1;

                const double delta = v.angle();
                const Vec2 v_next = points[3 * i2] - points[3 * i1];
                const double length_v_next = v_next.length();
                double psi = delta - delta_prev;
                while (psi <= -M_PI) psi += 2 * M_PI;
                while (psi > M_PI) psi -= 2 * M_PI;

                m[i * cols + rows] = -psi;

                m[i * cols + i] = 1;
                m[i * cols + j_1] = 1;

                // A_i
                m[j * cols + i] = length_v_next * tension[i2].u * tension[i1].u * tension[i1].u;
                // B_{i+1}
                m[j * cols + i1] = -length_v * tension[i].v * tension[i1].v * tension[i1].v *
                                   (1 - 3 * tension[i2].u);
                // C_{i+1}
                m[j * cols + j] = length_v_next * tension[i2].u * tension[i1].u * tension[i1].u *
                                  (1 - 3 * tension[i].v);
                // D_{i+2}
                m[j * cols + j1] = -length_v * tension[i].v * tension[i1].v * tension[i1].v;

                v_prev = v;
                v = v_next;
                length_v = length_v_next;
                delta_prev = delta;
            }

            gauss_jordan_elimination(m, pivots, rows, cols);
            // NOTE: re-use the first row of m to temporarily hold the angles
            double* theta = m;
            double* phi = theta + count;
            for (uint64_t r = 0; r < count; r++) {
                theta[r] = m[pivots[r] * cols + rows];
                phi[r] = m[pivots[count + r] * cols + rows];
            }

            Vec2* cta = points + 1;
            Vec2* ctb = points + 2;
            v = points[3] - points[0];
            Vec2 w = cplx_from_angle(theta[0] + v.angle());
            for (uint64_t i = 0; i < count; i++, cta += 3, ctb += 3) {
                const uint64_t i1 = i == count - 1 ? 0 : i + 1;
                const uint64_t i2 = i < count - 2 ? i + 2 : (i == count - 2 ? 0 : 1);
                const double st = sin(theta[i]);
                const double ct = cos(theta[i]);
                const double sp = sin(phi[i]);
                const double cp = cos(phi[i]);
                const double alpha = A * (st - B * sp) * (sp - B * st) * (ct - cp);
                const Vec2 v_next = points[3 * i2] - points[3 * i1];
                const double _length_v = v.length();
                const double delta_next = v_next.angle();
                const Vec2 w_next = cplx_from_angle(theta[i1] + delta_next);
                *cta = points[3 * i] + w * _length_v * ((2 + alpha) / (1 + (1 - C) * ct + C * cp)) /
                                           (3 * tension[i].v);
                *ctb = points[3 * i1] - w_next * _length_v *
                                            ((2 - alpha) / (1 + (1 - C) * cp + C * ct)) /
                                            (3 * tension[i1].u);
                v = v_next;
                w = w_next;
            }
            free_allocation(m);
            free_allocation(pivots);
            return;
        }

        // Cycle, but with  angle constraint.
        // Rotate inputs and add append last point to cycle,
        // then use open curve solver.
        points_size++;

        pts = (Vec2*)allocate(sizeof(Vec2) * 3 * points_size);
        memcpy(pts, points + 3 * rotate, sizeof(Vec2) * 3 * (count - rotate));
        memcpy(pts + 3 * (count - rotate), points, sizeof(Vec2) * 3 * (rotate + 1));

        tens = (Vec2*)allocate(sizeof(Vec2) * points_size);
        memcpy(tens, tension + rotate, sizeof(Vec2) * (count - rotate));
        memcpy(tens + count - rotate, tension, sizeof(Vec2) * (rotate + 1));

        ang = (double*)allocate(sizeof(double) * points_size);
        memcpy(ang, angles + rotate, sizeof(double) * (count - rotate));
        memcpy(ang + count - rotate, angles, sizeof(double) * (rotate + 1));

        ang_c = (bool*)allocate(sizeof(bool) * points_size);
        memcpy(ang_c, angle_constraints + rotate, sizeof(bool) * (count - rotate));
        memcpy(ang_c + count - rotate, angle_constraints, sizeof(bool) * (rotate + 1));
    }

    {
        // Open curve solver
        const uint64_t n = points_size - 1;
        double* theta = (double*)allocate(sizeof(double) * n);
        double* phi = (double*)allocate(sizeof(double) * n);
        if (ang_c[0]) theta[0] = ang[0] - (pts[3] - pts[0]).angle();

        uint64_t i = 0;
        while (i < n) {
            uint64_t j = i + 1;
            while (j < n + 1 && !ang_c[j]) j++;
            if (j == n + 1)
                j--;
            else {
                phi[j - 1] = (pts[3 * j] - pts[3 * (j - 1)]).angle() - ang[j];
                if (j < n) theta[j] = ang[j] - (pts[3 * (j + 1)] - pts[3 * j]).angle();
            }

            // Solve curve pts[i] thru pts[j]
            const uint64_t range = j - i;
            const uint64_t rows = 2 * range;
            const uint64_t cols = rows + 1;
            memset(m, 0, sizeof(double) * rows * cols);

            Vec2 v_prev = pts[3 * (i + 1)] - pts[3 * i];
            double delta_prev = v_prev.angle();
            double length_v_prev = v_prev.length();
            for (uint64_t k = 0; k < range - 1; k++) {  // [0; range - 2]
                const uint64_t k1 = k + 1;              // [1; range - 1]
                const uint64_t i0 = i + k;              // [i; j - 2]
                const uint64_t i1 = i0 + 1;             // [i + 1; j - 1]
                const uint64_t i2 = i0 + 2;             // [i + 2; j]
                const uint64_t l = k + range;           // [range; 2 * range - 2]
                const uint64_t l1 = l + 1;              // [range + 1; range - 1]

                Vec2 v = pts[3 * i2] - pts[3 * i1];
                const double delta = v.angle();
                const double length_v = v.length();

                double psi = delta - delta_prev;
                while (psi <= -M_PI) psi += 2 * M_PI;
                while (psi > M_PI) psi -= 2 * M_PI;
                m[k1 * cols + rows] = -psi;

                m[k1 * cols + k1] = 1;
                m[k1 * cols + l] = 1;

                // A_k
                m[l * cols + k] = length_v * tens[i2].u * tens[i1].u * tens[i1].u;
                // B_{k+1}
                m[l * cols + k1] =
                    -length_v_prev * tens[i0].v * tens[i1].v * tens[i1].v * (1 - 3 * tens[i2].u);
                // C_{k+1}
                m[l * cols + l] =
                    length_v * tens[i2].u * tens[i1].u * tens[i1].u * (1 - 3 * tens[i0].v);
                // D_{k+2}
                m[l * cols + l1] = -length_v_prev * tens[i0].v * tens[i1].v * tens[i1].v;

                delta_prev = delta;
                length_v_prev = length_v;
            }
            if (ang_c[i]) {
                m[0 * cols + rows] = theta[i];
                // B_0
                m[0] = 1;
                // D_1
                // m[0 * cols + range] = 0;
            } else {
                const double to3 = tens[0].v * tens[0].v * tens[0].v;
                const double cti3 = initial_curl * tens[1].u * tens[1].u * tens[1].u;
                // B_0
                m[0] = to3 * (1 - 3 * tens[1].u) - cti3;
                // D_1
                m[0 * cols + range] = to3 - cti3 * (1 - 3 * tens[0].v);
            }
            if (ang_c[j]) {
                m[(rows - 1) * cols + rows] = phi[j - 1];
                // A_{range-1}
                // m[(rows - 1) * cols + (range - 1)] = 0;
                // C_range
                m[(rows - 1) * cols + (rows - 1)] = 1;
            } else {
                const double ti3 = tens[n].u * tens[n].u * tens[n].u;
                const double cto3 = final_curl * tens[n - 1].v * tens[n - 1].v * tens[n - 1].v;
                // A_{range-1}
                m[(rows - 1) * cols + (range - 1)] = ti3 - cto3 * (1 - 3 * tens[n].u);
                // C_range
                m[(rows - 1) * cols + (rows - 1)] = ti3 * (1 - 3 * tens[n - 1].v) - cto3;
            }
            if (range > 1 || !ang_c[i] || !ang_c[j]) {
                // printf("Solving range [%" PRIu64 ", %" PRIu64 "]\n\n", i, j);
                // print_matrix(m, rows, cols);

                gauss_jordan_elimination(m, pivots, rows, cols);
                for (uint64_t r = 0; r < range; r++) {
                    theta[i + r] = m[pivots[r] * cols + rows];
                    phi[i + r] = m[pivots[range + r] * cols + rows];
                }

                // printf("\n");
                // print_matrix(m, rows, cols);
                // printf("\n");
            }
            i = j;
        }

        Vec2 v = pts[3] - pts[0];
        Vec2 w = cplx_from_angle(theta[0] + v.angle());
        for (uint64_t ii = 0; ii < n; ii++) {
            const uint64_t i1 = ii + 1;
            const uint64_t i2 = ii == n - 1 ? 0 : ii + 2;
            const uint64_t ci = ii + rotate >= count ? ii + rotate - count : ii + rotate;
            const double st = sin(theta[ii]);
            const double ct = cos(theta[ii]);
            const double sp = sin(phi[ii]);
            const double cp = cos(phi[ii]);
            const double alpha = A * (st - B * sp) * (sp - B * st) * (ct - cp);
            const Vec2 v_next = pts[3 * i2] - pts[3 * i1];
            const double length_v = v.length();
            const Vec2 w_next = ii == n - 1 ? cplx_from_angle(v.angle() - phi[n - 1])
                                            : cplx_from_angle(theta[i1] + v_next.angle());
            points[3 * ci + 1] = pts[3 * ii] + w * length_v *
                                                   ((2 + alpha) / (1 + (1 - C) * ct + C * cp)) /
                                                   (3 * tens[ii].v);
            points[3 * ci + 2] = pts[3 * i1] - w_next * length_v *
                                                   ((2 - alpha) / (1 + (1 - C) * cp + C * ct)) /
                                                   (3 * tens[i1].u);
            v = v_next;
            w = w_next;
        }

        if (cycle) {
            free_allocation(pts);
            free_allocation(tens);
            free_allocation(ang);
            free_allocation(ang_c);
        }

        free_allocation(theta);
        free_allocation(phi);
    }
    free_allocation(m);
    free_allocation(pivots);
}

void convex_hull(const Array<Vec2> points, Array<Vec2>& result) {
    if (points.count < 4) {
        result.extend(points);
        return;
    } else if (points.count > qh_POINTSmax) {
        Array<Vec2> partial;
        partial.count = qh_POINTSmax;
        partial.items = points.items;
        Array<Vec2> temp = {};
        convex_hull(partial, temp);

        partial.count = points.count - qh_POINTSmax;
        partial.items = points.items + qh_POINTSmax;
        temp.extend(partial);
        convex_hull(temp, result);
        temp.clear();
        return;
    }

    qhT qh;
    QHULL_LIB_CHECK;
    qh_zero(&qh, error_logger);
    char command[256] = "qhull";
    int exitcode = qh_new_qhull(&qh, 2, (int)points.count, (double*)points.items, false, command,
                                NULL, error_logger);

    if (exitcode == 0) {
        result.ensure_slots(qh.num_facets);
        Vec2* point = result.items + result.count;
        result.count += qh.num_facets;

        vertexT* qh_vertex = NULL;
        facetT* qh_facet = qh_nextfacet2d(qh.facet_list, &qh_vertex);
        for (int64_t i = qh.num_facets; i > 0; i--, point++) {
            point->x = qh_vertex->point[0];
            point->y = qh_vertex->point[1];
            qh_facet = qh_nextfacet2d(qh_facet, &qh_vertex);
        }
    } else if (exitcode == qh_ERRsingular) {
        // QHull errors for singular input (collinear points in 2D)
        Vec2 min = {DBL_MAX, DBL_MAX};
        Vec2 max = {-DBL_MAX, -DBL_MAX};
        Vec2* p = points.items;
        for (uint64_t num = points.count; num > 0; num--, p++) {
            if (p->x < min.x) min.x = p->x;
            if (p->x > max.x) max.x = p->x;
            if (p->y < min.y) min.y = p->y;
            if (p->y > max.y) max.y = p->y;
        }
        if (min.x < max.x) {
            result.append(min);
            result.append(max);
        }
    } else {
        // The least we can do
        result.extend(points);
    }

#ifdef qh_NOmem
    qh_freeqhull(&qh, qh_ALL);
#else
    int curlong, totlong;
    qh_freeqhull(&qh, !qh_ALL);               /* free long memory  */
    qh_memfreeshort(&qh, &curlong, &totlong); /* free short memory and memory allocator */
    if (curlong || totlong) {
        if (error_logger) {
            fprintf(
                error_logger,
                "[GDSTK] Qhull internal warning: did not free %d bytes of long memory (%d pieces)\n",
                totlong, curlong);
        }
    }
#endif
}

char* double_print(double value, uint32_t precision, char* buffer, size_t buffer_size) {
    uint64_t len = snprintf(buffer, buffer_size, "%.*f", precision, value);
    if (precision) {
        while (buffer[--len] == '0');
        if (buffer[len] != '.') len++;
        buffer[len] = 0;
    }
    return buffer;
}

tm* get_now(tm& result) {
    time_t t = time(NULL);
#ifdef _WIN32
    localtime_s(&result, &t);
#else
    localtime_r(&t, &result);
#endif
    return &result;
}

// Kenneth Kelly's 22 colors of maximum contrast (minus B/W: "F2F3F4", "222222")
const char* colors[] = {"F3C300", "875692", "F38400", "A1CAF1", "BE0032", "C2B280", "848482",
                        "008856", "E68FAC", "0067A5", "F99379", "604E97", "F6A600", "B3446C",
                        "DCD300", "882D17", "8DB600", "654522", "E25822", "2B3D26"};

inline static const char* default_color(Tag tag) {
    return colors[(2 + get_layer(tag) + get_type(tag) * 13) % COUNT(colors)];
}

const char* default_svg_shape_style(Tag tag) {
    static char buffer[] = "stroke: #XXXXXX; fill: #XXXXXX; fill-opacity: 0.5;";
    const char* c = default_color(tag);
    memcpy(buffer + 9, c, 6);
    memcpy(buffer + 24, c, 6);
    return buffer;
}

const char* default_svg_label_style(Tag tag) {
    static char buffer[] = "stroke: none; fill: #XXXXXX;";
    const char* c = default_color(tag);
    memcpy(buffer + 21, c, 6);
    return buffer;
}

}  // namespace gdstk
