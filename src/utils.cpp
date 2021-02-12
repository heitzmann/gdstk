/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "utils.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "allocator.h"
#include "vec.h"

extern "C" {
// Fortran is column-major!
extern void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b,
                   const int* ldb, int* info);
}

namespace gdstk {

char* copy_string(const char* str, uint64_t& len) {
    len = 1 + strlen(str);
    char* result = (char*)allocate(len);
    memcpy(result, str, len);
    return result;
}

double modulo(double x, double y) {
    double m = fmod(x, y);
    return m < 0 ? m + y : m;
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

uint64_t arc_num_points(double angle, double radius, double tol) {
    return (uint64_t)(0.5 + 0.5 * fabs(angle) / acos(1 - tol / radius));
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

// ut are unitary tangent vectors (in the segment direction)
void segments_intersection(const Vec2 p0, const Vec2 ut0, const Vec2 p1, const Vec2 ut1, double& u0,
                           double& u1) {
    const double den = ut0.cross(ut1);
    u0 = 0;
    u1 = 0;
    if (den >= PARALLEL_EPS || den <= -PARALLEL_EPS) {
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

// Calculated control points `a` and `b` are stored in `points`, which must
// have the appropriate count and layout:
// points[3 * count] = {p[0], ca[0], cb[0],
//                     p[1], ca[1], cb[1],
//                     ...,
//                     p[count - 1], ca[count - 1], cb[count - 1]};
// The last controls are only present if `cycle == true`.
void hobby_interpolation(uint64_t count, Vec2* points, double* angles, bool* angle_constraints,
                         Vec2* tension, double initial_curl, double final_curl, bool cycle) {
    const int nrhs = 1;
    const double A = sqrt(2.0);
    const double B = 1.0 / 16.0;
    const double C = 0.5 * (3.0 - sqrt(5.0));

    int info = 0;
    int* ipiv = (int*)allocate(sizeof(int) * 2 * count);
    double* a = (double*)allocate(sizeof(double) * (4 * count * count + 2 * count));
    double* b = a + 4 * count * count;

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
            const uint64_t dim = 2 * count;
            Vec2 v = points[3] - points[0];
            Vec2 v_prev = points[0] - points[3 * (count - 1)];
            double length_v = v.length();
            double delta_prev = v_prev.angle();
            memset(a, 0, sizeof(double) * dim * dim);
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

                b[i] = -psi;
                b[j] = 0;

                a[i + dim * i] = 1;
                a[i + dim * j_1] = 1;
                // A_i
                a[j + dim * i] = length_v_next * tension[i2].u * tension[i1].u * tension[i1].u;
                // B_{i+1}
                a[j + dim * i1] = -length_v * tension[i].v * tension[i1].v * tension[i1].v *
                                  (1 - 3 * tension[i2].u);
                // C_{i+1}
                a[j + dim * j] = length_v_next * tension[i2].u * tension[i1].u * tension[i1].u *
                                 (1 - 3 * tension[i].v);
                // D_{i+2}
                a[j + dim * j1] = -length_v * tension[i].v * tension[i1].v * tension[i1].v;

                v_prev = v;
                v = v_next;
                length_v = length_v_next;
                delta_prev = delta;
            }

            dgesv_((const int*)&dim, &nrhs, a, (const int*)&dim, ipiv, b, (const int*)&dim, &info);
            double* theta = b;
            double* phi = b + count;

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
            free_allocation(ipiv);
            free_allocation(a);
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
            const uint64_t dim = 2 * range;
            memset(a, 0, sizeof(double) * dim * dim);
            memset(b, 0, sizeof(double) * dim);

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
                b[k1] = -psi;

                a[k1 + dim * k1] = 1;
                a[k1 + dim * l] = 1;
                // A_k
                a[l + dim * k] = length_v * tens[i2].u * tens[i1].u * tens[i1].u;
                // printf("a %"PRIu64" %"PRIu64" %lg\n", l, k, a[l + dim * k]);
                // B_{k+1}
                a[l + dim * k1] =
                    -length_v_prev * tens[i0].v * tens[i1].v * tens[i1].v * (1 - 3 * tens[i2].u);
                // printf("a %"PRIu64" %"PRIu64" %lg\n", l, k1, a[l + dim * k1]);
                // C_{k+1}
                a[l + dim * l] =
                    length_v * tens[i2].u * tens[i1].u * tens[i1].u * (1 - 3 * tens[i0].v);
                // printf("a %"PRIu64" %"PRIu64" %lg\n", l, l, a[l + dim * l]);
                // D_{k+2}
                a[l + dim * l1] = -length_v_prev * tens[i0].v * tens[i1].v * tens[i1].v;
                // printf("a %"PRIu64" %"PRIu64" %lg\n", l, l1, a[l + dim * l1]);

                delta_prev = delta;
                length_v_prev = length_v;
            }
            if (ang_c[i]) {
                b[0] = theta[i];
                // B_0
                a[0] = 1;
                // D_1
                a[dim * range] = 0;
            } else {
                const double to3 = tens[0].v * tens[0].v * tens[0].v;
                const double cti3 = initial_curl * tens[1].u * tens[1].u * tens[1].u;
                // B_0
                a[0] = to3 * (1 - 3 * tens[1].u) - cti3;
                // D_1
                a[dim * range] = to3 - cti3 * (1 - 3 * tens[0].v);
            }
            if (ang_c[j]) {
                b[dim - 1] = phi[j - 1];
                // A_{range-1}
                a[dim - 1 + dim * (range - 1)] = 0;
                // C_range
                a[dim - 1 + dim * (dim - 1)] = 1;
            } else {
                const double ti3 = tens[n].u * tens[n].u * tens[n].u;
                const double cto3 = final_curl * tens[n - 1].v * tens[n - 1].v * tens[n - 1].v;
                // A_{range-1}
                a[dim - 1 + dim * (range - 1)] = ti3 - cto3 * (1 - 3 * tens[n].u);
                // C_range
                a[dim - 1 + dim * (dim - 1)] = ti3 * (1 - 3 * tens[n - 1].v) - cto3;
            }
            if (range > 1 || !ang_c[i] || !ang_c[j]) {
                // printf("Solving range [%"PRIu64", %"PRIu64"]\n\n", i, j);
                // for (int _l = 0; _l < dim; _l++) {
                //     printf("%s[", _l == (dim - 1) / 2 ? "A = " : "    ");
                //     for (int _c = 0; _c < dim; _c++) printf(" %lg ", a[_l + dim * _c]);
                //     printf("]\n");
                // }
                // printf("\nb' = [");
                // for (int _l = 0; _l < dim; _l++) printf(" %lg ", b[_l]);
                // printf("]\n");
                dgesv_((const int*)&dim, &nrhs, a, (const int*)&dim, ipiv, b, (const int*)&dim,
                       &info);
                // printf("\nx' = [");
                // for (int _l = 0; _l < dim; _l++) printf(" %lg ", b[_l]);
                // printf("]\n");
                memcpy(theta + i, b, sizeof(double) * range);
                memcpy(phi + i, b + range, sizeof(double) * range);
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
    free_allocation(ipiv);
    free_allocation(a);
}

}  // namespace gdstk
