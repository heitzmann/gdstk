/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_UTILS
#define GDSTK_HEADER_UTILS

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#define MIN_POINTS 4

#define PARALLEL_EPS 1e-8

#define COUNT(a) (sizeof(a) / sizeof(0 [a]))

// Linear interpoaltion
#define LERP(a, b, u) ((a) * (1 - (u)) + (b) * (u))

// Smooth interpolation (3rd order polynomial with zero derivatives at 0 and 1)
#define SERP(a, b, u) ((a) + ((b) - (a)) * (3 - 2 * (u)) * (u) * (u))

#define DEBUG_PRINT                               \
    printf("%s:%d: DEBUG\n", __FILE__, __LINE__); \
    fflush(stdout);

// From http://esr.ibiblio.org/?p=5095
#define IS_BIG_ENDIAN (*(uint16_t*)"\0\xFF" < 0x100)

#include <math.h>
#include <stdint.h>

#include "array.h"
#include "vec.h"

namespace gdstk {

// Argument between 0 and 1, plus user data
typedef double (*ParametricDouble)(double, void*);

// Argument between 0 and 1, plus user data
typedef Vec2 (*ParametricVec2)(double, void*);

// Arguments: first_point, first_direction, second_point, second_direction,
// user data
typedef Array<Vec2> (*EndFunction)(const Vec2, const Vec2, const Vec2, const Vec2, void*);

// Arguments: first_point, first_direction, second_point, second_direction,
// center, width, user data
typedef Array<Vec2> (*JoinFunction)(const Vec2, const Vec2, const Vec2, const Vec2, const Vec2,
                                    double, void*);

// Arguments: radius, initial_angle, final_angle, center, user data
typedef Array<Vec2> (*BendFunction)(double, double, double, const Vec2, void*);

// Returns new dynamically allocated memory
char* copy_string(const char* str, uint64_t& len);

// If true, m is set to the multiplicative factor, i.e., angle = 0.5 * M_PI * m
bool is_multiple_of_pi_over_2(double angle, int64_t& m);

// Number of points needed to approximate an arc within some tolerance
uint64_t arc_num_points(double angle, double radius, double tol);

double elliptical_angle_transform(double angle, double radius_x, double radius_y);

// Distance squared from p to the line defined by p1 and p2
double distance_to_line_sq(const Vec2 p, const Vec2 p1, const Vec2 p2);

// Distance from p to the line defined by p1 and p2
double distance_to_line(const Vec2 p, const Vec2 p1, const Vec2 p2);

// Finds the intersection between lines defined by point p0 and direction ut0
// (unit vector along the line) and by point p1 and direction ut1.  Scalars u0
// and u1 can be used to determine the intersection point:
// p = p0 + u0 * ut0 == p1 + u1 * ut1
void segments_intersection(const Vec2 p0, const Vec2 ut0, const Vec2 p1, const Vec2 ut1, double& u0,
                           double& u1);

void scale_and_round_array(const Array<Vec2> points, double scaling, Array<IntVec2>& scaled_points);

// Swap to big-endian (do nothing if the host is big-endian)
void big_endian_swap16(uint16_t* buffer, uint64_t n);
void big_endian_swap32(uint32_t* buffer, uint64_t n);
void big_endian_swap64(uint64_t* buffer, uint64_t n);

// Swap to little-endian (do nothing if the host is little-endian)
void little_endian_swap16(uint16_t* buffer, uint64_t n);
void little_endian_swap32(uint32_t* buffer, uint64_t n);
void little_endian_swap64(uint64_t* buffer, uint64_t n);

// Update the checksum32 of checksum with count values from bytes
uint32_t checksum32(uint32_t checksum, const uint8_t* bytes, uint64_t count);

Vec2 eval_line(double t, const Vec2 p0, const Vec2 p1);

// Quadratic Bézier defined by control points p0, p1 and p2 at 0 ≤ t ≤ 1
Vec2 eval_bezier2(double t, const Vec2 p0, const Vec2 p1, const Vec2 p2);

// Cubic Bézier defined by control points p0 through p3 at 0 ≤ t ≤ 1
Vec2 eval_bezier3(double t, const Vec2 p0, const Vec2 p1, const Vec2 p2, const Vec2 p3);

// Evaluate a Bézier curve defined by count control points at 0 ≤ t ≤ 1
Vec2 eval_bezier(double t, const Vec2* ctrl, uint64_t count);

// Calculates the control points for a smooth cubic Bézier interpolation
// following:
//
// John D. Hobby. “Smooth, easy to compute interpolating splines.” Discrete
// Comput. Geom., 1:123–140, 1986.
//
// Calculated control points ca and cb are stored in points, which must have
// the appropriate count and layout:
//
// points[3 * count] = {p[0], ca[0], cb[0],
//                      p[1], ca[1], cb[1],
//                      …,
//                      p[count - 1], ca[count - 1], cb[count - 1]};
//
// The last controls are only present if cycle == true.  Parameter angles can
// be used to constrain the angle at any interpolation point by setting the
// respective angle_constraints to true.  Defaults for tension (at each
// interpolation point) and curl should be 1.
void hobby_interpolation(uint64_t count, Vec2* points, double* angles, bool* angle_constraints,
                         Vec2* tension, double initial_curl, double final_curl, bool cycle);

// Stores the convex hull of points into result
void convex_hull(const Array<Vec2> points, Array<Vec2>& result);

}  // namespace gdstk

#endif
