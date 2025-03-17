/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_UTILS
#define GDSTK_HEADER_UTILS

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#define GDSTK_PRINT_BUFFER_COUNT 1024
#define GDSTK_DOUBLE_BUFFER_COUNT 1024

#define GDSTK_MIN_POINTS 4

#define GDSTK_PARALLEL_EPS 1e-8

#define GDSTK_MAP_GROWTH_FACTOR 2
#define GDSTK_INITIAL_MAP_CAPACITY 8
#define GDSTK_MAP_CAPACITY_THRESHOLD 5  // in tenths

#define COUNT(a) (sizeof(a) / sizeof(0 [a]))

// Linear interpoaltion
#define LERP(a, b, u) ((a) * (1 - (u)) + (b) * (u))

// Smooth interpolation (3rd order polynomial with zero derivatives at 0 and 1)
#define SERP(a, b, u) ((a) + ((b) - (a)) * (3 - 2 * (u)) * (u) * (u))

#ifdef NDEBUG
#define DEBUG_HERE ((void)0)
#define DEBUG_PRINT(...) ((void)0)
#else
#define DEBUG_HERE                                                         \
    do {                                                                   \
        fprintf(error_logger, "%s:%d:%s\n", __FILE__, __LINE__, __func__); \
        fflush(error_logger);                                              \
    } while (false)
#define DEBUG_PRINT(...)                    \
    do {                                    \
        fprintf(error_logger, __VA_ARGS__); \
        fflush(error_logger);               \
    } while (false)
#endif

// From http://esr.ibiblio.org/?p=5095
#define IS_BIG_ENDIAN (*(uint16_t*)"\0\xFF" < 0x100)

#ifdef _WIN32
#define FSEEK64 _fseeki64
#else
// Assuming sizeof(long) == 8
#define FSEEK64 fseek
#endif

#include <stdint.h>
#include <time.h>

#include "array.hpp"
#include "vec.hpp"

namespace gdstk {

// Error codes
enum struct ErrorCode {
    NoError = 0,
    // Warnings
    BooleanError,
    EmptyPath,
    IntersectionNotFound,
    MissingReference,
    UnsupportedRecord,
    UnofficialSpecification,
    InvalidRepetition,
    Overflow,
    // Errors
    ChecksumError,
    OutputFileOpenError,
    InputFileOpenError,
    InputFileError,
    FileError,
    InvalidFile,
    InsufficientMemory,
    ZlibError,
};

// Tag encapsulates layer and data (text) type.  The implementation details
// might change in the future.  The only guarantee is that a zeroed Tag
// indicates layer 0 and type 0.
typedef uint64_t Tag;
inline Tag make_tag(uint32_t layer, uint32_t type) {
    return ((uint64_t)type << 32) | (uint64_t)layer;
};
inline uint32_t get_layer(Tag tag) { return (uint32_t)tag; };
inline uint32_t get_type(Tag tag) { return (uint32_t)(tag >> 32); };
inline void set_layer(Tag& tag, uint32_t layer) { tag = make_tag(layer, get_type(tag)); };
inline void set_type(Tag& tag, uint32_t type) { tag = make_tag(get_layer(tag), type); };

enum struct GdsiiRecord : uint8_t;
void tag_to_gds(FILE* out, Tag tag, GdsiiRecord type_record);

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

// Returns new dynamically allocated memory.  If len if not NULL, it is set to
// the length of the string (including the null termination).
char* copy_string(const char* str, uint64_t* len);

// If true, m is set to the multiplicative factor, i.e., angle = 0.5 * M_PI * m
bool is_multiple_of_pi_over_2(double angle, int64_t& m);

// Number of points needed to approximate an arc within some tolerance
uint64_t arc_num_points(double angle, double radius, double tolerance);

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

// Return a global buffer with the representation of the value in fixed format
// with a maximal precision set.  This function is meant for internal use only.
char* double_print(double value, uint32_t precision, char* buffer, size_t buffer_size);

// Returns the default SVG style for a given tag.  The return value points to a
// statically allocated buffer that is overwritten in future calls to this
// function.
const char* default_svg_shape_style(Tag tag);
const char* default_svg_label_style(Tag tag);

// Thread-safe version of localtime.
tm* get_now(tm& result);

// FNV-1a hash function (64 bits)
#define HASH_FNV_PRIME 0x00000100000001b3
#define HASH_FNV_OFFSET 0xcbf29ce484222325
template <class T>
inline uint64_t hash(T key) {
    uint64_t result = HASH_FNV_OFFSET;
    uint8_t* byte = (uint8_t*)(&key);
    for (unsigned i = sizeof(T); i > 0; i--) {
        result ^= *byte++;
        result *= HASH_FNV_PRIME;
    }
    return result;
}

inline uint64_t hash(const char* key) {
    uint64_t result = HASH_FNV_OFFSET;
    for (const char* c = key; *c; c++) {
        result ^= (uint64_t)(*c);
        result *= HASH_FNV_PRIME;
    }
    return result;
}

extern FILE* error_logger;
void set_error_logger(FILE* log);

}  // namespace gdstk

#endif
