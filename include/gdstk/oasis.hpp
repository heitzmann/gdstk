/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_OASIS
#define GDSTK_HEADER_OASIS

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include "array.hpp"
#include "map.hpp"
#include "repetition.hpp"
#include "utils.hpp"

namespace gdstk {

// Configuration flags for Library.write_oas
#define OASIS_CONFIG_PROPERTY_MAX_COUNTS 0x0001
#define OASIS_CONFIG_PROPERTY_TOP_LEVEL 0x0002
#define OASIS_CONFIG_PROPERTY_BOUNDING_BOX 0x0004
#define OASIS_CONFIG_PROPERTY_CELL_OFFSET 0x0008

#define OASIS_CONFIG_DETECT_RECTANGLES 0x0010
#define OASIS_CONFIG_DETECT_TRAPEZOIDS 0x0020

#define OASIS_CONFIG_INCLUDE_CRC32 0x0040
#define OASIS_CONFIG_INCLUDE_CHECKSUM32 0x0080

#define OASIS_CONFIG_STANDARD_PROPERTIES                                  \
    (OASIS_CONFIG_PROPERTY_MAX_COUNTS | OASIS_CONFIG_PROPERTY_TOP_LEVEL | \
     OASIS_CONFIG_PROPERTY_BOUNDING_BOX | OASIS_CONFIG_PROPERTY_CELL_OFFSET)
#define OASIS_CONFIG_DETECT_ALL (OASIS_CONFIG_DETECT_RECTANGLES | OASIS_CONFIG_DETECT_TRAPEZOIDS)

// Types and functions in this header are not meant to be used by the end user
// of the library.  They are for internal use only.

extern const char s_max_int_size_property_name[];
extern const char s_max_uint_size_property_name[];
extern const char s_max_string_size_property_name[];
extern const char s_max_polygon_property_name[];
extern const char s_max_path_property_name[];
extern const char s_top_level_property_name[];
extern const char s_bounding_box_available_property_name[];
extern const char s_bounding_box_property_name[];
extern const char s_cell_offset_property_name[];
extern const char s_gds_property_name[];

enum struct OasisDataType : uint8_t {
    RealPositiveInteger = 0,
    RealNegativeInteger = 1,
    RealPositiveReciprocal = 2,
    RealNegativeReciprocal = 3,
    RealPositiveRatio = 4,
    RealNegativeRatio = 5,
    RealFloat = 6,
    RealDouble = 7,
    UnsignedInteger = 8,
    SignedInteger = 9,
    AString = 10,  // printable characters + space (0x20 through 0x7e)
    BString = 11,  // any bytes
    NString = 12,  // printable characters (0x21 through 0x7e), count > 0
    ReferenceA = 13,
    ReferenceB = 14,
    ReferenceN = 15
};

enum struct OasisRepetition : uint8_t {
    Previous = 0,
    Rectangular = 1,
    RectangularX = 2,
    RectangularY = 3,
    ExplicitX = 4,
    ExplicitXGrid = 5,
    ExplicitY = 6,
    ExplicitYGrid = 7,
    Regular = 8,
    Linear = 9,
    Explicit = 10,
    ExplicitGrid = 11
};

enum struct OasisPointList : uint8_t {
    ManhattanHorizontalFirst = 0,
    ManhattanVerticalFirst = 1,
    Manhattan = 2,
    Octangular = 3,
    General = 4,
    Relative = 5
};

enum struct OasisValidation : uint8_t { None = 0, Crc32 = 1, CheckSum = 2 };

enum struct OasisInterval : uint8_t {
    AllValues = 0,
    UpperBound = 1,
    LowerBound = 2,
    SingleValue = 3,
    Bounded = 4
};

enum struct OasisDirection : uint8_t { E = 0, N = 1, W = 2, S = 3, NE = 4, NW = 5, SW = 6, SE = 7 };

enum struct OasisRecord : uint8_t {
    PAD = 0,
    START = 1,
    END = 2,
    CELLNAME_IMPLICIT = 3,
    CELLNAME = 4,
    TEXTSTRING_IMPLICIT = 5,
    TEXTSTRING = 6,
    PROPNAME_IMPLICIT = 7,
    PROPNAME = 8,
    PROPSTRING_IMPLICIT = 9,
    PROPSTRING = 10,
    LAYERNAME_DATA = 11,
    LAYERNAME_TEXT = 12,
    CELL_REF_NUM = 13,
    CELL = 14,
    XYABSOLUTE = 15,
    XYRELATIVE = 16,
    PLACEMENT = 17,
    PLACEMENT_TRANSFORM = 18,
    TEXT = 19,
    RECTANGLE = 20,
    POLYGON = 21,
    PATH = 22,
    TRAPEZOID_AB = 23,
    TRAPEZOID_A = 24,
    TRAPEZOID_B = 25,
    CTRAPEZOID = 26,
    CIRCLE = 27,
    PROPERTY = 28,
    LAST_PROPERTY = 29,
    XNAME_IMPLICIT = 30,
    XNAME = 31,
    XELEMENT = 32,
    XGEOMETRY = 33,
    CBLOCK = 34
};

struct OasisStream {
    FILE* file;
    uint8_t* data;
    uint8_t* cursor;
    uint64_t data_size;
    uint32_t signature;
    bool crc32;
    bool checksum32;
    ErrorCode error_code;
};

struct OasisState {
    double scaling;
    double circle_tolerance;
    Map<uint64_t> property_name_map;
    Array<PropertyValue*> property_value_array;
    uint16_t config_flags;
};

ErrorCode oasis_read(void* buffer, size_t size, size_t count, OasisStream& in);

size_t oasis_write(const void* buffer, size_t size, size_t count, OasisStream& out);

int oasis_putc(int c, OasisStream& out);

uint8_t* oasis_read_string(OasisStream& in, bool append_terminating_null, uint64_t& len);

uint64_t oasis_read_unsigned_integer(OasisStream& in);

int64_t oasis_read_integer(OasisStream& in);

inline int64_t oasis_read_1delta(OasisStream& in) { return oasis_read_integer(in); };

void oasis_read_2delta(OasisStream& in, int64_t& x, int64_t& y);

void oasis_read_3delta(OasisStream& in, int64_t& x, int64_t& y);

void oasis_read_gdelta(OasisStream& in, int64_t& x, int64_t& y);

double oasis_read_real_by_type(OasisStream& in, OasisDataType type);

inline double oasis_read_real(OasisStream& in) {
    OasisDataType type;
    if (oasis_read(&type, 1, 1, in) != ErrorCode::NoError) return 0;
    return oasis_read_real_by_type(in, type);
}

// result must have at least 1 point in it, which will be used as reference for the relative deltas.
// polygon indicates whether this is supposed to be a polygon point list (in which case there will
// be an implicit extra delta for Manhattan types).
uint64_t oasis_read_point_list(OasisStream& in, double scaling, bool closed, Array<Vec2>& result);

void oasis_read_repetition(OasisStream& in, double scaling, Repetition& repetition);

void oasis_write_unsigned_integer(OasisStream& out, uint64_t value);

void oasis_write_integer(OasisStream& out, int64_t value);

inline void oasis_write_1delta(OasisStream& out, int64_t value) {
    oasis_write_integer(out, value);
};

void oasis_write_2delta(OasisStream& out, int64_t x, int64_t y);

void oasis_write_3delta(OasisStream& out, int64_t x, int64_t y);

void oasis_write_gdelta(OasisStream& out, int64_t x, int64_t y);

void oasis_write_real(OasisStream& out, double value);

// Uses first point as reference, does not output it.  Deltas between neighboring points
// will be calculated in-place: points[i] = points[i] - points[i - 1] (i = 1...count - 1)
void oasis_write_point_list(OasisStream& out, Array<IntVec2>& points, bool closed);
// Uses first point as reference, does not output it.
void oasis_write_point_list(OasisStream& out, const Array<Vec2> points, double scaling,
                            bool closed);

// This should only be called with repetition.get_count() > 1
void oasis_write_repetition(OasisStream& out, const Repetition repetition, double scaling);

}  // namespace gdstk

#endif
