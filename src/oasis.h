/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __OASIS_H__
#define __OASIS_H__

#define _USE_MATH_DEFINES

#include <cstdint>
#include <cstdio>

#include "array.h"
#include "repetition.h"

namespace gdstk {

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
    NString = 12,  // printable characters (0x21 through 0x7e), size > 0
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

enum struct OasisDirection { E = 0, N = 1, W = 2, S = 3, NE = 4, NW = 5, SW = 6, SE = 7 };

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
    CELL_REFNAME = 13,
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

uint8_t* oasis_read_string(FILE* in, bool append_terminating_null, uint64_t& len);

uint64_t oasis_read_unsigned_integer(FILE* in);

int64_t oasis_read_integer(FILE* in);

inline int64_t oasis_read_1delta(FILE* in) { return oasis_read_integer(in); };

void oasis_read_2delta(FILE* in, int64_t& x, int64_t& y);

void oasis_read_3delta(FILE* in, int64_t& x, int64_t& y);

void oasis_read_gdelta(FILE* in, int64_t& x, int64_t& y);

double oasis_read_real_by_type(FILE* in, OasisDataType type);

inline double oasis_read_real(FILE* in) {
    OasisDataType type;
    if (fread(&type, 1, 1, in) < 1) return 0;
    return oasis_read_real_by_type(in, type);
}

// result must have at least 1 point in it, which will be used as reference for the relative deltas.
// polygon indicates whether this is supposed to be a polygon point list (in which case there will
// be an implicit extra delta for Manhattan types).
uint64_t oasis_read_point_list(FILE* in, double scaling, bool polygon, Array<Vec2>& result);

void oasis_read_repetition(FILE* in, double scaling, Repetition& repetition);

void oasis_write_unsigned_integer(FILE* out, uint64_t value);

void oasis_write_integer(FILE* out, int64_t value);

inline void oasis_write_1delta(FILE* out, int64_t value) { oasis_write_integer(out, value); };

void oasis_write_2delta(FILE* out, int64_t x, int64_t y);

void oasis_write_3delta(FILE* out, int64_t x, int64_t y);

void oasis_write_gdelta(FILE* out, int64_t x, int64_t y);

void oasis_write_real(FILE* out, double value);

void oasis_write_point_list(FILE* out, const Array<Vec2> points, double scaling, bool polygon);

}  // namespace gdstk

#endif

