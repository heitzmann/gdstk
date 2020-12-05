/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "oasis.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "utils.h"

namespace gdstk {

const char* oasis_record_names[] = {"PAD",
                                    "START",
                                    "END",
                                    "CELLNAME_IMPLICIT",
                                    "CELLNAME",
                                    "TEXTSTRING_IMPLICIT",
                                    "TEXTSTRING",
                                    "PROPNAME_IMPLICIT",
                                    "PROPNAME",
                                    "PROPSTRING_IMPLICIT",
                                    "PROPSTRING",
                                    "LAYERNAME_DATA",
                                    "LAYERNAME_TEXT",
                                    "CELL_REFNAME",
                                    "CELL",
                                    "XYABSOLUTE",
                                    "XYRELATIVE",
                                    "PLACEMENT",
                                    "PLACEMENT_TRANSFORM",
                                    "TEXT",
                                    "RECTANGLE",
                                    "POLYGON",
                                    "PATH",
                                    "TRAPEZOID_AB",
                                    "TRAPEZOID_A",
                                    "TRAPEZOID_B",
                                    "CTRAPEZOID",
                                    "CIRCLE",
                                    "PROPERTY",
                                    "LAST_PROPERTY",
                                    "XNAME_IMPLICIT",
                                    "XNAME",
                                    "XELEMENT",
                                    "XGEOMETRY",
                                    "CBLOCK"};

uint64_t oasis_read_uint(FILE* in) {
    uint8_t byte;
    if (fread(&byte, 1, 1, in) < 1) return 0;

    uint64_t result = (uint64_t)(byte & 0x7f);

    uint16_t num_bits = 7;
    while (byte & 0x80) {
        if (fread(&byte, 1, 1, in) < 1) return result;
        if (num_bits == 63 && byte > 1) {
            fputs("[GDSTK] Integer above maximal limit found. Clipping.\n", stderr);
            return 0xFFFFFFFFFFFFFFFF;
        }
        result |= ((uint64_t)(byte & 0x7f)) << num_bits;
        num_bits += 7;
    }

    return result;
}

void oasis_write_uint(FILE* out, uint64_t value) {
    uint8_t bytes[10] = {(uint8_t)(value & 0x7f)};
    uint8_t* b = bytes;
    value >>= 7;

    while (value > 0) {
        *b++ |= 0x80;
        *b = value & 0x7f;
        value >>= 7;
    }
    fwrite(bytes, 1, b - bytes + 1, out);
}

int64_t oasis_read_int(FILE* in) {
    uint8_t byte;
    if (fread(&byte, 1, 1, in) < 1) return 0;

    bool negative = ((byte & 0x01) != 0);
    uint64_t result = ((uint64_t)(byte & 0x7e) >> 1);

    uint16_t num_bits = 6;
    while (byte & 0x80) {
        if (fread(&byte, 1, 1, in) < 1) return result;
        if (num_bits == 62 && byte > 3) {
            fputs("[GDSTK] Integer above maximal limit found. Clipping.\n", stderr);
            return 0xFFFFFFFFFFFFFFFF;
        }
        result |= ((uint64_t)(byte & 0x7f)) << num_bits;
        num_bits += 7;
    }

    if (negative) return -result;
    return result;
}

void oasis_write_int(FILE* out, int64_t value) {
    uint8_t bytes[10];
    uint8_t* b = bytes;

    if (value < 0) {
        value = -value;
        *b = 1;
    } else {
        *b = 0;
    }
    *b |= (uint8_t)(value & 0x3f) << 1;
    value >>= 6;

    while (value > 0) {
        *b++ |= 0x80;
        *b = value & 0x7f;
        value >>= 7;
    }
    fwrite(bytes, 1, b - bytes + 1, out);
}

double oasis_read_real(FILE* in) {
    uint8_t byte;
    if (fread(&byte, 1, 1, in) < 1) return 0;
    switch ((OasisDataType)byte) {
        case OasisDataType::RealPositiveInteger:
            return (double)oasis_read_uint(in);
        case OasisDataType::RealNegativeInteger:
            return -(double)oasis_read_uint(in);
        case OasisDataType::RealPositiveReciprocal:
            return 1.0 / (double)oasis_read_uint(in);
        case OasisDataType::RealNegativeReciprocal:
            return -1.0 / (double)oasis_read_uint(in);
        case OasisDataType::RealPositiveRatio: {
            double num = oasis_read_uint(in);
            double den = oasis_read_uint(in);
            return num / den;
        }
        case OasisDataType::RealNegativeRatio: {
            double num = oasis_read_uint(in);
            double den = oasis_read_uint(in);
            return -num / den;
        }
        case OasisDataType::RealFloat: {
            float value;
            fread(&value, sizeof(float), 1, in);
            little_endian_swap32((uint32_t*)&value, 1);
            return (double)value;
        }
        case OasisDataType::RealDouble: {
            double value;
            fread(&value, sizeof(double), 1, in);
            little_endian_swap64((uint64_t*)&value, 1);
            return value;
        }
        default:
            fputs("[GDSTK] Unable to determine real value.\n", stderr);
    }
    return 0;
}

void oasis_write_real(FILE* out, double value) {
    if (trunc(value) == value && fabs(value) < UINT64_MAX) {
        // value is integer
        if (value >= 0) {
            fputc((uint8_t)OasisDataType::RealPositiveInteger, out);
            oasis_write_uint(out, (uint64_t)value);
            return;
        } else {
            fputc((uint8_t)OasisDataType::RealNegativeInteger, out);
            oasis_write_uint(out, (uint64_t)(-value));
            return;
        }
    }

    double inverse = 1.0 / value;
    if (trunc(inverse) == inverse && fabs(inverse) < UINT64_MAX) {
        // inverse is integer
        if (inverse >= 0) {
            fputc((uint8_t)OasisDataType::RealPositiveReciprocal, out);
            oasis_write_uint(out, (uint64_t)inverse);
            return;
        } else {
            fputc((uint8_t)OasisDataType::RealNegativeReciprocal, out);
            oasis_write_uint(out, (uint64_t)(-inverse));
            return;
        }
    }

    fputc((uint8_t)OasisDataType::RealDouble, out);
    little_endian_swap64((uint64_t*)&value, 1);
    fwrite(&value, sizeof(double), 1, out);
}

}  // namespace gdstk
