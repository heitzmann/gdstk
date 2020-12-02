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
    switch (byte) {
        case 0:
            return (double)oasis_read_uint(in);
        case 1:
            return -(double)oasis_read_uint(in);
        case 2:
            return 1.0 / (double)oasis_read_uint(in);
        case 3:
            return -1.0 / (double)oasis_read_uint(in);
        case 4: {
            double num = oasis_read_uint(in);
            double den = oasis_read_uint(in);
            return num / den;
        }
        case 5: {
            double num = oasis_read_uint(in);
            double den = oasis_read_uint(in);
            return -num / den;
        }
        case 6: {
            float value;
            fread(&value, sizeof(float), 1, in);
            little_endian_swap32((uint32_t*)&value, 1);
            return (double)value;
        }
        case 7: {
            double value;
            fread(&value, sizeof(double), 1, in);
            little_endian_swap64((uint64_t*)&value, 1);
            return value;
        }
    }
    return 0;
}

void oasis_write_real(FILE* out, double value) {
    if (trunc(value) == value && fabs(value) < UINT64_MAX) {
        // value is integer
        if (value >= 0) {
            fputc(0, out);
            oasis_write_uint(out, (uint64_t)value);
            return;
        } else {
            fputc(1, out);
            oasis_write_uint(out, (uint64_t)(-value));
            return;
        }
    }

    double inverse = 1.0 / value;
    if (trunc(inverse) == inverse && fabs(inverse) < UINT64_MAX) {
        // inverse is integer
        if (inverse >= 0) {
            fputc(2, out);
            oasis_write_uint(out, (uint64_t)inverse);
            return;
        } else {
            fputc(3, out);
            oasis_write_uint(out, (uint64_t)(-inverse));
            return;
        }
    }

    fputc(7, out);
    little_endian_swap64((uint64_t*)&value, 1);
    fwrite(&value, sizeof(double), 1, out);
}

}  // namespace gdstk
