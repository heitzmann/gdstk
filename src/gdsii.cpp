/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "gdsii.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "utils.h"

namespace gdstk {

uint64_t gdsii_real_from_double(double value) {
    if (value == 0) return 0;
    uint8_t u8_1 = 0;
    if (value < 0) {
        u8_1 = 0x80;
        value = -value;
    }
    const double fexp = 0.25 * log2(value);
    double exponent = ceil(fexp);
    if (exponent == fexp) exponent++;
    const uint64_t mantissa = (uint64_t)(value * pow(16, 14 - exponent));
    u8_1 += (uint8_t)(64 + exponent);
    const uint64_t result = ((uint64_t)u8_1 << 56) | (mantissa & 0x00FFFFFFFFFFFFFF);
    return result;
}

double gdsii_real_to_double(uint64_t real) {
    const int64_t exponent = ((real & 0x7F00000000000000) >> 54) - 256;
    const double mantissa = ((double)(real & 0x00FFFFFFFFFFFFFF)) / 72057594037927936.0;
    const double result = mantissa * exp2((double)exponent);
    return (real & 0x8000000000000000) ? -result : result;
}

// Read record and make necessary swaps
uint32_t gdsii_read_record(FILE* in, uint8_t* buffer) {
    uint64_t read_length = fread(buffer, 1, 4, in);
    if (read_length < 4) {
        if (feof(in) != 0)
            fputs("[GDSTK] Unable to read input file. End of file reached unexpectedly.\n", stderr);
        else
            fprintf(stderr, "[GDSTK] Unable to read input file. Error number %d\n.", ferror(in));
        return 0;
    }
    big_endian_swap16((uint16_t*)buffer, 1);  // second word is interpreted byte-wise (no swaping);
    const uint32_t record_length = *((uint16_t*)buffer);
    if (record_length < 4) return 0;
    if (record_length == 4) return record_length;
    read_length = fread(buffer + 4, 1, record_length - 4, in);
    if (read_length < record_length - 4) {
        if (feof(in) != 0)
            fputs("[GDSTK] Unable to read input file. End of file reached unexpectedly.\n", stderr);
        else
            fprintf(stderr, "[GDSTK] Unable to read input file. Error number %d\n.", ferror(in));
        return 0;
    }
    return record_length;
}

}  // namespace gdstk
