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
    if (fread(&byte, 1, 1, in) < 1) {
        fputs("[GDSTK] Error reading file.\n", stderr);
        return 0;
    }

    uint64_t result = (uint64_t)(byte & 0x7F);

    uint8_t num_bits = 7;
    while (byte & 0x80) {
        if (fread(&byte, 1, 1, in) < 1) {
            fputs("[GDSTK] Error reading file.\n", stderr);
            return result;
        }
        if (num_bits == 63 && byte > 1) {
            fputs("[GDSTK] Integer above maximal limit found. Clipping.\n", stderr);
            return 0xFFFFFFFFFFFFFFFF;
        }
        result |= ((uint64_t)(byte & 0x7F)) << num_bits;
        num_bits += 7;
    }
    return result;
}

static uint8_t oasis_read_int_internal(FILE* in, uint8_t skip_bits, int64_t& result) {
    uint8_t byte;
    if (fread(&byte, 1, 1, in) < 1) {
        fputs("[GDSTK] Error reading file.\n", stderr);
        return 0;
    }
    result = ((uint64_t)(byte & 0x7F)) >> skip_bits;
    uint8_t bits = byte & ((1 << skip_bits) - 1);
    uint8_t num_bits = 7 - skip_bits;
    while (byte & 0x80) {
        if (fread(&byte, 1, 1, in) < 1) {
            fputs("[GDSTK] Error reading file.\n", stderr);
            return bits;
        }
        if (num_bits > 56 && (byte >> (63 - num_bits)) > 0) {
            fputs("[GDSTK] Integer above maximal limit found. Clipping.\n", stderr);
            result = 0x7FFFFFFFFFFFFFFF;
            return bits;
        }
        result |= ((uint64_t)(byte & 0x7F)) << num_bits;
        num_bits += 7;
    }
    return bits;
}

int64_t oasis_read_int(FILE* in) {
    int64_t value;
    if (oasis_read_int_internal(in, 1, value) > 0) return -value;
    return value;
}

void oasis_read_2delta(FILE* in, int64_t& x, int64_t& y) {
    int64_t value;
    switch ((OasisDirection)oasis_read_int_internal(in, 2, value)) {
        case OasisDirection::E:
            x = value;
            y = 0;
            break;
        case OasisDirection::N:
            x = 0;
            y = value;
            break;
        case OasisDirection::W:
            x = -value;
            y = 0;
            break;
        case OasisDirection::S:
            x = 0;
            y = -value;
            break;
        default:
            x = y = 0;
    }
}

void oasis_read_3delta(FILE* in, int64_t& x, int64_t& y) {
    int64_t value;
    switch ((OasisDirection)oasis_read_int_internal(in, 3, value)) {
        case OasisDirection::E:
            x = value;
            break;
        case OasisDirection::N:
            y = value;
            break;
        case OasisDirection::W:
            x = -value;
            break;
        case OasisDirection::S:
            y = -value;
            break;
        case OasisDirection::NE:
            x = value;
            y = value;
            break;
        case OasisDirection::NW:
            x = -value;
            y = value;
            break;
        case OasisDirection::SW:
            x = -value;
            y = -value;
            break;
        case OasisDirection::SE:
            x = value;
            y = -value;
    }
}

void oasis_read_gdelta(FILE* in, int64_t& x, int64_t& y) {
    uint8_t bits;
    if (fread(&bits, 1, 1, in) < 1) return;
    // TODO: How inefficient is this?
    fseek(in, -1, SEEK_CUR);

    if ((bits & 0x01) == 0) {
        int64_t value;
        switch ((OasisDirection)(oasis_read_int_internal(in, 4, value) >> 1)) {
            case OasisDirection::E:
                x = value;
                break;
            case OasisDirection::N:
                y = value;
                break;
            case OasisDirection::W:
                x = -value;
                break;
            case OasisDirection::S:
                y = -value;
                break;
            case OasisDirection::NE:
                x = value;
                y = value;
                break;
            case OasisDirection::NW:
                x = -value;
                y = value;
                break;
            case OasisDirection::SW:
                x = -value;
                y = -value;
                break;
            case OasisDirection::SE:
                x = value;
                y = -value;
        }
    } else {
        if ((oasis_read_int_internal(in, 2, x) & 0x02) > 0) x = -x;
        if ((oasis_read_int_internal(in, 1, y) & 0x01) > 0) y = -y;
    }
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

static void oasis_write_int_internal(FILE* out, int64_t value, uint8_t num_bits, uint8_t bits) {
    uint8_t bytes[10];
    uint8_t* b = bytes;
    *b = bits;
    *b |= (uint8_t)(value & ((1 << (7 - num_bits)) - 1)) << num_bits;
    value >>= 7 - num_bits;
    while (value > 0) {
        *b++ |= 0x80;
        *b = value & 0x7f;
        value >>= 7;
    }
    fwrite(bytes, 1, b - bytes + 1, out);
}

void oasis_write_int(FILE* out, int64_t value) {
    if (value < 0) {
        oasis_write_int_internal(out, -value, 1, 1);
    } else {
        oasis_write_int_internal(out, value, 1, 0);
    }
}

void oasis_write_2delta(FILE* out, int64_t x, int64_t y) {
    if (x == 0) {
        if (y < 0) {
            oasis_write_int_internal(out, -y, 2, (uint8_t)OasisDirection::S);
        } else {
            oasis_write_int_internal(out, y, 2, (uint8_t)OasisDirection::N);
        }
    } else if (y == 0) {
        if (x < 0) {
            oasis_write_int_internal(out, -x, 2, (uint8_t)OasisDirection::W);
        } else {
            oasis_write_int_internal(out, x, 2, (uint8_t)OasisDirection::E);
        }
    } else {
        fputs("[GDSTK] Error writing 2-delta.\n", stderr);
    }
}

void oasis_write_3delta(FILE* out, int64_t x, int64_t y) {
    if (x == 0) {
        if (y < 0) {
            oasis_write_int_internal(out, -y, 3, (uint8_t)OasisDirection::S);
        } else {
            oasis_write_int_internal(out, y, 3, (uint8_t)OasisDirection::N);
        }
    } else if (y == 0) {
        if (x < 0) {
            oasis_write_int_internal(out, -x, 3, (uint8_t)OasisDirection::W);
        } else {
            oasis_write_int_internal(out, x, 3, (uint8_t)OasisDirection::E);
        }
    } else if (x == y) {
        if (x < 0) {
            oasis_write_int_internal(out, -x, 3, (uint8_t)OasisDirection::SW);
        } else {
            oasis_write_int_internal(out, x, 3, (uint8_t)OasisDirection::NE);
        }
    } else if (x == -y) {
        if (x < 0) {
            oasis_write_int_internal(out, -x, 3, (uint8_t)OasisDirection::NW);
        } else {
            oasis_write_int_internal(out, x, 3, (uint8_t)OasisDirection::SE);
        }
    } else {
        fputs("[GDSTK] Error writing 3-delta.\n", stderr);
    }
}

void oasis_write_gdelta(FILE* out, int64_t x, int64_t y) {
    if (x == 0) {
        if (y < 0) {
            oasis_write_int_internal(out, -y, 4, (uint8_t)OasisDirection::S << 1);
        } else {
            oasis_write_int_internal(out, y, 4, (uint8_t)OasisDirection::N << 1);
        }
    } else if (y == 0) {
        if (x < 0) {
            oasis_write_int_internal(out, -x, 4, (uint8_t)OasisDirection::W << 1);
        } else {
            oasis_write_int_internal(out, x, 4, (uint8_t)OasisDirection::E << 1);
        }
    } else if (x == y) {
        if (x < 0) {
            oasis_write_int_internal(out, -x, 4, (uint8_t)OasisDirection::SW << 1);
        } else {
            oasis_write_int_internal(out, x, 4, (uint8_t)OasisDirection::NE << 1);
        }
    } else if (x == -y) {
        if (x < 0) {
            oasis_write_int_internal(out, -x, 4, (uint8_t)OasisDirection::NW << 1);
        } else {
            oasis_write_int_internal(out, x, 4, (uint8_t)OasisDirection::SE << 1);
        }
    } else {
        if (x < 0) {
            oasis_write_int_internal(out, -x, 2, 3);
        } else {
            oasis_write_int_internal(out, x, 2, 1);
        }
        if (y < 0) {
            oasis_write_int_internal(out, -y, 1, 1);
        } else {
            oasis_write_int_internal(out, y, 1, 0);
        }
    }
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
