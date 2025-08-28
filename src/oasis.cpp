/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <zlib.h>

#include <gdstk/oasis.hpp>
#include <gdstk/sort.hpp>
#include <gdstk/utils.hpp>

namespace gdstk {

ErrorCode oasis_read(void* buffer, size_t size, size_t count, OasisStream& in) {
    if (in.data) {
        uint64_t total = size * count;
        memcpy(buffer, in.cursor, total);
        in.cursor += total;
        if (in.cursor >= in.data + in.data_size) {
            if (in.cursor > in.data + in.data_size) {
                if (error_logger)
                    fputs("[GDSTK] Error reading compressed data in file.\n", error_logger);
                in.error_code = ErrorCode::InputFileError;
            }
            free_allocation(in.data);
            in.data = NULL;
        }
    } else if (fread(buffer, size, count, in.file) < count) {
        if (error_logger) fputs("[GDSTK] Error reading OASIS file.\n", error_logger);
        in.error_code = ErrorCode::InputFileError;
    }
    return in.error_code;
}

static uint8_t oasis_peek(OasisStream& in) {
    uint8_t byte;
    if (in.data) {
        byte = *in.cursor;
    } else {
        if (fread(&byte, 1, 1, in.file) < 1) {
            if (error_logger) fputs("[GDSTK] Error reading OASIS file.\n", error_logger);
            if (in.error_code == ErrorCode::NoError) in.error_code = ErrorCode::InputFileError;
        }
        FSEEK64(in.file, -1, SEEK_CUR);
    }
    return byte;
}

size_t oasis_write(const void* buffer, size_t size, size_t count, OasisStream& out) {
    if (out.cursor) {
        uint64_t total = size * count;
        uint64_t available = out.data + out.data_size - out.cursor;
        if (total > available) {
            uint64_t used = out.cursor - out.data;
            out.data_size += (total > out.data_size ? 2 * total : out.data_size);
            out.data = (uint8_t*)reallocate(out.data, out.data_size);
            out.cursor = out.data + used;
        }
        memcpy(out.cursor, buffer, total);
        out.cursor += total;
        return total;
    }
    if (out.crc32) {
        uint64_t remaining = size * count;
        uint8_t* b = (uint8_t*)buffer;
        while (remaining > UINT_MAX) {
            out.signature = crc32(out.signature, b, UINT_MAX);
            remaining -= UINT_MAX;
            b += UINT_MAX;
        }
        if (remaining > 0) {
            out.signature = crc32(out.signature, b, (unsigned int)remaining);
        }
    } else if (out.checksum32) {
        out.signature = checksum32(out.signature, (uint8_t*)buffer, size * count);
    }
    return fwrite(buffer, size, count, out.file);
}

int oasis_putc(int c, OasisStream& out) {
    if (out.cursor) {
        uint64_t available = out.data + out.data_size - out.cursor;
        if (available == 0) {
            uint64_t used = out.cursor - out.data;
            out.data_size *= 2;
            out.data = (uint8_t*)reallocate(out.data, out.data_size);
            out.cursor = out.data + used;
        }
        uint8_t c_cast = (uint8_t)c;
        *out.cursor++ = c_cast;
        return (int)c_cast;
    }
    if (out.crc32) {
        uint8_t c_cast = (uint8_t)c;
        out.signature = crc32(out.signature, &c_cast, 1);
    } else if (out.checksum32) {
        uint8_t c_cast = (uint8_t)c;
        out.signature = checksum32(out.signature, &c_cast, 1);
    }
    return putc(c, out.file);
}

uint64_t oasis_read_unsigned_integer(OasisStream& in) {
    uint8_t byte;
    if (oasis_read(&byte, 1, 1, in) != ErrorCode::NoError) return 0;

    uint64_t result = (uint64_t)(byte & 0x7F);
    uint8_t num_bits = 7;
    while (byte & 0x80) {
        if (oasis_read(&byte, 1, 1, in) != ErrorCode::NoError) return result;
        if (num_bits == 63 && byte > 1) {
            if (error_logger)
                fputs("[GDSTK] Integer above maximal limit found. Clipping.\n", error_logger);
            if (in.error_code == ErrorCode::NoError) in.error_code = ErrorCode::Overflow;
            return 0xFFFFFFFFFFFFFFFF;
        }
        result |= ((uint64_t)(byte & 0x7F)) << num_bits;
        num_bits += 7;
    }
    return result;
}

uint8_t* oasis_read_string(OasisStream& in, bool append_terminating_null, uint64_t& count) {
    uint8_t* bytes;
    count = oasis_read_unsigned_integer(in);
    if (append_terminating_null) {
        bytes = (uint8_t*)allocate(count + 1);
    } else if (count > 0) {
        bytes = (uint8_t*)allocate(count);
    } else {
        return NULL;
    }
    if (oasis_read(bytes, 1, count, in) != ErrorCode::NoError) {
        free_allocation(bytes);
        bytes = NULL;
        count = -1;
    }
    if (append_terminating_null) {
        bytes[count++] = 0;
    }

    // printf("String (%d): [", count);
    // for (uint64_t i = 0; i < count; i++)
    //     if (bytes[i] >= 0x20 && bytes[i] < 0x7f)
    //         putchar(bytes[i]);
    //     else
    //         printf("\\%02x", bytes[i]);
    // puts("]");

    return bytes;
}

static uint8_t oasis_read_int_internal(OasisStream& in, uint8_t skip_bits, int64_t& result) {
    uint8_t byte;
    if (oasis_read(&byte, 1, 1, in) != ErrorCode::NoError) return 0;

    result = ((uint64_t)(byte & 0x7F)) >> skip_bits;
    uint8_t bits = byte & ((1 << skip_bits) - 1);
    uint8_t num_bits = 7 - skip_bits;
    while (byte & 0x80) {
        if (oasis_read(&byte, 1, 1, in) != ErrorCode::NoError) return bits;
        if (num_bits > 56 && (byte >> (63 - num_bits)) > 0) {
            if (error_logger)
                fputs("[GDSTK] Integer above maximal limit found. Clipping.\n", error_logger);
            if (in.error_code == ErrorCode::NoError) in.error_code = ErrorCode::Overflow;
            result = 0x7FFFFFFFFFFFFFFF;
            return bits;
        }
        result |= ((uint64_t)(byte & 0x7F)) << num_bits;
        num_bits += 7;
    }
    return bits;
}

int64_t oasis_read_integer(OasisStream& in) {
    int64_t value;
    if (oasis_read_int_internal(in, 1, value) > 0) return -value;
    return value;
}

void oasis_read_2delta(OasisStream& in, int64_t& x, int64_t& y) {
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

void oasis_read_3delta(OasisStream& in, int64_t& x, int64_t& y) {
    int64_t value;
    switch ((OasisDirection)oasis_read_int_internal(in, 3, value)) {
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

void oasis_read_gdelta(OasisStream& in, int64_t& x, int64_t& y) {
    uint8_t bits = oasis_peek(in);
    if (in.error_code != ErrorCode::NoError) return;

    if ((bits & 0x01) == 0) {
        int64_t value;
        switch ((OasisDirection)(oasis_read_int_internal(in, 4, value) >> 1)) {
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

double oasis_read_real_by_type(OasisStream& in, OasisDataType type) {
    switch ((OasisDataType)type) {
        case OasisDataType::RealPositiveInteger:
            return (double)oasis_read_unsigned_integer(in);
        case OasisDataType::RealNegativeInteger:
            return -(double)oasis_read_unsigned_integer(in);
        case OasisDataType::RealPositiveReciprocal:
            return 1.0 / (double)oasis_read_unsigned_integer(in);
        case OasisDataType::RealNegativeReciprocal:
            return -1.0 / (double)oasis_read_unsigned_integer(in);
        case OasisDataType::RealPositiveRatio: {
            double num = (double)oasis_read_unsigned_integer(in);
            double den = (double)oasis_read_unsigned_integer(in);
            return num / den;
        }
        case OasisDataType::RealNegativeRatio: {
            double num = (double)oasis_read_unsigned_integer(in);
            double den = (double)oasis_read_unsigned_integer(in);
            return -num / den;
        }
        case OasisDataType::RealFloat: {
            float value;
            if (oasis_read(&value, sizeof(float), 1, in) != ErrorCode::NoError) return 0;
            little_endian_swap32((uint32_t*)&value, 1);
            return (double)value;
        }
        case OasisDataType::RealDouble: {
            double value;
            if (oasis_read(&value, sizeof(double), 1, in) != ErrorCode::NoError) return 0;
            little_endian_swap64((uint64_t*)&value, 1);
            return value;
        }
        default:
            if (error_logger) fputs("[GDSTK] Unable to determine real value.\n", error_logger);
            if (in.error_code == ErrorCode::NoError) in.error_code = ErrorCode::InvalidFile;
    }
    return 0;
}

uint64_t oasis_read_point_list(OasisStream& in, double scaling, bool closed, Array<Vec2>& result) {
    assert(result.count > 0);

    uint8_t byte;
    if (oasis_read(&byte, 1, 1, in) != ErrorCode::NoError) return 0;

    uint64_t num = oasis_read_unsigned_integer(in);
    if (in.error_code != ErrorCode::NoError) return 0;

    switch ((OasisPointList)byte) {
        case OasisPointList::ManhattanHorizontalFirst:
        case OasisPointList::ManhattanVerticalFirst: {
            result.ensure_slots(closed ? num + 1 : num);
            Vec2* cur = result.items + result.count;
            Vec2* ref = cur - 1;
            Vec2 initial = *ref;
            bool horizontal = (OasisPointList)byte == OasisPointList::ManhattanHorizontalFirst;
            for (uint64_t i = num; i > 0; i--) {
                if (horizontal) {
                    cur->x = ref->x + oasis_read_1delta(in) * scaling;
                    cur->y = ref->y;
                    horizontal = false;
                } else {
                    cur->x = ref->x;
                    cur->y = ref->y + oasis_read_1delta(in) * scaling;
                    horizontal = true;
                }
                cur++;
                ref++;
            }
            if (closed) {
                if (horizontal) {
                    cur->x = initial.x;
                    cur->y = ref->y;
                } else {
                    cur->x = ref->x;
                    cur->y = initial.y;
                }
                num++;
            }
            result.count += num;
        } break;
        case OasisPointList::Manhattan: {
            result.ensure_slots(num);
            Vec2* cur = result.items + result.count;
            Vec2* ref = cur - 1;
            for (uint64_t i = num; i > 0; i--) {
                int64_t x, y;
                oasis_read_2delta(in, x, y);
                *cur++ = Vec2{scaling * x, scaling * y} + *ref++;
            }
            result.count += num;
        } break;
        case OasisPointList::Octangular: {
            result.ensure_slots(num);
            Vec2* cur = result.items + result.count;
            Vec2* ref = cur - 1;
            for (uint64_t i = num; i > 0; i--) {
                int64_t x, y;
                oasis_read_3delta(in, x, y);
                *cur++ = Vec2{scaling * x, scaling * y} + *ref++;
            }
            result.count += num;
        } break;
        case OasisPointList::General: {
            result.ensure_slots(num);
            Vec2* cur = result.items + result.count;
            Vec2* ref = cur - 1;
            for (uint64_t i = num; i > 0; i--) {
                int64_t x, y;
                oasis_read_gdelta(in, x, y);
                *cur++ = Vec2{scaling * x, scaling * y} + *ref++;
            }
            result.count += num;
        } break;
        case OasisPointList::Relative: {
            Vec2 delta = {0, 0};
            result.ensure_slots(num);
            Vec2* cur = result.items + result.count;
            Vec2* ref = cur - 1;
            for (uint64_t i = num; i > 0; i--) {
                int64_t x, y;
                oasis_read_gdelta(in, x, y);
                delta.x += scaling * x;
                delta.y += scaling * y;
                *cur++ = delta + *ref++;
            }
            result.count += num;
        } break;
        default:
            if (error_logger) fputs("[GDSTK] Point list type not supported.\n", error_logger);
            if (in.error_code == ErrorCode::NoError) in.error_code = ErrorCode::InvalidFile;
            return 0;
    }
    return num;
}

void oasis_read_repetition(OasisStream& in, double scaling, Repetition& repetition) {
    uint8_t type;

    if (oasis_read(&type, 1, 1, in) != ErrorCode::NoError) return;

    if (type == 0) return;

    repetition.clear();

    switch (type) {
        case 1: {
            repetition.type = RepetitionType::Rectangular;
            repetition.columns = 2 + oasis_read_unsigned_integer(in);
            repetition.rows = 2 + oasis_read_unsigned_integer(in);
            repetition.spacing.x = scaling * oasis_read_unsigned_integer(in);
            repetition.spacing.y = scaling * oasis_read_unsigned_integer(in);
        } break;
        case 2: {
            repetition.type = RepetitionType::Rectangular;
            repetition.columns = 2 + oasis_read_unsigned_integer(in);
            repetition.rows = 1;
            repetition.spacing.x = scaling * oasis_read_unsigned_integer(in);
            repetition.spacing.y = 0;
        } break;
        case 3: {
            repetition.type = RepetitionType::Rectangular;
            repetition.columns = 1;
            repetition.rows = 2 + oasis_read_unsigned_integer(in);
            repetition.spacing.x = 0;
            repetition.spacing.y = scaling * oasis_read_unsigned_integer(in);
        } break;
        case 4:
        case 5: {
            repetition.type = RepetitionType::ExplicitX;
            uint64_t count = 1 + oasis_read_unsigned_integer(in);
            repetition.coords.ensure_slots(count);
            double grid_factor = scaling;
            if (type == 5) {
                grid_factor *= oasis_read_unsigned_integer(in);
            }
            for (double x = 0; count > 0; count--) {
                x += grid_factor * oasis_read_unsigned_integer(in);
                repetition.coords.append_unsafe(x);
            }
        } break;
        case 6:
        case 7: {
            repetition.type = RepetitionType::ExplicitY;
            uint64_t count = 1 + oasis_read_unsigned_integer(in);
            repetition.coords.ensure_slots(count);
            double grid_factor = scaling;
            if (type == 7) {
                grid_factor *= oasis_read_unsigned_integer(in);
            }
            for (double y = 0; count > 0; count--) {
                y += grid_factor * oasis_read_unsigned_integer(in);
                repetition.coords.append_unsafe(y);
            }
        } break;
        case 8: {
            repetition.type = RepetitionType::Regular;
            repetition.columns = 2 + oasis_read_unsigned_integer(in);
            repetition.rows = 2 + oasis_read_unsigned_integer(in);
            int64_t x, y;
            oasis_read_gdelta(in, x, y);
            repetition.v1.x = scaling * x;
            repetition.v1.y = scaling * y;
            oasis_read_gdelta(in, x, y);
            repetition.v2.x = scaling * x;
            repetition.v2.y = scaling * y;
        } break;
        case 9: {
            repetition.type = RepetitionType::Regular;
            repetition.columns = 2 + oasis_read_unsigned_integer(in);
            repetition.rows = 1;
            int64_t x, y;
            oasis_read_gdelta(in, x, y);
            repetition.v1.x = scaling * x;
            repetition.v1.y = scaling * y;
            repetition.v2.x = -repetition.v1.y;
            repetition.v2.y = repetition.v1.x;
        } break;
        case 10:
        case 11: {
            repetition.type = RepetitionType::Explicit;
            uint64_t count = 1 + oasis_read_unsigned_integer(in);
            repetition.offsets.ensure_slots(count);
            double grid_factor = scaling;
            if (type == 11) {
                grid_factor *= oasis_read_unsigned_integer(in);
            }
            for (Vec2 v = {0, 0}; count > 0; count--) {
                int64_t x, y;
                oasis_read_gdelta(in, x, y);
                v.x += grid_factor * x;
                v.y += grid_factor * y;
                repetition.offsets.append_unsafe(v);
            }
        } break;
    }
}

void oasis_write_unsigned_integer(OasisStream& out, uint64_t value) {
    uint8_t bytes[10] = {(uint8_t)(value & 0x7f)};
    uint8_t* b = bytes;
    value >>= 7;

    while (value > 0) {
        *b++ |= 0x80;
        *b = value & 0x7f;
        value >>= 7;
    }
    oasis_write(bytes, 1, b - bytes + 1, out);
}

static void oasis_write_int_internal(OasisStream& out, int64_t value, uint8_t num_bits,
                                     uint8_t bits) {
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
    oasis_write(bytes, 1, b - bytes + 1, out);
}

void oasis_write_integer(OasisStream& out, int64_t value) {
    if (value < 0) {
        oasis_write_int_internal(out, -value, 1, 1);
    } else {
        oasis_write_int_internal(out, value, 1, 0);
    }
}

void oasis_write_2delta(OasisStream& out, int64_t x, int64_t y) {
    assert(x == 0 || y == 0);
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
        if (error_logger) fputs("[GDSTK] Error writing 2-delta.\n", error_logger);
    }
}

void oasis_write_3delta(OasisStream& out, int64_t x, int64_t y) {
    assert(x == 0 || y == 0 || x == y || x == -y);
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
        if (error_logger) fputs("[GDSTK] Error writing 3-delta.\n", error_logger);
    }
}

void oasis_write_gdelta(OasisStream& out, int64_t x, int64_t y) {
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

void oasis_write_real(OasisStream& out, double value) {
    if (trunc(value) == value && fabs(value) < (double)UINT64_MAX) {
        // value is integer
        if (value >= 0) {
            oasis_putc((uint8_t)OasisDataType::RealPositiveInteger, out);
            oasis_write_unsigned_integer(out, (uint64_t)value);
            return;
        } else {
            oasis_putc((uint8_t)OasisDataType::RealNegativeInteger, out);
            oasis_write_unsigned_integer(out, (uint64_t)(-value));
            return;
        }
    }

    double inverse = 1.0 / value;
    if (trunc(inverse) == inverse && fabs(inverse) < (double)UINT64_MAX) {
        // inverse is integer
        if (inverse >= 0) {
            oasis_putc((uint8_t)OasisDataType::RealPositiveReciprocal, out);
            oasis_write_unsigned_integer(out, (uint64_t)inverse);
            return;
        } else {
            oasis_putc((uint8_t)OasisDataType::RealNegativeReciprocal, out);
            oasis_write_unsigned_integer(out, (uint64_t)(-inverse));
            return;
        }
    }

    oasis_putc((uint8_t)OasisDataType::RealDouble, out);
    little_endian_swap64((uint64_t*)&value, 1);
    oasis_write(&value, sizeof(double), 1, out);
}

void oasis_write_point_list(OasisStream& out, Array<IntVec2>& points, bool closed) {
    if (points.count < 1) return;
    IntVec2 last_delta = points[0] - points[points.count - 1];
    // We never use Relative to write point lists.  It serves only to indicate
    // the initial state and will be overwritten in the first iteration.
    OasisPointList list_type = OasisPointList::Relative;
    bool prev_delta_is_horizontal = false;
    IntVec2 previous = points[0];
    for (uint64_t i = 1; i < points.count; i++) {
        IntVec2 v = points[i];
        points[i] -= previous;
        previous = v;
        v = points[i];
        switch (list_type) {
            case OasisPointList::Relative:
                // This is the initial state
                if (v.y == 0) {
                    list_type = OasisPointList::ManhattanHorizontalFirst;
                    prev_delta_is_horizontal = true;
                } else if (v.x == 0) {
                    list_type = OasisPointList::ManhattanVerticalFirst;
                    prev_delta_is_horizontal = false;
                } else if (v.x == v.y || v.x == -v.y) {
                    list_type = OasisPointList::Octangular;
                } else {
                    list_type = OasisPointList::General;
                }
                break;
            case OasisPointList::ManhattanHorizontalFirst:
            case OasisPointList::ManhattanVerticalFirst:
                if (v.y == 0) {
                    if (prev_delta_is_horizontal) {
                        list_type = OasisPointList::Manhattan;
                    } else {
                        prev_delta_is_horizontal = true;
                    }
                } else if (v.x == 0) {
                    if (!prev_delta_is_horizontal) {
                        list_type = OasisPointList::Manhattan;
                    } else {
                        prev_delta_is_horizontal = false;
                    }
                } else if (v.x == v.y || v.x == -v.y) {
                    list_type = OasisPointList::Octangular;
                } else {
                    list_type = OasisPointList::General;
                }
                break;
            case OasisPointList::Manhattan:
                if (v.y != 0 && v.x != 0) {
                    if (v.x == v.y || v.x == -v.y) {
                        list_type = OasisPointList::Octangular;
                    } else {
                        list_type = OasisPointList::General;
                    }
                }
                break;
            case OasisPointList::Octangular:
                if (v.y != 0 && v.x != 0 && v.x != v.y && v.x != -v.y) {
                    list_type = OasisPointList::General;
                }
                break;
            case OasisPointList::General:
                break;
        }
    }

    if (closed) {
        switch (list_type) {
            case OasisPointList::ManhattanHorizontalFirst:
            case OasisPointList::ManhattanVerticalFirst:
                if (last_delta.y == 0) {
                    if (prev_delta_is_horizontal) {
                        list_type = OasisPointList::Manhattan;
                    }
                } else if (last_delta.x == 0) {
                    if (!prev_delta_is_horizontal) {
                        list_type = OasisPointList::Manhattan;
                    }
                } else if (last_delta.x == last_delta.y || last_delta.x == -last_delta.y) {
                    list_type = OasisPointList::Octangular;
                } else {
                    list_type = OasisPointList::General;
                }
                break;
            case OasisPointList::Manhattan:
                if (last_delta.y != 0 && last_delta.x != 0) {
                    if (last_delta.x == last_delta.y || last_delta.x == -last_delta.y) {
                        list_type = OasisPointList::Octangular;
                    } else {
                        list_type = OasisPointList::General;
                    }
                }
                break;
            case OasisPointList::Octangular:
                if (last_delta.y != 0 && last_delta.x != 0 && last_delta.x != last_delta.y &&
                    last_delta.x != -last_delta.y) {
                    list_type = OasisPointList::General;
                }
                break;
            default:
                break;
        }
    }

    uint64_t count = points.count - 1;
    if (list_type == OasisPointList::ManhattanHorizontalFirst ||
        list_type == OasisPointList::ManhattanVerticalFirst) {
        if (closed) {
            --count;
            if (count < 2 || count % 2 == 1) {
                // There is probably a duplicate vertex in the polygon,
                // otherwise the point count should be even.
                count = points.count - 1;
                list_type = OasisPointList::Manhattan;
            }
        }
    }

    switch (list_type) {
        case OasisPointList::ManhattanHorizontalFirst: {
            oasis_putc((uint8_t)OasisPointList::ManhattanHorizontalFirst, out);
            oasis_write_unsigned_integer(out, count);
            prev_delta_is_horizontal = false;
            IntVec2* delta = points.items + 1;
            for (uint64_t i = count; i > 0; i--) {
                oasis_write_1delta(out, prev_delta_is_horizontal ? delta->y : delta->x);
                prev_delta_is_horizontal = !prev_delta_is_horizontal;
                delta++;
            }
        } break;
        case OasisPointList::ManhattanVerticalFirst: {
            oasis_putc((uint8_t)OasisPointList::ManhattanVerticalFirst, out);
            oasis_write_unsigned_integer(out, count);
            prev_delta_is_horizontal = true;
            IntVec2* delta = points.items + 1;
            for (uint64_t i = count; i > 0; i--) {
                oasis_write_1delta(out, prev_delta_is_horizontal ? delta->y : delta->x);
                prev_delta_is_horizontal = !prev_delta_is_horizontal;
                delta++;
            }
        } break;
        case OasisPointList::Manhattan: {
            oasis_putc((uint8_t)OasisPointList::Manhattan, out);
            oasis_write_unsigned_integer(out, count);
            IntVec2* delta = points.items + 1;
            for (uint64_t i = count; i > 0; i--) {
                oasis_write_2delta(out, delta->x, delta->y);
                delta++;
            }
        } break;
        case OasisPointList::Octangular: {
            oasis_putc((uint8_t)OasisPointList::Octangular, out);
            oasis_write_unsigned_integer(out, count);
            IntVec2* delta = points.items + 1;
            for (uint64_t i = count; i > 0; i--) {
                oasis_write_3delta(out, delta->x, delta->y);
                delta++;
            }
        } break;
        default: {
            oasis_putc((uint8_t)OasisPointList::General, out);
            oasis_write_unsigned_integer(out, count);
            IntVec2* delta = points.items + 1;
            for (uint64_t i = count; i > 0; i--) {
                oasis_write_gdelta(out, delta->x, delta->y);
                delta++;
            }
        }
    }
}

void oasis_write_point_list(OasisStream& out, const Array<Vec2> points, double scaling,
                            bool closed) {
    Array<IntVec2> scaled_points = {};
    scale_and_round_array(points, scaling, scaled_points);
    oasis_write_point_list(out, scaled_points, closed);
    scaled_points.clear();
}

void oasis_write_repetition(OasisStream& out, const Repetition repetition, double scaling) {
    switch (repetition.type) {
        case RepetitionType::Rectangular: {
            if (repetition.columns > 1 && repetition.rows > 1) {
                if (repetition.spacing.x >= 0 && repetition.spacing.y >= 0) {
                    oasis_putc(1, out);
                    oasis_write_unsigned_integer(out, repetition.columns - 2);
                    oasis_write_unsigned_integer(out, repetition.rows - 2);
                    oasis_write_unsigned_integer(out,
                                                 (uint64_t)llround(repetition.spacing.x * scaling));
                    oasis_write_unsigned_integer(out,
                                                 (uint64_t)llround(repetition.spacing.y * scaling));
                } else {
                    oasis_putc(8, out);
                    oasis_write_unsigned_integer(out, repetition.columns - 2);
                    oasis_write_unsigned_integer(out, repetition.rows - 2);
                    oasis_write_gdelta(out, (int64_t)llround(repetition.spacing.x * scaling), 0);
                    oasis_write_gdelta(out, 0, (int64_t)llround(repetition.spacing.y * scaling));
                }
            } else if (repetition.columns > 1) {
                if (repetition.spacing.x >= 0) {
                    oasis_putc(2, out);
                    oasis_write_unsigned_integer(out, repetition.columns - 2);
                    oasis_write_unsigned_integer(out,
                                                 (uint64_t)llround(repetition.spacing.x * scaling));
                } else {
                    oasis_putc(9, out);
                    oasis_write_unsigned_integer(out, repetition.columns - 2);
                    oasis_write_gdelta(out, (int64_t)llround(repetition.spacing.x * scaling), 0);
                }
            } else {
                if (repetition.spacing.y >= 0) {
                    oasis_putc(3, out);
                    oasis_write_unsigned_integer(out, repetition.rows - 2);
                    oasis_write_unsigned_integer(out,
                                                 (uint64_t)llround(repetition.spacing.y * scaling));
                } else {
                    oasis_putc(9, out);
                    oasis_write_unsigned_integer(out, repetition.rows - 2);
                    oasis_write_gdelta(out, 0, (int64_t)llround(repetition.spacing.y * scaling));
                }
            }
        } break;
        case RepetitionType::Regular: {
            if (repetition.columns > 1 && repetition.rows > 1) {
                oasis_putc(8, out);
                oasis_write_unsigned_integer(out, repetition.columns - 2);
                oasis_write_unsigned_integer(out, repetition.rows - 2);
                oasis_write_gdelta(out, (int64_t)llround(repetition.v1.x * scaling),
                                   (int64_t)llround(repetition.v1.y * scaling));
                oasis_write_gdelta(out, (int64_t)llround(repetition.v2.x * scaling),
                                   (int64_t)llround(repetition.v2.y * scaling));
            } else if (repetition.columns > 1) {
                oasis_putc(9, out);
                oasis_write_unsigned_integer(out, repetition.columns - 2);
                oasis_write_gdelta(out, (int64_t)llround(repetition.v1.x * scaling),
                                   (int64_t)llround(repetition.v1.y * scaling));
            } else {
                oasis_putc(9, out);
                oasis_write_unsigned_integer(out, repetition.rows - 2);
                oasis_write_gdelta(out, (int64_t)llround(repetition.v2.x * scaling),
                                   (int64_t)llround(repetition.v2.y * scaling));
            }
        } break;
        case RepetitionType::ExplicitX:
            if (repetition.coords.count > 0) {
                oasis_putc(4, out);
                oasis_write_unsigned_integer(out, repetition.coords.count - 1);
                double* items = (double*)allocate(sizeof(double) * repetition.coords.count);
                memcpy(items, repetition.coords.items, sizeof(double) * repetition.coords.count);
                sort(items, repetition.coords.count);
                double* c0 = items;
                double* c1 = c0 + 1;
                oasis_write_unsigned_integer(out, (uint64_t)llround(*c0 * scaling));
                for (uint64_t i = repetition.coords.count - 1; i > 0; --i) {
                    oasis_write_unsigned_integer(out, (uint64_t)llround((*c1++ - *c0++) * scaling));
                }
                free_allocation(items);
            }
            break;
        case RepetitionType::ExplicitY:
            if (repetition.coords.count > 0) {
                oasis_putc(6, out);
                oasis_write_unsigned_integer(out, repetition.coords.count - 1);
                double* items = (double*)allocate(sizeof(double) * repetition.coords.count);
                memcpy(items, repetition.coords.items, sizeof(double) * repetition.coords.count);
                sort(items, repetition.coords.count);
                double* c0 = items;
                double* c1 = c0 + 1;
                oasis_write_unsigned_integer(out, (uint64_t)llround(*c0 * scaling));
                for (uint64_t i = repetition.coords.count - 1; i > 0; --i) {
                    oasis_write_unsigned_integer(out, (uint64_t)llround((*c1++ - *c0++) * scaling));
                }
                free_allocation(items);
            }
            break;
        case RepetitionType::Explicit:
            if (repetition.offsets.count > 0) {
                oasis_putc(10, out);
                oasis_write_unsigned_integer(out, repetition.offsets.count - 1);
                Vec2* v0 = repetition.offsets.items;
                Vec2* v1 = v0 + 1;
                oasis_write_gdelta(out, (int64_t)llround(v0->x * scaling),
                                   (int64_t)llround(v0->y * scaling));
                for (uint64_t i = repetition.offsets.count - 1; i > 0; --i, ++v0, ++v1) {
                    oasis_write_gdelta(out, (int64_t)llround((v1->x - v0->x) * scaling),
                                       (int64_t)llround((v1->y - v0->y) * scaling));
                }
            }
            break;
        case RepetitionType::None:
            break;
    }
}

}  // namespace gdstk
