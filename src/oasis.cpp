/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "oasis.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "utils.h"

namespace gdstk {

size_t oasis_read(void* buffer, size_t size, size_t count, OasisStream& in) {
    if (in.data) {
        uint64_t total = size * count;
        memcpy(buffer, in.cursor, size * count);
        in.cursor += total;
        if (in.cursor >= in.data + in.data_size) {
            free_allocation(in.data);
            in.data = NULL;
        }
        return total;
    }
    return fread(buffer, size, count, in.file);
}

static uint8_t oasis_peek(OasisStream& in) {
    uint8_t byte;
    if (in.data) {
        byte = *in.cursor;
    } else {
        fread(&byte, 1, 1, in.file);
        fseek(in.file, -1, SEEK_CUR);
    }
    return byte;
}

size_t oasis_write(void* buffer, size_t size, size_t count, OasisStream& out) {
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
    return fputc(c, out.file);
}

uint8_t* oasis_read_string(OasisStream& in, bool append_terminating_null, uint64_t& count) {
    uint8_t* bytes;
    count = oasis_read_unsigned_integer(in);
    if (append_terminating_null) {
        bytes = (uint8_t*)allocate(count + 1);
    } else {
        bytes = (uint8_t*)allocate(count);
    }
    if (oasis_read(bytes, 1, count, in) < count) {
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

uint64_t oasis_read_unsigned_integer(OasisStream& in) {
    uint8_t byte;
    if (oasis_read(&byte, 1, 1, in) < 1) {
        fputs("[GDSTK] Error reading file.\n", stderr);
        return 0;
    }

    uint64_t result = (uint64_t)(byte & 0x7F);

    uint8_t num_bits = 7;
    while (byte & 0x80) {
        if (oasis_read(&byte, 1, 1, in) < 1) {
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

static uint8_t oasis_read_int_internal(OasisStream& in, uint8_t skip_bits, int64_t& result) {
    uint8_t byte;
    if (oasis_read(&byte, 1, 1, in) < 1) {
        fputs("[GDSTK] Error reading file.\n", stderr);
        return 0;
    }
    result = ((uint64_t)(byte & 0x7F)) >> skip_bits;
    uint8_t bits = byte & ((1 << skip_bits) - 1);
    uint8_t num_bits = 7 - skip_bits;
    while (byte & 0x80) {
        if (oasis_read(&byte, 1, 1, in) < 1) {
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
            double num = oasis_read_unsigned_integer(in);
            double den = oasis_read_unsigned_integer(in);
            return num / den;
        }
        case OasisDataType::RealNegativeRatio: {
            double num = oasis_read_unsigned_integer(in);
            double den = oasis_read_unsigned_integer(in);
            return -num / den;
        }
        case OasisDataType::RealFloat: {
            float value;
            oasis_read(&value, sizeof(float), 1, in);
            little_endian_swap32((uint32_t*)&value, 1);
            return (double)value;
        }
        case OasisDataType::RealDouble: {
            double value;
            oasis_read(&value, sizeof(double), 1, in);
            little_endian_swap64((uint64_t*)&value, 1);
            return value;
        }
        default:
            fputs("[GDSTK] Unable to determine real value.\n", stderr);
    }
    return 0;
}

uint64_t oasis_read_point_list(OasisStream& in, double scaling, bool polygon, Array<Vec2>& result) {
    uint8_t byte;
    if (oasis_read(&byte, 1, 1, in) < 1) return 0;
    uint64_t num = oasis_read_unsigned_integer(in);
    switch ((OasisPointList)byte) {
        case OasisPointList::ManhattanHorizontalFirst: {
            result.ensure_slots(polygon ? num + 1 : num);
            Vec2* cur = result.items + result.count;
            Vec2* ref = cur - 1;
            double initial_x = ref->x;
            for (uint64_t i = num; i > 0; i--) {
                cur->x = ref->x + oasis_read_1delta(in) * scaling;
                cur->y = ref->y;
                cur++;
                ref++;
                if (--i > 0) {
                    cur->x = ref->x;
                    cur->y = ref->y + oasis_read_1delta(in) * scaling;
                    cur++;
                    ref++;
                }
            }
            if (polygon) {
                cur->x = initial_x;
                cur->y = ref->y;
                num++;
            }
            result.count += num;
        } break;
        case OasisPointList::ManhattanVerticalFirst: {
            result.ensure_slots(polygon ? num + 1 : num);
            Vec2* cur = result.items + result.count;
            Vec2* ref = cur - 1;
            double initial_y = ref->y;
            for (uint64_t i = num; i > 0; i--) {
                cur->x = ref->x;
                cur->y = ref->y + oasis_read_1delta(in) * scaling;
                cur++;
                ref++;
                if (--i > 0) {
                    cur->x = ref->x + oasis_read_1delta(in) * scaling;
                    cur->y = ref->y;
                    cur++;
                    ref++;
                }
            }
            if (polygon) {
                cur->x = ref->x;
                cur->y = initial_y;
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
            fputs("[GDSTK] Point list type not supported.\n", stderr);
            return 0;
    }
    return num;
}

void oasis_read_repetition(OasisStream& in, double scaling, Repetition& repetition) {
    uint8_t type;
    if (oasis_read(&type, 1, 1, in) < 1) {
        fputs("[GDSTK] Error reading file.\n", stderr);
        return;
    }
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

void oasis_write_3delta(OasisStream& out, int64_t x, int64_t y) {
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
    if (trunc(value) == value && fabs(value) < UINT64_MAX) {
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
    if (trunc(inverse) == inverse && fabs(inverse) < UINT64_MAX) {
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

void oasis_write_point_list(OasisStream& out, const Array<Vec2> points, double scaling,
                            bool polygon) {
    // TODO: choose point list type to decrease file count
    if (points.count < 1) return;
    oasis_putc((uint8_t)OasisPointList::General, out);
    oasis_write_unsigned_integer(out, points.count - 1);
    Vec2* ref = points.items;
    Vec2* cur = ref + 1;
    for (uint64_t i = points.count - 1; i > 0; i--) {
        Vec2 v = *cur++ - *ref++;
        oasis_write_gdelta(out, llround(scaling * v.x), llround(scaling * v.y));
    }
}

void oasis_write_repetition(OasisStream& out, const Repetition repetition, double scaling) {
    switch (repetition.type) {
        case RepetitionType::Rectangular: {
            if (repetition.columns > 1 && repetition.rows > 1) {
                if (repetition.spacing.x >= 0 && repetition.spacing.y >= 0) {
                    oasis_putc(1, out);
                    oasis_write_unsigned_integer(out, repetition.columns - 2);
                    oasis_write_unsigned_integer(out, repetition.rows - 2);
                    oasis_write_unsigned_integer(out, repetition.spacing.x * scaling);
                    oasis_write_unsigned_integer(out, repetition.spacing.y * scaling);
                } else {
                    oasis_putc(8, out);
                    oasis_write_unsigned_integer(out, repetition.columns - 2);
                    oasis_write_unsigned_integer(out, repetition.rows - 2);
                    oasis_write_gdelta(out, repetition.spacing.x * scaling, 0);
                    oasis_write_gdelta(out, 0, repetition.spacing.y * scaling);
                }
            } else if (repetition.columns > 1) {
                if (repetition.spacing.x >= 0) {
                    oasis_putc(2, out);
                    oasis_write_unsigned_integer(out, repetition.columns - 2);
                    oasis_write_unsigned_integer(out, repetition.spacing.x * scaling);
                } else {
                    oasis_putc(9, out);
                    oasis_write_unsigned_integer(out, repetition.columns - 2);
                    oasis_write_gdelta(out, repetition.spacing.x * scaling, 0);
                }
            } else {
                if (repetition.spacing.y >= 0) {
                    oasis_putc(3, out);
                    oasis_write_unsigned_integer(out, repetition.rows - 2);
                    oasis_write_unsigned_integer(out, repetition.spacing.y * scaling);
                } else {
                    oasis_putc(9, out);
                    oasis_write_unsigned_integer(out, repetition.rows - 2);
                    oasis_write_gdelta(out, 0, repetition.spacing.y * scaling);
                }
            }
        } break;
        case RepetitionType::Regular: {
            if (repetition.columns > 1 && repetition.rows > 1) {
                oasis_putc(8, out);
                oasis_write_unsigned_integer(out, repetition.columns - 2);
                oasis_write_unsigned_integer(out, repetition.rows - 2);
                oasis_write_gdelta(out, repetition.v1.x * scaling, repetition.v1.y * scaling);
                oasis_write_gdelta(out, repetition.v2.x * scaling, repetition.v2.y * scaling);
            } else if (repetition.columns > 1) {
                oasis_putc(9, out);
                oasis_write_unsigned_integer(out, repetition.columns - 2);
                oasis_write_gdelta(out, repetition.v1.x * scaling, repetition.v1.y * scaling);
            } else {
                oasis_putc(9, out);
                oasis_write_unsigned_integer(out, repetition.rows - 2);
                oasis_write_gdelta(out, repetition.v2.x * scaling, repetition.v2.y * scaling);
            }
        } break;
        case RepetitionType::ExplicitX:
            if (repetition.coords.count > 0) {
                oasis_putc(4, out);
                oasis_write_unsigned_integer(out, repetition.coords.count - 1);
                double* items = (double*)allocate(sizeof(double) * repetition.coords.count);
                memcpy(items, repetition.coords.items, sizeof(double) * repetition.coords.count);
                std::sort(items, items + repetition.coords.count);
                double* c0 = items;
                double* c1 = c0 + 1;
                oasis_write_unsigned_integer(out, *c0 * scaling);
                for (uint64_t i = repetition.coords.count - 1; i > 0; --i) {
                    oasis_write_unsigned_integer(out, (*c1++ - *c0++) * scaling);
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
                std::sort(items, items + repetition.coords.count);
                double* c0 = items;
                double* c1 = c0 + 1;
                oasis_write_unsigned_integer(out, *c0 * scaling);
                for (uint64_t i = repetition.coords.count - 1; i > 0; --i) {
                    oasis_write_unsigned_integer(out, (*c1++ - *c0++) * scaling);
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
                oasis_write_gdelta(out, v0->x * scaling, v0->y * scaling);
                for (uint64_t i = repetition.coords.count - 1; i > 0; --i, ++v0, ++v1) {
                    oasis_write_gdelta(out, (v1->x - v0->x) * scaling, (v1->y - v0->y) * scaling);
                }
            }
            break;
        case RepetitionType::None:
            break;
    }
}

}  // namespace gdstk
