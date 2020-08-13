/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __GDSWRITER_H__
#define __GDSWRITER_H__

#include <cstdio>

#include "cell.h"
#include "rawcell.h"

namespace gdstk {

struct GdsWriter {
    FILE* out;
    double unit;
    double precision;
    int64_t max_points;
    std::tm timestamp;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void write_gds(const char* name) const {
        int64_t len = strlen(name);
        if (len % 2) len++;
        uint16_t buffer_start[] = {6,
                                   0x0002,
                                   0x0258,
                                   28,
                                   0x0102,
                                   (uint16_t)(timestamp.tm_year + 1900),
                                   (uint16_t)(timestamp.tm_mon + 1),
                                   (uint16_t)timestamp.tm_mday,
                                   (uint16_t)timestamp.tm_hour,
                                   (uint16_t)timestamp.tm_min,
                                   (uint16_t)timestamp.tm_sec,
                                   (uint16_t)(timestamp.tm_year + 1900),
                                   (uint16_t)(timestamp.tm_mon + 1),
                                   (uint16_t)timestamp.tm_mday,
                                   (uint16_t)timestamp.tm_hour,
                                   (uint16_t)timestamp.tm_min,
                                   (uint16_t)timestamp.tm_sec,
                                   (uint16_t)(4 + len),
                                   0x0206};
        swap16(buffer_start, COUNT(buffer_start));
        fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
        fwrite(name, sizeof(char), len, out);

        uint16_t buffer_units[] = {20, 0x0305};
        swap16(buffer_units, COUNT(buffer_units));
        fwrite(buffer_units, sizeof(uint16_t), COUNT(buffer_units), out);
        uint64_t units[] = {gdsii_real_from_double(precision / unit),
                            gdsii_real_from_double(precision)};
        swap64(units, COUNT(units));
        fwrite(units, sizeof(uint64_t), COUNT(units), out);
    }

    void write_cell(Cell& cell) const {
        cell.to_gds(out, unit / precision, max_points, precision, &timestamp);
    }

    void write_rawcell(RawCell& rawcell) const { rawcell.to_gds(out); }

    void close() {
        uint16_t buffer_end[] = {4, 0x0400};
        swap16(buffer_end, COUNT(buffer_end));
        fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
        fclose(out);
    }
};

}  // namespace gdstk

#endif
