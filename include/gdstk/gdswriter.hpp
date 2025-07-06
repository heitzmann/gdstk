/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_GDSWRITER
#define GDSTK_HEADER_GDSWRITER

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdio.h>

#include "cell.hpp"
#include "rawcell.hpp"

namespace gdstk {

// Struct used to write GDSII files incrementally, so that not all cells need
// to be held in memory simultaneously.  It should not be created manually, but
// through gdswriter_init.  Once created, use write_cell and write_rawcell to
// output all layout cells as needed, then call close to finalize the GDSII
// file.
struct GdsWriter {
    FILE* out;
    double unit;
    double precision;
    uint64_t max_points;
    tm timestamp;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    ErrorCode write_cell(Cell& cell) const {
        return cell.to_gds(out, unit / precision, max_points, precision, &timestamp);
    }

    ErrorCode write_rawcell(RawCell& rawcell) const { return rawcell.to_gds(out); }

    void close() {
        uint16_t buffer_end[] = {4, 0x0400};
        big_endian_swap16(buffer_end, COUNT(buffer_end));
        fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
        fclose(out);
    }
};

// Create a GDSII file with the given filename for output and write the header
// information to the file.  If max_points > 4, polygons written from
// write_cell will be fractured to max_points.  GDSII files include a
// timestamp, which can be specified by the caller or left NULL, in which case
// the current time will be used.  If not NULL, any errors will be reported
// through error_code.
inline GdsWriter gdswriter_init(const char* filename, const char* library_name, double unit,
                                double precision, uint64_t max_points, tm* timestamp,
                                ErrorCode* error_code) {
    GdsWriter result = {NULL, unit, precision, max_points};

    if (timestamp) {
        result.timestamp = *timestamp;
    } else {
        get_now(result.timestamp);
    }

    result.out = fopen(filename, "wb");
    if (result.out == NULL) {
        fputs("[GDSTK] Unable to open GDSII file for output.\n", error_logger);
        if (error_code) *error_code = ErrorCode::OutputFileOpenError;
        return result;
    }

    uint64_t len = strlen(library_name);
    if (len % 2) len++;
    uint16_t buffer_start[] = {6,
                               0x0002,
                               0x0258,
                               28,
                               0x0102,
                               (uint16_t)(result.timestamp.tm_year),
                               (uint16_t)(result.timestamp.tm_mon + 1),
                               (uint16_t)result.timestamp.tm_mday,
                               (uint16_t)result.timestamp.tm_hour,
                               (uint16_t)result.timestamp.tm_min,
                               (uint16_t)result.timestamp.tm_sec,
                               (uint16_t)(result.timestamp.tm_year),
                               (uint16_t)(result.timestamp.tm_mon + 1),
                               (uint16_t)result.timestamp.tm_mday,
                               (uint16_t)result.timestamp.tm_hour,
                               (uint16_t)result.timestamp.tm_min,
                               (uint16_t)result.timestamp.tm_sec,
                               (uint16_t)(4 + len),
                               0x0206};
    big_endian_swap16(buffer_start, COUNT(buffer_start));
    fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), result.out);
    fwrite(library_name, 1, len, result.out);

    uint16_t buffer_units[] = {20, 0x0305};
    big_endian_swap16(buffer_units, COUNT(buffer_units));
    fwrite(buffer_units, sizeof(uint16_t), COUNT(buffer_units), result.out);
    uint64_t units[] = {gdsii_real_from_double(precision / unit),
                        gdsii_real_from_double(precision)};
    big_endian_swap64(units, COUNT(units));
    fwrite(units, sizeof(uint64_t), COUNT(units), result.out);
    return result;
}

}  // namespace gdstk

#endif
