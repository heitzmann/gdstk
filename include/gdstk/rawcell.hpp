/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_RAWCELL
#define GDSTK_HEADER_RAWCELL

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "array.hpp"
#include "map.hpp"
#include "utils.hpp"

namespace gdstk {

struct RawSource {
    FILE* file;
    uint32_t uses;

    // Read num_bytes into buffer from fd starting at offset
    int64_t offset_read(void* buffer, uint64_t num_bytes, uint64_t offset) const {
#ifdef _WIN32
        // The POSIX version (pread) does not change the file cursor, this
        // does.  Furthermore, this is not thread-safe!
        FSEEK64(file, offset, SEEK_SET);
        return fread(buffer, 1, num_bytes, file);
#else
        return pread(fileno(file), buffer, num_bytes, offset);
#endif
    };
};

// Rawcells are not meant to be created except through read_rawcells.  They are
// not explicitly loaded in memory until used, so the GDSII file where they are
// loaded from, remains open until it is no longer needed.  That is done by
// reference counting.
struct RawCell {
    char* name;
    RawSource* source;
    union {
        uint8_t* data;
        uint64_t offset;
    };
    uint64_t size;
    Array<RawCell*> dependencies;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;

    void clear();

    // Append dependencies of this cell to result.  If recursive is true, also
    // includes dependencies of any dependencies recursively.
    void get_dependencies(bool recursive, Map<RawCell*>& result) const;

    // This function outputs the rawcell in the GDSII.  It is not supposed to
    // be called by the user.
    ErrorCode to_gds(FILE* out);
};

// Load a GDSII file and extract its cells as RawCell.
Map<RawCell*> read_rawcells(const char* filename, ErrorCode* error_code);

}  // namespace gdstk

#endif
