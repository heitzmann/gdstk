/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_RAWSOURCE
#define GDSTK_HEADER_RAWSOURCE

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace gdstk {

struct RawSource {
    FILE* file;
    uint32_t uses;

    // Read num_bytes into buffer from fd starting at offset
    int64_t offset_read(void* buffer, uint64_t num_bytes, uint64_t offset) const {
#ifdef _WIN32
        // The POSIX version (pread) does not change the file cursor, this does.
        // Furthermore, this is not thread-safe!
        _fseeki64(file, offset, SEEK_SET);
        return fread(buffer, 1, num_bytes, file);
#else
        return pread(fileno(file), buffer, num_bytes, offset);
#endif
    };
};

}  // namespace gdstk

#endif

