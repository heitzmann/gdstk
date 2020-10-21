/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __RAWSOURCE_H__
#define __RAWSOURCE_H__

#include <cstdint>

// TODO: Windows, OSX, BSD, support.
#include <unistd.h>

namespace gdstk {

struct RawSource {
    FILE* file;
    uint32_t uses;

    // Read num_bytes into buffer from fd starting at offset
    ssize_t offset_read(void* buffer, size_t num_bytes, off_t offset) const {
        return pread(fileno(file), buffer, num_bytes, offset);
    };
};

}  // namespace gdstk

#endif

