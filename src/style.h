/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_STYLE
#define GDSTK_HEADER_STYLE

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include <stdint.h>

namespace gdstk {

struct Style {
    uint32_t layer;
    uint32_t type;
    char* value;
    Style* next;
};

struct StyleMap {
    uint64_t capacity;
    uint64_t count;
    Style* style;

    void clear();
    void copy_from(const StyleMap& map);
    void resize(uint64_t new_capacity);
    void set(uint32_t layer, uint32_t type, const char* value);
    const char* get(uint32_t layer, uint32_t type) const;
    void del(uint32_t layer, uint32_t type);
    Style* next(const Style* current) const;
};

}  // namespace gdstk

#endif
