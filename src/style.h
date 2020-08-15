/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __STYLE_H__
#define __STYLE_H__

// FNV-1a hash function (32 bits)
#define STYLE_FNV_PRIME 0x01000193
#define STYLE_FNV_OFFSET 0x811c9dc5
#define HASH2(a, b)                                       \
    ((uint32_t)(STYLE_FNV_PRIME *                         \
                ((((b)&0x00FF) >> 8) ^                    \
                 (STYLE_FNV_PRIME *                       \
                  (((b)&0x00FF) ^ (STYLE_FNV_PRIME *      \
                                   ((((a)&0xFF00) >> 8) ^ \
                                    (STYLE_FNV_PRIME * (((a)&0x00FF) ^ STYLE_FNV_OFFSET)))))))))

#include <cstdint>

namespace gdstk {

struct Style {
    int16_t layer;
    int16_t type;
    char* value;
    Style* next;
};

struct StyleMap {
    int64_t capacity;
    int64_t size;
    Style* style;

    void clear();
    void copy_from(const StyleMap& map);
    void resize(int64_t new_capacity);
    void set(int16_t layer, int16_t type, const char* value);
    const char* get(int16_t layer, int16_t type) const;
    void del(int16_t layer, int16_t type);
    Style* next(const Style* current) const;
};

}  // namespace gdstk

#endif
