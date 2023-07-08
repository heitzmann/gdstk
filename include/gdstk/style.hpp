/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_STYLE
#define GDSTK_HEADER_STYLE

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>

#include "utils.hpp"

namespace gdstk {

// Style used in SVG output. Value is the SVG style to be applied to elements
// with the given tag, e.g., "stroke: #D04030; fill: #D89080;"
struct Style {
    Tag tag;
    char* value;
};

// Hash map of styles indexed by tag
struct StyleMap {
    uint64_t capacity;  // allocated capacity
    uint64_t count;     // number of items in the map
    Style* items;       // array with length capacity

    void print(bool all) const;

    // Deallocates the whole map, including its contents
    void clear();

    // The instance should be zeroed before using copy_from
    void copy_from(const StyleMap& map);

    // Internal use
    void resize(uint64_t new_capacity);
    Style* get_slot(Tag tag) const;

    // Value is internally allocated and copied
    void set(Tag tag, const char* value);
    const char* get(Tag tag) const;
    bool del(Tag tag);

    // Function to iterate over all values in the map:
    // for (Style* style = style_map.next(NULL); style;
    //      style = style_map.next(style)) {…}
    Style* next(const Style* current) const;
};

}  // namespace gdstk

#endif
