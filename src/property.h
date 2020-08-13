/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __PROPERTY_H__
#define __PROPERTY_H__

#include <cstdint>
#include <cstdio>

namespace gdstk {

struct Property {
    int16_t key;
    char* value;
    Property* next;
};

void properties_clear(Property* properties);
Property* properties_copy(const Property* properties);
void set_property(Property*& properties, int16_t key, const char* value);
const char* get_property(const Property* properties, int16_t key);
void delete_property(Property*& properties, int16_t key);
void properties_to_gds(const Property* properties, FILE* out);

}  // namespace gdstk

#endif
