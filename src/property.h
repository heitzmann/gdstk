/*
Copyright 2020 Lucas Heitzmann Gabrielli.
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

// Properties (and their members) are assumed to always be allocated through allocate,
// allocate_clear, or reallocate.

extern const char gds_property_name[];

enum struct PropertyType { UnsignedInteger, Integer, Real, String };

struct PropertyValue {
    PropertyType type;
    union {
        uint64_t unsigned_integer;
        int64_t integer;
        double real;
        struct {
            uint64_t size;
            uint8_t* bytes;
        };
    };
    PropertyValue* next;
};

struct Property {
    char* name;
    PropertyValue* value;
    Property* next;
};

void properties_clear(Property* properties);
Property* properties_copy(const Property* properties);

// set_property functions add values to the first existing property with the given name.  A new
// one is created if none exists or create_new == true.  New values are added to the start of the
// value list, so in the OASIS file, they will appear in reverse of the insertion order (last added
// will first).
Property* set_property(Property* properties, const char* name, uint64_t unsigned_integer,
                       bool create_new);
Property* set_property(Property* properties, const char* name, int64_t integer, bool create_new);
Property* set_property(Property* properties, const char* name, double real, bool create_new);
Property* set_property(Property* properties, const char* name, const char* string, bool create_new);
Property* set_property(Property* properties, const char* name, const uint8_t* bytes, uint64_t size,
                       bool create_new);

// Overwrite properties with the same attribute number
Property* set_gds_property(Property* properties, uint16_t attribute, const char* value);

Property* remove_property(Property* properties, const char* name);
Property* remove_gds_property(Property* properties, uint16_t attribute);
PropertyValue* get_property(Property* properties, const char* name);
PropertyValue* get_gds_property(Property* properties, uint16_t attribute);
void properties_to_gds(const Property* properties, FILE* out);

}  // namespace gdstk

#endif
