/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_PROPERTY
#define GDSTK_HEADER_PROPERTY

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include "utils.hpp"

namespace gdstk {

struct OasisStream;
struct OasisState;

// Properties (and their members) are assumed to always be allocated through
// allocate, allocate_clear, or reallocate.

enum struct PropertyType { UnsignedInteger, Integer, Real, String };

// Each property can hold a series of values, which are represented by a
// NULL-terminated linked list of PropertyValue
struct PropertyValue {
    PropertyType type;
    union {
        uint64_t unsigned_integer;
        int64_t integer;
        double real;
        struct {
            uint64_t count;
            uint8_t* bytes;
        };
    };
    PropertyValue* next;
};

// Properties are stored as a NULL-terminated linked list.  Their name is a
// NULL-terminated string.  Property names in the OASIS should be strings of
// characters within the range 0x21 and 0x7E with at least 1 character.
struct Property {
    char* name;
    PropertyValue* value;
    Property* next;
};

void properties_print(Property* properties);
void properties_clear(Property*& properties);
Property* properties_copy(const Property* properties);

// property_values_copy is used in the OASIS reader; it is not intended to be
// used elsewhere.
PropertyValue* property_values_copy(const PropertyValue* values);

// The set_property functions add values to the first existing property with
// the given name.  A new one is created if none exists or create_new == true.
// New values are added to the start of the value list, so in the OASIS file,
// they will appear in reverse of the insertion order (last added will appear
// first).  The NULL byte at the end of string is NOT included in the property
// value.
void set_property(Property*& properties, const char* name, uint64_t unsigned_integer,
                  bool create_new);
void set_property(Property*& properties, const char* name, int64_t integer, bool create_new);
void set_property(Property*& properties, const char* name, double real, bool create_new);
void set_property(Property*& properties, const char* name, const char* string, bool create_new);
void set_property(Property*& properties, const char* name, const uint8_t* bytes, uint64_t count,
                  bool create_new);

// Overwrite properties with the same attribute number. The NULL byte is
// included in the property value.
void set_gds_property(Property*& properties, uint16_t attribute, const char* value, uint64_t count);

uint64_t remove_property(Property*& properties, const char* name, bool all_occurences);
bool remove_gds_property(Property*& properties, uint16_t attribute);

PropertyValue* get_property(Property* properties, const char* name);
PropertyValue* get_gds_property(Property* properties, uint16_t attribute);

// These functions output the properties in the GDSII and OASIS formats.  They
// are not supposed to be called by the user.
ErrorCode properties_to_gds(const Property* properties, FILE* out);
ErrorCode properties_to_oas(const Property* properties, OasisStream& out, OasisState& state);

}  // namespace gdstk

#endif
