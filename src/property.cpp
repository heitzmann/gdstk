/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "property.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "allocator.h"
#include "utils.h"

namespace gdstk {

const char gds_property_name[] = "S_GDS_PROPERTY";

void properties_print(Property* properties) {
    if (!properties) return;
    puts("Properties:");
    for (; properties; properties = properties->next) {
        printf("- %s:", properties->name);
        for (PropertyValue* value = properties->value; value; value = value->next) {
            switch (value->type) {
                case PropertyType::UnsignedInteger:
                    printf(" %" PRIu64, value->unsigned_integer);
                    break;
                case PropertyType::Integer:
                    printf(" %" PRId64, value->integer);
                    break;
                case PropertyType::Real:
                    printf(" %lg", value->real);
                    break;
                case PropertyType::String: {
                    putchar(' ');
                    uint8_t* c = value->bytes;
                    for (uint64_t i = 0; i < value->size; i++, c++)
                        if (*c >= 0x20 && *c < 0x7f)
                            putchar(*c);
                        else
                            printf("[%02x]", *c);
                }
            }
        }
        putchar('\n');
    }
}

static bool is_gds_property(const Property* property) {
    if (strcmp(property->name, gds_property_name) != 0 || property->value == NULL) return false;
    PropertyValue* attribute = property->value;
    PropertyValue* value = attribute->next;
    if (attribute->type != PropertyType::UnsignedInteger || value == NULL ||
        value->type != PropertyType::String)
        return false;
    return true;
}

static void property_values_clear(PropertyValue* values) {
    while (values) {
        if (values->type == PropertyType::String) {
            free_allocation(values->bytes);
        }
        PropertyValue* next_value = values->next;
        free_allocation(values);
        values = next_value;
    }
}

void properties_clear(Property*& properties) {
    while (properties) {
        property_values_clear(properties->value);
        Property* next = properties->next;
        free_allocation(properties);
        properties = next;
    }
}

PropertyValue* property_values_copy(const PropertyValue* values) {
    PropertyValue* result = NULL;
    PropertyValue* dst;
    for (; values; values = values->next) {
        if (result == NULL) {
            result = (PropertyValue*)allocate(sizeof(PropertyValue));
            dst = result;
        } else {
            dst->next = (PropertyValue*)allocate(sizeof(PropertyValue));
            dst = dst->next;
        }
        dst->type = values->type;
        switch (values->type) {
            case PropertyType::UnsignedInteger:
                dst->unsigned_integer = values->unsigned_integer;
                break;
            case PropertyType::Integer:
                dst->integer = values->integer;
                break;
            case PropertyType::Real:
                dst->real = values->real;
                break;
            case PropertyType::String: {
                dst->size = values->size;
                dst->bytes = (uint8_t*)allocate(sizeof(uint8_t) * dst->size);
                memcpy(dst->bytes, values->bytes, dst->size);
            }
        }
        dst->next = NULL;
    }
    return result;
}

Property* properties_copy(const Property* properties) {
    Property* result = NULL;
    Property* dst;
    for (; properties; properties = properties->next) {
        if (result == NULL) {
            result = (Property*)allocate(sizeof(Property));
            dst = result;
        } else {
            dst->next = (Property*)allocate(sizeof(Property));
            dst = dst->next;
        }
        uint64_t len = strlen(properties->name) + 1;
        dst->name = (char*)allocate(sizeof(char) * len);
        memcpy(dst->name, properties->name, len);
        dst->value = property_values_copy(properties->value);
        dst->next = NULL;
    }
    return result;
}

static PropertyValue* get_or_add_property(Property*& properties, const char* name,
                                          bool create_new) {
    if (!create_new) {
        Property* property = properties;
        while (property && strcmp(property->name, name) != 0) property = property->next;
        if (property) {
            PropertyValue* value = (PropertyValue*)allocate_clear(sizeof(PropertyValue));
            value->next = property->value;
            property->value = value;
            return value;
        }
    }
    Property* property = (Property*)allocate(sizeof(Property));
    property->next = properties;
    properties = property;

    uint64_t len = strlen(name) + 1;
    property->name = (char*)allocate(sizeof(char) * len);
    memcpy(property->name, name, len);
    properties->value = (PropertyValue*)allocate_clear(sizeof(PropertyValue));
    return properties->value;
}

void set_property(Property*& properties, const char* name, uint64_t unsigned_integer,
                  bool create_new) {
    PropertyValue* value = get_or_add_property(properties, name, create_new);
    value->type = PropertyType::UnsignedInteger;
    value->unsigned_integer = unsigned_integer;
}

void set_property(Property*& properties, const char* name, int64_t integer, bool create_new) {
    PropertyValue* value = get_or_add_property(properties, name, create_new);
    value->type = PropertyType::Integer;
    value->integer = integer;
}

void set_property(Property*& properties, const char* name, double real, bool create_new) {
    PropertyValue* value = get_or_add_property(properties, name, create_new);
    value->type = PropertyType::Real;
    value->real = real;
}

void set_property(Property*& properties, const char* name, const char* string, bool create_new) {
    PropertyValue* value = get_or_add_property(properties, name, create_new);
    value->type = PropertyType::String;
    value->size = strlen(string);
    value->bytes = (uint8_t*)allocate(sizeof(uint8_t) * value->size);
    memcpy(value->bytes, string, value->size);
}

void set_property(Property*& properties, const char* name, const uint8_t* bytes, uint64_t size,
                  bool create_new) {
    PropertyValue* value = get_or_add_property(properties, name, create_new);
    value->type = PropertyType::String;
    value->size = size;
    value->bytes = (uint8_t*)allocate(sizeof(uint8_t) * size);
    memcpy(value->bytes, bytes, size);
}

void set_gds_property(Property*& properties, uint16_t attribute, const char* value) {
    PropertyValue* gds_attribute;
    PropertyValue* gds_value;
    Property* property = properties;
    for (; property; property = property->next) {
        if (is_gds_property(property) && property->value->unsigned_integer == attribute) {
            gds_value = property->value->next;
            gds_value->size = strlen(value) + 1;
            gds_value->bytes =
                (uint8_t*)reallocate(gds_value->bytes, sizeof(uint8_t) * gds_value->size);
            memcpy(gds_value->bytes, value, gds_value->size);
            return;
        }
    }
    gds_attribute = (PropertyValue*)allocate(sizeof(PropertyValue));
    gds_value = (PropertyValue*)allocate(sizeof(PropertyValue));
    gds_attribute->type = PropertyType::UnsignedInteger;
    gds_attribute->unsigned_integer = attribute;
    gds_attribute->next = gds_value;
    gds_value->type = PropertyType::String;
    gds_value->size = strlen(value) + 1;
    gds_value->bytes = (uint8_t*)allocate(sizeof(uint8_t) * gds_value->size);
    memcpy(gds_value->bytes, value, gds_value->size);
    gds_value->next = NULL;
    property = (Property*)allocate(sizeof(Property));
    property->name = (char*)allocate(sizeof(char) * COUNT(gds_property_name));
    memcpy(property->name, gds_property_name, COUNT(gds_property_name));
    property->value = gds_attribute;
    property->next = properties;
    properties = property;
}

bool remove_property(Property*& properties, const char* name) {
    if (properties == NULL) return false;
    if (strcmp(properties->name, name) == 0) {
        property_values_clear(properties->value);
        Property* next = properties->next;
        free_allocation(properties);
        properties = next;
        return true;
    }
    Property* property = properties;
    while (property->next && strcmp(property->next->name, name) != 0) property = property->next;
    if (property->next) {
        Property* rem = property->next;
        property_values_clear(rem->value);
        property->next = rem->next;
        free_allocation(rem);
        return true;
    }
    return false;
}

bool remove_gds_property(Property*& properties, uint16_t attribute) {
    if (properties == NULL) return false;
    if (is_gds_property(properties) && properties->value->unsigned_integer == attribute) {
        property_values_clear(properties->value);
        Property* next = properties->next;
        free_allocation(properties);
        properties = next;
        return true;
    }
    Property* property = properties;
    while (property->next &&
           (!is_gds_property(property->next) || property->value->unsigned_integer != attribute))
        property = property->next;
    if (property->next) {
        Property* rem = property->next;
        property_values_clear(rem->value);
        property->next = rem->next;
        free_allocation(rem);
        return true;
    }
    return false;
}

PropertyValue* get_property(Property* properties, const char* name) {
    while (properties && strcmp(properties->name, name) != 0) properties = properties->next;
    if (properties) return properties->value;
    return NULL;
}

PropertyValue* get_gds_property(Property* properties, uint16_t attribute) {
    while (properties &&
           (!is_gds_property(properties) || properties->value->unsigned_integer != attribute))
        properties = properties->next;
    if (properties) return properties->value->next;
    return NULL;
}

void properties_to_gds(const Property* properties, FILE* out) {
    uint64_t size = 0;
    for (; properties; properties = properties->next) {
        if (!is_gds_property(properties)) continue;
        PropertyValue* attribute = properties->value;
        PropertyValue* value = attribute->next;
        uint64_t len = value->size;
        uint8_t* bytes = value->bytes;
        bool free_bytes = false;
        if (len % 2) {
            if (bytes[len - 1] == 0) {
                len--;
            } else {
                free_bytes = true;
                bytes = (uint8_t*)allocate(sizeof(uint8_t) * ++len);
                memcpy(bytes, value->bytes, len - 1);
                bytes[len - 1] = 0;
            }
        }

        uint16_t buffer_prop[] = {6, 0x2B02, (uint16_t)attribute->unsigned_integer,
                                  (uint16_t)(4 + len), 0x2C06};
        size += len;
        big_endian_swap16(buffer_prop, COUNT(buffer_prop));
        fwrite(buffer_prop, sizeof(uint16_t), COUNT(buffer_prop), out);
        fwrite(bytes, sizeof(uint8_t), len, out);

        if (free_bytes) free_allocation(bytes);
    }
    if (size > 128)
        fputs(
            "[GDSTK] Properties with size larger than 128 bytes are not officially supported by the GDSII specification.  This file might not be compatible with all readers.\n",
            stderr);
}

}  // namespace gdstk
