/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gdstk/allocator.hpp>
#include <gdstk/map.hpp>
#include <gdstk/oasis.hpp>
#include <gdstk/property.hpp>
#include <gdstk/utils.hpp>

namespace gdstk {

// Library level
const char s_max_int_size_property_name[] = "S_MAX_SIGNED_INTEGER_WIDTH";
const char s_max_uint_size_property_name[] = "S_MAX_UNSIGNED_INTEGER_WIDTH";
const char s_max_string_size_property_name[] = "S_MAX_STRING_LENGTH";
const char s_max_polygon_property_name[] = "S_POLYGON_MAX_VERTICES";
const char s_max_path_property_name[] = "S_PATH_MAX_VERTICES";
const char s_top_level_property_name[] = "S_TOP_CELL";
const char s_bounding_box_available_property_name[] = "S_BOUNDING_BOXES_AVAILABLE";

// Cell level
const char s_bounding_box_property_name[] = "S_BOUNDING_BOX";
const char s_cell_offset_property_name[] = "S_CELL_OFFSET";

// Element level
const char s_gds_property_name[] = "S_GDS_PROPERTY";

void properties_print(Property* properties) {
    if (!properties) return;
    puts("Properties:");
    for (; properties; properties = properties->next) {
        printf("- <%p> %s:", properties, properties->name);
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
                    for (uint64_t i = 0; i < value->count; i++, c++)
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
    if (strcmp(property->name, s_gds_property_name) != 0 || property->value == NULL) return false;
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
        free_allocation(properties->name);
        Property* next = properties->next;
        free_allocation(properties);
        properties = next;
    }
}

PropertyValue* property_values_copy(const PropertyValue* values) {
    PropertyValue* result = NULL;
    PropertyValue* dst = NULL;
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
                dst->count = values->count;
                dst->bytes = (uint8_t*)allocate(dst->count);
                memcpy(dst->bytes, values->bytes, dst->count);
            }
        }
        dst->next = NULL;
    }
    return result;
}

Property* properties_copy(const Property* properties) {
    Property* result = NULL;
    Property* dst = NULL;
    for (; properties; properties = properties->next) {
        if (result == NULL) {
            result = (Property*)allocate(sizeof(Property));
            dst = result;
        } else {
            dst->next = (Property*)allocate(sizeof(Property));
            dst = dst->next;
        }
        dst->name = copy_string(properties->name, NULL);
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

    property->name = copy_string(name, NULL);
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
    value->count = strlen(string);
    value->bytes = (uint8_t*)allocate(value->count);
    memcpy(value->bytes, string, value->count);
}

void set_property(Property*& properties, const char* name, const uint8_t* bytes, uint64_t count,
                  bool create_new) {
    PropertyValue* value = get_or_add_property(properties, name, create_new);
    value->type = PropertyType::String;
    value->count = count;
    value->bytes = (uint8_t*)allocate(count);
    memcpy(value->bytes, bytes, count);
}

void set_gds_property(Property*& properties, uint16_t attribute, const char* value, uint64_t count) {
    PropertyValue* gds_attribute;
    PropertyValue* gds_value;
    Property* property = properties;
    for (; property; property = property->next) {
        if (is_gds_property(property) && property->value->unsigned_integer == attribute) {
            gds_value = property->value->next;
            gds_value->count = count;
            gds_value->bytes = (uint8_t*)reallocate(gds_value->bytes, count);
            memcpy(gds_value->bytes, value, count);
            return;
        }
    }
    gds_attribute = (PropertyValue*)allocate(sizeof(PropertyValue));
    gds_value = (PropertyValue*)allocate(sizeof(PropertyValue));
    gds_attribute->type = PropertyType::UnsignedInteger;
    gds_attribute->unsigned_integer = attribute;
    gds_attribute->next = gds_value;
    gds_value->type = PropertyType::String;
    gds_value->bytes = (uint8_t*)allocate(count);
    memcpy(gds_value->bytes, value, count);
    gds_value->count = count;
    gds_value->next = NULL;
    property = (Property*)allocate(sizeof(Property));
    property->name = (char*)allocate(COUNT(s_gds_property_name));
    memcpy(property->name, s_gds_property_name, COUNT(s_gds_property_name));
    property->value = gds_attribute;
    property->next = properties;
    properties = property;
}

uint64_t remove_property(Property*& properties, const char* name, bool all_occurences) {
    uint64_t removed = 0;
    if (properties == NULL) return removed;
    while (strcmp(properties->name, name) == 0) {
        property_values_clear(properties->value);
        free_allocation(properties->name);
        Property* next = properties->next;
        free_allocation(properties);
        properties = next;
        removed++;
        if (!all_occurences) return removed;
    }
    Property* property = properties;
    while (true) {
        while (property->next && strcmp(property->next->name, name) != 0) property = property->next;
        if (property->next) {
            Property* rem = property->next;
            property_values_clear(rem->value);
            free_allocation(rem->name);
            property->next = rem->next;
            free_allocation(rem);
            removed++;
            if (!all_occurences) return removed;
        } else {
            return removed;
        }
    }
}

bool remove_gds_property(Property*& properties, uint16_t attribute) {
    if (properties == NULL) return false;
    if (is_gds_property(properties) && properties->value->unsigned_integer == attribute) {
        property_values_clear(properties->value);
        free_allocation(properties->name);
        Property* next = properties->next;
        free_allocation(properties);
        properties = next;
        return true;
    }
    Property* property = properties;
    while (property->next &&
           (!is_gds_property(property->next) || property->next->value->unsigned_integer != attribute))
        property = property->next;
    if (property->next) {
        Property* rem = property->next;
        property_values_clear(rem->value);
        free_allocation(rem->name);
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

ErrorCode properties_to_gds(const Property* properties, FILE* out) {
    uint64_t count = 0;
    for (; properties; properties = properties->next) {
        if (!is_gds_property(properties)) continue;
        PropertyValue* attribute = properties->value;
        PropertyValue* value = attribute->next;
        uint64_t len = value->count;
        uint8_t* bytes = value->bytes;
        bool free_bytes = false;
        if (len % 2) {
            if (bytes[len - 1] == 0) {
                len--;
            } else {
                free_bytes = true;
                bytes = (uint8_t*)allocate(++len);
                memcpy(bytes, value->bytes, len - 1);
                bytes[len - 1] = 0;
            }
        }

        uint16_t buffer_prop[] = {6, 0x2B02, (uint16_t)attribute->unsigned_integer,
                                  (uint16_t)(4 + len), 0x2C06};
        count += len;
        big_endian_swap16(buffer_prop, COUNT(buffer_prop));
        fwrite(buffer_prop, sizeof(uint16_t), COUNT(buffer_prop), out);
        fwrite(bytes, 1, len, out);

        if (free_bytes) free_allocation(bytes);
    }
    if (count > 128) {
        if (error_logger)
            fputs(
                "[GDSTK] Properties with count larger than 128 bytes are not officially supported by the GDSII specification.  This file might not be compatible with all readers.\n",
                error_logger);
        return ErrorCode::UnofficialSpecification;
    }
    return ErrorCode::NoError;
}

ErrorCode properties_to_oas(const Property* properties, OasisStream& out, OasisState& state) {
    while (properties) {
        uint8_t info = 0x06;
        if (is_gds_property(properties)) info |= 0x01;

        uint64_t value_count = 0;
        for (PropertyValue* value = properties->value; value; value = value->next) value_count++;
        if (value_count > 14) {
            info |= 0xF0;
        } else {
            info |= (uint8_t)(value_count & 0x0F) << 4;
        }

        oasis_putc((int)OasisRecord::PROPERTY, out);
        oasis_putc(info, out);

        uint64_t index;
        if (state.property_name_map.has_key(properties->name)) {
            index = state.property_name_map.get(properties->name);
        } else {
            index = state.property_name_map.count;
            state.property_name_map.set(properties->name, index);
        }
        oasis_write_unsigned_integer(out, index);

        if (value_count > 14) {
            oasis_write_unsigned_integer(out, value_count);
        }

        for (PropertyValue* value = properties->value; value; value = value->next) {
            switch (value->type) {
                case PropertyType::Real:
                    oasis_write_real(out, value->real);
                    break;
                case PropertyType::UnsignedInteger:
                    oasis_putc(8, out);
                    oasis_write_unsigned_integer(out, value->unsigned_integer);
                    break;
                case PropertyType::Integer:
                    oasis_putc(9, out);
                    oasis_write_integer(out, value->integer);
                    break;
                case PropertyType::String: {
                    bool space = false;
                    bool binary = false;
                    uint8_t* byte = value->bytes;
                    for (uint64_t i = value->count; i > 0; i--, byte++) {
                        if (*byte < 0x20 || *byte > 0x7E) {
                            binary = true;
                            break;
                        } else if (*byte == 0x20) {
                            space = true;
                        }
                    }
                    if (binary) {
                        oasis_putc(14, out);
                    } else if (space) {
                        oasis_putc(13, out);
                    } else {
                        oasis_putc(15, out);
                    }
                    for (index = 0; index < state.property_value_array.count; index++) {
                        PropertyValue* it = state.property_value_array[index];
                        if (it->count == value->count &&
                            memcmp(it->bytes, value->bytes, it->count) == 0)
                            break;
                    }
                    if (index == state.property_value_array.count)
                        state.property_value_array.append(value);
                    oasis_write_unsigned_integer(out, index);
                }
            }
        }

        properties = properties->next;
    }
    return ErrorCode::NoError;
}

}  // namespace gdstk
