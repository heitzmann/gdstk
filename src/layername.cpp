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
#include <gdstk/layername.hpp>
#include <gdstk/utils.hpp>
// #include "layername.hpp"

namespace gdstk {

// not sure what funciton properties print has other than diagnostic
// void properties_print(Property* properties) {
//     if (!properties) return;
//     puts("Properties:");
//     for (; properties; properties = properties->next) {
//         printf("- <%p> %s:", properties, properties->name);
//         for (PropertyValue* value = properties->value; value; value = value->next) {
//             switch (value->type) {
//                 case PropertyType::UnsignedInteger:
//                     printf(" %" PRIu64, value->unsigned_integer);
//                     break;
//                 case PropertyType::Integer:
//                     printf(" %" PRId64, value->integer);
//                     break;
//                 case PropertyType::Real:
//                     printf(" %lg", value->real);
//                     break;
//                 case PropertyType::String: {
//                     putchar(' ');
//                     uint8_t* c = value->bytes;
//                     for (uint64_t i = 0; i < value->count; i++, c++)
//                         if (*c >= 0x20 && *c < 0x7f)
//                             putchar(*c);
//                         else
//                             printf("[%02x]", *c);
//                 }
//             }
//         }
//         putchar('\n');
//     }
// }







// bool remove_gds_property(Property*& properties, uint16_t attribute) {
//     if (properties == NULL) return false;
//     if (is_gds_property(properties) && properties->value->unsigned_integer == attribute) {
//         property_values_clear(properties->value);
//         free_allocation(properties->name);
//         Property* next = properties->next;
//         free_allocation(properties);
//         properties = next;
//         return true;
//     }
//     Property* property = properties;
//     while (property->next &&
//            (!is_gds_property(property->next) || property->next->value->unsigned_integer != attribute))
//         property = property->next;
//     if (property->next) {
//         Property* rem = property->next;
//         property_values_clear(rem->value);
//         free_allocation(rem->name);
//         property->next = rem->next;
//         free_allocation(rem);
//         return true;
//     }
//     return false;
// }

// PropertyValue* get_property(Property* properties, const char* name) {
//     while (properties && strcmp(properties->name, name) != 0) properties = properties->next;
//     if (properties) return properties->value;
//     return NULL;
// }


// void set_layername(LayerName*& layer_names, const char* name, Interval layertype, Interval datatype){
//     LayerName

//                   }


void layernames_clear(LayerName*& layer_names) {
    while (layer_names){
        free_allocation(layer_names->name);
        free_allocation(layer_names->LayerInterval);
        free_allocation(layer_names->TypeInterval);
        LayerName* next = layer_names->next;
        free_allocation(layer_names);
        layer_names = next;
    }
}


ErrorCode layernames_to_oas(const Property* properties, OasisStream& out, OasisState& state) {
    // needs full rewrite
    while (properties) {
        uint8_t info = 0x06;
        // if (is_gds_property(properties)) info |= 0x01;

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

