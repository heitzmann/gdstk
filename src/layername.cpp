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



void set_layername(LayerName*& layer_names, const char* name, uint64_t layertype, uint64_t datatype) {
    Interval* layertype_interval = (Interval*)allocate_clear(sizeof(Interval)); 
    layertype_interval->type = OasisInterval::SingleValue;
    layertype_interval->bound = layertype;
    Interval* datatype_interval = (Interval*)allocate_clear(sizeof(Interval));
    datatype_interval->type = OasisInterval::SingleValue;
    datatype_interval->bound = datatype;
    set_layername(layer_names, name, layertype_interval, datatype_interval);
}
void set_textlayername(LayerName*& layer_names, const char* name, uint64_t layertype, uint64_t datatype) {
    Interval* layertype_interval = (Interval*)allocate_clear(sizeof(Interval)); 
    layertype_interval->type = OasisInterval::SingleValue;
    layertype_interval->bound = layertype;
    Interval* datatype_interval = (Interval*)allocate_clear(sizeof(Interval));
    datatype_interval->type = OasisInterval::SingleValue;
    datatype_interval->bound = datatype;
    set_textlayername(layer_names, name, layertype_interval, datatype_interval);
}

void set_layername(LayerName*& layer_names, const char* name, Interval* layertype, Interval* datatype) {
    LayerName* ln = (LayerName*)allocate_clear(sizeof(LayerName));
    ln->type = LayerNameType::DATA;
    ln->name = copy_string(name, NULL);
    ln->LayerInterval = layertype;
    ln->TypeInterval = datatype;
    ln->next = layer_names;
    layer_names = ln;
}
void set_textlayername(LayerName*& layer_names, const char* name, Interval* layertype, Interval* datatype) {
    LayerName* ln = (LayerName*)allocate_clear(sizeof(LayerName));
    ln->type = LayerNameType::TEXT;
    ln->name = copy_string(name, NULL);
    ln->LayerInterval = layertype;
    ln->TypeInterval = datatype;
    ln->next = layer_names;
    layer_names = ln;
}

void remove_layername(LayerName*& layer_names, const char* target_name, bool all_occurences) {
    if (layer_names == NULL) return;
    LayerName* current = layer_names;
    LayerName* previous = NULL;
    while (current) {
        if (strcmp(current->name, target_name) == 0) {
            
            if (previous) {
                previous->next = current->next;
            } else {
                layer_names = current->next; // Update head if needed
            }
            free_allocation(current->name);
            free_allocation(current->LayerInterval);
            free_allocation(current->TypeInterval);
            LayerName* to_delete = current;
            current = current->next; // Move to next before deleting
            free_allocation(to_delete);
            if (!all_occurences) {
                return; // Exit if only the first occurrence should be removed
            }
        } else {
            previous = current;
            current = current->next;
        }
    }
}
// this is suspect as it only returns the first match
// I may have to create a new data structure to hold multiple matches
LayerName* get_mapped_layers(LayerName* layer_names, const char* name) {
    while (layer_names && strcmp(layer_names->name, name) != 0) layer_names = layer_names->next;
    return layer_names;
}

bool layername_contains_layer_and_type(const LayerName* layer_name, uint64_t layertype, uint64_t datatype) {
    bool matches = false;
    

    // Check layertype
    switch (layer_name->LayerInterval->type) {
        case OasisInterval::AllValues:
            matches = true;
            break;
        case OasisInterval::UpperBound:
            matches = (layertype <= layer_name->LayerInterval->bound);
            break;
        case OasisInterval::LowerBound:
            matches = (layertype >= layer_name->LayerInterval->bound);
            break;  
        case OasisInterval::SingleValue:
            matches = (layer_name->LayerInterval->bound == layertype);
            break;
        case OasisInterval::Bounded:
            matches = (layertype >= layer_name->LayerInterval->bound_a &&
                                    layertype <= layer_name->LayerInterval->bound_b);
            break;
    }
    if (matches == false) {
        return false;
    }
    switch (layer_name->TypeInterval->type) {
        case OasisInterval::AllValues:
            matches = true;
            break;
        case OasisInterval::UpperBound:
            matches = (datatype <= layer_name->TypeInterval->bound);
            break;
        case OasisInterval::LowerBound:
            matches = (datatype >= layer_name->TypeInterval->bound);
            break;  
        case OasisInterval::SingleValue:
            matches = (layer_name->TypeInterval->bound == datatype);
            break;
        case OasisInterval::Bounded:
            matches = (layertype >= layer_name->TypeInterval->bound_a &&
                        layertype <= layer_name->TypeInterval->bound_b);
            break;
    }
    return matches;
}

LayerName* get_names_from_layers(LayerName* layer_names, uint64_t layertype, uint64_t datatype) {
    LayerName* result_head = NULL;
    LayerName* result_tail = NULL;

    while (layer_names) {
        // this might not need a deep copy since it just returning a result 
        // the dealocaiton might be messy though
        if (layername_contains_layer_and_type(layer_names, layertype, datatype)) {
            // Create a new LayerName node for the result list
            LayerName* new_node = (LayerName*)allocate_clear(sizeof(LayerName));
            new_node->type = layer_names->type;
            new_node->name = copy_string(layer_names->name, NULL);
            new_node->LayerInterval = (Interval*)allocate_clear(sizeof(Interval));
            *(new_node->LayerInterval) = *(layer_names->LayerInterval); // Deep copy
            new_node->TypeInterval = (Interval*)allocate_clear(sizeof(Interval));
            *(new_node->TypeInterval) = *(layer_names->TypeInterval); // Deep copy
            new_node->next = NULL;

            // Append to the result list
            if (result_tail) {
                result_tail->next = new_node;
                result_tail = new_node;
            } else {
                result_head = new_node;
                result_tail = new_node;
            }
        }
    }
    return result_head;
}
        

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


ErrorCode layernames_to_oas(const LayerName* layer_names, OasisStream& out, OasisState& state) {
    while (layer_names) {
        oasis_write_integer(out,(uint8_t)layer_names->type);
        size_t name_length = strlen(layer_names->name); 
        oasis_write_unsigned_integer(out, (uint64_t)name_length);
        oasis_write(layer_names->name, 1, name_length, out);
        
        // write layertype interval
        oasis_putc((uint8_t)layer_names->LayerInterval->type, out);
        if (layer_names->LayerInterval->type == OasisInterval::Bounded) {
            oasis_write_unsigned_integer(out, layer_names->LayerInterval->bound_a);
            oasis_write_unsigned_integer(out, layer_names->LayerInterval->bound_b);
        } else if (layer_names->LayerInterval->type != OasisInterval::AllValues) {
            oasis_write_unsigned_integer(out, layer_names->LayerInterval->bound);
        }

        oasis_putc((uint8_t)layer_names->TypeInterval->type, out);
        if (layer_names->TypeInterval->type == OasisInterval::Bounded) {
            oasis_write_unsigned_integer(out, layer_names->TypeInterval->bound_a);
            oasis_write_unsigned_integer(out, layer_names->TypeInterval->bound_b);
        } else if (layer_names->TypeInterval->type != OasisInterval::AllValues) {
            oasis_write_unsigned_integer(out, layer_names->TypeInterval->bound);
        }
        
        layer_names = layer_names->next;
        
    }
    return ErrorCode::NoError;
}

}  // namespace gdstk

