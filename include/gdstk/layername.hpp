/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_LAYERNAME
#define GDSTK_HEADER_LAYERNAME

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include "utils.hpp"
// #include "oasis.hpp"

namespace gdstk {

struct OasisStream;
struct OasisState;

// based roughly on property.hpp, please see that file for someone who knows what they're doing

// the spec lists tag types 11 and 12 which are representing layer and textlayer data respectively
enum struct LayerNameType : uint8_t { 
    DATA = static_cast<uint8_t>(OasisRecord::LAYERNAME_DATA),
    TEXT = static_cast<uint8_t>(OasisRecord::LAYERNAME_TEXT),
};
// only bound type index 4 (bound to bound) requires two bound values
struct Interval {
    OasisInterval type;
    union {
        uint64_t bound;
        struct {
            uint64_t bound_a;
            uint64_t bound_b;
        };
    };
};

// Properties are stored as a NULL-terminated linked list.  Their name is a
// NULL-terminated string.  Property names in the OASIS should be strings of
// characters within the range 0x21 and 0x7E with at least 1 character.
struct LayerName {
    LayerNameType type;
    char* name;
    Interval* LayerInterval;
    Interval* TypeInterval;
    LayerName* next;
};

// The set_layername functions add layer mappings to the library with the 
// given name.  A new one is created if none exists or create_new == true.
// New values are added to the start of the value list, so in the OASIS file,
// they will appear in reverse of the insertion order (last added will appear
// first).  The NULL byte at the end of string is NOT included in the layer
// name.
void set_layername(LayerName*& layer_names, const char* name, uint64_t layertype, uint64_t datatype);
void set_textlayername(LayerName*& layer_names, const char* name, uint64_t layertype, uint64_t datatype);
void set_layername(LayerName*& layer_names, const char* name, Interval* layertype, Interval* datatype);
void set_textlayername(LayerName*& layer_names, const char* name, Interval* layertype, Interval* datatype);
// Still thinking about how to deal with bounds that aren't type 3

// removes first instance of layername present in the linked list or all occourances if
// all occourences is set
void remove_layername(LayerName*& layer_names, const char* name, bool all_occurences);

LayerName* get_mapped_layers(LayerName* layer_names, const char* name);

LayerName* get_names_from_layers(LayerName* layer_names, uint64_t layertype, uint64_t datatype);

void layernames_clear(LayerName*& layer_names);

// These functions output the properties in the OASIS format.  They
// are not supposed to be called by the user.
ErrorCode layernames_to_oas(const LayerName* layer_names, OasisStream& out, OasisState& state);

}  // namespace gdstk

#endif
