/*
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

#include "oasis.hpp"
#include "utils.hpp"

namespace gdstk {

// OASIS LAYERNAME record types (spec section 19):
//   Record type 11 maps (layer, datatype) ranges to a layer name.
//   Record type 12 maps (textlayer, texttype) ranges to a layer name.
enum struct LayerNameType : uint8_t {
    DATA = static_cast<uint8_t>(OasisRecord::LAYERNAME_DATA),  // 11
    TEXT = static_cast<uint8_t>(OasisRecord::LAYERNAME_TEXT),   // 12
};

// OASIS interval for LAYERNAME records (spec section 19.4, Table 13).
// Uses OasisInterval enum from oasis.hpp:
//   AllValues  (0): 0 to infinity (no bounds)
//   UpperBound (1): 0 to bound_a
//   LowerBound (2): bound_a to infinity
//   SingleValue(3): exactly bound_a
//   Bounded    (4): bound_a to bound_b
struct LayerNameInterval {
    OasisInterval type;
    uint64_t bound_a;  // Used by types 1, 2, 3, 4
    uint64_t bound_b;  // Used only by type 4 (Bounded)
};

// A single LAYERNAME record from an OASIS file.
// Multiple records can share the same name; the complete mapping for
// a layer name is the union of all associated intervals (spec 19.5).
struct LayerName {
    LayerNameType type;  // DATA (11) or TEXT (12)
    char* name;          // NULL-terminated layer name string (n-string)
    LayerNameInterval layer_interval;
    LayerNameInterval type_interval;  // datatype or texttype interval
};

// Free all LayerName entries in the array (freeing name strings) and clear it.
void layernames_clear(Array<LayerName>& layer_names);

// Write all LayerName records to an OASIS stream.
void layernames_to_oas(const Array<LayerName>& layer_names, OasisStream& out);

// Read a single LAYERNAME record from an OASIS stream and append to array.
// record_type must be LAYERNAME_DATA or LAYERNAME_TEXT.
void layername_from_oas(OasisRecord record_type, OasisStream& in, Array<LayerName>& layer_names);

}  // namespace gdstk

#endif
