/*
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <gdstk/allocator.hpp>
#include <gdstk/layername.hpp>
#include <gdstk/oasis.hpp>
#include <gdstk/utils.hpp>

namespace gdstk {

void layernames_clear(Array<LayerName>& layer_names) {
    for (uint64_t i = 0; i < layer_names.count; i++) {
        if (layer_names[i].name) {
            free_allocation(layer_names[i].name);
            layer_names[i].name = NULL;
        }
    }
    layer_names.clear();
}

static void write_interval(OasisStream& out, const LayerNameInterval& interval) {
    oasis_write_unsigned_integer(out, (uint64_t)interval.type);
    switch (interval.type) {
        case OasisInterval::AllValues:
            // No bounds to write
            break;
        case OasisInterval::UpperBound:
        case OasisInterval::LowerBound:
        case OasisInterval::SingleValue:
            oasis_write_unsigned_integer(out, interval.bound_a);
            break;
        case OasisInterval::Bounded:
            oasis_write_unsigned_integer(out, interval.bound_a);
            oasis_write_unsigned_integer(out, interval.bound_b);
            break;
    }
}

static void read_interval(OasisStream& in, LayerNameInterval& interval) {
    uint64_t itype = oasis_read_unsigned_integer(in);
    interval.type = (OasisInterval)itype;
    interval.bound_a = 0;
    interval.bound_b = 0;
    switch (interval.type) {
        case OasisInterval::AllValues:
            break;
        case OasisInterval::UpperBound:
        case OasisInterval::LowerBound:
        case OasisInterval::SingleValue:
            interval.bound_a = oasis_read_unsigned_integer(in);
            break;
        case OasisInterval::Bounded:
            interval.bound_a = oasis_read_unsigned_integer(in);
            interval.bound_b = oasis_read_unsigned_integer(in);
            break;
    }
}

void layernames_to_oas(const Array<LayerName>& layer_names, OasisStream& out) {
    for (uint64_t i = 0; i < layer_names.count; i++) {
        const LayerName& ln = layer_names[i];
        // Record type byte: 11 for DATA, 12 for TEXT
        oasis_putc((int)ln.type, out);
        // layername-string (n-string): length + bytes
        uint64_t name_len = strlen(ln.name);
        oasis_write_unsigned_integer(out, name_len);
        oasis_write(ln.name, 1, name_len, out);
        // layer-interval (or textlayer-interval)
        write_interval(out, ln.layer_interval);
        // datatype-interval (or texttype-interval)
        write_interval(out, ln.type_interval);
    }
}

void layername_from_oas(OasisRecord record_type, OasisStream& in,
                        Array<LayerName>& layer_names) {
    LayerName ln = {};
    ln.type = (record_type == OasisRecord::LAYERNAME_DATA) ? LayerNameType::DATA
                                                           : LayerNameType::TEXT;
    uint64_t len;
    ln.name = (char*)oasis_read_string(in, true, len);
    read_interval(in, ln.layer_interval);
    read_interval(in, ln.type_interval);
    layer_names.append(ln);
}

}  // namespace gdstk
