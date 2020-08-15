/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define _USE_MATH_DEFINES

#ifndef __RAWCELL_H__
#define __RAWCELL_H__

#include <cstdint>
#include <cstdio>

#include "array.h"
#include "map.h"

namespace gdstk {

struct RawCell {
    char* name;
    uint8_t* data;
    int64_t size;
    Array<RawCell*> dependencies;
    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void* owner;

    void print(bool all) const;
    void clear();
    void get_dependencies(bool recursive, Array<RawCell*>& result) const;
    void to_gds(FILE* out) const;
};

Map<RawCell*> read_rawcells(FILE* in);

}  // namespace gdstk

#endif
